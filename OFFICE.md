# 网易邮箱、飞书与原生 MCP

CLI 和飞书入口共用 `create_assistant_agent` / `AssistantSession.run_turn`，工具转换使用固定框架自带的 `langchain.mcp.MCPAdapter`。账号连接不经过 Skills；Skills 的依赖安装规则保持只新增。未配置 MCP 时不会导入 FastMCP、启动 Node 或连接办公服务。

## 安装与配置

保留 LangChain 1.4.7、LangChain Core 1.6.13、LangGraph 1.2.16。新增 `mcp` extra；Pydantic 2.14.1、FastMCP 4.0.11、FastAPI 0.162.0、unstructured-client 0.46.4 已固定。`constraints/office.lock` 记录本次 Linux / Python 3.12 完整验证环境，`constraints/rag.lock` 同步了已有 RAG 依赖版本。完整环境中原有 Docling 2.65 的 pypdfium2<5 限制与新 unstructured-client 冲突，验证环境已升级至 Docling 2.142.1；完整锁文件包含这组配套版本。没有新增对 Docling 的运行依赖。

```bash
# GPU 安装继续使用已有 RAG 文档中的 torch CUDA wheel 步骤。
.venv/bin/python -m pip install -e '.[mcp,sessions,skills,rag,openai]' -c constraints/office.lock
npm ci --prefix deploy/feishu --ignore-scripts
.venv/bin/python -m pip check
```

Node >=20；运行时直接使用本地 0.5.1 官方 MCP 和 1.76.0 Node SDK，pg 固定 8.23.1。`--ignore-scripts` 避免编译可选的系统钥匙串组件；OAuth 令牌由 Python 私有存储管理，官方 MCP 不启用内置 OAuth。Node 程序只接收所需配置，密钥不放在进程命令行，不继承模型或数据库等无关凭据。

参考 `assistant.office.example.toml` 将所需配置合并到自己的私有配置，不要覆盖已有配置。支持 163.com、126.com、yeah.net 的客户端授权码。邮箱目录映射使用服务返回的真实目录名，不猜测中文目录。首先通过 `folders` 查看目录。

```bash
.venv/bin/python -m ai_assistant config check --config assistant.toml
.venv/bin/python -m ai_assistant mcp check --config assistant.toml
# 显式幂等迁移：包含原生 checkpoint、会话索引和集成业务表。
.venv/bin/python -m ai_assistant mcp init --config assistant.toml
.venv/bin/python -m ai_assistant mcp auth feishu login --config assistant.toml
.venv/bin/python -m ai_assistant mcp auth feishu status --config assistant.toml
# 只有 --network 才执行账号认证和只读连通检查。
.venv/bin/python -m ai_assistant mcp check --network --config assistant.toml
.venv/bin/python -m ai_assistant mcp list --config assistant.toml
.venv/bin/python -m ai_assistant mcp inspect feishu_bot --config assistant.toml
.venv/bin/python -m ai_assistant mcp tools feishu_bot --config assistant.toml
.venv/bin/python -m ai_assistant chat --config assistant.toml
```

OAuth 只从本机命令发起，打开终端打印的授权地址。回调地址必须是带显式端口的 loopback HTTP，并与应用后台配置一致。授权完成后核对 `open_id` 是否为配置主人。私有状态目录必须在工作区之外，目录 0700、文件 0600；跨进程 flock 保护刷新和原子替换。刷新前持久化不确定状态；若响应丢失，要求重新登录，不重复使用可能已经轮换的 refresh token。`logout` 清除本机令牌；远端授权撤销需在飞书授权管理中完成。

## 能力与授权

`src/ai_assistant/integrations/feishu-capabilities.json` 是版本化清单，包含 68 个官方工具、业务域、身份、读写分类、API 路径及权限申请参考。执行 `node deploy/feishu/build-manifest.mjs` 可从锁定官方包重新生成；清单随 Python 包分发。只有 `tools` 显式列出的工具才开放；`write_tools` 还须标明写入，且 `allow_write=true`，外部写操作必须配置持久会话。MCP annotations 不授予权限。永久删除、共享权限及组织管理不在飞书清单中；日程 delete 表示取消日程，是明确允许的日常写入。

| 业务域 | 功能 | 申请权限参考 |
| --- | --- | --- |
| 消息 | 查询会话、读取/发送/回复/转发/编辑消息 | im:message、im:message:readonly、im:chat:readonly |
| 文档 | 读取/创建文档、读取/创建/更新内容块 | docx:document |
| 云空间 | 查询文件、创建目录、移动文件，另有二进制桥接 | drive:drive |
| 知识库 | 查询空间/节点、创建/移动节点、修改标题 | wiki:wiki |
| 多维表格 | 查询应用/表/字段/记录、创建表/字段、创建/更新记录 | bitable:app |
| 日历 | 查询日历/日程、创建/更新/取消日程、管理日程参与人 | calendar:calendar |
| 任务 | 查询/创建/修改任务、负责人、截止时间、任务清单；patch completed_at 完成任务 | task:task |
| 联系人 | 必要用户查询、手机号/邮箱解析 ID；只读 | contact:user.base:readonly |

权限参考按业务域列出。应按实际启用的 API、应用身份支持情况和开放平台权限页面申请对应应用权限及 OAuth scope；这些权限仍受具体文档、日历、任务清单等资源可访问性限制。应用发布、资源共享与用户 OAuth 是不同步骤，scope 不代表全部资源可访问。

服务 `identity=user` 固定使用用户令牌；`identity=bot` 固定使用 tenant token，模型传入 `useUAT` 不能切换身份。官方 MCP 每次业务调用使用当前有效令牌建立受控连接，重新核对原始工具 schema 后执行并关闭；服务内调用串行化，避免刷新与在途调用交错。目录在 CLI 生命周期或机器人进程内固定，schema 变化要求重启并重新加载，不重放写操作。这一实现会增加 Node 冷启动开销。

原生发现工具为 `search_mcp_tools`、`load_mcp_tools`、`unload_mcp_tools`。分页不截断目录，模型只看到基础工具、发现工具与已激活工具，工具选择发生在最终输入预算检查之前。执行时再次验证激活摘要、身份、写权限与群资源目标。checkpoint 只保存 ID/schema hash，客户端和令牌留在运行时。可通过服务的 `resources`/`prompts` 显式 URI/名称白名单配合 `discover_mcp_content` / `read_mcp_content` 读取外部数据；不会把提示模板提升为系统指令。群中不开放这些内容接口。elicitation 明确拒绝，不触发图中断；不授权 sampling、动态 roots 或任意本地目录。

网易工具包括 folders、search、read、download、send、draft、update_draft、mark、move。send 的 reply_ref / reply_all / forward_ref 提供回复、回复全部、转发；默认不回复全部。UID 引用同时绑定账号、目录和 UIDVALIDITY。阅读使用 BODY.PEEK，不改变已读；mark 是显式写入。移动优先 MOVE，否则要求 UIDPLUS，精确 UID EXPUNGE，不调用整个目录 EXPUNGE。草稿更新保存替代项后仅标记原 UID deleted，不清空邮箱。HTML 只提取文本，不抓取远程资源。

附件使用受控 `outputs/mail` / `outputs/feishu` 新路径，拒绝穿越、符号链接、凭据文件与工作区外读取，不覆盖已有文件。邮箱单附件默认 25 MiB、整封编码后邮件 50 MiB；飞书桥接单文件 25 MiB。`feishu_files` 提供云盘/消息文件上传下载，通过工作区路径传递，二进制不进入模型上下文。主人私聊可用上传工具后经消息工具发送文件；群默认不开放本地文件桥接。

SMTP 回执区分 server_accepted、partially_accepted、rejected、unknown 和 sent_copy 失败；server_accepted 只表示服务器接受。发送的 Message-ID 绑定可信操作 UUID，超时后可用 search 的 message_id 在明确的已发送目录核查；没有可证明结果时保留 unknown。

## 飞书机器人

后台启用机器人和长连接消息接收事件 `im.message.receive_v1`，授予机器人接收私聊/被 @ 群消息及发送消息的权限。群必须显式加入 `channels.feishu.groups`，并登记该群可访问的机器人服务资源；例如配置示例只授权一个 chat_id 的消息查询与发送。每个群工具还须通过 `resource_arguments` 指定真实资源参数路径，缺失或目标不在群白名单中一律拒绝。私人邮箱、用户身份、终端、RAG、Skills 和本地文件工具均不向群开放。

```bash
.venv/bin/python -m ai_assistant feishu serve --config assistant.toml
```

接收进程只校验主人、群和 @，落入 PostgreSQL 后返回 SDK 回调。Python 单一执行器全局并发 1，普通请求队列最多 100，超限回复忙碌；/cancel 和 /status 保持可用。会话按 app/chat/owner 绑定，空闲时释放会话数据库锁。CLI、私聊、各群隔离。支持 /help、/new、/status、/cancel；主人私聊支持 /skills 和 /skill <ID> <请求>。长文本按顺序分段，回复不包含模型私有推理或原始异常。

重启后 received 可以继续；running 只有在会话成功输入指针匹配时补发答案，否则报告中断，不自动重跑。outbox 的 pending 可以补发，sending 崩溃后变为 unknown，不盲目重发；已送达正文清除。群配置和主人身份在执行时再次核对。

外部工具写入前持久化规范化参数摘要和操作标识；同 input_id、调用者、工具与参数组合不会重复提交。消息、日程、任务及单条表格记录创建使用可信层生成的远端幂等键，模型提供的键会被替换。网络失败、超时、取消和部分结果不能视为回滚。pending 在进程崩溃后表示需要核实，也不会重新提交。外部效果由回执解释，原有成功 checkpoint 分支规则保持不变。

```bash
.venv/bin/python -m ai_assistant mcp operations list --config assistant.toml
.venv/bin/python -m ai_assistant mcp operations inspect OPERATION_ID --config assistant.toml
```

## 验证与显式联网验收

自动测试不读取真实凭据、不使用付费模型、不发送真实邮件、不修改真实飞书数据。测试使用本地 MCP stdio/HTTP、IMAP/SMTP mock、HTTP Mock、伪飞书事件、脚本模型与独立 PostgreSQL。额外官方 Node 检查：

```bash
node deploy/feishu/test-office.mjs
node deploy/feishu/test-receiver.mjs
# 使用可销毁数据库；RAG 与 sessions URL 的驱动格式不同。
SESSION_TEST_DATABASE_URL=postgresql://... RAG_TEST_DATABASE_URL=postgresql+asyncpg://... \
  .venv/bin/python -m pytest -q
```

CSV、DOCX、PPTX、XLSX、Markdown 实际执行离线本地解析。测试环境需要提前准备 NLTK punkt_tab 和 averaged_perceptron_tagger_eng；解析子进程始终禁止联网及资源下载。

`scripts/check_office_live.py` 接受显式 JSON 场景，默认只校验；加 `--execute` 后才连接账号并执行。场景必须指定专用测试收件人和资源，工具名/参数可从 `mcp tools SERVER` 查看。它使用脚本模型驱动生产原生图，不调用付费模型；生成 0600、拒绝覆盖的报告，记录版本、会话、input_id、工具结果与操作回执。示例：

```json
[
  {"server":"mail","tool":"search","arguments":{"folder":"INBOX","subject":"OFFICE-ACCEPTANCE","limit":5}},
  {"server":"feishu_bot","tool":"im_v1_message_create","arguments":{"params":{"receive_id_type":"chat_id"},"data":{"receive_id":"oc_TEST_ONLY","msg_type":"text","content":"{\"text\":\"OFFICE-ACCEPTANCE\"}"}}}
]
```

```bash
.venv/bin/python scripts/check_office_live.py --config assistant.toml --scenario office-test.json
.venv/bin/python scripts/check_office_live.py --config assistant.toml --scenario office-test.json --execute
```

需要上一步返回 ID 的操作分两次显式场景执行，将观察到的真实 UID 引用或资源 ID 填入下一场景，不猜测标识。联合验收还应通过 chat/机器人请求检查 RAG、Skills 与办公工具协作：

1. 查询测试邮件、阅读和下载附件，结合已安装 Skills 生成摘要。
2. 发至测试群，创建文档、知识库节点和多维表格记录，再分别读取、更新。
3. 从测试事项创建日程和任务，读取、修改、取消日程与完成任务。
4. 验证测试邮件发送、回复、转发、草稿更新、已读/未读和安全移动。
5. 主人私聊与授权群分别请求，验证私人能力和群资源隔离。
6. 模拟令牌过期、网络中断、取消、进程重启；检查回执未知状态和成功输入补发，确认没有自动重复发送。

真实账号、授权码、应用权限、OAuth 和测试目标由部署者提供。本次实现的本地验证不能代替这些真实业务验收；未进行真实联网写入。外部未知结果需要通过只读业务查询与远端记录核实，不能仅凭本地超时判定失败或再次提交。
