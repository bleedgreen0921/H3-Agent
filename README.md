# AI Assistant

基于 LangChain/LangGraph 的个人 AI 助手，以飞书作为日常对话入口，终端负责配置、授权、数据管理与排障。

## 主要能力

- 持久会话、长对话摘要、飞书最终文本回复与模型用量查询。
- 长任务工作卡：超过阈值时私聊出现秒表、当前阶段/工具与答案预览；最终答案仍以文本送达，卡片更新为已完成终态并保留。
- 普通聊天：可恢复的只读错误允许纠正；写入失败或效果未知时停止本轮并保留回执；已派发的并发调用需结束或完成取消清理。
- 本地及飞书资料入库、来源同步与带引用的知识检索。
- 通过 MCP 在主人私聊中使用飞书工具。
- 工作区读写、联网与本地 Skills。

模型通过 HTTP 服务调用；知识库、MCP、Skills 等能力按配置启用。不提供通用命令执行或群聊入口；Skills 和依赖需提前准备。

MCP 按 `mcp.servers.<服务>.tools` 白名单确定允许范围，全量工具登记在注册表中；每轮只把核心工具和检索到的相关工具 schema 绑定给模型，能力不足时可调用 `search_tools` 渐进扩展。执行时仍检查主人身份、工具白名单、写开关和资源范围。`[tools.selection]` 的 `mode = "all"` 可回退到全量绑定。详见 [运行契约](docs/runtime.md)。

Skills 由用户在项目 `skills/` 下管理，显式 roots 不追加私人目录。发现只读 `SKILL.md`，激活时准备包快照；依赖提前安装，修改后重开会话或重启共享服务。Skill 目录也按意图和 `skills.directory_tokens` 选择；`/skill` 显式指定优先，模型也可通过 `search_skills` / `load_skill` 自主加载，路由不会自动激活正文。单个 MCP 初始化失败会标记当前不可用，其余服务与聊天继续；CLI 诊断仍返回失败。

## 快速开始

需要 Python 3.12；飞书入口还需 Node.js 20+、PostgreSQL，以及相应的飞书应用配置。知识库使用 pgvector 和兼容的 Embedding/Reranker 服务。

先按 [安装与部署](docs/setup.md) 准备依赖与外部服务，再在项目根目录指定解释器并创建配置：

```bash
printf '%s\n' /ABSOLUTE/PATH/TO/EXISTING/ENV/bin/python > .python-path
cp config.example/assistant.example.toml assistant.toml
cp config.example/assistant.example.component.toml ./
# 填写组件中的实际配置。
```

`assistant.sh` 和 `models.sh` 优先使用非空 `AI_ASSISTANT_PYTHON`，否则读取项目根目录 `.python-path` 首行；指定解释器无效或不是 Python 3.12 时直接失败，不回退到系统 Python。两个脚本均切换到项目根目录执行，默认配置和相对 `--config` 参数从项目根目录定位。

填写模型、工作区和数据库连接，并按 [办公配置模板](config.example/assistant.office.example.toml) 合并飞书配置。配置完成后：

```bash
./assistant.sh init
./assistant.sh mcp auth feishu login  # 使用主人 user 身份的服务时执行
./assistant.sh feishu serve
```

`init` 显式初始化已配置存储；服务启动不自动安装依赖、升级数据库或下载模型。已有部署升级前先备份；旧结构不再升级，详见 [升级步骤](docs/COMMANDS.md#升级步骤)。

在飞书私聊中正常提问，使用 `/new` 取消当前执行并开始新会话、`/usage` 查看用量。启用知识库后，可用 `/rag add <飞书文档链接>` 提交资料入库，使用 `/rag status <任务ID>` 查看进度。普通私聊链接、附件和 MCP 阅读不自动入库。只有显式添加、用户明确要求入库及配置来源会登记；现有 `pilot_wiki` 整空间范围保留，主动同步可发现新文档。提交不代表可检索；查询只使用已发布且当前允许访问的快照，不会自动同步、等待或自动续答。需刷新资料时主动使用 `/rag sync [来源ID]`。

上下文由固定规则、结构化会话记忆、压缩摘要、最近对话和本轮请求/工具结果组成。默认每完成 5 轮增量更新记忆；完整请求超过模型输入窗口的 90% 时才压缩上下文，摘要目标为窗口的 10%，目标保留最近 30 轮。撤销资源访问不清除已有会话信息，新访问仍按当前授权检查。外部会话按可信范围与绑定隔离，共享会话不继承个人能力；没有历史成果检索/读取工具或引用链恢复。详见 [运行契约](docs/runtime.md)。

异常后保留最新真实对话和工具结果，在同一会话的新分支处理新请求；不自动续跑旧工具。回答先保存并完成最终回复处理，再做定期摘要；预览、展示回调及独立用量统计故障只记录整理错误。

`/usage` 查询当前会话已保存的服务报告用量，进程重启后沿用同一会话累计，`/new` 单独统计。模型返回后按调用 ID 幂等保存；统计连接或表不可用不阻断聊天。仅显式 `init` 创建统计表，空记录不代表没有模型调用。详见 [运行契约](docs/runtime.md)。

轮次结果由 runtime 保存；正常回复直接展示正文，`complete` 仅表示本轮回复完整结束。只读错误允许纠正，写入失败或效果未知停止；历史终端回执中的非零退出保留退出事实，并按可信输入与规范化参数去重。

## 文档导航

| 文档 | 内容 |
| --- | --- |
| [安装与部署](docs/setup.md) | Python 依赖、数据库、模型服务和飞书授权 |
| [命令说明](docs/COMMANDS.md) | 完整终端命令、飞书知识库指令及升级步骤 |
| [飞书会话管理](docs/FEISHU-CONVERSATIONS.md) | 主人私聊会话指令与本地 Skills |
| [知识库](docs/knowledge.md) | 支持的资料、同步、索引与检索行为 |
| [配置与运行行为](docs/configuration.md) | Skills、个人风格、摘要与用量、具名配置组合 |
| [测试与验收](docs/testing.md) | 自动测试入口、合成评测资产与真实验收边界 |
| [开发约定](AGENTS.md) | 项目分层、权限与开发约束 |


飞书输入执行与 outbox 投递使用独立循环和数据库连接，慢发送不阻塞取消处理；取消回执仍按投递顺序发送。发送中关闭的未确认消息不会自动重发。

记录、执行边界、模块归属和投递恢复见 [运行与结果契约](docs/runtime.md)。旧数据格式不再迁移，也不自动补发或重跑历史业务；旧 interface 兼容入口已删除，当前入口为 `channels/`、`interfaces/`、`storage/` 和 `runtime/`。Plan 与知识等待的移除范围见 [运行契约](docs/runtime.md)。

飞书任务按请求由原生 Agent 选择工具，不隐式读取或写后回读。保留主人私聊默认负责人和本轮确定性时间保护；结果附加本地时间说明，原始字段完整保存。回执区分接口接受、字段未确认和明确不符，失败或未知效果不自动重放。详见 [运行契约](docs/runtime.md)。

## 配置方案切换与 Spark

配置可按任意现有配置节拆分为 TOML 组件，再由具名方案组合；示例见 [配置入口](config.example/profiles.toml)。当前私有入口仍默认 `current`，有效配置与拆分前一致；组件和原配置备份位于已忽略的 `.data/config/`，不要提交其中的凭据。相对路径统一以入口文件目录为基准。

```bash
./assistant.sh config list
./assistant.sh config check --profile current
./assistant.sh feishu serve --profile current
# 填好 .data/config/main/spark.toml 的 api_key 后，再选择 spark。
./assistant.sh model check --profile spark
./assistant.sh feishu serve --profile spark
```

`--profile` 只影响本次命令，不记住上次选择；修改入口 `default_profile` 可改变默认方案。所有现有命令均支持 `--config` 与 `--profile`。服务配置切换需重新启动，不会迁移数据库或重放旧请求。

Spark 使用现有 OpenAI 兼容 provider，模型为 `spark-x2.6`，首版关闭思考；无需增加依赖，环境缺少所选集成时手动执行 `"$(cat .python-path)" -m pip install -e '.[openai]'`。离线配置校验不验证 API Key 或远端权限；`model check` 会发送真实请求并可能产生费用，本次交付不自动执行。

主人飞书私聊附件阅读通过 `[tools.pdf] enabled = true` 启用，覆盖 PDF、DOCX、Markdown（`.md`），无需迁移配置。每个附件最多 25 MiB；PDF 使用现有 pypdf，DOCX/Markdown 使用标准库，均在可终止子进程中解析。统一工具 `read_document_attachment` 返回正文、位置、覆盖范围及续读游标；只提供当前附件工具，不保留旧 PDF 工具接口。DOCX 仅覆盖正文和表格文字，不提供物理页码，解压总量最多 100 MiB；Markdown 仅接受 UTF-8/BOM 原文，不加载链接或图片。读取不自动入库或授权外部写入。

## Trace 与 LangSmith

观测默认关闭，可显式启用 LangSmith；默认隐藏正文，通过 `input_id` 关联接收日志、Agent 调用树、结果提交、摘要与独立投递。示例见 [观测配置](config.example/observability.toml)，部署、显式存储扩展与模拟验收见 [观测说明](docs/observability.md)。不接入评测集或评测流水线。

## 当前接口与会话历史重置

配置入口仅支持具名组合方案；单文件业务配置、`context.mcp_schema_tokens` 及 MCP server 的 `resources`/`prompts` 字段均拒绝。模型内部保留原生流式执行；飞书等待最终回复后按 Markdown 文本分段投递。长任务仅额外使用一张临时工作卡（终态保留，不撤回），最终答案不使用卡片。

会话存储使用 v5，旧结构不自动升级。一次性清理使用 `scripts/reset_runtime.py`（默认仅预览，显式 `--execute` 才删除），再执行 `init`。该脚本按固定表清单删除会话、checkpoint、用量、渠道队列及工具操作历史，完整保留知识库和知识库管理回执，不重放旧业务。

测试、文档和示例仍沿用当前 Git 忽略规则，其本地更新不会随仓库完整分发。
