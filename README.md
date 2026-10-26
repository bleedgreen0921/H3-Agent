# AI Assistant

基于 LangChain/LangGraph 的个人 AI 助手。目标是持续解决实际工作问题，优先复用框架原生 Agent、工具、中间件、MCP 和 checkpoint，减少自建代码。

保留本机管理命令与飞书机器人、多模型配置、持久会话和上下文管理、工作区读写、联网、RAG、飞书 MCP、本地 Skills 与受限命令执行。Skills 由用户手动准备，不提供下载、更新或依赖安装；暂不提供邮箱集成。

业务与部署环境分离：模型通过 HTTP 服务调用；硬件推理由服务管理；平台差异集中在启动配置与终端隔离层。CLI 和飞书共用同一 Agent；tools 负责业务适配，runtime 负责轮次和取消，interfaces 负责入口，storage 负责持久化。开发约定见 [AGENTS.md](AGENTS.md)。

## 首次配置

需要 Python 3.12。指定已有解释器，不需要 activate 或新建环境：

```bash
printf '%s\n' /ABSOLUTE/PATH/TO/EXISTING/ENV/bin/python > .python-path
# 或设置环境变量 AI_ASSISTANT_PYTHON；其优先于 .python-path。
```

安装前先检查共享环境，保留其他项目依赖。以下只安装明确选择的项目 extras，不卸载无关包：

```bash
PYTHON=/ABSOLUTE/PATH/TO/EXISTING/ENV/bin/python
"$PYTHON" -m pip install --dry-run -e '.[deepseek,openai,sessions,rag,skills,mcp,dev]' -c constraints/runtime.txt
"$PYTHON" -m pip install -e '.[deepseek,openai,sessions,rag,skills,mcp,dev]' -c constraints/runtime.txt
"$PYTHON" -m pip check
"$PYTHON" -m nltk.downloader punkt_tab averaged_perceptron_tagger_eng
```

约束文件仅覆盖项目依赖。共享环境如有冲突，先重新解析兼容约束，不强行同步整个环境。模型推理不依赖应用环境中的 PyTorch；解析器需要的 NLTK 资源显式准备，运行时不下载。macOS 缺少 libmagic 时执行 `brew install libmagic`。PDF/OCR 重量解析资源须按实际格式另行准备。

```bash
cp assistant.example.toml assistant.toml
./assistant.sh config check
```

填写真实模型 ID、密钥、工作区及数据库连接。相对路径以配置文件目录为基准，凭据不读取 `.env` 或自动展开变量。已移除的邮箱、Skills 安装、终端档位和本地权重字段会被拒绝。

## 数据库与模型服务

开发数据库使用 PostgreSQL 17/pgvector；先启动 Docker，再显式部署和初始化：

```bash
docker compose -p h3-agent-dev -f deploy/compose.dev.yaml up -d --wait
./assistant.sh mcp init
# 启用 knowledge 后：
./assistant.sh rag init
```

该 Compose 仅供受信任的本机开发使用：loopback 端口、专用库/账号免密；其他部署应配置自己的认证。应用启动不自动迁移数据库。已有数据卷不会删除重建。

本机固定使用 llama.cpp 与 Qwen3 Embedding/Reranker 0.6B Q8_0 GGUF。两个实例分别提供 `/v1/embeddings` 和 `/v1/rerank`；应用同样可以配置其他机器上兼容的 HTTP 服务。

```bash
cp models.example.toml models.local.toml
# 在 models.local.toml 填写两个 GGUF 文件位置。
./models.sh embedding    # 独立终端，默认 127.0.0.1:1236
./models.sh reranker     # 另一终端，默认 127.0.0.1:1235
./check.sh models
```

启动命令前台运行，Ctrl+C 停止；不自动安装程序或下载模型，GPU 不兼容时应检查服务日志。默认上下文 2048，可在部署配置中调整。两个服务具有各自 Web UI，重排与向量检查使用 API。模型文件或量化版本变化时，重新验证并为 Embedding 更换明确模型名、重建索引。

## 日常使用

```bash
./assistant.sh feishu serve
./assistant.sh --help
```

飞书是日常对话入口，支持 `/help`、`/new`、`/status`、`/cancel`、`/skills`、`/skill ID 请求`。本机命令只用于配置、授权、数据管理和排障。旧 CLI 会话保留，可用 session 命令管理，不自动绑定飞书。

启用示例中的 knowledge 配置后：

```bash
./assistant.sh rag submit path/to/material.csv --tag personal
./assistant.sh rag worker     # 仅使用管理命令时需独立 Worker；feishu serve 自动管理 Worker
./assistant.sh rag status JOB_ID
./assistant.sh rag search '具体问题' --tag personal
./assistant.sh rag check --services
./assistant.sh rag sources
./assistant.sh rag sync --source personal --wait
./assistant.sh rag documents
```

入库保留不可变快照、任务去重、租约、心跳与版本发布；检索使用原生向量/关键词混合检索和重排，返回可追踪来源。片段长度以字符计，默认 512/64。Embedding 输出固定为 1024 维。模型服务不可用或输出无效时明确失败，不回退模型、不跳过重排。排障使用 `rag status JOB_ID --diagnostics`。

云文档与本地文档默认长期保存。扫描范围在 `[knowledge.sources.NAME]` 中明确配置，示例见配置模板；不默认扫描磁盘或整个飞书账号。支持 TXT/Markdown/文本型 PDF/DOCX/XLSX/PPTX/CSV、飞书新版文档、Wiki 文档、电子表格及多维表格。扫描件、图片理解、旧版云文档、全文总结和表格精确统计不在本阶段范围。

每次跨库检索前重新枚举全部配置范围并刷新变化内容；指定文档时只检查目标。没有可靠版本号的云表格每轮重新读取后比较 hash。正文与定位保存在受控快照中，PostgreSQL 保存来源、版本、片段和 1024 维向量。未变内容不重复生成向量；新构建完整成功后才发布。删除、失权或移出范围的文档保留存档，但退出正常检索。暂时读取失败不误判为删除，本次回答排除未核实部分并说明缺口。

飞书主人私聊会自动登记发送的云文档链接；飞书文件附件也会登记。可直接说“把 path/to/file.pdf 加入知识库”，助手使用 `ingest_document`。`search_knowledge` 接受可选的 `source_id` 和 `document_id`，返回来源、版本、核实时间和页/块/单元格/记录定位。模型只接收最多约 8,000 字符的完整证据片段，不加载整份长文档。

大量更新时，原生 LangGraph checkpoint 保存等待位置，先通知同步进度，完成后自动回答原问题；同一会话后续消息排队，其他会话继续。飞书 `/status` 查询、`/cancel` 取消本次等待、`/new` 取消旧等待并创建新会话；入库资料仍保留。同步默认截止 4 小时，返回已核实部分；没有证据时明确说明。只有明确等待中的知识请求可自动恢复，普通失败或未知外部写入不自动重放。

升级现有数据库需要显式运行以下命令；建议先备份。启动只检查 schema，不自动迁移。飞书来源需要实际授予对应的 Wiki/文档/电子表格/多维表格读取权限，配置本身不能授予平台权限。

```bash
./assistant.sh session init
./assistant.sh rag init
./assistant.sh feishu serve
```

## 本地 Skills 与终端

手动将 Skill 目录放入工作区 `.agents/skills`、用户 `~/.agents/skills`，或 `[skills].roots` 指定目录。每个目录应包含带 name/description 的 `SKILL.md`；依赖由用户提前安装，资源随会话建立只读快照。

```bash
./assistant.sh skill list
./assistant.sh skill inspect SKILL_ID
./assistant.sh skill check SKILL_ID
```

终端只有一种权限：工作区可写，Python 环境、依赖和激活 Skill 快照只读；默认断网，以 `[tools.terminal].allow_network=true` 显式开启。凭据只能通过 `env` 或明确的额外只读 `read_paths` 授权；不会自动开放用户目录。联网型 Skill 需要对应授权和预装命令。

本项目当前仅维护 macOS 的 Seatbelt 受限终端；沙箱启动失败时拒绝执行。macOS 为保证后台进程清理，禁止改变进程组/session 及 `posix_spawn`；Homebrew Python 使用同一解释器的直接启动适配，部分依赖这些进程操作的命令会明确失败。macOS 升级后应重跑沙箱测试。Node 的已链接运行库按实际依赖开放，不开放整个 Homebrew 前缀。

## 飞书

按 `assistant.office.example.toml` 合并所需配置；只复制实际需要的 MCP 服务与权限，不覆盖已有配置。

```bash
npm ci --prefix deploy/feishu --ignore-scripts
./assistant.sh mcp check
./assistant.sh mcp auth feishu login
./assistant.sh mcp check --network
./assistant.sh feishu serve
```

Node 需 >=20。OAuth 使用本机 loopback 回调；应用内主人 ID 必须与授权用户匹配。写入保留持久回执，失败或未知结果不自动重发。群聊只能使用明确授权的飞书机器人资源，不继承私人能力。

## 测试与真实验收

```bash
"$AI_ASSISTANT_PYTHON" -m pytest -q  # 或直接使用 .python-path 中的解释器
./check.sh models
./check.sh rag
./check.sh rag --agent  # 同时验证真实 Agent 检索与回答，可能产生费用
./check.sh agent
./check.sh skills
./check.sh terminal
./check.sh web
```

`models` 只调用真实本地模型；`rag` 使用独立测试知识库验证真实解析、后台入库、检索与重排，要求已初始化 schema，测试文档结束后退出检索范围。其余命令调用真实生成模型，可能产生费用；不自动运行。可用 `--output smoke-results.json` 新建私有结果文件，默认只输出简短结果。

飞书工具流程通过显式测试目标执行：

```bash
./check.sh feishu --prompt '在指定测试资源中执行具体操作并回读' \
  --required-tool NATIVE_TOOL_NAME --expect '已知核对值'
```

写操作须在 prompt 中明确指定测试目标；随后人工核对远端结果与 `mcp operations list` 回执。机器人入口另通过真实测试会话发送消息验证。自动测试使用 fixture/mock 和独立数据库，不读取真实凭据；数据库集成测试需显式设置 `SESSION_TEST_DATABASE_URL` 和 `RAG_TEST_DATABASE_URL`。未提供依赖时的跳过不代表真实业务验收通过。

## 个人回答风格

在 `[agent]` 中设置 `persona_file = "PERSONA.md"`，即可加载独立的个人风格文件。路径相对 TOML 配置文件解析，支持绝对路径。风格文件使用 UTF-8，最大 16 KiB；读取失败会明确报错。未配置则使用默认风格。

飞书在处理每条消息时建立会话，个人风格文件修改后下一轮生效。首次启用需要重启现有机器人进程。风格作为系统提示的一部分参与上下文预算，不改变工具权限或证据要求。


## 上下文、个人记忆与流式回复

`models.<名称>.max_input_tokens` 是应用输入窗口配置，必须与实际模型能力匹配；当前 DeepSeek Flash 示例使用官方 1,048,576 token 上下文，不再限制为 24,000。输入预算保留 10% 余量，先扣除系统提示、PERSONA、Skills、已选 MCP schema 和个人记忆；消息达到剩余预算的 75% 时通过原生摘要压缩。这里使用本地保守估算，服务实际计费 token 单独统计。历史能放入窗口时完整保留，不因累计工具正文超过 4,000 token 就提前收起。摘要约 1,500 token 是压缩目标，有效摘要超过目标不会单独导致失败，最终完整请求仍须通过总预算检查。MCP 目录每页最多 5 项、1,500 字符；`load_mcp_tools` 每次替换选择集合，schema 总预算默认 6,000 token。普通工具单次返回默认最多 4,000 token，完整内容保存在 checkpoint artifact，可用 `read_tool_result` 分页；单次工具分页不等同于会话上下文窗口。必需回执无法容纳时明确报错。

旧配置默认关闭个人记忆和飞书流式显示。先备份数据库及配置，再显式升级；这些管理命令不通过模型开放：

```toml
[context]
summary_tokens = 1500
tool_result_tokens = 4000
mcp_schema_tokens = 6000

[memory]
enabled = true

[channels.feishu]
enabled = true
streaming = true
stream_interval_seconds = 1
card_max_bytes = 24576
```

记忆需要已配置的 sessions、context、飞书主人身份和本机 knowledge.embedding，复用现有 Qwen 1024 维向量服务。依赖已包含在 sessions/rag extras 中；如环境缺少依赖，由用户显式运行 `python -m pip install -e '.[sessions,rag]'`，启动不会安装。

```bash
./assistant.sh session init
./assistant.sh mcp init       # 显式扩展 outbox
./assistant.sh memory init    # 原生 AsyncPostgresStore 与版本表
./assistant.sh config check
./assistant.sh feishu serve
```

个人记忆只在主人私聊中使用，按配置中的应用 ID 和主人 ID 隔离；群聊不读取、不写入。普通短对话没有额外提取调用；压缩时一次结构化调用同时生成摘要与候选，校验用户原话及消息 ID 后保存在 checkpoint，整轮成功才登记后台持久化任务。自动候选只接受用户原话的连续内容，冲突事实不会自动覆盖。明确说“记住”时调用工具并等待持久化回执。保存失败不能声称已记住；向量召回失败时降级读取近期偏好。

飞书支持 `/memory`、`/memory list OFFSET`、`/memory delete ID`、`/memory clear`、`/memory pause`、`/memory resume`，也可用自然语言请求管理。本机对应命令为 `memory list --offset N`、`memory delete ID`、`memory clear/pause/resume`。更正、删除和暂停更新版本及提取水位，旧任务失效；恢复不回扫旧消息。`/new` 保留个人记忆；删除记忆不删除会话历史。

流式回复使用一张可更新卡片，按约 1 秒合并正文预览；只显示主模型正文，工具前的预览会替换为执行状态。完成校验及 checkpoint 提交后才显示“已完成”，失败或取消显示“未完成”。正文超出 24 KiB 卡片预算时，完整答案按现有文本分段交付。创建结果未知保留 outbox 回执，不重复发送；明确拒绝卡片时回退文本。重启只恢复已确认结果的投递，不重跑业务动作。卡片接口见[飞书官方消息 API](https://open.feishu.cn/document/uAjLw4CM/ukTMukTMukTM/reference/im-v1)。

可分别将 `memory.enabled` 或 `channels.feishu.streaming` 设为 false 回退，再重启服务；会话和知识库仍保留。日志只记录轮次 ID、首次状态/正文事件耗时、总耗时、工具耗时、预算、摘要次数和模型用量，不记录正文或凭据；首次事件时间不等于飞书客户端实际显示时间，流式显示不代表模型推理加速。

真实验收由主人在指定私聊中发起：长对话压缩后检查目标和约束；明确记住后 `/new` 检查召回，再更正与删除；知识查询检查阶段与正文增量；取消、重启或网络故障后检查状态和操作回执。自动测试使用模拟模型、模拟飞书与独立数据库，不代替真实飞书验收。

### 任务时间、上下文与会话用量

主人私聊的任务时间使用 `agent.timezone`（默认 `Asia/Shanghai`）。模型先明确日期，再调用 `resolve_datetime`；任务 `start`/`due` 的具体时间戳必须与本轮转换一致，全天任务以明确日期转换。无效时间和未指定偏移的夏令时重复时间会被拒绝。`integrations.feishu.default_task_assignee = "owner"` 可为未提供成员的个人任务设置主人负责人；兼容默认值为 `"none"`，显式成员或空列表优先，群聊不继承默认负责人。

用户身份任务工具应显式允许 `task_v2_task_get`、`task_v2_task_addMembers`，后者还需加入 `write_tools`。修改前回读；负责人通过 addMembers 添加并去重。写后核对时间、成员与提醒，缺失时只回读一次。远端成功但业务不符会保留成功操作回执并显示“未完成”，不补写、不重建、不自动重放。

`search_mcp_tools` 支持 `server`/`domain` 筛选、中英文关键词排序，每页最多 5 项和 1,500 字符。每条新消息清空激活 schema，知识同步续接保留选择。模型视图中目录合计最多 2,000 token，普通读取合计最多 4,000 token（本地保守估算）；优先保留最新正文，旧结果通过 `read_tool_result` 读取 checkpoint artifact。写回执和当前用户请求受保护。自动摘要和最终请求共用投影，保持原输入预算及 10% 余量；预算明细仅进入开发日志。

飞书 `/usage` 直接展示当前完整会话**服务已报告的累计用量**，不调用模型、不创建会话、不等待正在运行的 Agent 锁。主模型和摘要模型按角色及配置名分别累计输入、输出、合计 token，并显示调用数、进行中、未返回用量和部分用量。`/new` 开始新统计会话，重启不清零；旧 checkpoint 不回扫估算，统计覆盖起点会明确显示。缺失数值不填零，SDK 内部未报告的重试消耗不推算，因此不等同供应商完整账单。

升级前备份配置与数据库，再显式执行 `./assistant.sh session init` 创建用量表。启动仅检查 schema，不迁移或安装依赖。用量独立于业务 checkpoint 提交，按原生调用 ID 去重；删除会话会级联删除其用量记录。恢复投递不会重新调用模型或累计用量。

重启和人工验收见 [任务可靠性与用量验收](docs/task-reliability-usage.md)。

本次摘要误判与百万窗口修复、真实长输入验收结果见 [长上下文验收](docs/long-context-verification.md)。
