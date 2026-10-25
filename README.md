# AI Assistant

基于 LangChain/LangGraph 的个人 AI 助手。目标是持续解决实际工作问题，优先复用框架原生 Agent、工具、中间件、MCP 和 checkpoint，减少自建代码。

保留 CLI 与飞书机器人、多模型配置、持久会话和上下文管理、工作区读写、联网、RAG、飞书 MCP、本地 Skills 与受限命令执行。Skills 由用户手动准备，不提供下载、更新或依赖安装；暂不提供邮箱集成。

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
./assistant.sh chat
./assistant.sh chat --session SESSION_ID
./assistant.sh chat --model PROFILE
./assistant.sh --help
```

CLI 支持 `/help`、`/new`、`/skills`、`/skill ID 请求`、`/exit`。执行中 Ctrl+C 取消本轮并等待清理；空闲 Ctrl+C 退出。失败轮次不自动重放，文件或远端效果不会回滚。会话由原生 PostgreSQL checkpoint 持久化，上下文使用原生摘要、编辑与裁剪。

启用示例中的 knowledge 配置后：

```bash
./assistant.sh rag submit path/to/material.csv --tag personal
./assistant.sh rag worker     # 独立终端；--once 只处理一个任务
./assistant.sh rag status JOB_ID
./assistant.sh rag search '具体问题' --tag personal
./assistant.sh rag check --services
```

入库保留不可变快照、任务去重、租约、心跳与版本发布；检索使用原生向量/关键词混合检索和重排，返回可追踪来源。片段长度以字符计，默认 512/64。Embedding 输出固定为 1024 维。模型服务不可用或输出无效时明确失败，不回退模型、不跳过重排。排障使用 `rag status JOB_ID --diagnostics`。

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
./check.sh chat
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
