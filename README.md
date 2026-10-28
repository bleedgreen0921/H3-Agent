# AI Assistant

基于 LangChain/LangGraph 的个人 AI 助手，以飞书作为日常对话入口，终端负责配置、授权、数据管理与排障。

## 主要能力

- 持久会话、长对话摘要、流式回复与模型用量查询。
- 本地及飞书资料入库、来源同步与带引用的知识检索。
- 通过 MCP 使用飞书工具，区分主人私聊与群聊授权。
- 工作区读写、联网、本地 Skills 与受限终端执行。

模型通过 HTTP 服务调用；知识库、MCP、Skills 等能力按配置启用。当前受限终端仅维护 macOS Seatbelt 实现；Skills 和依赖需提前准备。

## 快速开始

需要 Python 3.12；飞书入口还需 Node.js 20+、PostgreSQL，以及相应的飞书应用配置。知识库使用 pgvector 和兼容的 Embedding/Reranker 服务。

先按 [安装与部署](docs/setup.md) 准备依赖与外部服务，再在项目根目录指定解释器并创建配置：

```bash
printf '%s\n' /ABSOLUTE/PATH/TO/EXISTING/ENV/bin/python > .python-path
cp assistant.example.toml assistant.toml
```

填写模型、工作区和数据库连接，并按 [办公配置模板](assistant.office.example.toml) 合并飞书配置。配置完成后：

```bash
./assistant.sh init
./assistant.sh mcp auth feishu login  # 使用主人 user 身份的服务时执行
./assistant.sh feishu serve
```

`init` 显式初始化已配置存储；服务启动不自动安装依赖、升级数据库或下载模型。已有部署升级前先备份，详见 [升级步骤](docs/COMMANDS.md#升级步骤)。

在飞书私聊中正常提问，使用 `/new` 开始新会话、`/usage` 查看用量。启用知识库后，可用 `/rag add <飞书文档链接>` 添加资料，再询问资料中的问题。

## 文档导航

| 文档 | 内容 |
| --- | --- |
| [安装与部署](docs/setup.md) | Python 依赖、数据库、模型服务和飞书授权 |
| [命令说明](docs/COMMANDS.md) | 完整终端命令、飞书知识库指令及升级步骤 |
| [飞书会话管理](docs/FEISHU-CONVERSATIONS.md) | 会话指令、群聊与 Skills 授权 |
| [知识库](docs/knowledge.md) | 支持的资料、同步、索引与检索行为 |
| [配置与运行行为](docs/configuration.md) | Skills、终端权限、个人风格、摘要与用量 |
| [测试与验收](docs/testing.md) | 自动测试、真实业务验收入口和专项记录 |
| [开发约定](AGENTS.md) | 项目分层、权限与开发约束 |
| [待修复事项](TODO.md) | 已知问题与当前优先级 |
