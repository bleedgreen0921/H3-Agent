# AI Assistant

基于 LangChain/LangGraph 的本地个人 AI 助手。日常入口是本地 Web UI，终端负责配置、授权、初始化和知识库管理。飞书只作为**只读云文档与多维表格来源**，不再承担聊天、任务、日历或写入。

## 主要能力

- 本地 Web 对话：多轮会话、流式过程、引用/来源侧栏、取消和重载恢复。
- 本地附件：上传 PDF、DOCX、Markdown 后由模型按需读取；支持页码/段落/行号和续读；不自动入库。
- 知识库：显式登记、不可变快照、后台同步/索引、混合检索、重排和带定位引用。
- 飞书只读：读取 docx、Wiki、电子表格、多维表格和云空间文件夹；不包含飞书写入。
- 本地 Skills：按需加载本地指导与资源，不执行脚本、不安装依赖。
- 规划模式：先生成 DAG 草稿，再执行只读并行/写任务串行，保留真实回执。
- 可选联网：通过 DeepSeek Search/Fetch API；默认不启用。
- 可选 LangSmith 观测：默认关闭，默认不采集正文。
- 持久会话：PostgreSQL + LangGraph checkpoint；外部文件写入和操作回执独立保存。

## 入口

```text
本地 Web UI   → 日常对话、附件、知识库、规划
CLI           → 配置、init、飞书授权、知识库管理、诊断
飞书云文档     → 只读来源，不是聊天入口
```

当前不支持群聊、飞书私聊、飞书任务/日历、通用命令执行、手机远端访问、多用户和 Web API 对外服务。

## 快速开始

需要 Python 3.12；知识库使用 PostgreSQL 17 + pgvector；Web 入口默认只监听 `127.0.0.1`。

```bash
printf '%s\n' /ABSOLUTE/PATH/TO/EXISTING/ENV/bin/python > .python-path
cp config.example/assistant.example.toml assistant.toml
# 按示例组件填写模型、数据库和可选的飞书文档配置。
./assistant.sh init
./assistant.sh feishu auth login   # 需要读取个人飞书云文档时执行
./assistant.sh web serve
```

`web serve` 会输出带一次性本地令牌的访问地址。默认只监听 `127.0.0.1`，不提供公网或多用户认证。

启动不会自动安装依赖、迁移数据库或下载模型。已有部署升级前先备份；当前会话结构为 v5，旧结构不自动升级。

## 配置

- 只支持具名 profiles + `includes` 组件组合。
- 未启用集成不连接外部服务。
- 未知字段、旧字段会在连接服务前拒绝。
- 不读取 dotenv；模型密钥写入受保护配置，LangSmith Key 通过显式环境变量名注入。
- 飞书 OAuth 令牌保存在工作区外的私有目录（默认 `~/.local/state/ai-assistant`）。

关键配置节：

```toml
[sessions]       # 会话/checkpoint/操作/用量
[web]            # Web 监听地址、令牌、状态目录
[attachments]    # 附件大小、类型、解析超时
[integrations.feishu]  # 只读文档 OAuth
[knowledge]      # pgvector、Embedding、Reranker 和来源
[planning]       # /plan 预算
[tools.selection]# 按轮工具暴露
[skills]         # 本地 Skills
```

## 飞书只读文档

支持的只读能力：

- Wiki 节点解析
- docx 正文
- 电子表格
- 多维表格表/字段/记录筛选
- 云空间文件夹枚举

模型工具只暴露只读操作；知识库同步使用同一个 Python 连接器，不经过模型工具。飞书 OAuth 使用本机 loopback 回调，不启动 Node 接收器。

## 附件读取

1. 在 Web 输入框上传 PDF/DOCX/Markdown。
2. 附件只关联当前会话，不自动进入模型上下文。
3. 模型需要时调用 `read_attachment`，按页码/段落/行号返回，并给出 `next_cursor`。
4. 用户点击“加入知识库”或明确要求时，才提交后台索引。

默认限制：单附件 25 MiB，单次读取 16,000 字符；不支持 OCR、图片理解、音频/视频。

## 规划模式

在 Web 顶部切换到“规划模式”，或输入 `/plan`：

```text
/plan 整理本周会议并创建待办
/plan run
/plan status
/plan exit
```

顶部持续显示当前模式；输入框固定在页面底部，长会话只滚动消息区。规划面板可折叠查看目标、待补充问题和任务进度，执行中会定时刷新；页面普通刷新即可加载最新前端资源。

Web 端的模式切换和按钮等价于 `/plan`、`/plan run`、`/plan exit`，不需要手打命令。规划模式内也可以直接回复“开始执行”“退出规划模式”“计划状态”；系统会用同一个对话模型做一次短意图判断，置信度低于 0.8 或判断失败时不执行，按普通补充处理。

Planner 只生成草稿，不执行工具；信息不足时先返回问题并暂停执行，用户补充后重新规划。`/plan run` 后只读任务并行、写任务串行。规划模式由用户手动进入和退出；任务完成、失败或取消后仍保留计划状态，发送新目标可重新规划，`/plan exit` 才回到普通对话；不自动重跑旧计划。

## 知识库

- 显式登记：CLI `rag submit`、Web 知识库页面提交、用户明确要求时 `ingest_document`。
- 只查询已发布快照，不自动同步、等待新索引或自动续答。
- 支持本地目录、飞书 Wiki/文件夹、单个飞书文档链接。
- 本地支持 TXT/Markdown/PDF/DOCX/XLSX/PPTX/CSV；飞书支持 docx/Wiki/电子表格/多维表格。
- 扫描件 OCR、图片理解、旧版云文档和全文总结不在当前范围。
- 检索使用 1024 维 Embedding、向量+关键词混合召回和 Reranker。

## 运维

```bash
./assistant.sh init
./assistant.sh config list
./assistant.sh config check
./assistant.sh web serve
./assistant.sh feishu auth status
./assistant.sh rag worker
./assistant.sh rag search "查询"
./assistant.sh session list
```

`init` 显式创建/核对会话、操作回执、附件、Web 消息、知识库和用量表；普通启动不建表。`scripts/reset_runtime.py` 默认只预览，`--execute` 才删除会话/checkpoint/操作/用量/附件/Web 消息，知识库保留；`scripts/delete_feishu_sessions.py` 只删除旧渠道绑定会话及其投递/操作数据，知识库保留。

## 文档导航

| 文档 | 内容 |
|---|---|
| [安装与部署](docs/setup.md) | Python、数据库、模型服务和飞书授权 |
| [命令说明](docs/COMMANDS.md) | CLI 与知识库命令 |
| [配置与运行行为](docs/configuration.md) | profiles、Web、附件、飞书、Skills |
| [知识库](docs/knowledge.md) | 登记、同步、索引与检索 |
| [测试与验收](docs/testing.md) | 自动测试、合成评测与真实验收 |
| [开发约定](AGENTS.md) | 分层、权限与开发约束 |
