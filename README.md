# AI Assistant

基于 LangChain 和 LangGraph 逐步构建的个人 AI 助理，面向任务规划、工具调用、长期记忆和 Skills 扩展。

目前已完成多模型接入、统一配置、模型能力检查、DeepSeek 原生结构化输出，以及工作区读写、沙箱命令执行和联网搜索/网页抓取工具。现已支持原生单 Agent 工具循环和持久多轮 CLI；MCP、长期记忆及计算机操作等能力将逐步实现。

## 快速开始

需要 Python 3.12。使用项目虚拟环境安装依赖并准备配置：

```bash
.venv/bin/python -m pip install -e '.[deepseek,sessions]'
cp assistant.example.toml assistant.toml
```

编辑 `assistant.toml`，填写模型、API 密钥和 PostgreSQL 会话连接，按实际模型设置输入预算，然后检查配置：

```bash
.venv/bin/python -m ai_assistant config check
.venv/bin/python -m ai_assistant session init
```

默认使用 DeepSeek，关闭 thinking 模式。

## 文档

- [使用指南](docs/usage.md)：安装、配置、模型调用、检查命令及测试。
- [持久会话与上下文](docs/sessions.md)：原生 Saver、会话命令、摘要、预算与恢复规则。
- [工具使用](docs/tools.md)：只读工作区、DeepSeek 联网工具与原生框架装配。
- [写权限工具](docs/write-tools.md)：文件创建、精确编辑与 Bubblewrap 终端。
- [架构说明](docs/architecture.md)：目录职责、依赖边界与后续能力规划。
- [框架复用记录](docs/llm-framework-reuse.md)：原生能力对照、行为差异与验证证据。

可选的本地知识库支持 PostgreSQL/pgvector 持久化、可靠后台入库、元数据过滤、中文混合检索及本地重排，并以原生 LangChain 工具向 Agent 提供证据。安装、CLI、Agent 装配及验收见 [知识检索说明](docs/rag.md)。

## 多轮 CLI 助理

```bash
python -m ai_assistant chat --config assistant.toml
python -m ai_assistant chat --config assistant.toml --model PROFILE
python -m ai_assistant chat --config assistant.toml --session SESSION_ID
```

每行输入一个请求，空行忽略。`/help` 显示帮助，`/new` 开始新对话，`/exit` 或 EOF 退出。
使用配置中的模型参数和 SDK 重试策略；真实请求可能产生费用。
默认开放三个只读工作区工具；联网、写入和终端分别按现有工具配置启用。
配置知识库时增加固定范围的检索工具，未配置时不导入可选 RAG 集成或加载 GPU。

可选 `[agent]` 参数见示例配置，默认每轮模型 10 次、工具 20 次、知识检索 5 次、期限 300 秒。
全局限额结束本轮；知识检索专属限额允许模型读取错误提示后继续回答。
执行中 Ctrl+C 只取消本轮并等待清理，空闲 Ctrl+C 退出（130）。
失败和超时后可以继续输入；上下文从最近完整轮次分支，已发生的文件或命令效果不会回滚。
会话由 PostgreSQL 原生 checkpoint 持久化；上下文使用原生工具结果编辑、摘要和预算裁剪。
不提供完整聊天记录回看、长期记忆或自动恢复失败任务。chat 要求 sessions/context 和所用模型预算；旧配置可继续执行其他命令。

Python 调用方式：

```python
from ai_assistant.config import load_config
from ai_assistant.runtime import assistant_session

async def example():
    async with assistant_session(load_config("assistant.toml")) as session:
        turn = await session.run_turn("列出工作区文件")
        print(turn.messages[-1].text)
        await session.new_conversation()
```

原生图接口为 `ai_assistant.agents.create_assistant_agent`；Python 调用方可传入原生 checkpointer、
模型和工具。事件回调接收原生工具事件；不要将工具参数、完整输出或诊断直接展示给终端用户。
显式手动验收：`python scripts/check_chat_live.py --config assistant.toml`，
加 `--write` 可显示文件创建及追问修改任务，`--session` 可重开已有会话。该脚本调用真实模型，可能产生费用；自动测试不运行它。


社区 Agent Skills 可选接入：配置 `[skills]` 后启用共享目录发现、原生按需加载、
只读脚本挂载和只新增依赖事务。安装方式、CLI、会话语义和显式联网验收见
[Skills 文档](src/ai_assistant/skills/README.md)。未配置时保持原有行为。
