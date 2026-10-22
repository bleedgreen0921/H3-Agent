# AI Assistant

基于 LangChain 和 LangGraph 逐步构建的个人 AI 助理，面向任务规划、工具调用、长期记忆和 Skills 扩展。

目前已完成多模型接入、统一配置、模型能力检查、DeepSeek 原生结构化输出，以及只读工作区和联网搜索/网页抓取工具。Agent 执行、MCP、记忆及计算机操作等能力将逐步实现。

## 快速开始

需要 Python 3.12。使用项目虚拟环境安装依赖并准备配置：

```bash
.venv/bin/python -m pip install -e '.[deepseek]'
cp assistant.example.toml assistant.toml
```

编辑 `assistant.toml`，填写模型和 API 密钥，然后检查配置：

```bash
.venv/bin/python -m ai_assistant config check
```

默认使用 DeepSeek，关闭 thinking 模式。

## 文档

- [使用指南](docs/usage.md)：安装、配置、模型调用、检查命令及测试。
- [工具使用](docs/tools.md)：只读工作区、DeepSeek 联网工具与原生框架装配。
- [架构说明](docs/architecture.md)：目录职责、依赖边界与后续能力规划。
- [框架复用记录](docs/llm-framework-reuse.md)：原生能力对照、行为差异与验证证据。
