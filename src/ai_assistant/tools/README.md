# 工具与外部能力

已实现只读工作区的 list_directory、read_text、search_text，以及 DeepSeek 的 web_search、web_fetch。

builtin.py 负责原生 BaseTool 声明；context.py 装配可信 ToolRuntime 上下文和客户端生命周期；web.py 实现独立的 Search/Fetch HTTP 适配。文件实际访问由 workspace 负责。

对外使用 create_builtin_tools、ToolContext、tool_context，不创建自定义工具协议或 Agent 执行循环。后续 MCP、浏览器、计算机工具按实际需要接入。详细配置、限制与示例见 [工具使用文档](../../../docs/tools.md)。
