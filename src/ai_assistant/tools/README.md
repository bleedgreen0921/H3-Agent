# 工具与外部能力

已实现八个原生工具：工作区 list_directory、read_text、search_text、create_file、edit_file，DeepSeek web_search、web_fetch，以及 Bubblewrap run_command。

builtin.py 声明 BaseTool；context.py 装配可信 ToolRuntime 依赖、共享修改锁和资源生命周期；web.py 接入搜索服务；terminal.py 复用 LangChain ShellSession，薄执行策略落实 Bubblewrap 三档权限。文件实际访问和原子修改位于 workspace。

不创建自定义工具协议或 Agent 循环。写操作和终端默认禁用，详见 [工具使用](../../../docs/tools.md) 和 [写权限工具](../../../docs/write-tools.md)。
