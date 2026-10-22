# 工作区文件访问

ReadOnlyWorkspace 实现受限目录列表、UTF-8 读取、字面量搜索及完整文件摘要。WritableWorkspace 复用其路径策略，实现 create-only 文件创建与带摘要检查的原子精确编辑。均不依赖 Agent 或 LangGraph。

共享相对路径、符号链接、凭据排除和大小限制；文件策略不是抵御恶意本地并发修改的操作系统沙箱。终端的 Bubblewrap 边界由 tools/terminal 负责。产物登记、媒体转换和持久元数据尚未实现。

详见 [工具使用](../../../docs/tools.md) 和 [写权限工具](../../../docs/write-tools.md)。
