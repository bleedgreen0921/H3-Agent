# 用户接口

状态：已实现 CLI，尚无 HTTP API、Web 或桌面界面。

cli.py 负责 argparse 参数、命令分派、终端输出和退出码。包根 __main__.py 仅启动此入口，python -m ai_assistant 的命令保持不变。

后续按需要增加 api/，负责请求与事件传输。业务流程放在 agents/runtime，模型和工具协议放在各自模块。UI 技术确定后再建立项目根 frontend/，当前不预建空前端。

`chat.py` 提供 POSIX 非阻塞 stdin 交互。执行中 Ctrl+C 取消当前轮次并等待清理，
清理期间重复信号被忽略；空闲 Ctrl+C 返回 130。EOF、/exit 返回 0。
只显示最终回答、工具事件名称与 ID、本轮真实检索来源，不输出参数或原始诊断。

chat 需要持久会话与上下文预算；--session 重开会话，/new 异步创建新会话。
session init/list/delete 负责显式初始化、列表和指定会话删除，不构造模型。
每轮增加调用数和实际报告的用量汇总；预算/限额错误安全展示，持久连接失效返回 2。
