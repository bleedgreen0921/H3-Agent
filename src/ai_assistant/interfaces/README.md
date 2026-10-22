# 用户接口

状态：已实现 CLI，尚无 HTTP API、Web 或桌面界面。

cli.py 负责 argparse 参数、命令分派、终端输出和退出码。包根 __main__.py 仅启动此入口，python -m ai_assistant 的命令保持不变。

后续按需要增加 api/，负责请求与事件传输。业务流程放在 agents/runtime，模型和工具协议放在各自模块。UI 技术确定后再建立项目根 frontend/，当前不预建空前端。
