# 会话执行与资源管理

`assistant_session(config, model_name=None, *, retriever=None, session_id=None)` 管理模型、工具上下文和原生 Saver。
配置 sessions 时使用 PostgreSQL，否则 Python 调用保留 InMemorySaver；CLI chat 要求持久化及上下文预算。
`await session.new_conversation()` 新建会话，`session.session_id` 返回稳定 UUID。

`await session.run_turn(text, on_event=callback)` 返回 `AgentTurn(messages, sources, usage)`。
消息和 Document 保持框架原生类型；每轮原生 callback 收集真实来源和实际报告的用量。
用户事件回调继续接收原生工具事件；不会输出中间模型文本。

只有图成功结束且应用成功基点提交完成才返回成功。失败、取消、超时或崩溃不推进基点，
后续从明确成功 checkpoint 分支；无成功历史则更换原生线程，不自动重放工具。
打开期间持有数据库独占会话锁，连接失效会取消执行并等待清理。

退出关闭客户端和连接，不删除持久历史。取消等待工具已有的写入和进程清理机制，
不是强制停止 GPU 或保证外部副作用恰好一次的协议。
详细配置、API 迁移和验收见 [会话说明](../../../docs/sessions.md)。
