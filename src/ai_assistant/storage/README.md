# 持久化

已实现 PostgreSQL RAG 存储，以及原生 LangGraph 异步 PostgreSQL Saver 的资源装配。
两者配置和可选依赖独立，可以复用同一数据库服务。

sessions 只保存会话索引、工作区绑定、线程归属、明确成功 checkpoint 和未完成标记。
消息、摘要、工具结果和执行状态由原生 Saver 保存，不建立消息副本或通用 CRUD 基类。
session init 是显式 schema 入口；启动只检查。会话锁、删除重试及版本检查也在此实现。

storage 不导入 Agent、runtime、模型或工具。长期记忆 Store 仍未实现。
详细说明见 [会话说明](../../../docs/sessions.md)。
