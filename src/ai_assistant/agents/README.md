# 原生单 Agent

`create_assistant_agent` 返回原生 create_agent 编译图，使用 AgentState、原生消息和 ToolContext。
不重写图调度。可注入 checkpointer、上下文配置、摘要模型和两个模型的显式输入预算。

中间件顺序为模型限额、工具限额、可选 RAG、原生摘要、原生工具结果编辑、最终输入检查。
摘要通过公开 summarizer 扩展点执行，保留原生切分与消息替换，不使用框架额外重试。
预算检查复用原生计数和 trim_messages；应用只补保护规则及工具配对校验。
系统提示要求证据引用、外部数据隔离、读取后精确编辑和基于已观察结果报告完成。
