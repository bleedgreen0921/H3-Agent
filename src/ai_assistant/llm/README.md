# 模型接入

状态：已实现基础接入。

- factory.py：从配置创建原生 LangChain 模型。
- providers.py：供应商默认地址、认证要求与 DeepSeek thinking 规则；只包含纯声明和校验。
- requests.py：诊断请求的限制副本、临时 HTTP 客户端生命周期和用量展示；保留 profile 重试设置。
- structured.py：DeepSeek 原生结构化 Runnable 的薄构造函数；不自建解析、结果或异常协议。
- checks.py：模型连通性和原生能力探测。

未来按实际需要增加模型能力档案及显式选择策略。模型接入不导入 Agent、记忆、工具或界面，不实现规划；其他模块复用原生模型与消息类型。

包 __init__.py 不急切导入子模块，避免配置读取时初始化模型依赖。

结构化接口已改为 create_structured_runnable。SDK 重试由 profile 管理，本层不叠加重试。完整对照、行为差异和测试证据见项目 docs/llm-framework-reuse.md。
