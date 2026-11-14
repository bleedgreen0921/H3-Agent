"""飞书适配器注册入口。"""
from ai_assistant.integrations import adapters


def install():
    """将飞书和飞书文件传输绑定到同一个业务适配器。"""
    from ai_assistant.integrations import feishu_adapter
    adapters.register_adapter('feishu', feishu_adapter)
    adapters.register_adapter('feishu_files', feishu_adapter)
