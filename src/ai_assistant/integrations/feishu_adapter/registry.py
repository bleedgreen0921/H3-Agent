"""飞书适配器注册入口。"""
from ai_assistant.integrations import adapters


def install():
    from ai_assistant.integrations import feishu_adapter
    adapters.register_adapter('feishu', feishu_adapter)
    adapters.register_adapter('feishu_files', feishu_adapter)
