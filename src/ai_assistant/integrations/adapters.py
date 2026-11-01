"""由可信 transport 显式选择服务适配；不根据模型文字猜测服务。"""

def service_adapter(entry):
    if entry.settings.transport in {'feishu', 'feishu_files'}:
        from ai_assistant.integrations import feishu_adapter
        return feishu_adapter
    return None
