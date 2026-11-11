"""传输适配器注册表：core 不根据具体产品 import 业务适配器。"""
_ADAPTERS = {}
_DEFAULTS_LOADED = False


def register_adapter(transport, adapter):
    _ADAPTERS[transport] = adapter
    return adapter


def _load_defaults():
    global _DEFAULTS_LOADED
    if _DEFAULTS_LOADED:
        return
    _DEFAULTS_LOADED = True
    from ai_assistant.integrations import feishu_adapter
    register_adapter('feishu', feishu_adapter)
    register_adapter('feishu_files', feishu_adapter)


def service_adapter(entry):
    _load_defaults()
    return _ADAPTERS.get(entry.settings.transport)
