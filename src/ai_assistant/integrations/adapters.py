"""传输适配器注册表；没有内置产品适配器，插件可显式注册。"""
_ADAPTERS = {}


def register_adapter(transport, adapter):
    _ADAPTERS[transport] = adapter
    return adapter


def service_adapter(entry):
    return _ADAPTERS.get(entry.settings.transport)
