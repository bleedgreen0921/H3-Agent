"""显式 TOML 组件组合；只读取所选方案，不执行插值或外部连接。"""
from copy import deepcopy
from pathlib import Path
import tomllib


def read_toml(path):
    from ai_assistant.config import ConfigError
    try:
        with Path(path).open('rb') as stream:
            return tomllib.load(stream)
    except FileNotFoundError:
        raise ConfigError('Configuration file not found; specify --config or create assistant.toml.') from None
    except PermissionError:
        raise ConfigError('Permission denied when reading the configuration file.') from None
    except (tomllib.TOMLDecodeError, UnicodeDecodeError):
        raise ConfigError('Invalid TOML or UTF-8 in the configuration file.') from None
    except OSError:
        raise ConfigError('Unable to read the configuration file.') from None


def profile_options(document):
    from ai_assistant.config import ConfigError
    if not {'profiles', 'default_profile', 'includes'}.intersection(document):
        return None
    if set(document) - {'profiles', 'default_profile', 'includes'}:
        raise ConfigError('组合配置入口只允许 includes、default_profile 和 profiles。')
    profiles = document.get('profiles')
    default = document.get('default_profile')
    if (not isinstance(profiles, dict) or not profiles or not isinstance(default, str)
            or default not in profiles or any(not name.strip() for name in profiles)):
        raise ConfigError('组合配置需要有效的 profiles 和 default_profile。')
    return profiles, default


def _include_paths(value, base):
    from ai_assistant.config import ConfigError
    if not isinstance(value, list) or any(not isinstance(item, str) or not item.strip() for item in value):
        raise ConfigError('includes 必须是非空文件路径组成的数组。')
    return [base / item for item in value]


def _merge(target, source):
    from ai_assistant.config import ConfigError
    for key, value in source.items():
        if key not in target:
            target[key] = deepcopy(value)
        elif isinstance(target[key], dict) and isinstance(value, dict):
            _merge(target[key], value)
        else:
            # 不输出配置键或值；键也可能包含私有资源标识。
            raise ConfigError('配置组件重复定义同一配置项；请拆分组件或使用 overrides。')


def resolve_config(path, profile=None):
    from ai_assistant.config import ConfigError
    path = Path(path).absolute()
    document = read_toml(path)
    options = profile_options(document)
    if options is None:
        if profile is not None:
            raise ConfigError('旧单文件配置不支持 --profile；请使用组合配置入口。')
        return document, (path.resolve(),), None
    profiles, default = options
    selected = default if profile is None else profile
    if selected not in profiles:
        raise ConfigError('未知配置方案。')
    spec = profiles[selected]
    if not isinstance(spec, dict) or set(spec) - {'includes', 'overrides'}:
        raise ConfigError('配置方案只允许 includes 和 overrides。')
    overrides = spec.get('overrides', {})
    if not isinstance(overrides, dict):
        raise ConfigError('overrides 必须是配置表。')
    files = [path.resolve()]
    data = {}
    includes = _include_paths(document.get('includes', []), path.parent)
    includes += _include_paths(spec.get('includes', []), path.parent)
    for component in includes:
        resolved = component.resolve()
        if resolved in files:
            raise ConfigError('配置组件不能重复引用或引用入口自身。')
        part = read_toml(component)
        if {'includes', 'profiles', 'default_profile'}.intersection(part):
            raise ConfigError('配置组件不支持嵌套引用或方案定义。')
        _merge(data, part)
        files.append(resolved)
    # 仅顶层整体替换，不将旧服务的权限或参数混入新服务。
    data.update(deepcopy(overrides))
    return data, tuple(files), selected
