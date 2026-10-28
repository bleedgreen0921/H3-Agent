"""Safe Agent Skills parsing and session-owned immutable package snapshots."""
from dataclasses import dataclass
import hashlib
from pathlib import Path
import re
import shutil
import stat
from tempfile import TemporaryDirectory


class SkillError(ValueError):
    """Safe, actionable skill diagnostic."""


def parse_skill(path):
    try:
        import yaml
    except ImportError:
        raise SkillError('Install the skills optional dependencies (PyYAML, packaging).') from None

    class Loader(yaml.SafeLoader):
        pass

    def mapping(loader, node, deep=False):
        result = {}
        for key, value in node.value:
            key = loader.construct_object(key, deep=True)
            if not isinstance(key, str) or key in result:
                raise SkillError('YAML keys must be unique strings.')
            result[key] = loader.construct_object(value, deep=True)
        return result

    Loader.add_constructor(yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG, mapping)
    try:
        text = Path(path).read_text(encoding='utf-8')
        if len(text.encode()) > 1048576:
            raise SkillError('SKILL.md exceeds 1 MiB.')
        lines = text.splitlines(keepends=True)
        if not lines or lines[0].strip() != '---':
            raise SkillError('SKILL.md requires YAML frontmatter.')
        end = next((i for i in range(1, len(lines)) if lines[i].strip() == '---'), None)
        if end is None:
            raise SkillError('Unterminated YAML frontmatter.')
        data = yaml.load(''.join(lines[1:end]), Loader=Loader)
    except (yaml.YAMLError, UnicodeError, RecursionError):
        raise SkillError('Invalid YAML structure or unsupported YAML tag.') from None
    if not isinstance(data, dict):
        raise SkillError('Frontmatter must be a mapping.')
    remaining = 10000
    def validate_structure(value, ancestors=frozenset(), depth=0):
        nonlocal remaining
        remaining -= 1
        if remaining < 0:
            raise SkillError('YAML structure exceeds the supported size.')
        if depth > 40:
            raise SkillError('YAML nesting exceeds the supported depth.')
        if isinstance(value, (list, dict)):
            if id(value) in ancestors:
                raise SkillError('Recursive YAML aliases are unsupported.')
            ancestors = ancestors | {id(value)}
            for child in (value.values() if isinstance(value, dict) else value):
                validate_structure(child, ancestors, depth + 1)
        elif not isinstance(value, (str, int, float, bool, type(None))):
            raise SkillError('YAML metadata must contain plain scalar values, lists or mappings.')
    validate_structure(data)
    name, description = data.get('name'), data.get('description')
    if not isinstance(name, str) or not re.fullmatch(r'[a-z0-9]+(?:-[a-z0-9]+)*', name) or len(name) > 64:
        raise SkillError('name must be 1–64 lowercase letters, numbers and single hyphens.')
    if not isinstance(description, str) or not description.strip() or len(description) > 1024:
        raise SkillError('description must be a nonempty string of at most 1024 characters.')
    if 'compatibility' in data and (not isinstance(data['compatibility'], str) or not 1 <= len(data['compatibility']) <= 500):
        raise SkillError('compatibility must be a string of 1–500 characters.')
    return data, ''.join(lines[end + 1:])


def copy_package(source, target):
    """Resolve only internal links; reject special files, cycles and oversized trees."""
    source, target = Path(source).resolve(strict=True), Path(target)
    total = 0
    count = 0
    digest = hashlib.sha256()

    def visit(path, destination, ancestors):
        nonlocal total, count
        count += 1
        if count > 100000:
            raise SkillError('Package contains too many entries.')
        resolved = path.resolve(strict=True)
        if not resolved.is_relative_to(source):
            raise SkillError('Package link escapes its root.')
        mode = resolved.stat().st_mode
        if stat.S_ISDIR(mode):
            if resolved in ancestors:
                raise SkillError('Cyclic package link.')
            destination.mkdir(parents=True, exist_ok=True)
            for child in sorted(resolved.iterdir()):
                visit(child, destination / child.name, ancestors | {resolved})
        elif stat.S_ISREG(mode):
            size = resolved.stat().st_size
            total += size
            if total > 512 * 1024 * 1024:
                raise SkillError('Expanded package exceeds 512 MiB.')
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(resolved, destination)
            destination.chmod(0o755 if mode & 0o111 else 0o644)
            digest.update(destination.relative_to(target).as_posix().encode() + b'\0' + str(mode & 0o111).encode() + b'\0')
            with destination.open('rb') as stream:
                for chunk in iter(lambda: stream.read(65536), b''):
                    digest.update(chunk)
            digest.update(b'\0')
        else:
            raise SkillError('Special package files are unsupported.')
    visit(source, target, set())
    return digest.hexdigest()


def configured_roots(config):
    base = config.config_path.parent if config.config_path else Path.cwd()
    workspace = Path(config.workspace.root)
    workspace = (base / workspace).resolve()
    roots = [workspace / '.agents/skills', Path.home() / '.agents/skills']
    if config.skills:
        roots += [(base / Path(p)).absolute() for p in config.skills.roots]
    return list(dict.fromkeys(p.absolute() for p in roots))


@dataclass(frozen=True)
class Skill:
    id: str
    metadata: dict
    body: str
    digest: str
    path: Path
    warnings: tuple[str, ...]

    @property
    def sandbox_path(self):
        return str(self.path)

    def info(self):
        return dict(id=self.id, name=self.metadata['name'], description=self.metadata['description'],
                    digest=self.digest, base_path=self.sandbox_path, warnings=self.warnings,
                    resources=sorted(p.relative_to(self.path).as_posix() for p in self.path.rglob('*') if p.is_file()))


class SkillCatalog:
    def __init__(self, roots, *, disabled=()):
        self._temporary = TemporaryDirectory(prefix='assistant-skills-')
        self.skills = {}
        self.diagnostics = []
        disabled = frozenset(disabled)
        try:
            for root in roots:
                root = Path(root).absolute()
                if not root.exists():
                    continue
                for entry in sorted(root.iterdir()):
                    if entry.name.startswith('.') or not entry.is_dir():
                        continue
                    identity = 'skill:' + hashlib.sha256(str(root / entry.name).encode()).hexdigest()[:24]
                    # 项目级禁用优先于自动发现，不删除共享包，也不授予隐含权限。
                    if entry.name in disabled or identity in disabled:
                        continue
                    destination = Path(self._temporary.name).resolve() / identity.replace(':', '-')
                    try:
                        digest = copy_package(entry, destination)
                        metadata, body = parse_skill(destination / 'SKILL.md')
                        if metadata['name'] in disabled:
                            shutil.rmtree(destination)
                            continue
                        warnings = []
                        if metadata['name'] != entry.name:
                            warnings.append('Directory name differs from skill name.')
                        for key in ('allowed-tools', 'context', 'agent', 'mcp', 'fork', 'model'):
                            if key in metadata:
                                warnings.append(f'{key} is advisory; client/MCP/fork permissions are not implemented.')
                        self.skills[identity] = Skill(identity, metadata, body, digest, destination, tuple(warnings))
                    except (SkillError, OSError) as exc:
                        self.diagnostics.append(f'{entry.name}: {exc}')
                        shutil.rmtree(destination, ignore_errors=True)
        except BaseException:
            self.close()
            raise

    def close(self):
        self._temporary.cleanup()

    def resolve(self, identity):
        if identity in self.skills:
            return self.skills[identity]
        matches = [s for s in self.skills.values() if s.metadata['name'] == identity]
        if len(matches) != 1:
            raise SkillError('Unknown or ambiguous skill; choose full ID: ' + ', '.join(s.id for s in matches))
        return matches[0]

    def search(self, query='', offset=0, limit=20):
        if offset < 0 or not 1 <= limit <= 100:
            raise SkillError('Invalid pagination.')
        matches = [dict(id=s.id, name=s.metadata['name'], description=s.metadata['description']) for s in self.skills.values()
                   if query.casefold() in (s.metadata['name'] + ' ' + s.metadata['description']).casefold()]
        return dict(skills=matches[offset:offset + limit], total=len(matches), next_offset=offset + limit if offset + limit < len(matches) else None)

    def active(self, records):
        return [s for identity, digest in (records or {}).items()
                if digest is not None and (s := self.skills.get(identity)) is not None and s.digest == digest]

    def read(self, identity, resource, offset=0, limit=20000):
        skill = self.resolve(identity)
        path = skill.path / resource
        if Path(resource).is_absolute() or not path.resolve().is_relative_to(skill.path) or not path.is_file():
            raise SkillError('Resource must be a file within the skill package.')
        if offset < 0 or not 1 <= limit <= 20000:
            raise SkillError('Invalid resource pagination.')
        result = dict(path=skill.sandbox_path + '/' + resource, bytes=path.stat().st_size)
        binary_suffixes = {'.pdf', '.png', '.jpg', '.jpeg', '.gif', '.webp', '.zip', '.gz', '.woff', '.woff2', '.docx', '.xlsx', '.pptx'}
        if path.suffix.lower() in binary_suffixes:
            return dict(result, binary=True)
        import codecs
        with path.open('rb') as stream:
            stream.seek(offset)
            raw = stream.read(limit)
            try:
                if b'\0' in raw:
                    raise UnicodeDecodeError('utf-8', raw, 0, 1, 'binary data')
                decoder = codecs.getincrementaldecoder('utf-8')()
                content = decoder.decode(raw, final=False)
                consumed = len(raw) - len(decoder.getstate()[0])
                if raw and not consumed:
                    extra = stream.read(3)
                    content += decoder.decode(extra, final=False)
                    consumed = len(raw) + len(extra) - len(decoder.getstate()[0])
                if not consumed and raw:
                    raise UnicodeDecodeError('utf-8', raw, 0, len(raw), 'incomplete data')
                result.update(content=content, offset=offset, next_offset=offset + consumed if offset + consumed < result['bytes'] else None)
            except UnicodeDecodeError:
                result['binary'] = True
        return result


class RestrictedSkillCatalog(SkillCatalog):
    """共享不可变包快照，只暴露当前群明确授权的项目。"""

    def __init__(self, source, identities):
        self.skills = {}
        self.diagnostics = []
        for identity in identities:
            if identity not in source.skills:
                raise SkillError('群聊授权的 Skill ID 不存在，请检查配置。')
            self.skills[identity] = source.skills[identity]

    def close(self):
        # 包生命周期由进程级原始目录管理。
        pass


def group_skill_ids(config, binding):
    groups = config.channels.feishu.groups
    matches = [value for chat, value in groups.items()
               if binding == f'{config.integrations.feishu.app_id}:{chat}:{config.integrations.feishu.owner_open_id}'] if config.integrations.feishu else []
    if len(matches) != 1:
        raise SkillError('无法确定当前群的 Skills 授权。')
    selected = matches[0].skill_ids
    return config.channels.feishu.default_group_skill_ids if selected is None else selected


def skills_for_caller(config, source, caller):
    if caller.channel != 'feishu_group':
        return source
    identities = group_skill_ids(config, caller.binding)
    if not identities:
        return None
    if source is None:
        raise SkillError('群聊已授权 Skills，但未配置本地 Skills。')
    return RestrictedSkillCatalog(source, identities)


def validate_group_skills(config, source):
    from ai_assistant.config import ConfigError
    lists = [config.channels.feishu.default_group_skill_ids]
    lists += [g.skill_ids for g in config.channels.feishu.groups.values() if g.skill_ids is not None]
    for identities in lists:
        for identity in identities:
            if source is None or identity not in source.skills:
                raise ConfigError('群聊授权的 Skill ID 不存在，请检查 default_group_skill_ids / skill_ids。')
