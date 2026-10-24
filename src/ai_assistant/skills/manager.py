"""Public GitHub/local acquisition and atomic managed entry publication."""
from contextlib import contextmanager
import fcntl
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import shutil
from tempfile import TemporaryDirectory
from urllib.parse import quote, urlparse
from uuid import uuid4
import zipfile

import httpx
from .catalog import SkillError, copy_package, parse_skill


@contextmanager
def file_lock(path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('a') as stream:
        try:
            fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise SkillError('Another installation or recovery is running; retry later.') from None
        try:
            yield
        finally:
            fcntl.flock(stream, fcntl.LOCK_UN)


def repository_name(source):
    if source.startswith('https://'):
        url = urlparse(source)
        if url.hostname != 'github.com' or url.query or url.fragment or url.username or url.port:
            raise SkillError('Only public github.com repository HTTPS URLs are supported.')
        source = url.path.strip('/').removesuffix('.git')
    if not re.fullmatch(r'[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+', source):
        raise SkillError('Use OWNER/REPO or repository HTTPS URL, with --path and --ref separately.')
    return source


def extract_archive(archive, destination):
    total = 0
    seen = set()
    with zipfile.ZipFile(archive) as bundle:
        if len(bundle.infolist()) > 100000:
            raise SkillError('Archive contains too many entries.')
        for member in bundle.infolist():
            path = PurePosixPath(member.filename)
            mode = member.external_attr >> 16
            if path.is_absolute() or '..' in path.parts or '\\' in member.filename or not path.parts:
                raise SkillError('Unsafe archive path.')
            if member.filename in seen:
                raise SkillError('Duplicate archive entry.')
            seen.add(member.filename)
            kind = mode & 0o170000
            if kind not in (0, 0o100000, 0o040000):
                raise SkillError('Archive links and special files are unsupported.')
            total += member.file_size
            if total > 512 * 1024 * 1024:
                raise SkillError('Expanded archive exceeds 512 MiB.')
            target = destination.joinpath(*path.parts)
            if member.is_dir():
                target.mkdir(parents=True, exist_ok=True)
            else:
                target.parent.mkdir(parents=True, exist_ok=True)
                with bundle.open(member) as incoming, target.open('xb') as output:
                    copied = 0
                    while chunk := incoming.read(65536):
                        copied += len(chunk)
                        if copied > member.file_size:
                            raise SkillError('Archive size mismatch.')
                        output.write(chunk)
                target.chmod(0o755 if mode & 0o111 else 0o644)
    roots = list(destination.iterdir())
    if len(roots) != 1 or not roots[0].is_dir():
        raise SkillError('Expected one repository archive root.')
    return roots[0]


class GitHubSource:
    def __init__(self, client=None):
        self.client = client

    @contextmanager
    def fetch(self, source, ref=None):
        repo = repository_name(source)
        with TemporaryDirectory(prefix='assistant-github-') as temporary:
            root = Path(temporary)
            with httpx.Client(timeout=30, follow_redirects=False, trust_env=False) if self.client is None else _borrow(self.client) as client:
                def get_json(path):
                    response = client.get('https://api.github.com/repos/' + repo + path)
                    response.raise_for_status()
                    return response.json()
                selected = ref or get_json('')['default_branch']
                commit = get_json('/commits/' + quote(selected, safe=''))['sha']
                if not re.fullmatch('[0-9a-f]{40}', commit):
                    raise SkillError('GitHub returned an invalid commit.')
                archive = root / 'archive.zip'
                with client.stream('GET', f'https://codeload.github.com/{repo}/zip/{commit}') as response:
                    response.raise_for_status()
                    size = 0
                    with archive.open('wb') as output:
                        for chunk in response.iter_bytes(65536):
                            size += len(chunk)
                            if size > 128 * 1024 * 1024:
                                raise SkillError('Download exceeds 128 MiB.')
                            output.write(chunk)
            unpacked = root / 'unpacked'
            unpacked.mkdir()
            try:
                checkout = extract_archive(archive, unpacked)
            except (zipfile.BadZipFile, RuntimeError):
                raise SkillError('Invalid or unsupported GitHub archive; existing entries preserved.') from None
            yield checkout, dict(kind='github', repository=repo, ref=selected, commit=commit)


@contextmanager
def _borrow(value):
    yield value


class SkillManager:
    def __init__(self, root, github=None):
        self.root = Path(root).absolute()
        self.store = self.root.parent / '.assistant-skills'
        self.github = github or GitHubSource()

    def _manifest(self, key):
        if not re.fullmatch('[0-9a-f]{24}', key):
            raise SkillError('Invalid managed entry.')
        return self.store / 'manifests' / (key + '.json')

    def _publish(self, source, provenance, *, key=None):
        key = key or hashlib.sha256(json.dumps({k: v for k, v in provenance.items() if k != 'commit'}, sort_keys=True).encode()).hexdigest()[:24]
        with TemporaryDirectory(prefix='prepare-', dir=self.store) as temporary:
            staged = Path(temporary) / 'package'
            digest = copy_package(source, staged)
            metadata, _ = parse_skill(staged / 'SKILL.md')
            entry = self.root / (metadata['name'] + '-' + key)
            manifest_path = self._manifest(key)
            if manifest_path.exists():
                old = self._read_record(manifest_path)
                entry = self.root / old['entry']
                if not entry.is_symlink() or str(entry.resolve()) != old['object']:
                    raise SkillError('Managed entry was externally modified.')
            elif entry.exists() or entry.is_symlink():
                raise SkillError('Entry already exists and is not managed by this installer.')
            obj = self.store / 'objects' / (digest + '-' + uuid4().hex)
            obj.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(staged), obj)
            record = dict(provenance, key=key, entry=entry.name, digest=digest, object=str(obj))
            manifest_path.parent.mkdir(parents=True, exist_ok=True)
            obj.with_suffix('.source.json').write_text(json.dumps(record, indent=2))
            temporary_manifest = manifest_path.with_suffix('.tmp')
            temporary_manifest.write_text(json.dumps(record, indent=2))
            temporary_link = self.root / ('.publish-' + uuid4().hex)
            temporary_link.symlink_to(obj, target_is_directory=True)
            # Entry is the commit point. Each immutable object also owns its
            # provenance, so crash recovery always follows the published version.
            previous_manifest = manifest_path.read_bytes() if manifest_path.exists() else None
            try:
                os.replace(temporary_manifest, manifest_path)
                os.replace(temporary_link, entry)
            except BaseException:
                temporary_link.unlink(missing_ok=True)
                if previous_manifest is None:
                    manifest_path.unlink(missing_ok=True)
                else:
                    temporary_manifest.write_bytes(previous_manifest)
                    os.replace(temporary_manifest, manifest_path)
                raise
            return record

    def install(self, source, path=None, ref=None, *, key=None):
        self.root.mkdir(parents=True, exist_ok=True)
        with file_lock(self.store / 'manage.lock'):
            return self._install(source, path, ref, key=key)

    def _install(self, source, path=None, ref=None, *, key=None):
        if Path(source).is_dir():
            if path or ref:
                raise SkillError('--path/--ref apply only to GitHub sources.')
            local = Path(source).resolve()
            return self._publish(local, dict(kind='local', source=str(local)), key=key)
        with self.github.fetch(source, ref) as (checkout, provenance):
            relative = PurePosixPath(path or '.')
            if relative.is_absolute() or '..' in relative.parts:
                raise SkillError('Skill path must be inside the repository.')
            return self._publish(checkout / str(relative), dict(provenance, path=str(relative)), key=key)

    def _read_record(self, manifest):
        data = json.loads(manifest.read_text())
        if not isinstance(data.get('entry'), str) or Path(data['entry']).name != data['entry'] or data['entry'].startswith('.'):
            raise SkillError('Invalid managed entry path.')
        entry = self.root / data['entry']
        if entry.is_symlink():
            target = entry.resolve()
            if not target.is_relative_to(self.store / 'objects'):
                raise SkillError('Managed entry points outside the object store.')
            # The atomic entry points to both package and provenance of that version.
            record = target.with_suffix('.source.json')
            if record.is_file():
                data = json.loads(record.read_text())
        return data

    def record_for(self, skill):
        for manifest in (self.store / 'manifests').glob('*.json'):
            data = self._read_record(manifest)
            entry = self.root / data['entry']
            if 'skill:' + hashlib.sha256(str(entry).encode()).hexdigest()[:24] == skill.id:
                return data
        raise SkillError('This entry is managed by another installer; use its original management tool.')

    def update(self, skill, ref=None):
        with file_lock(self.store / 'manage.lock'):
            record = self.record_for(skill)
            if record['kind'] == 'local':
                if ref:
                    raise SkillError('--ref applies only to GitHub sources.')
                return self._install(record['source'], key=record['key'])
            return self._install(record['repository'], record['path'], ref or record['ref'], key=record['key'])

    def remove(self, skill):
        with file_lock(self.store / 'manage.lock'):
            record = self.record_for(skill)
            entry = self.root / record['entry']
            if not entry.is_symlink() or str(entry.resolve()) != record['object']:
                raise SkillError('Managed entry was externally modified.')
            entry.unlink()
            self._manifest(record['key']).unlink()

    def discover(self, source, ref=None):
        with self.github.fetch(source, ref) as (checkout, provenance):
            result = []
            for path in sorted(checkout.rglob('SKILL.md')):
                try:
                    metadata, _ = parse_skill(path)
                    result.append(dict(path=path.parent.relative_to(checkout).as_posix(), metadata=metadata))
                except SkillError as exc:
                    result.append(dict(path=path.parent.relative_to(checkout).as_posix(), error=str(exc)))
            return dict(source=provenance, skills=result)
