import io
import json
from pathlib import Path
import zipfile

import httpx
import pytest

from ai_assistant.skills.catalog import SkillCatalog, SkillError, parse_skill
from ai_assistant.skills.manager import GitHubSource, SkillManager, extract_archive
from ai_assistant.skills.dependencies import Transaction, inventory


def package(root, name='example', body='Follow these rules.', extra=''):
    root.mkdir(parents=True, exist_ok=True)
    (root / 'SKILL.md').write_text(f'---\nname: {name}\ndescription: Test skill\n{extra}---\n{body}')
    (root / 'examples').mkdir(exist_ok=True)
    (root / 'examples/test.txt').write_text('Example')
    return root


def test_snapshot_conflicts_resources_and_root_links(tmp_path):
    roots = [tmp_path / 'a', tmp_path / 'b']
    one = package(roots[0] / 'example')
    roots[1].mkdir()
    (roots[1] / 'linked').symlink_to(one)
    catalog = SkillCatalog(roots)
    try:
        assert len(catalog.skills) == 2
        with pytest.raises(SkillError, match='ambiguous'):
            catalog.resolve('example')
        skill = next(iter(catalog.skills.values()))
        (one / 'examples/test.txt').write_text('Changed')
        assert catalog.read(skill.id, 'examples/test.txt')['content'] == 'Example'
        assert catalog.search(limit=1)['next_offset'] == 1
        with pytest.raises(SkillError):
            catalog.read(skill.id, '../SKILL.md')
        assert catalog.active({skill.id: 'stale'}) == []
    finally:
        catalog.close()
    assert not skill.path.exists()


@pytest.mark.parametrize('extra', ['name: duplicate\n', 'metadata: !!python/object:os.system {}\n', 'metadata: {a: 1, a: 2}\n'])
def test_reject_yaml(tmp_path, extra):
    p = package(tmp_path / 'example', extra=extra)
    with pytest.raises(SkillError):
        parse_skill(p / 'SKILL.md')


def test_unknown_fields_and_escape(tmp_path):
    p = package(tmp_path / 'skills/example', extra='x-extension: [a, b]\nallowed-tools: Bash\n')
    metadata, body = parse_skill(p / 'SKILL.md')
    assert metadata['x-extension'] == ['a', 'b']
    (p / 'escape').symlink_to(tmp_path)
    catalog = SkillCatalog([p.parent])
    try:
        assert not catalog.skills
        assert 'escapes' in catalog.diagnostics[0]
    finally:
        catalog.close()


def test_managed_local_lifecycle(tmp_path):
    source = package(tmp_path / 'source')
    root = tmp_path / 'shared'
    manager = SkillManager(root)
    first = manager.install(str(source))
    catalog = SkillCatalog([root])
    try:
        skill = catalog.resolve('example')
        (source / 'SKILL.md').write_text('invalid')
        with pytest.raises(SkillError):
            manager.update(skill)
        assert (root / first['entry']).resolve() == Path(first['object'])
        package(source, body='New version')
        updated = manager.update(skill)
        assert updated['digest'] != first['digest']
        assert skill.body == 'Follow these rules.'
        fresh = SkillCatalog([root])
        try:
            assert fresh.resolve(skill.id).body == 'New version'
            manager.remove(fresh.resolve(skill.id))
            assert not (root / first['entry']).exists()
        finally:
            fresh.close()
    finally:
        catalog.close()


def test_github_default_branch_commit_and_discover(tmp_path):
    sha = 'a' * 40
    blob = io.BytesIO()
    with zipfile.ZipFile(blob, 'w') as z:
        z.writestr('repo/skills/example/SKILL.md', '---\nname: example\ndescription: Test\n---\nRules')
    urls = []
    def handler(request):
        urls.append(str(request.url))
        if request.url.host == 'codeload.github.com':
            return httpx.Response(200, content=blob.getvalue())
        if '/commits/' in request.url.path:
            return httpx.Response(200, json={'sha': sha})
        return httpx.Response(200, json={'default_branch': 'main'})
    with httpx.Client(transport=httpx.MockTransport(handler)) as client:
        manager = SkillManager(tmp_path / 'skills', GitHubSource(client))
        assert manager.discover('owner/repo')['skills'][0]['path'] == 'skills/example'
        installed = manager.install('owner/repo', 'skills/example')
        assert installed['commit'] == sha
        assert any(url.endswith('/zip/' + sha) for url in urls)


@pytest.mark.parametrize('name', ['../escape', '/absolute', 'repo/../../escape', 'repo/a\\b'])
def test_unsafe_archive(tmp_path, name):
    archive = tmp_path / 'archive.zip'
    with zipfile.ZipFile(archive, 'w') as z:
        z.writestr(name, 'bad')
    output = tmp_path / 'output'
    output.mkdir()
    with pytest.raises(SkillError):
        extract_archive(archive, output)


def test_recovery_only_removes_matching_files(tmp_path):
    source = tmp_path / 'stage'
    source.write_text('new')
    live = tmp_path / 'live'
    live.mkdir()
    old = live / 'package.json'
    old.write_text('old')
    tx = Transaction(tmp_path / 'journal', [live])
    tx.publish(source, live / 'new')
    tx.replace(old, source)
    Transaction(tx.root, [live]).recover()
    assert not (live / 'new').exists()
    assert old.read_text() == 'old'
    tx.publish(source, live / 'new')
    (live / 'new').write_text('external')
    with pytest.raises(SkillError, match='externally'):
        Transaction(tx.root, [live]).recover()
    assert (live / 'new').read_text() == 'external'


def test_atomic_entry_failure_preserves_old_provenance(tmp_path, monkeypatch):
    import os
    source = package(tmp_path / 'source')
    root = tmp_path / 'shared'
    manager = SkillManager(root)
    first = manager.install(str(source))
    catalog = SkillCatalog([root])
    try:
        skill = catalog.resolve('example')
        package(source, body='Changed')
        original = os.replace
        def fail_link(incoming, target):
            if Path(incoming).name.startswith('.publish-'):
                raise OSError('fixture publication failure')
            return original(incoming, target)
        monkeypatch.setattr(os, 'replace', fail_link)
        with pytest.raises(OSError):
            manager.update(skill)
        assert manager.record_for(skill) == first
        assert (root / first['entry']).resolve() == Path(first['object'])
        assert not list(root.glob('.publish-*'))
    finally:
        catalog.close()


def test_archive_links_and_size_rejected(tmp_path, monkeypatch):
    archive = tmp_path / 'linked.zip'
    with zipfile.ZipFile(archive, 'w') as z:
        info = zipfile.ZipInfo('repo/link')
        info.external_attr = 0o120777 << 16
        z.writestr(info, '/etc/passwd')
    with pytest.raises(SkillError, match='links and special'):
        extract_archive(archive, tmp_path / 'unpacked')
    class FakeArchive:
        def __enter__(self):
            return self
        def __exit__(self, *args):
            pass
        def infolist(self):
            info = zipfile.ZipInfo('repo/bomb')
            info.file_size = 513 * 1024 * 1024
            return [info]
    monkeypatch.setattr(zipfile, 'ZipFile', lambda path: FakeArchive())
    with pytest.raises(SkillError, match='512 MiB'):
        extract_archive(archive, tmp_path / 'unpacked')


def test_github_update_failure_and_commit_pin(tmp_path):
    current = 'a' * 40
    failed = False
    blob = io.BytesIO()
    with zipfile.ZipFile(blob, 'w') as z:
        z.writestr('repo/example/SKILL.md', '---\nname: example\ndescription: Test\n---\nRules')
    requested = []
    def handler(request):
        requested.append(str(request.url))
        if failed:
            return httpx.Response(503)
        if request.url.host == 'codeload.github.com':
            return httpx.Response(200, content=blob.getvalue())
        if '/commits/' in request.url.path:
            ref = request.url.path.rsplit('/', 1)[-1]
            return httpx.Response(200, json={'sha': ref if len(ref) == 40 else current})
        return httpx.Response(200, json={'default_branch': 'main'})
    with httpx.Client(transport=httpx.MockTransport(handler)) as client:
        manager = SkillManager(tmp_path / 'shared', GitHubSource(client))
        original = manager.install('owner/repo', 'example', ref=current)
        catalog = SkillCatalog([manager.root])
        try:
            skill = catalog.resolve('example')
            current = 'b' * 40
            updated = manager.update(skill)
            assert updated['commit'] == original['commit']
            failed = True
            with pytest.raises(httpx.HTTPError):
                manager.update(skill)
            assert manager.record_for(skill) == updated
            assert (manager.root / original['entry']).is_symlink()
        finally:
            catalog.close()


def test_unicode_resource_pagination_and_binary(tmp_path):
    source = package(tmp_path / 'skills/example')
    (source / 'examples/test.txt').write_text('中文资源')
    (source / 'sample.pdf').write_bytes(b'%PDF-1.7\nASCII binary PDF')
    (source / 'binary').write_bytes(b'abc\0def')
    catalog = SkillCatalog([source.parent])
    try:
        skill = catalog.resolve('example')
        offset, text = 0, ''
        while offset is not None:
            page = catalog.read(skill.id, 'examples/test.txt', offset, limit=1)
            text += page['content']
            offset = page['next_offset']
        assert text == '中文资源'
        assert catalog.read(skill.id, 'sample.pdf')['binary']
        assert catalog.read(skill.id, 'binary')['binary']
    finally:
        catalog.close()
