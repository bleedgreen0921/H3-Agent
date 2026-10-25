
import pytest

from ai_assistant.skills.catalog import SkillCatalog, SkillError, parse_skill


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
