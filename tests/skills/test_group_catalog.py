"""群聊只共享显式授权的不可变 Skill 包。"""
import pytest
from ai_assistant.config import ConfigError
from ai_assistant.integrations.policy import Caller
from ai_assistant.skills.catalog import SkillCatalog, SkillError, skills_for_caller, validate_group_skills
from tests.integrations.test_mcp import config


def test_group_defaults_overrides_and_resources(tmp_path):
    root = tmp_path / 'skills'
    for name in ('summary', 'private'):
        path = root / name
        path.mkdir(parents=True)
        (path / 'SKILL.md').write_text(f'---\nname: {name}\ndescription: fixture\n---\n指导{name}')
        (path / 'resource.txt').write_text(name)
    with_catalog = SkillCatalog([root])
    try:
        public = with_catalog.resolve('summary').id
        private = with_catalog.resolve('private').id
        c = config(tmp_path, skills={}, integrations={'feishu': {'app_id': 'cli_app', 'app_secret': 'x', 'owner_open_id': 'owner'}}, channels={'feishu': {
            'default_group_skill_ids': [public], 'groups': {'group': {}, 'disabled': {'skill_ids': []}, 'override': {'skill_ids': [private]}}}})
        validate_group_skills(c, with_catalog)
        caller = Caller('feishu_group', binding='cli_app:group:owner')
        allowed = skills_for_caller(c, with_catalog, caller)
        assert list(allowed.skills) == [public]
        assert allowed.diagnostics == []
        assert list(allowed.active({private: with_catalog.skills[private].digest})) == []
        with pytest.raises(SkillError):
            allowed.read(private, 'resource.txt')
        assert allowed.read(public, 'resource.txt')['content'] == 'summary'
        assert skills_for_caller(c, with_catalog, Caller('feishu_group', binding='cli_app:disabled:owner')) is None
        assert list(skills_for_caller(c, with_catalog, Caller('feishu_group', binding='cli_app:override:owner')).skills) == [private]
        assert skills_for_caller(c, with_catalog, Caller()) is with_catalog
        with pytest.raises(ConfigError):
            validate_group_skills(c, None)
    finally:
        with_catalog.close()
