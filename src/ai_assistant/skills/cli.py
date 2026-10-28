"""Inspect user-managed local Skills without acquiring or installing anything."""

import json
from .catalog import SkillCatalog, SkillError, configured_roots


def add_commands(commands):
    parser = commands.add_parser("skill", help="Inspect local Skills")
    actions = parser.add_subparsers(dest="action", required=True)
    for action in ("list", "check"):
        command = actions.add_parser(action)
        command.add_argument("--config", default="assistant.toml")
        if action != "list":
            command.add_argument("skill_id")


async def execute(config, args):
    if config.skills is None:
        raise SkillError("Configure [skills] before loading skills.")
    catalog = SkillCatalog(configured_roots(config), disabled=config.skills.disabled)
    try:
        if args.action == "list":
            result = dict(
                skills=[s.info() for s in catalog.skills.values()],
                diagnostics=catalog.diagnostics,
            )
        else:
            skill = catalog.resolve(args.skill_id)
            result = dict(skill.info(), metadata=skill.metadata)
            if args.action == "check":
                from .diagnostics import check_skill

                result["check"] = check_skill(
                    skill, terminal_enabled=config.tools.terminal is not None
                )
        channel = config.channels.feishu
        groups = {name: {'mode': 'default' if grant.skill_ids is None else
                         ('disabled' if not grant.skill_ids else 'override'),
                         'skill_ids': channel.default_group_skill_ids if grant.skill_ids is None else grant.skill_ids}
                  for name, grant in channel.groups.items()}
        invalid = sorted(set(channel.default_group_skill_ids + [s for g in groups.values() for s in g['skill_ids']]) - set(catalog.skills))
        result['group_authorization'] = dict(default=channel.default_group_skill_ids, groups=groups, invalid_ids=invalid)
        print(json.dumps(result, ensure_ascii=False, indent=2))
        unmet = any(not dep.get('satisfied', False) for dep in result.get('check', {}).get('known_python_dependencies', []))
        return 2 if invalid or catalog.diagnostics or unmet else 0
    finally:
        catalog.close()
