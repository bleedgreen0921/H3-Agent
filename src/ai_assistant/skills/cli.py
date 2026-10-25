"""Inspect user-managed local Skills without acquiring or installing anything."""

import json
from .catalog import SkillCatalog, SkillError, configured_roots


def add_commands(commands):
    parser = commands.add_parser("skill", help="Inspect local Skills")
    actions = parser.add_subparsers(dest="action", required=True)
    for action in ("list", "inspect", "check"):
        command = actions.add_parser(action)
        command.add_argument("--config", default="assistant.toml")
        if action != "list":
            command.add_argument("skill_id")


async def execute(config, args):
    if config.skills is None:
        raise SkillError("Configure [skills] before loading skills.")
    catalog = SkillCatalog(configured_roots(config))
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
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0
    finally:
        catalog.close()
