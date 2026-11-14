"""Inspect user-managed local Skills without acquiring or installing anything."""

import json
from .catalog import SkillCatalog, SkillError, configured_roots


def add_commands(commands):
    """注册本地 Skill 列表与检查命令。"""
    parser = commands.add_parser("skill", help="Inspect local Skills")
    actions = parser.add_subparsers(dest="action", required=True)
    for action in ("list", "check"):
        command = actions.add_parser(action)
        command.add_argument("--config", default="assistant.toml")
        if action != "list":
            command.add_argument("skill_id")


async def execute(config, args):
    """发现本地 Skills，展示目录或检查指定包，然后释放临时快照。"""
    if config.skills is None:
        raise SkillError("Configure [skills] before loading skills.")

    # 只扫描显式配置的根目录，并保留禁用项及配置文件保护规则。
    catalog = SkillCatalog(configured_roots(config), disabled=config.skills.disabled, config_files=config.config_files)
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
                # 检查本地包能否准备，不执行其中的脚本或安装依赖。
                result["check"] = dict(catalog.prepare(skill.id).info(), format='valid')
        print(json.dumps(result, ensure_ascii=False, indent=2))

        # 发现阶段有诊断时返回非零，即使其他 Skill 可用。
        return 2 if catalog.diagnostics else 0
    finally:
        catalog.close()
