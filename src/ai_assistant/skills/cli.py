"""Skill management commands, shared with the native agent integration."""
import json
from pathlib import Path
import httpx
from .catalog import SkillCatalog, SkillError, configured_roots
from .manager import SkillManager


def add_commands(commands):
    parser = commands.add_parser('skill', help='Manage community Agent Skills')
    actions = parser.add_subparsers(dest='action', required=True)
    for action in ('discover', 'install', 'list', 'inspect', 'check', 'update', 'remove'):
        command = actions.add_parser(action)
        command.add_argument('--config', default='assistant.toml')
        command.add_argument('--global', dest='global_scope', action='store_true')
        if action in ('discover', 'install'):
            command.add_argument('source')
        elif action != 'list':
            command.add_argument('skill_id')
        if action in ('discover', 'install', 'update'):
            command.add_argument('--ref')
        if action == 'install':
            command.add_argument('--path')
    deps = actions.add_parser('deps')
    sub = deps.add_subparsers(dest='deps_action', required=True)
    install = sub.add_parser('install')
    install.add_argument('skill_id')
    install.add_argument('--config', default='assistant.toml')
    install.add_argument('--manager', choices=('pip', 'npm'), required=True)
    install.add_argument('--requirement', action='append', required=True)


async def execute(config, args):
    if config.skills is None:
        raise SkillError('Configure [skills] before managing or loading skills.')
    roots = configured_roots(config)
    manager = SkillManager(roots[1 if getattr(args, 'global_scope', False) else 0])
    try:
        if args.action == 'discover':
            result = manager.discover(args.source, args.ref)
        elif args.action == 'install':
            result = manager.install(args.source, args.path, args.ref)
        else:
            catalog = SkillCatalog(roots)
            try:
                if args.action == 'list':
                    result = dict(skills=[s.info() for s in catalog.skills.values()], diagnostics=catalog.diagnostics)
                else:
                    skill = catalog.resolve(args.skill_id)
                    if args.action in ('inspect', 'check'):
                        result = dict(skill.info(), metadata=skill.metadata)
                        if args.action == 'check':
                            from .diagnostics import check_skill
                            result['check'] = check_skill(skill, terminal_enabled=config.tools.terminal is not None)
                        for root in roots[:2]:
                            try:
                                result['source'] = SkillManager(root).record_for(skill)
                                break
                            except SkillError:
                                pass
                    elif args.action == 'deps':
                        if not config.skills.dependency_install:
                            raise SkillError('Controlled dependency installation is disabled.')
                        from .dependencies import DependencyService
                        base = config.config_path.parent if config.config_path else Path.cwd()
                        service = DependencyService(base / config.workspace.root, config.skills)
                        result = await service.install(args.manager, args.requirement)
                    else:
                        # Resolve ownership across project/global roots without needing --global.
                        for root in roots[:2]:
                            candidate = SkillManager(root)
                            try:
                                candidate.record_for(skill)
                            except SkillError:
                                continue
                            manager = candidate
                            break
                        if args.action == 'update':
                            result = manager.update(skill, args.ref)
                        else:
                            manager.remove(skill)
                            result = dict(removed=skill.id)
            finally:
                catalog.close()
    except (httpx.HTTPError, KeyError, json.JSONDecodeError):
        raise SkillError('Public GitHub acquisition or manifest validation failed; existing entries preserved.') from None
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0
