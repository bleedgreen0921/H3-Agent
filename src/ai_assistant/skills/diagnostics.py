"""仅检查 Skill 声明，不扫描依赖、不执行代码。"""


def check_skill(skill, *, terminal_enabled=False):
    return dict(format='valid', environment_declaration=skill.metadata.get('compatibility'),
                permissions='Application tool settings remain authoritative; allowed-tools is advisory.',
                unsupported=list(skill.warnings), terminal_enabled=terminal_enabled,
                limitation='Dependencies must be prepared by the user; compatibility is not verified.')
