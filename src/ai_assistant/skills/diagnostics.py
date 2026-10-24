"""Read-only compatibility hints; never imports or executes a skill's code."""
import importlib.metadata
from pathlib import Path
import shutil
import sys


def check_skill(skill, *, terminal_enabled=False):
    from packaging.requirements import Requirement, InvalidRequirement
    dependencies = []
    for path in skill.path.rglob('requirements*.txt'):
        if not path.is_file() or path.stat().st_size > 1048576:
            continue
        for line in path.read_text(errors='replace').splitlines():
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            try:
                requirement = Requirement(line)
                try:
                    version = importlib.metadata.version(requirement.name)
                except importlib.metadata.PackageNotFoundError:
                    version = None
                dependencies.append(dict(requirement=str(requirement), installed=version,
                    satisfied=version is not None and requirement.specifier.contains(version),
                    source=path.relative_to(skill.path).as_posix()))
            except InvalidRequirement:
                dependencies.append(dict(requirement=line, supported=False))
    return dict(format='valid', environment_declaration=skill.metadata.get('compatibility'),
                permissions='Application tool settings remain authoritative; allowed-tools is advisory.',
                unsupported=list(skill.warnings), known_python_dependencies=dependencies,
                python=sys.version.split()[0], node=bool(shutil.which('node')), npm=bool(shutil.which('npm')),
                terminal_enabled=terminal_enabled,
                limitation='Only declared requirements files are checked; scripts are not executed and full compatibility is not proven.')
