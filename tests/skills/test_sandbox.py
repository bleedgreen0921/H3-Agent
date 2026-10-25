"""Local Skill resources and manually provisioned script dependencies."""

import asyncio
import shlex
import shutil
import sys
import pytest
from ai_assistant.skills.catalog import SkillCatalog
from ai_assistant.tools.terminal import TerminalService
from ai_assistant.workspace.files import ReadOnlyWorkspace
from tests.skills.test_catalog import package

pytestmark = pytest.mark.skipif(
    sys.platform != "darwin",
    reason="macOS Seatbelt required",
)


def test_local_python_and_node_skills_are_readonly(tmp_path):
    project = tmp_path / "project"
    project.mkdir()
    source = package(tmp_path / "skills/example")
    (source / "script.py").write_text(
        'from pathlib import Path; print(Path(__file__).with_name("examples").joinpath("test.txt").read_text())'
    )
    dep = source / "node_modules/fixture"
    dep.mkdir(parents=True)
    (dep / "package.json").write_text('{"name":"fixture","main":"index.js"}')
    (dep / "index.js").write_text("module.exports=42;")
    (source / "test.cjs").write_text('console.log(require("fixture"))')
    (source / "test.mjs").write_text('import n from "fixture"; console.log(n)')
    catalog = SkillCatalog([source.parent])
    skill = catalog.resolve("example")
    service = TerminalService(ReadOnlyWorkspace(project), timeout=10)

    async def scenario():
        try:
            result = await service.run_command(
                "python " + shlex.quote(skill.sandbox_path + "/script.py"),
                skills=[skill],
            )
            assert result["exit_code"] == 0 and "Example" in result["output"], result
            assert (
                await service.run_command(
                    "echo bad > " + shlex.quote(skill.sandbox_path + "/SKILL.md"),
                    skills=[skill],
                )
            )["exit_code"] != 0
            result = await service.run_command(
                "python " + shlex.quote(skill.sandbox_path + "/script.py")
            )
            assert result["exit_code"] != 0
            if shutil.which("node"):
                for script in ["test.cjs", "test.mjs"]:
                    result = await service.run_command(
                        "node " + shlex.quote(skill.sandbox_path + "/" + script),
                        skills=[skill],
                    )
                    assert result["exit_code"] == 0 and "42" in result["output"], result
        finally:
            await service.aclose()

    try:
        asyncio.run(scenario())
    finally:
        catalog.close()
