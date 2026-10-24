"""Opt-in real community acceptance; uses network, installs new deps, calls paid models.

Run from the existing project venv. A dedicated temporary installation root is
used, while controlled Python dependencies persist in the existing environment.
"""
import argparse
import asyncio
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import sysconfig

from ai_assistant.config import load_config
from ai_assistant.runtime import assistant_session, final_text
from ai_assistant.skills.catalog import SkillCatalog
from ai_assistant.skills.dependencies import DependencyService, inventory, unchanged
from ai_assistant.skills.manager import SkillManager


def snapshot(root):
    result = SkillCatalog([root])
    return result


async def run(args):
    config = load_config(args.config)
    if config.skills is None or config.tools.terminal is None or not config.skills.dependency_install:
        raise ValueError('Acceptance requires [skills], dependency_install=true and a configured terminal.')
    base = config.config_path.parent
    workspace = (base / config.workspace.root).resolve()
    before = inventory(Path(sysconfig.get_path('purelib')))
    records = []
    with TemporaryDirectory(prefix='assistant-community-acceptance-') as temporary:
        root = Path(temporary) / 'skills'
        manager = SkillManager(root)
        first = manager.install('anthropics/skills', 'skills/internal-comms')
        second = manager.install('anthropics/skills', 'skills/pdf')
        records += [first, second]
        print('Installed actual commits: ' + json.dumps([r['commit'] for r in records]), flush=True)
        catalog = snapshot(root)
        try:
            internal, pdf = catalog.resolve('internal-comms'), catalog.resolve('pdf')
            examples = [p for p in internal.info()['resources'] if p.startswith('examples/')]
            if not examples:
                raise AssertionError('Community internal-comms examples are missing.')
            print(catalog.read(internal.id, examples[0]), flush=True)
            # The same public dependency service is used by the CLI and native tool.
            service = DependencyService(workspace, config.skills)
            print(await service.install('pip', ['pypdf']), flush=True)
            unchanged(before, inventory(Path(sysconfig.get_path('purelib'))))
            from pypdf import PdfWriter
            from pypdf.generic import DictionaryObject, NameObject, TextStringObject, ArrayObject, NumberObject
            writer = PdfWriter()
            page = writer.add_blank_page(width=300, height=200)
            field = DictionaryObject({NameObject('/Type'): NameObject('/Annot'), NameObject('/Subtype'): NameObject('/Widget'),
                NameObject('/FT'): NameObject('/Tx'), NameObject('/T'): TextStringObject('acceptance_name'),
                NameObject('/V'): TextStringObject('Skills test'),
                NameObject('/Rect'): ArrayObject([NumberObject(v) for v in (20, 30, 200, 60)])})
            reference = writer._add_object(field)
            page[NameObject('/Annots')] = ArrayObject([reference])
            writer._root_object[NameObject('/AcroForm')] = DictionaryObject({NameObject('/Fields'): ArrayObject([reference])})
            input_path = workspace / 'skills-acceptance-input.pdf'
            if input_path.exists():
                raise ValueError('skills-acceptance-input.pdf already exists; choose a clean workspace.')
            with input_path.open('xb') as stream:
                writer.write(stream)
            selected = config.model_copy(update={'skills': config.skills.model_copy(update={'roots': [str(root)]})})
            async with assistant_session(selected, args.model) as session:
                reply = await session.run_turn('Read the relevant examples resource and write a short Chinese project progress announcement to skills-acceptance-announcement.txt. Report actual output.', skill_ids=[internal.id])
                print(final_text(reply.messages), flush=True)
                commands = []
                def record(event):
                    if event['event'] == 'on_tool_start' and event.get('name') == 'run_command':
                        commands.append(event.get('data', {}).get('input', {}).get('command', ''))
                reply = await session.run_turn('Read the original PDF skill scripts and their usage. Execute its original Python form inspection script on skills-acceptance-input.pdf; save field information to skills-acceptance-fields.json. Do not replace the script with your own implementation. Report the observed fields.', skill_ids=[pdf.id], on_event=record)
                print(final_text(reply.messages), flush=True)
                if not any(pdf.sandbox_path in command and '.py' in command and 'skills-acceptance-input.pdf' in command for command in commands):
                    raise AssertionError('No observed original skill Python script invocation.')
                if not (workspace / 'skills-acceptance-announcement.txt').is_file() or not (workspace / 'skills-acceptance-fields.json').is_file():
                    raise AssertionError('The real model did not produce the requested acceptance artifacts.')
                if 'acceptance_name' not in (workspace / 'skills-acceptance-fields.json').read_text():
                    raise AssertionError('Original form field was not observed in output.')
                for skill in (internal, pdf):
                    manager.update(skill)
                assert session.context.skills.resolve(pdf.id).digest == pdf.digest
                await session.new_conversation()
                assert session.context.skills.resolve(pdf.id).digest == pdf.digest
            async with assistant_session(selected, args.model) as reopened:
                for identity in (internal.id, pdf.id):
                    refreshed = reopened.context.skills.resolve(identity)
                    reply = await reopened.run_turn('Read one bundled text resource using the native resource tool and report this skill identity and the resource path.', skill_ids=[refreshed.id])
                    print(final_text(reply.messages), flush=True)
            fresh = snapshot(root)
            try:
                for skill in (internal, pdf):
                    assert fresh.resolve(skill.id).digest
                    manager.remove(fresh.resolve(skill.id))
            finally:
                fresh.close()
            empty = snapshot(root)
            try:
                assert not empty.skills
            finally:
                empty.close()
            unchanged(before, inventory(Path(sysconfig.get_path('purelib'))))
            print('Existing Python files unchanged. Community lifecycle completed; inspect workspace output quality.', flush=True)
        finally:
            catalog.close()
    if args.report:
        Path(args.report).write_text(json.dumps(records, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default='assistant.toml')
    parser.add_argument('--model')
    parser.add_argument('--report', help='Write actual commit/provenance records as JSON')
    parser.add_argument('--accept-network-and-model-costs', action='store_true', required=True,
        help='Explicitly allow GitHub/package access, new persistent dependencies and real model calls')
    asyncio.run(run(parser.parse_args()))


if __name__ == '__main__':
    main()
