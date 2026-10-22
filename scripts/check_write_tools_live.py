"""Manual create/edit/Bubblewrap check. No model, network requests or package install.

From the project root:
    PYTHONPATH=src .venv/bin/python scripts/check_write_tools_live.py

By default only a temporary fixture is modified. --config explicitly uses your
configured workspace and enabled capabilities; its temporary sample is cleaned.
"""

import argparse
import asyncio
from pathlib import Path
import shlex
import sys
from tempfile import TemporaryDirectory

from langchain_core.messages import AIMessage
from langgraph.graph import END, START, MessagesState, StateGraph
from langgraph.prebuilt import ToolNode

from ai_assistant.config import AppConfig, ConfigError, load_config
from ai_assistant.tools import ToolContext, create_builtin_tools, tool_context
from ai_assistant.tools.terminal import TerminalError
from ai_assistant.workspace import WorkspaceError


async def check(args):
    if args.config:
        config = load_config(args.config)
        if not config.tools.files.enabled or config.tools.terminal is None:
            raise ConfigError('需要启用 [tools.files] enabled=true 并配置 [tools.terminal]。')
        # This script never needs Search/Fetch or their clients.
        config = config.model_copy(update={'tools': config.tools.model_copy(update={'web': None})})
        base = config.config_path.parent
        root = Path(config.workspace.root)
        project = (root if root.is_absolute() else base / root).resolve(strict=True)
        fixture = TemporaryDirectory(prefix='tool-check-', dir=project)
    else:
        fixture = TemporaryDirectory(prefix='tool-check-')
        project = Path(fixture.name)
        config = AppConfig(
            default_model='unused', models={'unused': {'provider': 'ollama', 'model': 'unused'}},
            workspace={'root': str(project)},
            tools={'files': {'enabled': True}, 'terminal': {'profile': 'project', 'timeout': 10}},
        )
    with fixture:
        path = (Path(fixture.name) / 'example.py').relative_to(project).as_posix()
        builder = StateGraph(MessagesState, context_schema=ToolContext)
        builder.add_node('tools', ToolNode(create_builtin_tools(include_write=True, include_terminal=True)))
        builder.add_edge(START, 'tools')
        builder.add_edge('tools', END)
        app = builder.compile()
        print('手动写入与沙箱检查：仅修改本次测试文件；不安装依赖、不调用模型或联网接口。', flush=True)
        async with tool_context(config) as context:
            async def invoke(name, arguments):
                result = await app.ainvoke({'messages': [AIMessage(content='', tool_calls=[{
                    'name': name, 'args': arguments, 'id': name, 'type': 'tool_call',
                }])]}, context=context)
                message = result['messages'][-1]
                print(f'\n{name}: {message.status}\n{str(message.content)[:1500]}', flush=True)
                if message.status != 'success':
                    raise TerminalError('工具检查失败，后续项目不再执行。')
                return message.artifact
            created = await invoke('create_file', {'path': path, 'content': 'print("before")\n'})
            await invoke('edit_file', {
                'path': path, 'expected_sha256': created['sha256'],
                'edits': [{'old_text': 'before', 'new_text': 'after'}],
            })
            # Scratch cannot write project data; the sample is created by file tools
            # before the read-only sandbox executes it through its absolute path.
            absolute = context.workspace.root / path
            await invoke('run_command', {'command': 'python ' + shlex.quote(str(absolute))})
    print('\n三项通过，测试目录已清理。', flush=True)
    return 0


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, help='显式使用真实配置及工作区；默认仅测试临时夹具')
    args = parser.parse_args(argv)
    try:
        return asyncio.run(check(args))
    except (ConfigError, WorkspaceError, TerminalError) as exc:
        print(f'检查失败：{exc}', file=sys.stderr)
        return 1
    except OSError:
        print('检查失败：测试目录或沙箱资源无法访问。', file=sys.stderr)
        return 2
    except KeyboardInterrupt:
        print('检查已取消。', file=sys.stderr)
        return 130
    except Exception as exc:
        print(f'检查失败，异常类型：{type(exc).__name__}', file=sys.stderr)
        return 1


if __name__ == '__main__':
    raise SystemExit(main())
