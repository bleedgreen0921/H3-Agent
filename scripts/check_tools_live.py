"""Explicit manual tool checks; never imported by automatic test discovery.

Run from the project root:
    PYTHONPATH=src .venv/bin/python scripts/check_tools_live.py

Default execution sends one DeepSeek Search and one Fetch request (possibly billed).
No chat model is called and no request is automatically retried.
"""

import argparse
import asyncio
from pathlib import Path
import sys
import time

from langchain_core.messages import AIMessage, ToolMessage
from langgraph.graph import END, START, MessagesState, StateGraph
from langgraph.prebuilt import ToolNode

from ai_assistant.config import ConfigError, load_config
from ai_assistant.tools import ToolContext, create_builtin_tools, tool_context
from ai_assistant.workspace import WorkspaceError


CAPABILITIES = ('list_directory', 'read_text', 'search_text', 'web_search', 'web_fetch')


async def check(args: argparse.Namespace) -> int:
    config = load_config(args.config)
    selected = CAPABILITIES if args.tool == 'all' else (args.tool,)
    include_web = any(name.startswith('web_') for name in selected)
    if include_web and config.tools.web is None:
        raise ConfigError('联网测试需要配置 [tools.web] 和 credential_profile。')

    inputs = {
        'list_directory': {'path': '.', 'limit': 10},
        'read_text': {'path': args.file, 'max_lines': 10},
        'search_text': {'query': args.text, 'path': args.file},
        'web_search': {'query': args.query, 'count': 3},
        'web_fetch': {'url': args.url},
    }
    graph = StateGraph(MessagesState, context_schema=ToolContext)
    graph.add_node('tools', ToolNode(create_builtin_tools(include_web=include_web)))
    graph.add_edge(START, 'tools')
    graph.add_edge('tools', END)
    app = graph.compile()

    print('手动真实工具检查：不调用聊天模型；各项执行一次，无自动重试。', flush=True)
    if include_web:
        print('联网接口可能产生费用；搜索与抓取各最多一次请求。', flush=True)
    failed = 0
    # Local-only checks do not create a web client even when web is configured.
    if not include_web:
        config = config.model_copy(update={'tools': config.tools.model_copy(update={'web': None})})
    async with tool_context(config) as context:
        for index, name in enumerate(selected):
            started = time.monotonic()
            call_id = f'live_check_{index}'
            print(f'\n=== {name} ===', flush=True)
            try:
                result = await app.ainvoke(
                    {'messages': [AIMessage(content='', tool_calls=[{
                        'name': name, 'args': inputs[name],
                        'id': call_id, 'type': 'tool_call',
                    }])]},
                    context=context,
                )
                message = result['messages'][-1]
                valid = (isinstance(message, ToolMessage)
                         and message.tool_call_id == call_id
                         and message.status == 'success'
                         and isinstance(message.artifact, dict))
                failed += not valid
                print(f"{'通过' if valid else '失败'}  {time.monotonic() - started:.2f}s", flush=True)
                content = str(message.content)
                print(content[:1500], flush=True)
                if len(content) > 1500:
                    print('[终端预览已截断]', flush=True)
            except Exception as exc:
                failed += 1
                # Raw exception text and artifacts may contain sensitive request data.
                print(f'失败  {time.monotonic() - started:.2f}s  异常类型：{type(exc).__name__}', flush=True)

    print(f'\n完成：{len(selected) - failed} 项通过，{failed} 项失败。', flush=True)
    return 1 if failed else 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=Path('assistant.toml'))
    parser.add_argument('--tool', choices=('all', *CAPABILITIES), default='all')
    parser.add_argument('--file', default='README.md', help='工作区内用于读取和搜索的相对路径')
    parser.add_argument('--text', default='LangChain', help='本地字面量搜索词')
    parser.add_argument('--query', default='DeepSeek API 网页搜索 site:api-docs.deepseek.com')
    parser.add_argument('--url', default='https://api-docs.deepseek.com/zh-cn/guides/search/')
    args = parser.parse_args(argv)
    try:
        return asyncio.run(check(args))
    except ConfigError as exc:
        print(f'配置失败：{exc}', file=sys.stderr)
        return 2
    except (WorkspaceError, OSError):
        print('工作区无法初始化；请检查根目录和访问权限。', file=sys.stderr)
        return 2
    except KeyboardInterrupt:
        print('检查已取消，后续项目不再执行。', file=sys.stderr)
        return 130


if __name__ == '__main__':
    raise SystemExit(main())
