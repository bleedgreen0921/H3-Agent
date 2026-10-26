"""显式真实长上下文验收，只发送合成资料，不连接飞书或业务数据库。

使用当前主模型、摘要模型配置及生产 Agent/上下文中间件；会产生模型费用。
"""
import argparse
import asyncio
from contextlib import AsyncExitStack
from dataclasses import asdict
import json
from pathlib import Path
import random
from tempfile import TemporaryDirectory

from langchain_core.messages import AIMessage, HumanMessage
from langgraph.checkpoint.memory import InMemorySaver

from ai_assistant.agents import create_assistant_agent
from ai_assistant.config import load_config
from ai_assistant.llm.factory import create_model
from ai_assistant.llm.requests import model_session
from ai_assistant.runtime import AssistantSession, final_text
from ai_assistant.tools.context import ToolContext
from ai_assistant.workspace.files import ReadOnlyWorkspace


def synthetic_history(records):
    rng = random.Random(20261026)
    lines = [f'row_{i:05d} | data={rng.getrandbits(160):040x}' for i in range(records)]
    expected = ['START-' + f'{rng.getrandbits(48):012x}',
                'MIDDLE-' + f'{rng.getrandbits(48):012x}',
                'END-' + f'{rng.getrandbits(48):012x}']
    lines.insert(0, 'KEY_START=' + expected[0])
    lines.insert(len(lines) // 2, 'KEY_MIDDLE=' + expected[1])
    lines.append('KEY_END=' + expected[2])
    return '\n'.join(lines), expected


async def main(args):
    config = load_config(args.config)
    selected = config.default_model
    profile = config.models[selected]
    summary_name = (config.context.summary_model or selected) if config.context else selected
    if not config.context or not profile.max_input_tokens or profile.max_input_tokens < 1000000:
        raise RuntimeError('验收要求启用 context，且当前主模型已配置百万上下文。')
    data, expected = synthetic_history(args.records)
    async with AsyncExitStack() as stack:
        model = await stack.enter_async_context(model_session(config, selected,
            profile.parameters.get('timeout', 300), factory=create_model))
        summary = model if summary_name == selected else await stack.enter_async_context(model_session(
            config, summary_name, config.models[summary_name].parameters.get('timeout', 300), factory=create_model))
        with TemporaryDirectory(prefix='assistant-long-context-') as root:
            agent = create_assistant_agent(model, [], checkpointer=InMemorySaver(), settings=config.agent,
                context_settings=config.context, summary_model=summary, summary_profile=summary_name,
                max_input_tokens=profile.max_input_tokens,
                summary_max_input_tokens=config.models[summary_name].max_input_tokens)
            session = AssistantSession(agent, ToolContext(ReadOnlyWorkspace(Path(root))),
                                       config.agent, model_name=selected)
            session._complete = await agent.aupdate_state({'configurable': {'thread_id': session._thread}},
                {'messages': [HumanMessage('以下是纯合成测试资料，请保留用于后续提问：\n' + data),
                              AIMessage('已收到合成资料。')]})
            turn = await session.run_turn('只返回之前资料中的 KEY_START、KEY_MIDDLE、KEY_END 的值，'
                                          '按顺序每行一个；不要推测，找不到则说明找不到。')
            answer = final_text(turn.messages)
            reported_input = turn.usage.token_usage.get(selected, {}).get('input_tokens', 0)
            passed = (turn.usage.main_model_calls == 1 and turn.usage.summary_model_calls == 0
                      and reported_input > 24000 and answer.splitlines() == expected)
            result = {'passed': passed, 'model_profile': selected, 'model': profile.model,
                      'configured_max_input_tokens': profile.max_input_tokens,
                      'synthetic_characters': len(data), 'records': args.records,
                      'all_three_markers_retrieved': answer.splitlines() == expected,
                      'usage': asdict(turn.usage),
                      'scope': '真实主模型，生产上下文中间件；仅合成资料，不访问飞书或业务数据库；未测试百万边界'}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x', encoding='utf-8') as output:
        output.write(json.dumps(result, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return passed


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default='assistant.toml')
    parser.add_argument('--records', type=int, default=6000)
    parser.add_argument('--output', type=Path, default=Path('.data/long-context-acceptance.json'))
    args = parser.parse_args()
    if args.output.exists():
        parser.error('输出文件已存在；请指定新的 --output，避免计费后才发现无法保存。')
    if not 1000 <= args.records <= 10000:
        parser.error('--records 必须为 1000 到 10000，验收不会自动重试或扩大发送量。')
    try:
        ok = asyncio.run(main(args))
    except Exception as error:
        print(f'长上下文验收未完成：{type(error).__name__}；请检查模型连接、预算和输出文件。')
        raise SystemExit(1)
    raise SystemExit(0 if ok else 1)
