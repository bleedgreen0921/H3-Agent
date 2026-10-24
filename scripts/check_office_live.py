"""Explicit office acceptance using a scripted model and the production native graph.

No paid model is called. A scenario contains exact configured server/tool names and
arguments. --execute is required before connecting to accounts or writing externally.
Use dedicated test recipients, groups, documents, tables, calendars and task lists.
"""
import argparse
import asyncio
from contextlib import AsyncExitStack
from dataclasses import replace
import importlib.metadata
import json
import os
from pathlib import Path
from uuid import uuid4

from langchain_core.language_models.fake_chat_models import FakeMessagesListChatModel
from langchain_core.messages import AIMessage, ToolMessage

from ai_assistant.agents import create_assistant_agent
from ai_assistant.config import load_config
from ai_assistant.integrations.mcp import mcp_resources
from ai_assistant.integrations.policy import Caller
from ai_assistant.runtime import AssistantSession
from ai_assistant.storage.integrations import operation_storage
from ai_assistant.storage.sessions import session_storage
from ai_assistant.tools.context import tool_context
from ai_assistant.tools.mcp import discovery_tools


class ScriptedAcceptanceModel(FakeMessagesListChatModel):
    def bind_tools(self, tools, **kwargs):
        return self


def call(name, args):
    return AIMessage(content='', tool_calls=[{'id': str(uuid4()), 'name': name, 'args': args}])


async def execute(config, scenario):
    if config.sessions is None:
        raise ValueError('Acceptance requires [sessions] and an explicit mcp init migration.')
    input_id = str(uuid4())
    async with AsyncExitStack() as stack:
        repo = await stack.enter_async_context(session_storage(config.sessions))
        operations = await stack.enter_async_context(operation_storage(config.sessions))
        context = await stack.enter_async_context(tool_context(config))
        catalog = await stack.enter_async_context(mcp_resources(config, context.workspace))
        selected = []
        for action in scenario:
            entry = next((e for e in catalog.entries.values() if e.server == action['server'] and e.original_name == action['tool']), None)
            if entry is None:
                raise ValueError('Scenario tool is outside the configured fixed directory.')
            selected.append(entry)
        responses = [call('load_mcp_tools', {'tool_ids': list(dict.fromkeys(e.id for e in selected))})]
        responses.extend(call(entry.id, action['arguments']) for entry, action in zip(selected, scenario))
        responses.append(AIMessage(content='Explicit acceptance scenario completed. Inspect every operation receipt and remote artifact.'))
        settings = config.agent.model_copy(update={'model_call_limit':len(responses)+1,
                                                   'tool_call_limit':len(responses)+1})
        context = replace(context, mcp=catalog, caller=Caller(), operations=operations)
        tools = discovery_tools() + [entry.tool for entry in catalog.entries.values()]
        agent = create_assistant_agent(ScriptedAcceptanceModel(responses=responses),tools,
            checkpointer=repo.saver,settings=settings,mcp_enabled=True)
        session = AssistantSession(agent,context,settings,repository=repo)
        await session.open()
        report = {'input_id':input_id,'session_id':session.session_id,'versions':{
            name:importlib.metadata.version(name) for name in ['langchain','langchain-core','langgraph','fastmcp','pydantic','unstructured-client']}}
        try:
            result = await session.run_turn('Execute the explicitly supplied test scenario.',input_id=input_id)
            report['tool_results'] = [m.model_dump(mode='json') for m in result.messages if isinstance(m,ToolMessage)]
        except Exception as error:
            # Keep secrets and raw transport diagnostics out of the report.
            report['error_type'] = type(error).__name__
        report['operations'] = await operations.list(input_id)
        return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',default='assistant.toml')
    parser.add_argument('--scenario',required=True,type=Path)
    parser.add_argument('--execute',action='store_true')
    parser.add_argument('--report',type=Path)
    args = parser.parse_args()
    config = load_config(args.config)
    scenario = json.loads(args.scenario.read_text())
    if not isinstance(scenario,list) or not 1 <= len(scenario) <= 100:
        parser.error('Scenario must contain 1..100 explicit actions.')
    for action in scenario:
        if set(action) != {'server','tool','arguments'} or not isinstance(action['arguments'],dict):
            parser.error('Each action requires server, tool and arguments.')
    if not args.execute:
        print('Offline scenario validated. --execute explicitly connects and performs these configured operations:')
        print('\n'.join(f'{a["server"]}: {a["tool"]}' for a in scenario))
        return
    report = asyncio.run(execute(config,scenario))
    path = args.report or Path('office-acceptance-' + report['input_id'] + '.json')
    fd = os.open(path,os.O_WRONLY|os.O_CREAT|os.O_EXCL|os.O_NOFOLLOW,0o600)
    with os.fdopen(fd,'w') as output:
        json.dump(report,output,default=str,ensure_ascii=False,indent=2)
    print(f'Private acceptance report: {path}')
    if ('error_type' in report or any(row['status'] != 'succeeded' for row in report['operations'])
            or any(row.get('status') == 'error' for row in report.get('tool_results', []))):
        raise SystemExit(1)


if __name__ == '__main__':
    main()
