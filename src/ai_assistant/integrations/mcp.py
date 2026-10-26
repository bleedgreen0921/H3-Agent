"""Native MCP adapter ownership and a fixed, identity-qualified tool directory."""
import asyncio
from contextlib import AsyncExitStack, asynccontextmanager
from dataclasses import dataclass
from hashlib import sha256
import json

from ai_assistant.config import ConfigError
from ai_assistant.executables import executable, executable_env
from ai_assistant.integrations.policy import Caller


TASK_DESCRIPTIONS = {
    'task_v2_task_get': '读取已有任务，含时间、成员角色和提醒。',
    'task_v2_task_list': '列出任务。',
    'task_v2_task_create': '新建任务；创建不用于修正已有任务。',
    'task_v2_task_patch': '修改已有任务字段；负责人通过 addMembers 修改。',
    'task_v2_task_addMembers': '添加成员；负责人通过 addMembers 修改，已有负责人不重复添加。',
}


def schema_tokens(tools):
    """目录和加载共享框架原生 schema 估算。"""
    from langchain_core.messages.utils import count_tokens_approximately
    return count_tokens_approximately([], tools=tools)


def search_terms(text):
    import re
    aliases = {'任务': 'task', '修改': 'patch', '更新': 'patch', '新建': 'create',
               '创建': 'create', '负责人': 'member', '成员': 'member', '联系人': 'contact',
               '文档': 'docx', '知识库': 'wiki', '读取': 'get', '查看': 'get',
               '添加': 'add', '删除': 'delete', '列出': 'list',
               'members': 'member', 'tasks': 'task', 'read': 'get',
               'update': 'patch', 'remove': 'delete'}
    text = re.sub(r'([a-z])([A-Z])', r'\1 \2', text).casefold()
    for word, alias in aliases.items():
        if not word.isascii():
            text = text.replace(word, ' ' + alias + ' ')
    return list(dict.fromkeys(aliases.get(w, w) for w in re.findall(r'[^\W_]+', text)))


@dataclass(frozen=True)
class Entry:
    id: str
    server: str
    original_name: str
    digest: str
    tool: object
    settings: object

    def info(self):
        return dict(id=self.id, server=self.server, identity=self.settings.identity,
                    name=self.original_name, description=TASK_DESCRIPTIONS.get(self.original_name, (self.tool.description or '')[:80]),
                    schema_tokens=schema_tokens([self.tool]),
                    write=self.original_name in self.settings.write_tools)


class Catalog:
    def __init__(self, entries=(), *, timeout=30, adapters=None, servers=None):
        self.entries = {e.id: e for e in entries}
        self.timeout = timeout
        self.adapters = adapters or {}
        self.servers = servers or {}

    def search(self, query='', offset=0, limit=5, server=None, domain=None):
        if offset < 0 or not 1 <= limit <= 100:
            raise ValueError('Invalid pagination')
        words = search_terms(query)
        domains = {'task', 'contact', 'docx', 'wiki', 'calendar', 'bitable', 'im'}
        requested_domains = set(words) & domains
        actions = [w for w in words if w not in domains]
        ranked = []
        matched_terms = set()
        for entry in self.entries.values():
            if server and entry.server != server:
                continue
            name_terms = set(search_terms(entry.original_name))
            if domain and domain.casefold() not in entry.original_name.casefold():
                continue
            if requested_domains and not requested_domains <= name_terms:
                continue
            text_terms = set(search_terms(entry.tool.description or ''))
            exact = query.casefold() in (entry.id.casefold(), entry.original_name.casefold())
            capabilities = {'get', 'list', 'create', 'patch', 'add', 'member', 'delete'}
            # 官方任务描述可能提到删除等其他接口，不能将引用误认成本工具能力。
            hits = {w for w in actions if w in name_terms or (w in text_terms and
                    (entry.original_name not in TASK_DESCRIPTIONS or w not in capabilities))}
            if actions and not hits and not exact:
                continue
            matched_terms.update(hits)
            if exact:
                matched_terms.update(actions)
            score = 1000 * exact + sum(20 if w in name_terms else 1 for w in hits)
            ranked.append((-score, entry.id, entry.info()))
        matches = [item for _, _, item in sorted(ranked)]
        # 极长查询不能令反馈本身突破目录预算。
        unmatched = [w[:60] for w in actions if w not in matched_terms][:12]
        items = []
        for item in matches[offset:offset + min(limit, 5)]:
            candidate = items + [item]
            end = offset + len(candidate)
            envelope = dict(unmatched_terms=unmatched, items=candidate, total=len(matches), returned=len(candidate),
                            next_offset=end if end < len(matches) else None)
            if len(json.dumps(envelope, ensure_ascii=False)) > 1500:
                if not items:
                    raise ValueError('单条目录记录超过分页预算，请缩小服务范围。')
                break
            items = candidate
        end = offset + len(items)
        import logging
        logging.getLogger(__name__).info("MCP search: results=%s returned=%s", len(matches), len(items))
        return dict(unmatched_terms=unmatched, items=items, total=len(matches), returned=len(items),
                    next_offset=end if end < len(matches) else None)

    def for_caller(self, caller):
        if caller.channel != 'feishu_group':
            return self
        return Catalog([e for e in self.entries.values() if e.settings.identity == 'bot' and e.settings.transport == 'feishu'
                        and e.server in caller.resources], timeout=self.timeout)

    def active(self, state):
        return [e for e in self.entries.values() if state.get(e.id) == e.digest]


async def decline(*args, **kwargs):
    from fastmcp.client.elicitation import ElicitResult
    return ElicitResult(action='decline')


def stable_id(server, identity, name):
    # Provider-safe identifier, digest protects against normalization collisions.
    import re
    prefix = re.sub('[^a-zA-Z0-9_-]', '_', f'{server}_{identity}_{name}')[:43]
    digest = sha256(f'{server}\0{identity}\0{name}'.encode()).hexdigest()[:16]
    return f'mcp_{prefix}_{digest}'


@asynccontextmanager
async def mcp_resources(config, workspace, *, caller=None):
    if config.mcp is None or not config.mcp.servers:
        yield Catalog()
        return
    try:
        from fastmcp import Client
        from fastmcp.client.transports import StdioTransport, StreamableHttpTransport
        from langchain.mcp import MCPAdapter
    except ImportError:
        raise ConfigError('Install the mcp extra to enable MCP services.') from None
    caller = caller or Caller()
    entries = []
    adapters = {}
    async with AsyncExitStack() as stack:
        for name, settings in config.mcp.servers.items():
            if caller.channel == 'feishu_group' and (settings.identity != 'bot' or settings.transport != 'feishu' or name not in caller.resources):
                continue
            if not settings.tools and not settings.resources and not settings.prompts:
                continue
            tools = None
            if settings.transport == 'feishu':
                from ai_assistant.integrations.feishu import OfficialConnection
                tools = await OfficialConnection(config, settings, workspace).discover()
            elif settings.transport == 'stdio':
                overrides = {k: v.get_secret_value() for k, v in settings.env.items()}
                binary = executable(settings.command, path=overrides.get('PATH'))
                env = {**executable_env(binary), **overrides}
                target = StdioTransport(command=binary, args=settings.args,
                                        env=env, keep_alive=True)
            elif settings.transport == 'feishu_files':
                import httpx
                from ai_assistant.integrations.feishu_files import FeishuAPI, create_server
                from ai_assistant.integrations.oauth import TokenStore
                if config.integrations.feishu is None:
                    raise ConfigError('Configure [integrations.feishu].')
                http = await stack.enter_async_context(httpx.AsyncClient(timeout=config.mcp.call_timeout,
                    trust_env=False, follow_redirects=False))
                tokens = TokenStore(config, workspace.root) if settings.identity == 'user' else None
                target = create_server(FeishuAPI(config.integrations.feishu, http, tokens), workspace)
            elif settings.transport == 'http':
                target = StreamableHttpTransport(settings.url,
                    headers={k: v.get_secret_value() for k, v in settings.headers.items()})
            if tools is None:
                client = Client(target, elicitation_handler=decline,
                                timeout=config.mcp.call_timeout,
                                init_timeout=config.mcp.connect_timeout)
                adapter = MCPAdapter(client)
                async with asyncio.timeout(config.mcp.connect_timeout):
                    await stack.enter_async_context(adapter)
                    tools = await adapter.list_tools()
                adapters[name] = adapter
            found = {tool.name for tool in tools}
            if set(settings.tools) - found:
                raise ConfigError(f'MCP server {name}: configured tools are missing.')
            for tool in tools:
                if tool.name not in settings.tools:
                    continue
                original = tool.name
                identity = stable_id(name, settings.identity, original)
                schema = tool.args_schema if isinstance(tool.args_schema, dict) else tool.args_schema.model_json_schema()
                digest = sha256(json.dumps(schema, sort_keys=True).encode()).hexdigest()
                native = tool.model_copy(update={'name': identity, 'metadata': {
                    **(tool.metadata or {}), 'office': {'server': name, 'identity': settings.identity,
                    'original_name': original, 'schema_digest': digest}}})
                entries.append(Entry(identity, name, original, digest, native, settings))
        yield Catalog(entries, timeout=config.mcp.call_timeout, adapters=adapters, servers=config.mcp.servers)
