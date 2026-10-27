"""Owner-only memory tools; identity and evidence come from trusted runtime."""
import hashlib
import json
from datetime import datetime

from langchain.tools import ToolRuntime, tool
from langchain_core.messages import HumanMessage, ToolMessage
from langgraph.types import Command

from ai_assistant.storage.memory import result
from ai_assistant.storage.memory_types import MemoryChange, MemoryRequest
from ai_assistant.tools.context import ToolContext


def current_source(runtime):
    return next((m for m in reversed(runtime.state['messages'])
                 if isinstance(m, HumanMessage) and 'user_source' in m.additional_kwargs), None)


def receipt_command(runtime, response, *, name, change=None, read=False, project=False):
    payload = json.loads(response) if isinstance(response, str) else response
    key = runtime.tool_call_id
    if change is not None:
        identity = [change.source_id, change.memory_id, change.action, change.values] if change.memory_id else [
            change.source_id, change.attribute_id or change.topic, change.scope, change.project_id or change.project_name]
        key = hashlib.sha256(json.dumps(identity).encode()).hexdigest()
    # checkpoint 仅额外保存安全摘要，完整候选仍由工具消息/分页管理。
    receipt = {k: payload[k] for k in ('status', 'reason', 'summary', 'memory_id', 'revision',
               'project_id', 'project_revision', 'total', 'history_count') if k in payload}
    if payload.get('deleted'):
        receipt['deleted_ids'] = [v['memory_id'] for v in payload['deleted']]
    update = {'memory_reads' if read else 'memory_results': {key: receipt}, 'messages': [ToolMessage(
        content=json.dumps(payload, ensure_ascii=False), tool_call_id=runtime.tool_call_id, name=name,
        artifact={'protected_receipt': True})]}
    if project:
        update['memory_project_id'] = payload['project_id']
        update['memory_epoch'] = runtime.context.memory.policy['version']
    return Command(update=update)


@tool
async def search_memories(runtime: ToolRuntime[ToolContext], query: str = '',
                          scope: str | None = None, status: str | None = None,
                          offset: int = 0, project_id: str | None = None,
                          attribute_id: str | None = None, include_global: bool = True,
                          effective_at: datetime | None = None, projects_offset: int = 0,
                          attributes_offset: int = 0, deleted_offset: int = 0) -> Command:
    """Read-only candidate search. Query really filters text terms (not vector search); empty query lists candidates. Filter by project/attribute IDs, scope, active/inactive and optional timezone-aware effective_at. Project filter includes global rules by default. Page items, projects, attributes and deletion markers using their separate offsets. Always search before writing; if a query misses, list with empty query before creating a duplicate. This does not select a project or register aliases. deleted means deleted, not index lag; unavailable cannot be treated as not_found."""
    memory = runtime.context.memory
    if memory is None:
        response = result('rejected', 'memory_disabled')
    elif effective_at is not None and effective_at.tzinfo is None:
        response = result('rejected', 'invalid_change')
    else:
        source = current_source(runtime)
        try:
            response = await memory.search(query, scope=scope, status=status, offset=offset,
                source_id=source.id if source else None, project_id=project_id, attribute_id=attribute_id,
                include_global=include_global, effective_at=effective_at, projects_offset=projects_offset,
                attributes_offset=attributes_offset, deleted_offset=deleted_offset)
        except Exception:
            response = {'status': 'unavailable', 'summary': '记忆存储暂不可用，无法确认查询结果'}
    return receipt_command(runtime, response, name='search_memories', read=True)


@tool
async def select_memory_project(runtime: ToolRuntime[ToolContext], project_id: str | None = None,
                                alias: str | None = None, expected_revision: int | None = None) -> Command:
    """Select an existing project ID from search results; null returns to global context. A user shorthand may be registered as alias only when the current main model finds exactly one semantic project match. Pass the exact user shorthand and project directory expected_revision; evidence comes from the current user message. Multiple plausible candidates require clarification. This does not merge existing projects or rename the official name; read-only requests must not pass alias."""
    memory = runtime.context.memory
    if memory is None:
        response = json.loads(result('rejected', 'memory_disabled'))
    else:
        try:
            response = await memory.select_project(project_id, alias=alias,
                expected_revision=expected_revision, source=current_source(runtime))
            if response['status'] in {'saved', 'unchanged'}:
                await memory.recall(memory.query)
        except Exception:
            response = json.loads(result('unknown', 'management_not_confirmed'))
    success = response['status'] in {'saved', 'unchanged'}
    return receipt_command(runtime, response, name='select_memory_project', project=success)


@tool
async def remember_memory(change: MemoryRequest, runtime: ToolRuntime[ToolContext]) -> Command:
    """Save a stable natural statement after search. values contain atomic canonical values only: ['Python'], never 'project now uses Python'. Reuse existing values for equivalent wording; unchanged statements must still be confirmed by this tool. New attribute cardinality is mandatory: known skills/languages are multiple even with one initial member; primary backend language/default preference are single. Attribute topic excludes concrete project names. Keep the full exact user-named project_name when creating projects, never shorten it. Reuse existing attribute/project IDs. Single replace requires memory_id and expected_revision; multiple append/remove operate member arrays; replace_all=true only for an explicitly authorized entire-set replacement. Plans use kind=plan, quotes/hypotheses/one-off scripts are not current long-term facts. The current source ID is bound by runtime; cite an exact quote from the current message; context_evidence can explain pronouns but cannot grant old authorization."""
    memory = runtime.context.memory
    source = current_source(runtime)
    bound = MemoryChange.model_validate(change.model_dump(mode='json') | {'source_id': source.id if source else ''})
    if memory is None:
        response = result('rejected', 'memory_disabled')
    else:
        try:
            response = await memory.remember(bound.model_dump(mode='json'), runtime.state['messages'])
        except Exception:
            memory.reference = '记忆提交状态未知，不得使用旧缓存或声称已保存；请查找核实，不自动重放。'
            response = result('unknown', 'commit_not_confirmed')
    return receipt_command(runtime, response, name='remember_memory', change=bound)


@tool
async def manage_memory(runtime: ToolRuntime[ToolContext], action: str = 'list',
                        memory_id: str | None = None, offset: int = 0) -> Command:
    """Manage memory: list, detail, history, delete, clear, pause, resume. detail returns history_count, not unloaded empty history. history is paged. deleted is confirmed by a non-content deletion barrier; never explain it as index lag. Delete removes all revisions and vectors, preserves chat. Unknown operations must be checked, never automatically replayed."""
    memory = runtime.context.memory
    read = action in {'list', 'detail', 'history'}
    if memory is None:
        response = result('rejected', 'memory_disabled')
    else:
        try:
            response = await memory.manage(action, memory_id, offset)
        except Exception:
            if read:
                response = {'status': 'unavailable', 'summary': '记忆存储暂不可用，无法确认结果', 'memory_id': memory_id}
            else:
                memory.reference = '记忆管理状态未知，请查询核实，不得使用旧缓存或自动重放。'
                response = result('unknown', 'management_not_confirmed')
    return receipt_command(runtime, response, name='manage_memory', read=read)
