"""飞书任务写入前检查与写入后业务核对；失败绝不重放。"""
import json
from datetime import datetime
from zoneinfo import ZoneInfo
from langchain.agents.middleware import AgentMiddleware
from langchain_core.messages import AIMessage, ToolMessage
from langchain_core.tools import ToolException

TASK_WRITES = {'task_v2_task_create', 'task_v2_task_patch', 'task_v2_task_addMembers'}


def payload(result):
    artifact = getattr(result, 'artifact', None)
    if isinstance(artifact, dict):
        structured = artifact.get('structured_content') or artifact.get('structuredContent')
        if isinstance(structured, dict):
            return structured
    content = getattr(result, 'content', result)
    if isinstance(content, dict):
        return content
    if isinstance(content, str):
        try:
            return json.loads(content)
        except ValueError:
            return {}
    if isinstance(content, list):
        for block in content:
            if isinstance(block, dict) and block.get('type') == 'text':
                try:
                    value = json.loads(block['text'])
                    if isinstance(value, dict):
                        return value
                except (ValueError, KeyError):
                    pass
    return {}


def task_data(result):
    data = payload(result)
    if not isinstance(data, dict):
        return {}
    if data.get('code', 0) != 0 or getattr(result, 'status', '') == 'error':
        return {}
    data = data.get('data', data)
    return data.get('task', {}) if isinstance(data, dict) else {}


def task_view(result, timezone):
    """展示已有时间，不把它登记为本轮可写目标时间。"""
    task = task_data(result)
    if not task:
        return None
    view = {k: task[k] for k in ('guid', 'task_id', 'summary', 'members', 'reminders') if k in task}
    view['timezone'] = timezone
    for field in ('start', 'due'):
        if field not in task:
            continue
        value = dict(task[field]) if isinstance(task[field], dict) else task[field]
        if isinstance(value, dict):
            try:
                value['local_datetime'] = datetime.fromtimestamp(
                    int(value['timestamp']) / 1000, ZoneInfo(timezone)).isoformat()
            except (KeyError, ValueError, TypeError, OverflowError, OSError):
                value['display_error'] = '时间戳无法转换，保留原值'
        view[field] = value
    return {'source': 'official_task_get', 'task': view}


def task_model_result(result, context):
    """只替换模型视图，原始回执继续作为核对和分页读取依据。"""
    artifact = result.artifact if isinstance(result.artifact, dict) else {}
    if artifact.get('task_model_view'):
        return result
    timezone = 'UTC' if context.caller and context.caller.channel == 'feishu_group' else context.timezone
    view = task_view(result, timezone)
    if view is None:
        return result
    return result.model_copy(update={'content': json.dumps(view, ensure_ascii=False), 'artifact': {
        **artifact, 'task_model_view': True, 'full_tool_content': result.content,
        'original_artifact': result.artifact}})


def expected_data(name, arguments):
    data = arguments.get('data', {})
    return data.get('task', {}) if name == 'task_v2_task_patch' else data


def prepare_task(context, entry, arguments):
    name = entry.original_name
    expected = expected_data(name, arguments)
    if name == 'task_v2_task_create' and 'members' not in expected:
        caller = context.caller
        if caller and caller.owner and caller.channel == 'feishu_private' and context.default_task_assignee == 'owner':
            if not context.owner_open_id:
                raise ToolException('未配置可信主人身份，无法设置默认负责人。')
            expected['members'] = [{'id': context.owner_open_id, 'type': 'user', 'role': 'assignee'}]
            arguments.setdefault('params', {})['user_id_type'] = 'open_id'
    if name == 'task_v2_task_patch' and ('members' in expected or 'members' in arguments.get('data', {}).get('update_fields', [])):
        raise ToolException('负责人修改必须使用 task_v2_task_addMembers，不能放入 patch。')
    for field in ('start', 'due'):
        value = expected.get(field)
        if not isinstance(value, dict) or not value.get('timestamp'):
            continue
        key = (str(value['timestamp']), bool(value.get('is_all_day', False)))
        if key not in context.resolved_datetimes:
            raise ToolException(f'{field} 未匹配本轮可信时间转换；请先调用 resolve_datetime（全天任务提供明确日期）。')
    return expected


def discrepancies(expected, actual, name, guid=None):
    issues = []
    if not actual.get('guid'):
        issues.append('缺少任务 GUID')
    elif guid and actual.get('guid') != guid:
        issues.append('返回任务 GUID 与目标不符')
    for key in ('summary', 'start', 'due', 'reminders', 'members'):
        if key not in expected:
            continue
        wanted, got = expected[key], actual.get(key)
        if key in ('start', 'due') and isinstance(wanted, dict):
            same = isinstance(got, dict) and all(str(got.get(k, False if k == 'is_all_day' else '')) == str(v) for k, v in wanted.items())
        elif key == 'members':
            def members(items):
                return {(m.get('id'), m.get('role')) for m in items or []}
            same = got is not None and members(wanted) <= members(got)
            if not wanted:
                same = got is not None and not any(m.get('role') == 'assignee' for m in got)
        elif key == 'reminders':
            same = got is not None and sorted(str(x.get('relative_fire_minute')) for x in wanted) == sorted(str(x.get('relative_fire_minute')) for x in got)
        else:
            same = wanted == got
        if not same:
            issues.append(f'{key} 与发送值不符或未返回')
    return issues


async def read_task(context, entry, guid, config=None, params=None):
    from ai_assistant.integrations.policy import authorize
    get = next((e for e in context.mcp.entries.values() if e.server == entry.server
                and e.original_name == 'task_v2_task_get'), None)
    if get is None:
        raise ToolException('未授权任务 get 能力，不能核实任务；请检查最小 allowlist。')
    args = {'path': {'task_guid': guid}, 'params': params or {'user_id_type': 'open_id'}}
    authorize(context.caller, get.server, get.settings, get.original_name, args)
    import asyncio
    async with asyncio.timeout(context.mcp.timeout):
        result = await get.tool.ainvoke(args, config=config)
    task = task_data(result)
    if not task or task.get('guid') != guid:
        raise ToolException('任务回读失败或 GUID 不符；没有自动重放写入。')
    return task


class TaskOutcomeMiddleware(AgentMiddleware):
    async def aafter_agent(self, state, runtime):
        outcomes = state.get('turn_outcome') or {}
        if not outcomes:
            return None
        lines = ['未完成：远端任务操作已发生，但业务核对未通过；未自动补写或重放。']
        for outcome in outcomes.values():
            lines.append(json.dumps(outcome, ensure_ascii=False))
        last = state['messages'][-1]
        return {'messages': [AIMessage(content='\n'.join(lines), id=last.id)]}


def outcome(context, operation, expected, actual, issues):
    times = {}
    for key in ('start', 'due'):
        try:
            times[key] = datetime.fromtimestamp(int(actual[key]['timestamp']) / 1000,
                                               ZoneInfo(context.timezone)).isoformat()
        except (KeyError, TypeError, ValueError, OverflowError, OSError):
            pass
    return {'operation_id': operation['id'], 'remote_status': operation['status'],
            'business_status': 'partial_failure', 'issues': issues,
            'actual': {k: actual[k] for k in ('guid', 'task_id', 'start', 'due', 'members', 'reminders') if k in actual},
            'actual_local_time': times}
