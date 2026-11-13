"""飞书任务写入参数保护与本次接口返回核对；失败绝不重放。"""
import json
from langchain_core.tools import ToolException

from ai_assistant.integrations.feishu_temporal import time_view
from ai_assistant.integrations.tool_policy import PreflightIssue, ToolPreflightError

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
    task = data.get('task', {}) if isinstance(data, dict) else {}
    return task if isinstance(task, dict) else {}


def task_view(result, timezone):
    """展示已有时间，不把它登记为本轮可写目标时间。"""
    task = task_data(result)
    if not task:
        return None
    view = time_view(task, timezone, ('start', 'due'), report_invalid=True)
    return {'source': '本次接口返回的本地时间说明', 'task': view} if view else None


def task_model_result(result, context):
    """保留原生内容，独立追加时间说明，不授权后续时间写入。"""
    artifact = result.artifact if isinstance(result.artifact, dict) else {'original_artifact': result.artifact}
    if artifact.get('task_model_view'):
        return result
    timezone = context.timezone
    full = artifact.get('full_tool_content', result.content)
    source = result.model_copy(update={'content': full})
    view = task_view(source, timezone)
    if view is None:
        return result
    blocks = list(result.content) if isinstance(result.content, list) else [{'type': 'text', 'text': result.content}]
    blocks.append({'type': 'text', 'text': json.dumps({'task_time_annotation': view}, ensure_ascii=False)})
    return result.model_copy(update={'content': blocks, 'artifact': {
        **artifact, 'task_model_view': True, 'full_tool_content': full}})


def expected_data(name, arguments):
    data = arguments.get('data', {})
    return {k: v for k, v in data.get('task', {}).items() if k in data.get('update_fields', [])} if name == 'task_v2_task_patch' else data


def preflight_task(context, entry, arguments):
    """只读校验；在默认成员注入和 operation 创建之前调用。"""
    name = entry.original_name
    expected = expected_data(name, arguments)
    _validate_task(context, name, expected)


def _owner_only(context):
    service = (getattr(context, 'services', None) or {}).get('feishu')
    return getattr(service, 'task_members', 'model') == 'owner_only'


def _owner_id(context):
    service = (getattr(context, 'services', None) or {}).get('feishu')
    return getattr(service, 'owner_open_id', None)


def _validate_task(context, name, expected):
    """可复用的任务前置条件；本地失败不产生外部副作用。"""
    if not isinstance(expected, dict):
        return
    if _owner_only(context) and name == 'task_v2_task_addMembers':
        raise ToolPreflightError(PreflightIssue(
            code='task.members_owner_only',
            message='当前部署的负责人由系统绑定，不能修改任务成员。',
            field='data.members'))
    reminders = expected.get('reminders')
    if name == 'task_v2_task_create' and reminders and not expected.get('due'):
        raise ToolPreflightError(PreflightIssue(
            code='task.reminders_require_due',
            message='飞书任务提醒只能按截止时间 due 触发。请提供截止时间；'
                    '如果要在会议或考试开始前提醒，请创建日历日程。',
            field='data.due'))
    members = expected.get('members')
    if _owner_only(context) and name == 'task_v2_task_create' and members:
        owner = _owner_id(context)
        if not isinstance(members, list) or any(
                not isinstance(member, dict) or member.get('id') != owner for member in members):
            raise ToolPreflightError(PreflightIssue(
                code='task.members_owner_only',
                message='当前部署的负责人由系统绑定，请不要传 members 或指定其他成员。',
                field='data.members'))
    for field in ('start', 'due'):
        value = expected.get(field)
        if not isinstance(value, dict) or not value.get('timestamp'):
            continue
        key = (str(value['timestamp']), bool(value.get('is_all_day', False)))
        if key not in context.resolved_datetimes:
            raise ToolPreflightError(PreflightIssue(
                code='task.time_not_resolved',
                message=f'{field} 未匹配本轮可信时间转换；请先调用 resolve_datetime（全天任务提供明确日期）。',
                field=f'data.{field}'))
    start, due = expected.get('start'), expected.get('due')
    if isinstance(start, dict) and isinstance(due, dict) and start.get('timestamp') and due.get('timestamp'):
        if bool(start.get('is_all_day', False)) != bool(due.get('is_all_day', False)):
            raise ToolPreflightError(PreflightIssue(
                code='task.time_flags_mismatch',
                message='开始和截止时间的全天标记必须一致。',
                field='data.start'))
        if int(start['timestamp']) > int(due['timestamp']):
            raise ToolPreflightError(PreflightIssue(
                code='task.start_after_due',
                message='开始时间不能晚于截止时间。',
                field='data.start'))


def prepare_task(context, entry, arguments):
    name = entry.original_name
    expected = expected_data(name, arguments)
    if name == 'task_v2_task_create' and _owner_only(context):
        owner = _owner_id(context)
        if not owner:
            raise ToolException('未配置可信主人身份，无法设置负责人。')
        if not expected.get('members'):
            expected['members'] = [{'id': owner, 'type': 'user', 'role': 'assignee'}]
        arguments.setdefault('params', {})['user_id_type'] = 'open_id'
    elif name == 'task_v2_task_create' and 'members' not in expected:
        caller = context.caller
        if caller and caller.owner and caller.scope == 'private' and caller.channel == 'feishu_private' and context.services['feishu'].default_task_assignee == 'owner':
            if not context.services['feishu'].owner_open_id:
                raise ToolException('未配置可信主人身份，无法设置默认负责人。')
            expected['members'] = [{'id': context.services['feishu'].owner_open_id, 'type': 'user', 'role': 'assignee'}]
            arguments.setdefault('params', {})['user_id_type'] = 'open_id'
    if name == 'task_v2_task_patch' and ('members' in arguments.get('data', {}).get('task', {}) or 'members' in arguments.get('data', {}).get('update_fields', [])):
        raise ToolException('负责人修改必须使用 task_v2_task_addMembers，不能放入 patch。')
    _validate_task(context, name, expected)
    return expected


def verification(expected, actual, guid=None):
    """缺字段为未确认，明确不同才判不符；不发起额外读取。"""
    checked, unchecked, issues = [], [], []
    fields = ['guid', *expected]
    supported = {'summary', 'start', 'due', 'members', 'reminders'}
    for key in fields:
        if key != 'guid' and key not in supported:
            unchecked.append(key)
            continue
        if key not in actual or (key == 'guid' and not actual[key]):
            unchecked.append(key)
            continue
        wanted, got = (guid, actual[key]) if key == 'guid' else (expected[key], actual[key])
        missing = False
        if key == 'guid':
            same = not guid or got == guid
        elif key in {'start', 'due'} and isinstance(wanted, dict) and wanted:
            missing = not isinstance(got, dict) or 'timestamp' not in got
            same = isinstance(got, dict) and str(wanted.get('timestamp', '')) == str(got.get('timestamp', '')) and bool(wanted.get('is_all_day', False)) == bool(got.get('is_all_day', False))
        elif key == 'members':
            missing = not isinstance(got, list) or any(not isinstance(m, dict) or 'id' not in m or 'role' not in m for m in got)
            same = not missing and {(m.get('id'), m.get('role')) for m in wanted} <= {(m.get('id'), m.get('role')) for m in got}
        elif key == 'reminders':
            missing = not isinstance(got, list) or any(not isinstance(m, dict) or 'relative_fire_minute' not in m for m in got)
            same = not missing and sorted(str(m.get('relative_fire_minute')) for m in wanted) == sorted(str(m.get('relative_fire_minute')) for m in got)
        else:
            same = wanted == got
        if missing:
            unchecked.append(key)
        elif same:
            checked.append(key)
        else:
            issues.append(f'{key} 与发送值不符')
    return {'verification': 'mismatch' if issues else 'unverified' if unchecked else 'passed',
            'verification_source': '本次接口返回', 'checked_fields': checked,
            'unchecked_fields': unchecked, 'issues': issues + [f'{key} 未核对或未返回' for key in unchecked]}
