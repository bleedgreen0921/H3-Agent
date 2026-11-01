"""飞书任务写入参数保护与本次接口返回核对；失败绝不重放。"""
import json
from datetime import datetime
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError
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
    task = data.get('task', {}) if isinstance(data, dict) else {}
    return task if isinstance(task, dict) else {}


def task_view(result, timezone):
    """展示已有时间，不把它登记为本轮可写目标时间。"""
    task = task_data(result)
    if not task:
        return None
    view = {}
    view['timezone'] = timezone
    for field in ('start', 'due'):
        if field not in task:
            continue
        value = dict(task[field]) if isinstance(task[field], dict) else task[field]
        if isinstance(value, dict):
            try:
                value['local_datetime'] = datetime.fromtimestamp(
                    int(value['timestamp']) / 1000, ZoneInfo(timezone)).isoformat()
            except (KeyError, ValueError, TypeError, OverflowError, OSError, ZoneInfoNotFoundError):
                value['display_error'] = '时间戳无法转换，保留原值'
        else:
            value = {'original_value': value, 'display_error': '时间格式无法转换，保留原值'}
        view[field] = value
    return {'source': '本次接口返回的本地时间说明', 'task': view}


def task_model_result(result, context):
    """保留原生内容，独立追加时间说明，不授权后续时间写入。"""
    artifact = result.artifact if isinstance(result.artifact, dict) else {'original_artifact': result.artifact}
    if artifact.get('task_model_view'):
        return result
    timezone = 'UTC' if context.caller and context.caller.scope == 'shared' else context.timezone
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


def prepare_task(context, entry, arguments):
    name = entry.original_name
    expected = expected_data(name, arguments)
    if name == 'task_v2_task_create' and 'members' not in expected:
        caller = context.caller
        if caller and caller.owner and caller.scope == 'private' and caller.channel == 'feishu_private' and context.default_task_assignee == 'owner':
            if not context.owner_open_id:
                raise ToolException('未配置可信主人身份，无法设置默认负责人。')
            expected['members'] = [{'id': context.owner_open_id, 'type': 'user', 'role': 'assignee'}]
            arguments.setdefault('params', {})['user_id_type'] = 'open_id'
    if name == 'task_v2_task_patch' and ('members' in arguments.get('data', {}).get('task', {}) or 'members' in arguments.get('data', {}).get('update_fields', [])):
        raise ToolException('负责人修改必须使用 task_v2_task_addMembers，不能放入 patch。')
    for field in ('start', 'due'):
        value = expected.get(field)
        if not isinstance(value, dict) or not value.get('timestamp'):
            continue
        key = (str(value['timestamp']), bool(value.get('is_all_day', False)))
        if key not in context.resolved_datetimes:
            raise ToolException(f'{field} 未匹配本轮可信时间转换；请先调用 resolve_datetime（全天任务提供明确日期）。')
    start, due = expected.get('start'), expected.get('due')
    if isinstance(start, dict) and isinstance(due, dict) and start.get('timestamp') and due.get('timestamp'):
        if bool(start.get('is_all_day', False)) != bool(due.get('is_all_day', False)):
            raise ToolException('开始和截止时间的全天标记必须一致。')
        if int(start['timestamp']) > int(due['timestamp']):
            raise ToolException('开始时间不能晚于截止时间。')
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


def discrepancies(expected, actual, name, guid=None):
    return verification(expected, actual, guid)['issues']
