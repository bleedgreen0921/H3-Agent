"""由真实回执生成统一轮次终态，入口不使用模型正文判断完成。"""
from dataclasses import asdict, dataclass, field

TERMINAL_STATES = frozenset({'complete', 'partial', 'failed', 'unknown', 'cancelled', 'needs_clarification'})
SUCCESS = frozenset({'saved', 'unchanged', 'found', 'deleted', 'not_found', 'succeeded'})


@dataclass(frozen=True)
class TurnResult:
    status: str
    items: list[dict] = field(default_factory=list)

    def to_dict(self):
        return asdict(self)


def turn_result(state, operations=(), *, interrupted=None):
    items = []
    issues = state.get('turn_outcome') or {}
    for operation in operations:
        problem = issues.get(operation['id']) or (operation.get('result') or {}).get('business_outcome')
        items.append({'kind': 'remote', 'operation_id': operation['id'],
                      'status': 'failed' if problem else operation['status'],
                      'summary': '业务核对未通过' if problem else '远端操作已有成功回执' if operation['status'] == 'succeeded' else '远端操作需核实'})
    if issues and not operations:
        items.extend({'kind': 'remote', 'status': 'failed', 'summary': '远端任务业务核对未通过'} for _ in issues)
    statuses = [item['status'] for item in items]
    success = any(status in SUCCESS for status in statuses)
    bad = [status for status in statuses if status not in SUCCESS]
    if interrupted:
        status = 'partial' if success else interrupted
    elif not bad:
        status = 'complete'
    elif success:
        status = 'partial'
    elif any(value in {'unknown', 'pending', 'sending'} for value in bad):
        status = 'unknown'
    elif all(value == 'needs_clarification' for value in bad):
        status = 'needs_clarification'
    else:
        status = 'failed'
    return TurnResult(status, items)


def receipt_lines(items):
    labels = {'saved': '已保存', 'unchanged': '未变化', 'needs_clarification': '需要补充信息',
              'rejected': '未执行', 'conflict': '版本冲突', 'failed': '操作失败', 'unknown': '状态待核实',
              'found': '已找到', 'deleted': '已删除', 'not_found': '未找到', 'unavailable': '暂不可用',
              'succeeded': '已完成', 'pending': '状态待核实'}
    return [(item['target_label'] + '：' if item.get('target_label') else '') + labels.get(item['status'], '未完成') + '：' + item.get('summary', '请查询操作结果核实') +
            ('；相同条件已停止' if item.get('stopped') else '') +
            ('\n读取警告：' + item['read_warning'] if item.get('read_warning') else '') for item in items]
