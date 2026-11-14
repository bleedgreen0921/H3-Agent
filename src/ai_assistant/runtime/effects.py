"""统一效果回执；TurnResult 以 item dict 形式输出。"""
from dataclasses import dataclass, field


@dataclass(frozen=True)
class EffectReceipt:
    """供轮次结果汇总的效果状态、核验和资源引用。"""
    kind: str
    status: str
    accepted: bool = False
    verification: str = 'unverified'
    references: dict = field(default_factory=dict)
    summary: str = ''

    def to_item(self):
        return {'kind': self.kind, 'status': self.status, 'accepted': self.accepted,
                'verification': self.verification, **self.references, 'summary': self.summary}


def operation_receipt(operation):
    """把外部操作记录转换为统一回执，不把接受等同于字段核验。"""
    result = operation.get('result') or {}
    terminal = operation.get('tool') == 'run_command'
    accepted = operation.get('status') == 'succeeded' and not terminal
    verification = result.get('verification', 'unverified')
    problem = result.get('business_outcome')
    if problem:
        verification = problem.get('verification', 'mismatch')
    status = 'failed' if problem or verification == 'mismatch' else operation.get('status', 'unknown')
    references = result.get('references')
    references = references if isinstance(references, dict) else {}
    return EffectReceipt(
        'terminal' if terminal else 'external_operation', status, accepted, verification,
        {**references, 'operation_id': operation.get('id')},
        ('进程已退出，业务效果未核验' if terminal else '接口已接受；' + (
            '字段核验通过' if verification == 'passed' else '字段核验不符'
            if verification == 'mismatch' else '未核验')) if operation.get('status') == 'succeeded'
        else '操作效果需核实')
