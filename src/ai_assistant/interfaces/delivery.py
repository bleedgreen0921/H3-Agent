"""消息投递的持久化限额和脱敏错误分类，不推测远端写入结果。"""
import time

import httpx

from ai_assistant.integrations.feishu_files import FeishuRejected


def metadata(settings):
    now = time.time()
    return {'version': 1, 'attempts': 0, 'terminal_attempts': 0,
            'first_terminal_at': now,
            'deadline': now + settings.final_delivery_timeout_seconds,
            'last_error': None}


def classify(error):
    if isinstance(error, FeishuRejected):
        return ('temporary' if error.code == 99991400 else 'rejected' if error.code in {230001, 230006} else 'unknown',
                {'category': 'feishu', 'code': error.code, 'status': error.http_status})
    if isinstance(error, httpx.HTTPStatusError):
        status = error.response.status_code
        return ('temporary' if status == 429 else 'rejected' if 400 <= status < 500 else 'unknown',
                {'category': 'http', 'status': status})
    return 'unknown', {'category': type(error).__name__}


def exhausted(delivery, settings):
    return (delivery.get('terminal_attempts', delivery['attempts']) >= settings.delivery_max_attempts
            or (delivery.get('deadline') is not None and time.time() >= delivery['deadline']))
