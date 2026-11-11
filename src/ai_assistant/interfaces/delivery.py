"""通用投递重试与截止策略；渠道错误分类由 channel classifier 提供。"""
import time

import httpx


def metadata(settings):
    now = time.time()
    return {'version': 1, 'attempts': 0, 'terminal_attempts': 0,
            'first_terminal_at': now,
            'deadline': now + settings.final_delivery_timeout_seconds,
            'last_error': None}


def classify(error, classifier=None):
    if classifier is not None:
        result = classifier(error)
        if result is not None:
            return result
    if isinstance(error, httpx.HTTPStatusError):
        status = error.response.status_code
        return ('temporary' if status == 429 else 'rejected' if 400 <= status < 500 else 'unknown',
                {'category': 'http', 'status': status})
    return 'unknown', {'category': type(error).__name__}


def exhausted(delivery, settings):
    return (delivery.get('terminal_attempts', delivery['attempts']) >= settings.delivery_max_attempts
            or (delivery.get('deadline') is not None and time.time() >= delivery['deadline']))
