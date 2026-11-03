"""只读 IMAP 子进程：凭据仅从 stdin 接收，stdout 只输出脱敏结果。"""
import email.policy
from email.parser import BytesParser
from email.header import decode_header, make_header
from html.parser import HTMLParser
import imaplib
import json
import re
import ssl
import sys


class VisibleHTML(HTMLParser):
    def __init__(self):
        super().__init__()
        self.hidden = 0
        self.text = []

    def handle_starttag(self, tag, attrs):
        if tag in {'script', 'style', 'head'}:
            self.hidden += 1
        if tag in {'p', 'br', 'div', 'li', 'tr'}:
            self.text.append('\n')

    def handle_endtag(self, tag):
        if tag in {'script', 'style', 'head'} and self.hidden:
            self.hidden -= 1

    def handle_data(self, data):
        if not self.hidden:
            self.text.append(data)


STAGES = {
    'request': '读取请求', 'connect': 'SSL 连接', 'login': '登录', 'id': '客户端 ID 协商',
    'select': '只读打开收件箱', 'uidvalidity': '读取 UIDVALIDITY', 'search': '查询邮件',
    'headers': '读取邮件头', 'structure': '解析 MIME 结构', 'body': '读取正文', 'decode': '处理正文',
    'worker': '子进程处理',
}
ERROR_MESSAGES = {
    'tls_error': 'TLS 连接或证书校验失败。',
    'timeout': '邮箱操作超时；未修改邮件。',
    'network_error': '网络连接失败。',
    'connection_lost': 'IMAP 连接已中断。',
    'authentication_rejected': '邮箱认证被拒绝，请核对授权码与 IMAP 服务设置。',
    'command_rejected': '服务器拒绝了本阶段 IMAP 命令；不能据此判断授权码失效。',
    'server_policy_rejected': '服务器安全策略拒绝访问，请检查客户端授权及 IMAP 服务设置。',
    'invalid_response': '本阶段 IMAP 响应缺失、不完整或无法解析。',
    'uidvalidity_changed': '邮箱 UIDVALIDITY 已变化，请重新查询。',
    'invalid_request': '邮箱读取请求无效。',
    'internal_error': '邮箱工具发生内部处理异常，原因尚未确认。',
}


class MailFailure(Exception):
    def __init__(self, code):
        self.code = code
        super().__init__(code)


def failure(error, stage):
    """只输出固定诊断，不把原始异常、协议响应或凭据带出子进程。"""
    if isinstance(error, MailFailure):
        code = error.code
    elif isinstance(error, ssl.SSLError):
        code = 'tls_error'
    elif isinstance(error, TimeoutError):
        code = 'timeout'
    elif isinstance(error, imaplib.IMAP4.abort):
        code = 'connection_lost'
    elif isinstance(error, imaplib.IMAP4.error):
        code = 'authentication_rejected' if stage == 'login' else 'command_rejected'
    elif isinstance(error, OSError):
        code = 'network_error'
    else:
        code = 'internal_error'
    return {'error': ERROR_MESSAGES[code], 'stage': stage, 'error_code': code}


def describe_error(result):
    stage = STAGES.get(result.get('stage'), '子进程处理')
    code = result.get('error_code')
    message = ERROR_MESSAGES.get(code, ERROR_MESSAGES['internal_error'])
    return f'邮箱读取失败（阶段：{stage}；类型：{code if code in ERROR_MESSAGES else "internal_error"}）：{message}'


def structure(raw):
    """按字节解析括号、转义字符串和 literal；限制深度且拒绝不完整结构。"""
    pos = 0

    def parse(depth=0):
        nonlocal pos
        if depth > 50:
            raise ValueError('MIME nesting limit')
        while pos < len(raw) and raw[pos:pos + 1].isspace():
            pos += 1
        if pos >= len(raw):
            raise ValueError('Incomplete structure')
        token = raw[pos:pos + 1]
        pos += 1
        if token == b'(':
            values = []
            while True:
                while pos < len(raw) and raw[pos:pos + 1].isspace():
                    pos += 1
                if pos >= len(raw):
                    raise ValueError('Unclosed structure')
                if raw[pos:pos + 1] == b')':
                    pos += 1
                    return values
                values.append(parse(depth + 1))
        if token == b'"':
            value = bytearray()
            while pos < len(raw):
                char = raw[pos:pos + 1]
                pos += 1
                if char == b'"':
                    return value.decode('utf-8', 'replace').lower()
                if char == b'\\':
                    if pos >= len(raw):
                        break
                    char = raw[pos:pos + 1]
                    pos += 1
                value.extend(char)
            raise ValueError('Unclosed string')
        if token == b'{':
            match = re.match(rb'(\d+)\+?}\r\n', raw[pos:])
            if not match:
                raise ValueError('Invalid literal length')
            pos += match.end()
            length = int(match[1])
            if length > len(raw) - pos:
                raise ValueError('Incomplete literal')
            value = raw[pos:pos + length]
            pos += length
            return value.decode('utf-8', 'replace').lower()
        if token == b')':
            raise ValueError('Unexpected closing bracket')
        start = pos - 1
        while pos < len(raw) and not raw[pos:pos + 1].isspace() and raw[pos:pos + 1] not in (b'(', b')'):
            pos += 1
        return raw[start:pos].decode('ascii').lower()

    result = parse()
    if raw[pos:].strip():
        raise ValueError('Trailing structure data')
    return result


def fetch_structure(data):
    """imaplib 将 literal 拆成 (前缀, 内容) 元组；重建时不在 literal 内插空格。"""
    chunks = []
    for item in data:
        if isinstance(item, tuple) and len(item) == 2 and all(isinstance(v, bytes) for v in item):
            chunks.extend((item[0], b'\r\n', item[1]))
        elif isinstance(item, bytes):
            chunks.append(item)
        else:
            raise ValueError('Invalid FETCH fragment')
    raw = b''.join(chunks)
    # 解析整个 FETCH 列表，避免正则把字符串中的括号误当结构边界。
    match = re.match(rb'\s*\d+\s+(\(.*)', raw, re.S)
    if not match:
        raise ValueError('Missing FETCH envelope')
    fields = structure(match[1])
    if not isinstance(fields, list) or len(fields) % 2:
        raise ValueError('Invalid FETCH fields')
    for index in range(0, len(fields), 2):
        if fields[index] == 'bodystructure' and isinstance(fields[index + 1], list):
            return fields[index + 1]
    raise ValueError('Missing BODYSTRUCTURE')


def text_parts(node, prefix=''):
    if not isinstance(node, list) or not node:
        raise ValueError('Invalid MIME node')
    if isinstance(node[0], list):
        # multipart 的扩展字段依次为参数、disposition、语言等。
        children = next((i for i, v in enumerate(node) if not isinstance(v, list)), None)
        if children is None:
            raise ValueError('Missing multipart subtype')
        disposition = node[children + 2] if len(node) > children + 2 else None
        if isinstance(disposition, list) and disposition[0] == 'attachment':
            return []
        return [part for i, child in enumerate(node[:children], 1)
                for part in text_parts(child, f'{prefix}.{i}' if prefix else str(i))]
    if len(node) < 7 or not isinstance(node[0], str) or not isinstance(node[1], str):
        raise ValueError('Incomplete MIME leaf')
    disposition_index = 9 if node[0] == 'text' else 8
    disposition = node[disposition_index] if len(node) > disposition_index else None
    params = node[2] if len(node) > 2 else None
    if (isinstance(disposition, list) and disposition[0] == 'attachment'
            or isinstance(params, list) and 'name' in params
            or isinstance(disposition, list) and len(disposition) > 1
            and isinstance(disposition[1], list) and 'filename' in disposition[1]):
        return []
    if node[0] == 'text' and node[1] in {'plain', 'html'}:
        return [(prefix or '1', node[1])]
    return []


def literal(data):
    if not isinstance(data, list):
        raise MailFailure('invalid_response')
    chunks = []
    for item in data:
        if isinstance(item, tuple):
            if len(item) != 2 or not all(isinstance(value, bytes) for value in item):
                raise MailFailure('invalid_response')
            chunks.append(item[1])
        elif item is not None and not isinstance(item, bytes):
            raise MailFailure('invalid_response')
    return b''.join(chunks)


def require_ok(status, data):
    if status != 'OK':
        # 仅识别固定安全标记，不向外输出服务器原文。
        policy = any(isinstance(item, bytes) and b'UNSAFELOGIN' in item.upper() for item in (data or []))
        raise MailFailure('server_policy_rejected' if policy else 'command_rejected')


def checked(client, command, *args):
    status, data = client.uid(command, *args)
    require_ok(status, data)
    return data


def header(value):
    try:
        return str(make_header(decode_header(str(value or ''))))
    except (LookupError, UnicodeError, ValueError):
        return str(value or '')


def run(request, factory=imaplib.IMAP4_SSL):
    client = None
    stage = 'connect'
    try:
        client = factory('imap.163.com', 993, ssl_context=ssl.create_default_context(), timeout=request['timeout'])
        stage = 'login'
        status, data = client.login(request['address'], request['authorization_code'])
        if status != 'OK':
            raise MailFailure('authentication_rejected')
        stage = 'id'
        capabilities = {value.decode('ascii').upper() if isinstance(value, bytes) else value.upper()
                        for value in client.capabilities}
        if 'ID' in capabilities:
            # 标准库没有公开 ID 方法；仅注册并调用这一条原生 IMAP 命令。
            imaplib.Commands.setdefault('ID', ('AUTH', 'SELECTED'))
            status, data = client._simple_command('ID', '("name" "personal-assistant" "version" "1")')
            require_ok(status, data)
        stage = 'select'
        status, data = client.select('INBOX', readonly=True)
        require_ok(status, data)
        stage = 'uidvalidity'
        _, values = client.response('UIDVALIDITY')
        if (not isinstance(values, list) or len(values) != 1 or not isinstance(values[0], bytes)
                or not values[0].isdigit() or int(values[0]) < 1):
            raise MailFailure('invalid_response')
        validity = values[0].decode('ascii')
        if request.get('validity') and validity != request['validity']:
            raise MailFailure('uidvalidity_changed')
        if request['action'] == 'search':
            stage = 'search'
            data = checked(client, 'SEARCH', None, 'SINCE', request['since'], 'BEFORE', request['before'])
            if not isinstance(data, list) or not all(isinstance(item, bytes) for item in data):
                raise MailFailure('invalid_response')
            tokens = b' '.join(data).split()
            if any(not token.isdigit() or int(token) < 1 for token in tokens):
                raise MailFailure('invalid_response')
            uids = sorted({int(uid) for uid in tokens}, reverse=True)
            return {'validity': validity, 'uids': uids[:request['search_limit']], 'limited': len(uids) > request['search_limit']}
        if request['action'] == 'headers':
            result = []
            stage = 'headers'
            for uid in request['uids']:
                data = checked(client, 'FETCH', str(uid), '(UID FLAGS BODY.PEEK[HEADER.FIELDS (SUBJECT FROM DATE)])')
                raw = literal(data)
                if not raw:
                    result.append({'uid': uid, 'gap': '邮件已移除或邮件头不可读'})
                    continue
                msg = BytesParser(policy=email.policy.default).parsebytes(raw)
                flags = b' '.join(item[0] if isinstance(item, tuple) else item for item in data if item)
                result.append({'uid': uid, 'subject': header(msg['Subject']), 'sender': header(msg['From']),
                               'date': str(msg['Date'] or ''), 'seen': b'\\Seen' in flags})
            return {'validity': validity, 'messages': result}
        if request['action'] != 'read':
            raise MailFailure('invalid_request')
        uid = str(int(request['uid']))
        stage = 'structure'
        data = checked(client, 'FETCH', uid, '(BODYSTRUCTURE)')
        try:
            parts = text_parts(fetch_structure(data))
        except (ValueError, TypeError, IndexError, UnicodeError):
            raise MailFailure('invalid_response') from None
        chosen = [p for p in parts if p[1] == 'plain'] or [p for p in parts if p[1] == 'html']
        texts, gaps = [], []
        remaining = request['body_bytes']
        for section, kind in chosen:
            if remaining <= 0:
                gaps.append('正文超过读取字节限制')
                break
            stage = 'body'
            mime = literal(checked(client, 'FETCH', uid, f'(BODY.PEEK[{section}.MIME])'))
            body_data = checked(client, 'FETCH', uid, f'(BODY.PEEK[{section}]<0.{remaining + 1}>)')
            body = literal(body_data)
            if len(body) > remaining:
                gaps.append('正文超过读取字节限制，编码可能不完整')
            body = body[:remaining]
            remaining -= len(body)
            if not mime or not any(isinstance(item, tuple) for item in body_data):
                raise MailFailure('invalid_response')
            stage = 'decode'
            msg = BytesParser(policy=email.policy.default).parsebytes(mime + b'\r\n' + body)
            decoded = msg.get_payload(decode=True) or b''
            try:
                text = decoded.decode(msg.get_content_charset() or 'utf-8', 'strict')
            except (LookupError, UnicodeError):
                text = decoded.decode('utf-8', 'replace')
                gaps.append('正文编码损坏或未知，已替换不可识别字符')
            if kind == 'html':
                parser = VisibleHTML()
                parser.feed(text)
                text = ''.join(parser.text)
            if msg.defects:
                gaps.append('MIME 内容存在解析缺陷')
            texts.append(text)
        if not chosen:
            gaps.append('没有可读取的纯文本或 HTML 正文；未读取附件及嵌套邮件')
        return {'validity': validity, 'text': '\n\n'.join(texts), 'gaps': gaps,
                'parts': [p[0] for p in chosen], 'format': 'plain' if any(p[1] == 'plain' for p in chosen) else 'html-visible'}
    except Exception as error:
        return failure(error, stage)
    finally:
        if client is not None:
            try:
                client.logout()
            except Exception:
                try:
                    client.shutdown()
                except Exception:
                    pass


def main():
    try:
        request = json.load(sys.stdin)
    except (ValueError, UnicodeError):
        result = failure(MailFailure('invalid_request'), 'request')
    else:
        try:
            result = run(request)
        except Exception as error:
            result = failure(error, 'worker')
    print(json.dumps(result, ensure_ascii=False))


if __name__ == '__main__':
    main()
