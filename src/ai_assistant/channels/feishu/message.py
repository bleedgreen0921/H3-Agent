"""原生 Markdown 消息及按最终请求字节数计算的无损分段。"""
import json
import re

# 每个飞书请求的字节预算，包含外层请求和消息内容的 JSON 编码开销。
REQUEST_BUDGET = 28_000


def markdown_message(text):
    """将文本包装为飞书 post 消息中的 Markdown 内容。"""
    return {'msg_type': 'post', 'content': json.dumps(
        {'zh_cn': {'content': [[{'tag': 'md', 'text': text}]]}}, ensure_ascii=False)}


def validate_message(message):
    """仅接受当前保存的消息结构，未知格式不能降级重发。"""
    try:
        # 严格校验外层字段及内层结构，避免把旧格式误当成当前格式发送。
        if not isinstance(message, dict) or set(message) != {'msg_type', 'content'} or message['msg_type'] != 'post':
            raise ValueError
        content = json.loads(message['content'])
        node = content['zh_cn']['content'][0][0]
        if not isinstance(node['text'], str) or content != {'zh_cn': {'content': [[{'tag': 'md', 'text': node['text']}]]}}:
            raise ValueError
    except (ValueError, TypeError, KeyError, IndexError):
        raise ValueError('不支持的飞书消息格式；未发送。') from None


def request_body(chat_id, message, operation_id):
    """组装发送接口请求，operation_id 作为消息 UUID。"""
    return {'receive_id': chat_id, **message, 'uuid': operation_id}


def request_size(text, chat_id, operation_id):
    """计算指定正文对应的完整请求在 UTF-8 编码后的字节数。"""
    # 与 httpx 的 json 请求编码保持一致，计入两层 JSON 转义。
    return len(json.dumps(request_body(chat_id, markdown_message(text), operation_id),
                          ensure_ascii=False, separators=(',', ':'), allow_nan=False).encode('utf-8'))


def markdown_blocks(text):
    """空行分段；围栏内空行保留在同一块，连续表格行自然保持完整。"""
    lines = []
    fence = None
    for line in text.splitlines(keepends=True):
        # 识别反引号或波浪号代码围栏，围栏内部的空行不触发分块。
        match = re.match(r'^ {0,3}(`{3,}|~{3,})(.*?)(?:\r?\n)?$', line)
        if match:
            marker, suffix = match.groups()
            if fence is None:
                fence = marker
            elif marker[0] == fence[0] and len(marker) >= len(fence) and not suffix.strip():
                fence = None
        lines.append(line)

        # 保留原始换行符，确保各块按顺序拼接后还原输入。
        if not line.strip() and fence is None:
            yield ''.join(lines)
            lines = []
    if lines:
        yield ''.join(lines)


def split_markdown(text, chat_id, budget=REQUEST_BUDGET):
    """按完整请求字节预算切分正文，尽量保留 Markdown 块。"""
    # 新批次 UUID 均为 36 个 ASCII 字符；登记前即可精确计算请求大小。
    operation_id = '0' * 36
    if request_size('', chat_id, operation_id) >= budget:
        raise ValueError('飞书消息预算不足，无法登记正文。')

    def fits(value):
        return request_size(value, chat_id, operation_id) <= budget

    pending = ''
    for block in markdown_blocks(text):
        # 优先合并完整块；放不下时先输出已累积的正文。
        if fits(pending + block):
            pending += block
            continue
        if pending:
            yield pending
            pending = ''
        while not fits(block):
            # 超长连续块用二分查找字符边界，再优先使用最近的换行。
            low, high = 0, len(block)
            while low < high:
                middle = (low + high + 1) // 2
                if fits(block[:middle]):
                    low = middle
                else:
                    high = middle - 1
            if low == 0:
                raise ValueError('飞书消息预算不足，无法登记正文。')
            newline = block.rfind('\n', 0, low)
            end = newline + 1 if newline >= 0 else low
            yield block[:end]
            block = block[end:]
        pending = block

    # 输出最后一个非空分段；空正文不产生分段。
    if pending:
        yield pending
