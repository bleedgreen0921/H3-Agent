"""检查聚合后的原生模型响应，阻止不完整结果进入业务流程。"""


class IncompleteResponseError(RuntimeError):
    """仅包含固定错误摘要，不携带服务端内容。"""


def validate_completion(message, *, responses=False):
    metadata = message.response_metadata
    status = metadata.get('status')
    if status == 'incomplete' or metadata.get('finish_reason') == 'length':
        raise IncompleteResponseError('模型输出被截断，本轮未完成。')
    if status == 'failed':
        raise IncompleteResponseError('模型请求失败，本轮未完成。')
    if (message.additional_kwargs.get('refusal')
            or metadata.get('finish_reason') in ('content_filter', 'refusal')
            or isinstance(message.content, list) and any(
                isinstance(block, dict) and block.get('type') == 'refusal'
                for block in message.content)):
        raise IncompleteResponseError('模型拒绝生成，本轮未完成。')
    if responses and status != 'completed':
        raise IncompleteResponseError('模型响应缺少完成状态，本轮未完成。')
    if message.invalid_tool_calls:
        raise IncompleteResponseError('模型工具参数无效，本轮未完成。')
    return message
