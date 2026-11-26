"""Web 上传附件的只读工具。"""
from langchain.tools import ToolRuntime, tool
from langchain_core.tools import ToolException
from ai_assistant.tools.context import ToolContext


def _service(runtime):
    value = runtime.context.services.get("attachments")
    if value is None:
        raise ToolException("附件读取尚未启用。")
    if not runtime.context.session_id:
        raise ToolException("附件读取需要当前会话。")
    return value


@tool(response_format="content_and_artifact")
async def read_attachment(
    attachment_id: str,
    runtime: ToolRuntime[ToolContext],
    cursor: str | None = None,
) -> tuple[dict, dict]:
    """读取当前会话上传的 PDF、DOCX、文本、代码、JSON 或 CSV 附件；cursor 续读。"""
    try:
        result = await _service(runtime).read(
            attachment_id, runtime.context.session_id, cursor)
    except Exception as error:
        raise ToolException(str(error) if str(error) else "附件读取失败。") from None
    import json
    return json.dumps(result, ensure_ascii=False), {**result, "full_tool_content": result}


def attachment_tools():
    return [read_attachment.model_copy(update={"handle_tool_error": True})]
