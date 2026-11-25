"""Web 上传附件的只读工具和显式入库工具。"""
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
    """读取当前会话上传的 PDF、DOCX 或 Markdown 附件正文；cursor 续读。"""
    try:
        result = await _service(runtime).read(
            attachment_id, runtime.context.session_id, cursor)
    except Exception as error:
        raise ToolException(str(error) if str(error) else "附件读取失败。") from None
    import json
    return json.dumps(result, ensure_ascii=False), {**result, "full_tool_content": result}


@tool
async def ingest_attachment(
    attachment_id: str, runtime: ToolRuntime[ToolContext]
) -> dict:
    """用户明确要求时，把当前会话附件提交到知识库；提交不等于可检索。"""
    try:
        result = await _service(runtime).submit_to_knowledge(
            attachment_id, runtime.context.session_id)
    except Exception as error:
        raise ToolException(str(error) if str(error) else "附件入库失败。") from None
    from ai_assistant.observability import event
    event("knowledge.registration", action="attachment", status=result.get("status"),
          job_id=result.get("job_id"))
    return result


def attachment_tools():
    return [item.model_copy(update={"handle_tool_error": True})
            for item in (read_attachment, ingest_attachment)]
