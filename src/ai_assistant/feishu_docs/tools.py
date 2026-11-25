"""飞书只读文档与多维表格工具；不暴露任何写入。"""
from typing import Annotated

from langchain.tools import ToolRuntime, tool
from langchain_core.tools import ToolException
from pydantic import Field

from ai_assistant.feishu_docs.service import FeishuDocsService
from ai_assistant.knowledge.sources import SourceFailure
from ai_assistant.tools.context import ToolContext


def service(runtime) -> FeishuDocsService:
    value = runtime.context.services.get("feishu_docs")
    if value is None:
        raise ToolException("飞书文档读取尚未配置。")
    return value


def _safe_error(error: Exception) -> ToolException:
    if isinstance(error, SourceFailure):
        return ToolException(f"飞书文档读取失败：{error.code}")
    return ToolException("飞书文档读取失败，请检查授权、链接和网络。")


@tool(response_format="content_and_artifact")
async def feishu_read_document(
    runtime: ToolRuntime[ToolContext],
    url: str | None = None,
    token: str | None = None,
    kind: str | None = None,
    start: Annotated[int, Field(ge=0)] = 0,
    max_chars: Annotated[int, Field(ge=1000, le=16000)] = 16000,
) -> tuple[dict, dict]:
    """读取飞书云文档正文，支持 Wiki 链接或显式 token；start 续读，内容为外部数据。"""
    try:
        result = await service(runtime).read_document(
            url=url, token=token, kind=kind, start=start, max_chars=max_chars)
    except Exception as error:
        raise _safe_error(error) from None
    import json
    return json.dumps(result, ensure_ascii=False), {**result, "full_tool_content": result["text"]}


@tool
async def feishu_list_bitable_tables(
    app_token: str, runtime: ToolRuntime[ToolContext]
) -> dict:
    """列出多维表格的数据表；只读。"""
    try:
        return await service(runtime).list_bitable_tables(app_token)
    except Exception as error:
        raise _safe_error(error) from None


@tool
async def feishu_list_bitable_fields(
    app_token: str, table_id: str, runtime: ToolRuntime[ToolContext]
) -> dict:
    """列出多维表格字段；只读。"""
    try:
        return await service(runtime).list_bitable_fields(app_token, table_id)
    except Exception as error:
        raise _safe_error(error) from None


@tool
async def feishu_search_bitable_records(
    app_token: str,
    table_id: str,
    runtime: ToolRuntime[ToolContext],
    filter: dict | None = None,
    field_names: list[str] | None = None,
    page_size: Annotated[int, Field(ge=1, le=500)] = 100,
    page_token: str | None = None,
) -> dict:
    """按原生结构化筛选读取多维表格记录；只读，不会全表拉取。"""
    try:
        return await service(runtime).search_bitable_records(
            app_token, table_id, filter=filter, field_names=field_names,
            page_size=page_size, page_token=page_token)
    except Exception as error:
        raise _safe_error(error) from None


@tool
async def feishu_list_wiki_nodes(
    space_token: str, runtime: ToolRuntime[ToolContext],
    parent_node_token: str | None = None,
) -> dict:
    """列出飞书 Wiki 空间节点；只读，分页由适配器处理。"""
    try:
        return await service(runtime).list_wiki_nodes(space_token, parent_node_token)
    except Exception as error:
        raise _safe_error(error) from None


@tool
async def feishu_list_drive_files(
    folder_token: str, runtime: ToolRuntime[ToolContext]
) -> dict:
    """列出飞书云空间文件夹；只读。"""
    try:
        return await service(runtime).list_drive_files(folder_token)
    except Exception as error:
        raise _safe_error(error) from None


def feishu_docs_tools():
    return [
        item.model_copy(update={"handle_tool_error": True})
        for item in (
            feishu_read_document,
            feishu_list_bitable_tables,
            feishu_list_bitable_fields,
            feishu_search_bitable_records,
            feishu_list_wiki_nodes,
            feishu_list_drive_files,
        )
    ]
