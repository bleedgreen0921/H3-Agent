"""只读飞书云文档服务：复用知识库同步使用的 FeishuReader。"""
from __future__ import annotations


from ai_assistant.feishu_docs.reader import FeishuReader, cloud_reference
from ai_assistant.knowledge.sources import SourceFailure


class FeishuDocsService:
    """一个明确身份的只读文档服务；不提供任何写入方法。"""

    def __init__(self, reader: FeishuReader):
        self.reader = reader

    async def resolve(self, url: str | None = None, token: str | None = None,
                      kind: str | None = None):
        if url:
            descriptor = cloud_reference(self.reader.config, url,
                                         self.reader.identity)
        elif token:
            descriptor = {"kind": kind or "docx", "token": token,
                          "identity": self.reader.identity, "url": ""}
        else:
            raise SourceFailure("document_reference_required")
        return await self.reader.resolve(descriptor)

    async def read_document(self, *, url: str | None = None, token: str | None = None,
                            kind: str | None = None, start: int = 0,
                            max_chars: int = 16000):
        descriptor = await self.resolve(url=url, token=token, kind=kind)
        docs = await self.reader.read(descriptor)
        chunks = []
        for index, doc in enumerate(docs, 1):
            metadata = doc.metadata or {}
            location = (metadata.get("heading_path") and " / ".join(metadata["heading_path"])) or \
                (metadata.get("sheet_name") and f"{metadata['sheet_name']}!{metadata.get('cell_range', '')}") or \
                (metadata.get("table_name") and f"{metadata['table_name']}:{metadata.get('record_id', '')}") or \
                f"part {index}"
            chunks.append(f"[{location}]\n{doc.page_content}")
        text = "\n\n".join(chunks)
        start = max(0, min(int(start), len(text)))
        end = min(len(text), start + max(1, int(max_chars)))
        return {
            "title": descriptor.get("name") or descriptor.get("token"),
            "url": descriptor.get("url") or "",
            "kind": descriptor.get("kind"),
            "text": text[start:end],
            "start": start,
            "end": end,
            "total_chars": len(text),
            "next_start": end if end < len(text) else None,
            "complete": end >= len(text),
            "external_data": True,
        }

    async def list_bitable_tables(self, app_token: str):
        tables = await self.reader.bitable_tables(app_token)
        return {"tables": tables, "total": len(tables)}

    async def list_bitable_fields(self, app_token: str, table_id: str):
        fields = await self.reader.bitable_fields(app_token, table_id)
        return {"fields": fields, "total": len(fields)}

    async def search_bitable_records(self, app_token: str, table_id: str, *,
                                     filter: dict | None = None,
                                     field_names: list[str] | None = None,
                                     page_size: int = 100,
                                     page_token: str | None = None):
        result = await self.reader.bitable_records(
            app_token, table_id, filter=filter, field_names=field_names,
            page_size=page_size, page_token=page_token)
        return {
            **result,
            "filter": filter,
            "field_names": field_names,
            "external_data": True,
        }

    async def list_wiki_nodes(self, space_token: str, parent_node_token: str | None = None):
        params = {"page_size": 50}
        if parent_node_token:
            params["parent_node_token"] = parent_node_token
        from urllib.parse import quote
        nodes = await self.reader.pages(
            "/wiki/v2/spaces/" + quote(space_token, safe="") + "/nodes", params)
        return {"nodes": nodes, "total": len(nodes)}

    async def list_drive_files(self, folder_token: str):
        from urllib.parse import quote
        files = await self.reader.pages(
            "/drive/v1/files",
            {"folder_token": quote(folder_token, safe=""), "page_size": 100})
        return {"files": files, "total": len(files)}
