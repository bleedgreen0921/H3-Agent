"""Explicit read-only source adapters; cloud content never becomes instructions."""

import asyncio
from contextlib import asynccontextmanager
import fnmatch
import hashlib
import json
import os
from pathlib import Path
from urllib.parse import urlsplit, quote

import httpx
from langchain_core.documents import Document
from ai_assistant.knowledge import KnowledgeError
from ai_assistant.workspace.files import ReadOnlyWorkspace, WorkspaceError

EXTENSIONS = {".txt", ".md", ".pdf", ".docx", ".xlsx", ".pptx", ".csv"}
BASE = "https://open.feishu.cn/open-apis"


class SourceFailure(KnowledgeError):
    def __init__(self, code):
        self.code = code
        super().__init__(code)


def cloud_uri(app, identity, kind, token):
    return f"feishu://{app}/{identity}/{kind}/{token}"


def cloud_reference(config, value, identity="user"):
    url = urlsplit(value)
    if (
        url.scheme != "https"
        or not url.hostname
        or not url.hostname.endswith((".feishu.cn", ".larksuite.com"))
    ):
        raise KnowledgeError("仅接受飞书或 Lark 的 HTTPS 文档链接。")
    parts = url.path.strip("/").split("/")
    kinds = {"docx": "docx", "wiki": "wiki", "sheets": "sheet", "base": "bitable"}
    if len(parts) != 2 or parts[0] not in kinds or not parts[1].isalnum():
        raise KnowledgeError("不支持该飞书文档链接类型。")
    return dict(
        kind=kinds[parts[0]],
        token=parts[1],
        identity=identity,
        url=value.split("?")[0].split("#")[0],
    )


def local_descriptor(path, root):
    path, root = Path(path), Path(root)
    return {
        "kind": "local",
        "path": str(path.resolve()),
        "root": str(root.resolve()),
        "name": path.name,
    }


def local_inventory(config, source):
    base = config.config_path.parent if config.config_path else Path.cwd()
    root = Path(source["path"]).expanduser()
    root = (root if root.is_absolute() else base / root).absolute()
    if root.is_symlink() or not root.is_dir():
        raise SourceFailure("local_directory_unavailable")
    root = root.resolve()
    workspace = ReadOnlyWorkspace(
        root,
        exclude_patterns=config.workspace.exclude_patterns,
        config_path=config.config_path,
    )
    blocked = [
        Path(getattr(config.knowledge, k)).resolve()
        for k in ("materials_dir", "diagnostics_dir")
    ]
    result = []

    def fail(error):
        raise SourceFailure("local_enumeration_failed")

    for directory, dirs, names in os.walk(root, followlinks=False, onerror=fail):
        dirs[:] = [
            d
            for d in dirs
            if not d.startswith(".")
            and not (Path(directory) / d).is_symlink()
            and not any(
                (Path(directory) / d).resolve().is_relative_to(p) for p in blocked
            )
        ]
        for name in sorted(names):
            path = Path(directory) / name
            relative = path.relative_to(root).as_posix()
            if not any(
                fnmatch.fnmatch(relative, p) for p in source.get("include", ["*"])
            ):
                continue
            if any(fnmatch.fnmatch(relative, p) for p in source.get("exclude", [])):
                continue
            try:
                checked = workspace._path(relative)
                if checked.is_symlink() or not checked.is_file():
                    continue
            except WorkspaceError:
                continue
            result.append((checked.as_uri(), local_descriptor(checked, root)))
    return result


class FeishuReader:
    """One explicit identity, reused token lifecycle and a bounded HTTP client."""

    def __init__(self, config, identity, client):
        self.config, self.identity, self.client = config, identity, client
        self.api = None
        self.tokens = None
        if identity == "user":
            from ai_assistant.integrations.oauth import TokenStore

            root = Path(config.workspace.root)
            base = config.config_path.parent if config.config_path else Path.cwd()
            self.tokens = TokenStore(
                config, (root if root.is_absolute() else base / root).resolve()
            )
        else:
            from ai_assistant.integrations.feishu_files import FeishuAPI

            self.api = FeishuAPI(config.integrations.feishu, client)

    async def request(self, method, path, **kwargs):
        for attempt in range(3):
            token = (
                await self.tokens.access() if self.tokens else await self.api.access()
            )
            try:
                response = await self.client.request(
                    method,
                    BASE + path,
                    headers={"Authorization": "Bearer " + token},
                    **kwargs,
                )
            except httpx.RequestError:
                raise SourceFailure("cloud_connection_failed") from None
            if response.status_code == 429 and attempt < 2:
                await asyncio.sleep(min(2**attempt, 4))
                continue
            if response.status_code in (403, 404):
                raise SourceFailure(
                    "access_denied" if response.status_code == 403 else "not_found"
                )
            if response.status_code != 200:
                raise SourceFailure("cloud_http_" + str(response.status_code))
            try:
                payload = response.json()
            except ValueError:
                raise SourceFailure("cloud_response_invalid") from None
            if payload.get("code", 0):
                # 业务错误码保留用于诊断，不输出可能含文档或凭据的 msg。
                raise SourceFailure("cloud_api_" + str(payload["code"]))
            return payload.get("data", payload)
        raise SourceFailure("cloud_rate_limited")

    async def pages(self, path, params=None, method="GET", body=None):
        params = dict(params or {})
        result = []
        seen = set()
        while True:
            data = await self.request(
                method,
                path,
                params=params,
                **({"json": body or {}} if method == "POST" else {}),
            )
            result.extend(data.get("items", []))
            if not data.get("has_more"):
                return result
            token = data.get("page_token") or data.get("next_page_token")
            if not token or token in seen:
                raise SourceFailure("cloud_pagination_invalid")
            seen.add(token)
            params["page_token"] = token

    async def resolve(self, descriptor):
        result = dict(descriptor)
        if result["kind"] == "wiki":
            node = (
                await self.request(
                    "GET", "/wiki/v2/spaces/get_node", params={"token": result["token"]}
                )
            )["node"]
            result.update(
                kind=node["obj_type"],
                token=node["obj_token"],
                name=node.get("title", ""),
            )
        if not result.get("url") or "/wiki/" in result.get("url", ""):
            try:
                meta = await self.request(
                    "POST",
                    "/drive/v1/metas/batch_query",
                    json={
                        "request_docs": [
                            {"doc_token": result["token"], "doc_type": result["kind"]}
                        ],
                        "with_url": True,
                    },
                )
                if meta.get("metas"):
                    result["url"] = meta["metas"][0].get("url") or result.get("url", "")
                    result["name"] = meta["metas"][0].get("title") or result.get(
                        "name", ""
                    )
            except SourceFailure:
                # 元数据 URL 非正文读取的授权依据，失败时保留用户提供的原链接。
                pass
        return result

    async def inventory(self, source):
        result = []
        visited = set()

        async def walk(token):
            if token in visited:
                return
            visited.add(token)
            if source["kind"] == "feishu_wiki":
                items = await self.pages(
                    "/wiki/v2/spaces/" + quote(source["token"], safe="") + "/nodes",
                    {
                        "page_size": 50,
                        **({"parent_node_token": token} if token else {}),
                    },
                )
                for item in items:
                    result.append(
                        dict(
                            kind=item["obj_type"],
                            token=item["obj_token"],
                            name=item.get("title", ""),
                            identity=source["identity"],
                            url=item.get("url") or "",
                        )
                    )
                    if item.get("has_child"):
                        await walk(item["node_token"])
            else:
                items = await self.pages(
                    "/drive/v1/files", {"folder_token": token, "page_size": 100}
                )
                for item in items:
                    if item["type"] == "folder":
                        await walk(item["token"])
                    else:
                        result.append(
                            dict(
                                kind=item["type"],
                                token=item["token"],
                                name=item.get("name", ""),
                                identity=source["identity"],
                                url=item.get("url", ""),
                            )
                        )

        if source["kind"] == "feishu_wiki" and source.get("node_token"):
            seed = await self.resolve(
                dict(
                    kind="wiki", token=source["node_token"], identity=source["identity"]
                )
            )
            result.append(seed)
        await walk(
            source.get("node_token")
            if source["kind"] == "feishu_wiki"
            else source["token"]
        )
        return result

    async def revision(self, d):
        token = quote(d["token"], safe="")
        if d["kind"] == "docx":
            doc = (await self.request("GET", f"/docx/v1/documents/{token}"))["document"]
            return str(doc["revision_id"]), doc.get("title", d.get("name", ""))
        # 不假设表格元数据更新时间涵盖单元格、公式和记录的全部变化。
        return None, d.get("name", "")

    async def read(self, d):
        token = quote(d["token"], safe="")
        kind = d["kind"]
        docs = []
        if kind == "docx":
            blocks = await self.pages(
                f"/docx/v1/documents/{token}/blocks", {"page_size": 500}
            )
            headings = []
            for index, block in enumerate(blocks, 1):
                parts = []
                for value in block.values():
                    if not isinstance(value, dict):
                        continue
                    for element in value.get("elements", []):
                        if "text_run" in element:
                            parts.append(element["text_run"].get("content", ""))
                        elif "mention_doc" in element:
                            parts.append(element["mention_doc"].get("title", ""))
                content = "".join(parts)
                block_type = block.get("block_type", 0)
                if 3 <= block_type <= 11:
                    level = block_type - 2
                    headings = headings[: level - 1] + [content]
                if content.strip():
                    docs.append(
                        Document(
                            page_content=content,
                            metadata={
                                "block_start": index,
                                "block_end": index,
                                "element_refs": [block["block_id"]],
                                "heading_path": list(headings),
                            },
                        )
                    )
        elif kind == "sheet":
            sheets = (
                await self.request(
                    "GET", f"/sheets/v3/spreadsheets/{token}/sheets/query"
                )
            ).get("sheets", [])
            for sheet in sorted(sheets, key=lambda item: item["sheet_id"]):
                sid = sheet["sheet_id"]
                grid = sheet.get("grid_properties", {})
                rows = grid.get("row_count", 0)
                columns = grid.get("column_count", 0)
                header = []

                def col(n):
                    value = ""
                    while n:
                        n, r = divmod(n - 1, 26)
                        value = chr(65 + r) + value
                    return value

                # 每页最多 5000 单元格，行/列均分块，避免稀疏大表一次拉满。
                for c in range(0, columns, 100):
                    width = min(columns - c, 100)
                    page_rows = max(1, min(200, 5000 // width))
                    header = []
                    for start in range(0, rows, page_rows):
                        end = min(start + page_rows, rows)
                        area = f"{sid}!{col(c + 1)}{start + 1}:{col(c + width)}{end}"
                        data = await self.request(
                            "GET",
                            f"/sheets/v2/spreadsheets/{token}/values/"
                            + quote(area, safe=""),
                            params={"valueRenderOption": "ToString"},
                        )
                        values = data.get("valueRange", {}).get("values", [])
                        formula_data = await self.request(
                            "GET",
                            f"/sheets/v2/spreadsheets/{token}/values/"
                            + quote(area, safe=""),
                            params={"valueRenderOption": "Formula"},
                        )
                        formulas = formula_data.get("valueRange", {}).get("values", [])
                        if start == 0 and values:
                            header = values[0]
                        for offset, row in enumerate(values):
                            if not any(v is not None and str(v).strip() for v in row):
                                continue
                            content = " | ".join(
                                f"{header[i] if i < len(header) else col(c + i + 1)}: {v}"
                                for i, v in enumerate(row)
                            )
                            if offset < len(formulas):
                                found = [
                                    f"{col(c + i + 1)}{start + offset + 1}: {value}"
                                    for i, value in enumerate(formulas[offset])
                                    if isinstance(value, str) and value.startswith("=")
                                ]
                                if found:
                                    content += "\n公式：" + "; ".join(found)
                            docs.append(
                                Document(
                                    page_content=content,
                                    metadata={
                                        "sheet_id": sid,
                                        "sheet_name": sheet.get("title", ""),
                                        "cell_range": f"{col(c + 1)}{start + offset + 1}:{col(c + width)}{start + offset + 1}",
                                    },
                                )
                            )
        elif kind == "bitable":
            tables = await self.pages(
                f"/bitable/v1/apps/{token}/tables", {"page_size": 100}
            )
            for table in sorted(tables, key=lambda item: item["table_id"]):
                tid = table["table_id"]
                path = f"/bitable/v1/apps/{token}/tables/{quote(tid, safe='')}"
                fields = await self.pages(path + "/fields", {"page_size": 100})
                names = {f["field_name"] for f in fields}
                records = await self.pages(
                    path + "/records/search",
                    {"page_size": 500},
                    "POST",
                    {"automatic_fields": True},
                )
                unique = {}
                for record in records:
                    rid = record["record_id"]
                    if rid in unique and unique[rid] != record:
                        raise SourceFailure("records_changed_during_pagination")
                    unique[rid] = record
                for record in sorted(
                    unique.values(), key=lambda item: item["record_id"]
                ):
                    content = "\n".join(
                        f"{name}: {json.dumps(value, ensure_ascii=False, sort_keys=True)}"
                        for name, value in sorted(record.get("fields", {}).items())
                        if name in names
                    )
                    if content.strip():
                        docs.append(
                            Document(
                                page_content=content,
                                metadata={
                                    "table_id": tid,
                                    "table_name": table.get("name", ""),
                                    "record_id": record["record_id"],
                                },
                            )
                        )
        else:
            raise SourceFailure("unsupported_cloud_type_" + kind)
        if not docs:
            raise SourceFailure("empty_document")
        return docs


@asynccontextmanager
async def cloud_reader(config, identity):
    if config.integrations.feishu is None or config.mcp is None:
        raise SourceFailure("feishu_not_configured")
    async with httpx.AsyncClient(
        timeout=30, trust_env=False, follow_redirects=False
    ) as client:
        yield FeishuReader(config, identity, client)


def save_documents(directory, docs):
    """使用稳定正文与定位信息计算版本，检查时间不参与 hash。"""
    raw = json.dumps(
        [{"page_content": d.page_content, "metadata": d.metadata} for d in docs],
        ensure_ascii=False,
        sort_keys=True,
    ).encode()
    digest = hashlib.sha256(raw).hexdigest()
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True, mode=0o700)
    path = directory / (digest + ".ragjson")
    import tempfile

    with tempfile.NamedTemporaryFile(dir=directory, delete=False) as stream:
        temporary = Path(stream.name)
        stream.write(raw)
        stream.flush()
        os.fsync(stream.fileno())
    try:
        temporary.chmod(0o400)
        try:
            os.link(temporary, path)
        except FileExistsError:
            if path.is_symlink() or path.read_bytes() != raw:
                raise KnowledgeError("不可变云文档快照校验失败。")
    finally:
        temporary.unlink(missing_ok=True)
    return digest, path
