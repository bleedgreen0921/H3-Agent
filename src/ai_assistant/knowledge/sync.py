"""On-demand inventory, immutable snapshots and leased background synchronization."""

import asyncio
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import unquote, urlsplit

from sqlalchemy import text
from ai_assistant.knowledge import KnowledgeError
from ai_assistant.knowledge.config import resolve_knowledge_paths
from ai_assistant.knowledge.ingestion import snapshot
from ai_assistant.knowledge.sources import (
    SourceFailure,
    cloud_reader,
    cloud_reference,
    cloud_uri,
    local_descriptor,
    local_inventory,
    save_documents,
    EXTENSIONS,
)
from ai_assistant.storage.rag import RagStore
from ai_assistant.workspace.files import ReadOnlyWorkspace


def settings(config):
    return resolve_knowledge_paths(
        config.knowledge,
        config.config_path.parent if config.config_path else Path.cwd(),
    )


async def resolve_submission(config, reference, *, identity="user", tags=None):
    """仅解析来源；云端只读请求不占用数据库事务。"""
    base = config.config_path.parent if config.config_path else Path.cwd()
    if reference.startswith("https://"):
        descriptor = cloud_reference(config, reference, identity)
        async with cloud_reader(config, identity) as reader:
            descriptor = await reader.resolve(descriptor)
        uri = cloud_uri(config.integrations.feishu.app_id, identity,
                        descriptor["kind"], descriptor["token"])
    else:
        root = Path(config.workspace.root)
        root = root if root.is_absolute() else base / root
        workspace = ReadOnlyWorkspace(root, exclude_patterns=config.workspace.exclude_patterns,
                                      config_path=config.config_path)
        path = workspace._path(reference)
        if not path.is_file() or path.suffix.lower() not in EXTENSIONS:
            raise KnowledgeError("不支持的文档类型或文件不存在。")
        uri, descriptor = path.as_uri(), local_descriptor(path, root)
    if tags:
        descriptor["tags"] = list(tags)
    return uri, descriptor


async def register(config, reference, *, identity="user", restore=True, tags=None):
    uri, descriptor = await resolve_submission(config, reference, identity=identity, tags=tags)
    store = RagStore(settings(config))
    try:
        from ai_assistant.knowledge.management import register_submission
        result = await register_submission(store, uri, descriptor, restore=restore)
        return result if result.get('status') == 'removed' else dict(result, source_uri=uri)
    finally:
        await store.close()


async def configure(config, store):
    base = config.config_path.parent if config.config_path else Path.cwd()
    sources = {}
    for name, source in config.knowledge.sources.items():
        value = source.model_dump(mode="json")
        if source.path:
            path = source.path.expanduser()
            value["path"] = str(
                (path if path.is_absolute() else base / path).absolute()
            )
        sources[name] = value
    await store.configure_sources(sources)
    # 将旧版显式提交的工作区文档纳入刷新，首次查询即可核实旧资料。
    known = {m["uri"] for m in await store.members(include_disabled=True)}
    root = Path(config.workspace.root)
    root = (root if root.is_absolute() else base / root).resolve()
    for doc in await store.document_list():
        if (
            not doc["deleted"]
            and doc["source_uri"] not in known
            and doc["source_uri"].startswith("file://")
        ):
            p = Path(unquote(urlsplit(doc["source_uri"]).path))
            if p.is_relative_to(root):
                await store.add_submission(doc["source_uri"], local_descriptor(p, root), restore=False)


async def synchronize(config, store, run):
    jobs = []
    errors = []
    filters = run["filters"]
    await configure(config, store)
    sources = await store.source_list()
    target_uri = None
    if filters.get("document_id"):
        async with store.engine.connect() as conn:
            target_uri = (
                await conn.execute(
                    text(
                        "SELECT source_uri FROM rag_documents WHERE kb=:kb AND id=:id AND NOT deleted"
                    ),
                    dict(kb=store.config.knowledge_base_id, id=filters["document_id"]),
                )
            ).scalar()
        if target_uri is None:
            await store.settle_scan(
                run, [], [{"source": "document", "reason": "document_not_found"}]
            )
            return
    if filters.get("source_id") and not any(
        s["enabled"] and s["id"] == filters["source_id"] for s in sources
    ):
        await store.settle_scan(
            run, [], [{"source": "source", "reason": "source_not_found"}]
        )
        return
    for source in sources:
        if not source["enabled"]:
            continue
        name = source["id"]
        definition = source["config"]
        if filters.get("source_id") and filters["source_id"] != name:
            continue
        previous = await store.members(name)
        if target_uri:
            previous = [m for m in previous if m["uri"] == target_uri]
            if not previous:
                continue
        descriptors = []
        try:
            if target_uri or definition["kind"] == "submitted":
                descriptors = [(r["uri"], r["descriptor"]) for r in previous]
            elif definition["kind"] == "local_directory":
                descriptors = await asyncio.to_thread(
                    local_inventory, config, definition
                )
            else:
                async with cloud_reader(config, definition["identity"]) as reader:
                    descriptors = [
                        (
                            cloud_uri(
                                config.integrations.feishu.app_id,
                                definition["identity"],
                                d["kind"],
                                d["token"],
                            ),
                            d,
                        )
                        for d in await reader.inventory(definition)
                    ]
            present = {uri for uri, _ in descriptors}
            # 只有完整枚举成功，才能判断移出来源范围。
            for old in previous:
                if old["uri"] not in present:
                    await store.observe(
                        name, old["uri"], old["descriptor"], "removed", sync_run=run
                    )
        except Exception as exc:
            from ai_assistant.knowledge import LeaseLost

            if isinstance(exc, LeaseLost):
                raise
            async with store.engine.begin() as conn:
                await conn.execute(text("UPDATE rag_sources SET checked_at=now(),error=:error WHERE kb=:kb AND id=:id"),
                    dict(kb=store.config.knowledge_base_id, id=name,
                         error=exc.code if isinstance(exc, SourceFailure) else type(exc).__name__))
            errors.append(
                {
                    "source": name,
                    "reason": exc.code
                    if isinstance(exc, SourceFailure)
                    else type(exc).__name__,
                }
            )
            continue
        for uri, descriptor in descriptors:
            async with store.engine.connect() as conn:
                previous_document = (
                    (
                        await conn.execute(
                            text(
                                "SELECT d.deleted,v.metadata FROM rag_documents d LEFT JOIN rag_versions v ON v.id=d.desired_version WHERE d.kb=:kb AND d.source_uri=:uri"
                            ),
                            dict(kb=store.config.knowledge_base_id, uri=uri),
                        )
                    )
                    .mappings()
                    .one_or_none()
                )
            if previous_document and previous_document["deleted"]:
                continue
            if filters.get("document_id"):
                async with store.engine.connect() as conn:
                    match = (
                        await conn.execute(
                            text(
                                "SELECT id FROM rag_documents WHERE kb=:kb AND source_uri=:uri"
                            ),
                            dict(kb=store.config.knowledge_base_id, uri=uri),
                        )
                    ).scalar()
                if match != filters["document_id"]:
                    continue
            try:
                await store.sync_heartbeat(run)
                if descriptor["kind"] == "local":
                    path = Path(descriptor["path"])
                    if path.suffix.lower() not in EXTENSIONS:
                        raise SourceFailure("unsupported_file_type")
                    workspace = ReadOnlyWorkspace(
                        Path(descriptor["root"]),
                        exclude_patterns=config.workspace.exclude_patterns,
                        config_path=config.config_path,
                    )
                    _, digest, path, title = await asyncio.to_thread(
                        snapshot,
                        workspace,
                        Path(descriptor["path"])
                        .relative_to(descriptor["root"])
                        .as_posix(),
                        store.config.materials_dir,
                    )
                    metadata = {
                        "source_name": title,
                        "media_type": __import__("mimetypes").guess_type(title)[0]
                        or "application/octet-stream",
                        "tags": descriptor.get("tags", []),
                    }
                else:
                    async with cloud_reader(config, descriptor["identity"]) as reader:
                        descriptor = await reader.resolve(descriptor)
                        current, title = await reader.revision(descriptor)
                        cached = (
                            await store.cached_revision(uri, current)
                            if current is not None
                            else None
                        )
                        if cached and Path(cached["snapshot"]).is_file():
                            digest, path = cached["digest"], Path(cached["snapshot"])
                            after = current
                        else:
                            # 多页读取完成后核对可靠版本；表格无可靠版本时每轮重新读取正文。
                            for attempt in range(3):
                                before, title = await reader.revision(descriptor)
                                docs = await reader.read(descriptor)
                                after, _ = await reader.revision(descriptor)
                                if before == after:
                                    break
                            else:
                                raise SourceFailure("document_changing")
                            digest, path = await asyncio.to_thread(
                                save_documents, store.config.materials_dir, docs
                            )
                    metadata = {
                        "source_name": title
                        or descriptor.get("name")
                        or descriptor["token"],
                        "media_type": "application/x-feishu-" + descriptor["kind"],
                        "tags": descriptor.get("tags", []),
                        "source_url": descriptor.get("url", ""),
                        "identity": descriptor["identity"],
                        "remote_revision": after,
                    }
                if previous_document and previous_document["metadata"]:
                    metadata["tags"] = descriptor.get("tags", previous_document["metadata"].get("tags", []))
                await store.sync_heartbeat(run)
                job = await store.submit(
                    source_uri=uri,
                    digest=digest,
                    snapshot=path,
                    metadata=metadata,
                    sync_run=run,
                )
                await store.observe(name, uri, descriptor, "available", sync_run=run)
                jobs.append({"job_id": job, "source_id": name, "uri": uri})
            except Exception as exc:
                from ai_assistant.knowledge import LeaseLost

                if isinstance(exc, LeaseLost):
                    raise
                reason = (
                    exc.code if isinstance(exc, SourceFailure) else type(exc).__name__
                )
                state = (
                    "unavailable"
                    if reason in ("access_denied", "not_found")
                    or isinstance(exc, FileNotFoundError)
                    else "failed"
                )
                await store.observe(name, uri, descriptor, state, reason, sync_run=run)
                errors.append(
                    {
                        "source": name,
                        "document": str(
                            descriptor.get("name")
                            or descriptor.get("path")
                            or descriptor.get("token", "")
                        )[:160],
                        "reason": reason,
                    }
                )
        async with store.engine.begin() as conn:
            await conn.execute(
                text(
                    "UPDATE rag_sources SET checked_at=now(),error=NULL WHERE kb=:kb AND id=:id"
                ),
                dict(kb=store.config.knowledge_base_id, id=name),
            )
    await store.settle_scan(run, jobs, errors)


async def sync_once(config, store):
    run = await store.claim_sync()
    if run is None:
        await store.finish_syncs()
        return False

    async def heartbeat():
        while True:
            await asyncio.sleep(10)
            await store.sync_heartbeat(run)

    task = asyncio.create_task(synchronize(config, store, run))
    pulse = asyncio.create_task(heartbeat())
    try:
        remaining = max(
            0, (run["deadline"] - datetime.now(timezone.utc)).total_seconds()
        )
        async with asyncio.timeout(remaining):
            done, _ = await asyncio.wait(
                (task, pulse), return_when=asyncio.FIRST_COMPLETED
            )
            for completed in done:
                completed.result()
    except asyncio.CancelledError:
        raise
    except Exception as exc:
        from ai_assistant.knowledge import LeaseLost

        if not isinstance(exc, LeaseLost):
            await store.settle_scan(
                run, [], [{"source": "sync", "reason": type(exc).__name__}]
            )
    finally:
        task.cancel()
        pulse.cancel()
        await asyncio.gather(task, pulse, return_exceptions=True)
    await store.finish_syncs()
    return True


async def wait_sync(config, ident):
    store = RagStore(settings(config))
    try:
        while True:
            await store.finish_syncs()
            row = await store.sync_status(ident)
            if row is None:
                raise KnowledgeError("同步任务不存在。")
            if row["state"] == "ready":
                return row
            await asyncio.sleep(1)
    finally:
        await store.close()


from contextlib import asynccontextmanager


@asynccontextmanager
async def background_worker(config):
    """入口显式拥有 Worker 生命周期；初始化仍只通过管理命令进行。"""
    if config.knowledge is None:
        yield None
        return
    from ai_assistant.knowledge.ingestion import worker

    store = RagStore(settings(config))
    try:
        await store.check_schema()
    except BaseException:
        await store.close()
        raise
    task = asyncio.create_task(worker(store, app_config=config))
    try:
        yield task
    finally:
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)
        await store.close()
