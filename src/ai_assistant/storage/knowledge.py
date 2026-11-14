"""Durable source observations and query-scoped publication sets."""

import hashlib
import json
from uuid import uuid4
from contextlib import asynccontextmanager
from sqlalchemy import text


@asynccontextmanager
async def transaction(engine, conn=None):
    """复用调用方事务，确保业务效果与管理回执一起提交。"""
    if conn is not None:
        yield conn
    else:
        async with engine.begin() as connection:
            yield connection


# 管理回执、来源及成员观察、同步任务与同范围活动任务的唯一约束。
DDL = (
    """CREATE TABLE IF NOT EXISTS rag_management_operations (
        kb text NOT NULL,principal text NOT NULL,input_id text NOT NULL,
        action text NOT NULL,target text,result jsonb NOT NULL,
        created_at timestamptz NOT NULL DEFAULT now(),PRIMARY KEY(kb,principal,input_id))""",
    """CREATE TABLE IF NOT EXISTS rag_sources (
        kb text NOT NULL,id text NOT NULL,config jsonb NOT NULL,enabled boolean NOT NULL DEFAULT true,
        checked_at timestamptz,error text,PRIMARY KEY(kb,id))""",
    """CREATE TABLE IF NOT EXISTS rag_source_documents (
        kb text NOT NULL,source_id text NOT NULL,uri text NOT NULL,descriptor jsonb NOT NULL,
        state text NOT NULL DEFAULT 'pending',checked_at timestamptz,error text,
        PRIMARY KEY(kb,source_id,uri))""",
    """CREATE TABLE IF NOT EXISTS rag_sync_runs (
        id text PRIMARY KEY,kb text NOT NULL,scope text NOT NULL,state text NOT NULL DEFAULT 'queued',
        filters jsonb NOT NULL,builds jsonb NOT NULL DEFAULT '[]',
        errors jsonb NOT NULL DEFAULT '[]',jobs jsonb NOT NULL DEFAULT '[]',
        token text,lease_until timestamptz,created_at timestamptz NOT NULL DEFAULT now(),
        deadline timestamptz NOT NULL,finished_at timestamptz)""",
    """CREATE UNIQUE INDEX IF NOT EXISTS rag_sync_live ON rag_sync_runs(kb,scope)
        WHERE state IN ('queued','running','indexing')""",
)


def packed(value):
    """将值规范化为 JSON 文本，供参数传递及同步范围摘要使用。"""
    return json.dumps(value, ensure_ascii=False, sort_keys=True, default=str)


class SourceStore:
    """管理知识来源、同步租约、业务回执和查询可用构建。"""

    async def configure_sources(self, sources):
        """更新配置来源，停用不再配置的非提交来源。"""
        kb = self.config.knowledge_base_id
        async with self.engine.begin() as conn:
            await conn.execute(
                text(
                    "UPDATE rag_sources SET enabled=false WHERE kb=:kb AND id NOT LIKE :prefix"
                ),
                dict(kb=kb, prefix="submitted:%"),
            )
            for name, source in sources.items():
                await conn.execute(
                    text("""INSERT INTO rag_sources(kb,id,config) VALUES(:kb,:id,CAST(:c AS jsonb))
                    ON CONFLICT(kb,id) DO UPDATE SET config=EXCLUDED.config,enabled=true"""),
                    dict(kb=kb, id=name, c=packed(source)),
                )

    async def add_submission(self, uri, descriptor, *, restore=True, conn=None):
        """登记显式提交来源，并按 restore 决定是否恢复已移除文档。"""
        source = "submitted:" + hashlib.sha256(uri.encode()).hexdigest()[:24]
        kb = self.config.knowledge_base_id
        async with transaction(self.engine, conn) as conn:
            # 与同步提交、移除使用相同文档行锁，避免并发绕过移除标记。
            removed = (await conn.execute(text(
                "SELECT deleted FROM rag_documents WHERE kb=:kb AND source_uri=:uri FOR UPDATE"
            ), dict(kb=kb, uri=uri))).scalar()
            if removed and not restore:
                return None
            await conn.execute(
                text("""INSERT INTO rag_sources(kb,id,config)
                VALUES(:kb,:id,CAST(:c AS jsonb)) ON CONFLICT(kb,id) DO UPDATE SET enabled=true"""),
                dict(kb=kb, id=source, c=packed({"kind": "submitted"})),
            )
            await conn.execute(
                text("""INSERT INTO rag_source_documents(kb,source_id,uri,descriptor)
                VALUES(:kb,:id,:uri,CAST(:d AS jsonb)) ON CONFLICT(kb,source_id,uri)
                DO UPDATE SET descriptor=rag_source_documents.descriptor || EXCLUDED.descriptor"""),
                dict(kb=kb, id=source, uri=uri, d=packed(descriptor)),
            )
            await conn.execute(
                text(
                    "UPDATE rag_documents SET deleted=false WHERE kb=:kb AND source_uri=:uri AND :restore"
                ),
                dict(kb=kb, uri=uri, restore=restore),
            )
        return source

    async def source_list(self):
        """列出当前知识库的来源记录。"""
        async with self.engine.connect() as conn:
            return [
                dict(r)
                for r in (
                    await conn.execute(
                        text("SELECT * FROM rag_sources WHERE kb=:kb ORDER BY id"),
                        dict(kb=self.config.knowledge_base_id),
                    )
                ).mappings()
            ]

    async def document_uri(self, ident):
        async with self.engine.connect() as conn:
            return (await conn.execute(text(
                "SELECT source_uri FROM rag_documents WHERE kb=:kb AND id=:id AND NOT deleted"),
                dict(kb=self.config.knowledge_base_id, id=ident))).scalar()

    async def document_for_uri(self, uri):
        async with self.engine.connect() as conn:
            return (await conn.execute(text(
                "SELECT d.deleted,v.metadata FROM rag_documents d "
                "LEFT JOIN rag_versions v ON v.id=d.desired_version "
                "WHERE d.kb=:kb AND d.source_uri=:uri"),
                dict(kb=self.config.knowledge_base_id, uri=uri))).mappings().one_or_none()

    async def document_id_for_uri(self, uri):
        async with self.engine.connect() as conn:
            return (await conn.execute(text(
                "SELECT id FROM rag_documents WHERE kb=:kb AND source_uri=:uri"),
                dict(kb=self.config.knowledge_base_id, uri=uri))).scalar()

    async def mark_source_error(self, source, error):
        async with self.engine.begin() as conn:
            await conn.execute(text(
                "UPDATE rag_sources SET checked_at=now(),error=:error WHERE kb=:kb AND id=:id"),
                dict(kb=self.config.knowledge_base_id, id=source, error=error))

    async def sync_status_counts(self, ident):
        """按状态统计同步任务关联的去重索引任务。"""
        async with self.engine.connect() as conn:
            return [dict(row) for row in (await conn.execute(text(
                "SELECT j.state,count(*) AS n FROM rag_jobs j "
                "WHERE j.id IN (SELECT DISTINCT x->>'job_id' FROM rag_sync_runs r, "
                "LATERAL jsonb_array_elements(r.jobs) x WHERE r.id=:id) "
                "GROUP BY j.state"), dict(id=ident))).mappings()]

    async def mark_source_ok(self, source):
        async with self.engine.begin() as conn:
            await conn.execute(text(
                "UPDATE rag_sources SET checked_at=now(),error=NULL WHERE kb=:kb AND id=:id"),
                dict(kb=self.config.knowledge_base_id, id=source))

    async def management_lock(self, conn, key):
        """在当前事务内串行化同一管理操作键。"""
        await conn.execute(text('SELECT pg_advisory_xact_lock(hashtextextended(:key,0))'), dict(key=key))

    async def record_management(self, conn, principal, input_id, action, target, result):
        """在业务事务中保存操作回执。"""
        await conn.execute(text(
            "INSERT INTO rag_management_operations(kb,principal,input_id,action,target,result) "
            "VALUES(:kb,:p,:i,:a,:t,CAST(:r AS jsonb))"),
            dict(kb=self.config.knowledge_base_id, p=principal, i=input_id,
                 a=action, t=target, r=packed(result)))

    async def management_receipt(self, principal, input_id, conn=None):
        """按知识库、主体和输入 ID 查询管理回执。"""
        if conn is None:
            async with self.engine.connect() as connection:
                return await self.management_receipt(principal, input_id, conn=connection)
        return (await conn.execute(text(
            "SELECT action,target,result FROM rag_management_operations "
            "WHERE kb=:kb AND principal=:p AND input_id=:i"),
            dict(kb=self.config.knowledge_base_id, p=principal, i=input_id))).mappings().first()

    async def management_receipts(self, ident=None):
        async with self.engine.connect() as conn:
            return [dict(row) for row in (await conn.execute(text(
                "SELECT input_id,action,target,result FROM rag_management_operations "
                "WHERE kb=:kb AND (CAST(:id AS text) IS NULL OR input_id=:id) "
                "ORDER BY created_at DESC LIMIT 100"),
                dict(kb=self.config.knowledge_base_id, id=ident))).mappings()]

    async def members(self, source=None, *, include_disabled=False):
        """读取来源成员，默认只包含启用来源。"""
        async with self.engine.connect() as conn:
            return [
                dict(r)
                for r in (
                    await conn.execute(
                        text("""SELECT m.* FROM rag_source_documents m
                JOIN rag_sources s ON s.kb=m.kb AND s.id=m.source_id WHERE m.kb=:kb AND (s.enabled OR :all_sources)
                AND (CAST(:source AS text) IS NULL OR m.source_id=:source)"""),
                        dict(
                            kb=self.config.knowledge_base_id,
                            source=source,
                            all_sources=include_disabled,
                        ),
                    )
                ).mappings()
            ]

    async def observe(
        self, source, uri, descriptor, state, error=None, *, sync_run=None
    ):
        """记录来源成员的观察状态；同步任务必须仍持有有效租约。"""
        async with self.engine.begin() as conn:
            if sync_run:
                from ai_assistant.knowledge import LeaseLost

                valid = (
                    await conn.execute(
                        text("""SELECT id FROM rag_sync_runs WHERE id=:id AND token=:token
                    AND state='running' AND lease_until>clock_timestamp() AND deadline>clock_timestamp() FOR UPDATE"""),
                        dict(id=sync_run["id"], token=sync_run["token"]),
                    )
                ).scalar()
                if valid is None:
                    raise LeaseLost("过期同步任务不能改写来源状态。")
            await conn.execute(
                text("""INSERT INTO rag_source_documents(kb,source_id,uri,descriptor,state,checked_at,error)
                VALUES(:kb,:s,:u,CAST(:d AS jsonb),:state,now(),:error)
                ON CONFLICT(kb,source_id,uri) DO UPDATE SET descriptor=EXCLUDED.descriptor,
                state=EXCLUDED.state,checked_at=now(),error=EXCLUDED.error"""),
                dict(
                    kb=self.config.knowledge_base_id,
                    s=source,
                    u=uri,
                    d=packed(descriptor),
                    state=state,
                    error=error,
                ),
            )

    async def request_sync(self, filters=None, *, conn=None):
        """按知识库和筛选范围合并正在进行的同步请求。"""
        filters = filters or {}
        scope = hashlib.sha256(packed(filters).encode()).hexdigest()
        kb = self.config.knowledge_base_id
        async with transaction(self.engine, conn) as conn:
            # 事务锁与唯一索引共同限制同范围并发提交。
            await conn.execute(
                text("SELECT pg_advisory_xact_lock(hashtextextended(:k,0))"),
                dict(k=kb + scope),
            )
            ident = (
                await conn.execute(
                    text("""SELECT id FROM rag_sync_runs WHERE kb=:kb AND scope=:scope
                AND state IN ('queued','running','indexing')"""),
                    dict(kb=kb, scope=scope),
                )
            ).scalar()
            if ident is None:
                ident = str(uuid4())
                await conn.execute(
                    text("""INSERT INTO rag_sync_runs(id,kb,scope,filters,deadline)
                    VALUES(:id,:kb,:scope,CAST(:f AS jsonb),now()+make_interval(secs=>:seconds))"""),
                    dict(
                        id=ident,
                        kb=kb,
                        scope=scope,
                        f=packed(filters),
                        seconds=self.config.sync_deadline_seconds,
                    ),
                )
            return ident

    async def sync_status(self, ident):
        """仅查询当前知识库中的同步任务。"""
        async with self.engine.connect() as conn:
            row = (
                (
                    await conn.execute(
                        text("SELECT * FROM rag_sync_runs WHERE kb=:kb AND id=:id"),
                        dict(kb=self.config.knowledge_base_id, id=ident),
                    )
                )
                .mappings()
                .one_or_none()
            )
            return dict(row) if row else None

    async def claim_sync(self):
        """领取排队或租约过期的同步任务，并签发新的执行 token。"""
        async with self.engine.begin() as conn:
            row = (
                (
                    await conn.execute(
                        text("""SELECT * FROM rag_sync_runs WHERE kb=:kb
                AND (state='queued' OR (state='running' AND lease_until<now())) AND deadline>now()
                ORDER BY created_at FOR UPDATE SKIP LOCKED LIMIT 1"""),
                        dict(kb=self.config.knowledge_base_id),
                    )
                )
                .mappings()
                .one_or_none()
            )
            if row is None:
                return None
            token = str(uuid4())
            await conn.execute(
                text(
                    "UPDATE rag_sync_runs SET state='running',token=:t,lease_until=now()+interval '60 seconds' WHERE id=:id"
                ),
                dict(t=token, id=row["id"]),
            )
            return dict(row) | {"token": token}

    async def sync_heartbeat(self, run):
        """仅允许仍持有有效租约的任务续期。"""
        from ai_assistant.knowledge import LeaseLost

        async with self.engine.begin() as conn:
            changed = await conn.execute(
                text("""UPDATE rag_sync_runs SET lease_until=now()+interval '60 seconds'
                WHERE id=:id AND token=:t AND state='running' AND lease_until>now() AND deadline>now()"""),
                dict(id=run["id"], t=run["token"]),
            )
            if changed.rowcount != 1:
                raise LeaseLost("同步任务执行权已过期。")

    async def record_scan_errors(self, run, errors):
        """在租约有效时保存扫描错误快照。"""
        async with self.engine.begin() as conn:
            await conn.execute(text("""UPDATE rag_sync_runs SET errors=CAST(:errors AS jsonb)
                WHERE id=:id AND token=:token AND state='running'
                AND lease_until>now() AND deadline>now()"""),
                dict(id=run['id'], token=run['token'], errors=packed(errors)))

    async def settle_scan(self, run, jobs, errors):
        """扫描结束后保存关联索引任务并进入 indexing 状态。"""
        async with self.engine.begin() as conn:
            await conn.execute(
                text("""UPDATE rag_sync_runs SET state='indexing',jobs=CASE WHEN jobs='[]'::jsonb THEN CAST(:jobs AS jsonb) ELSE jobs END,
                errors=CAST(:errors AS jsonb),lease_until=NULL WHERE id=:id AND token=:token
                AND state='running' AND lease_until>now() AND deadline>now()"""),
                dict(
                    id=run["id"],
                    token=run["token"],
                    jobs=packed(jobs),
                    errors=packed(errors),
                ),
            )

    async def finish_syncs(self):
        """汇总索引结果或截止时间，将同步任务推进到 ready。"""
        async with self.engine.begin() as conn:
            rows = (
                (
                    await conn.execute(
                        text("""SELECT * FROM rag_sync_runs WHERE kb=:kb
                AND (state='indexing' OR (state IN ('queued','running') AND deadline<=now()))
                FOR UPDATE SKIP LOCKED"""),
                        dict(kb=self.config.knowledge_base_id),
                    )
                )
                .mappings()
                .all()
            )
            for row in rows:
                # 分别累计仍在处理的任务、可用构建和覆盖错误。
                builds, errors, pending = [], list(row["errors"]), False
                for item in row["jobs"]:
                    job = (
                        (
                            await conn.execute(
                                text("""SELECT j.state,d.active_build,d.desired_job,d.deleted
                        FROM rag_jobs j JOIN rag_versions v ON v.id=j.version_id
                        JOIN rag_documents d ON d.id=v.document_id WHERE j.id=:id"""),
                                dict(id=item["job_id"]),
                            )
                        )
                        .mappings()
                        .one_or_none()
                    )
                    if job and job["state"] in ("queued", "running"):
                        pending = True
                    elif (
                        job
                        and job["state"] == "succeeded"
                        and not job["deleted"]
                        and job["desired_job"] == item["job_id"]
                    ):
                        builds.append(job["active_build"])
                    else:
                        errors.append(
                            {
                                "source": item["source_id"],
                                "reason": "index_not_available",
                            }
                        )
                expired = (
                    await conn.execute(
                        text("SELECT deadline<=now() FROM rag_sync_runs WHERE id=:id"),
                        dict(id=row["id"]),
                    )
                ).scalar()
                if pending and not expired:
                    continue
                if expired:
                    errors.append({"source": "sync", "reason": "deadline_exceeded"})

                # ready 表示汇总结束，不代表每个文档都已可检索。
                await conn.execute(
                    text("""UPDATE rag_sync_runs SET state='ready',builds=CAST(:b AS jsonb),
                    errors=CAST(:e AS jsonb),finished_at=now(),lease_until=NULL WHERE id=:id"""),
                    dict(
                        id=row["id"],
                        b=packed(list(dict.fromkeys(builds))),
                        e=packed(errors),
                    ),
                )

    async def document_list(self):
        """列出文档、期望版本、索引状态和所属来源。"""
        async with self.engine.connect() as conn:
            return [
                dict(r)
                for r in (
                    await conn.execute(
                        text("""SELECT d.id,d.source_uri,d.deleted,d.active_build,j.state AS index_state,
                v.metadata, ARRAY(SELECT m.source_id FROM rag_source_documents m
                    WHERE m.kb=d.kb AND m.uri=d.source_uri ORDER BY m.source_id) AS source_ids
                FROM rag_documents d LEFT JOIN rag_versions v ON v.id=d.desired_version
                LEFT JOIN rag_jobs j ON j.id=d.desired_job
                WHERE d.kb=:kb ORDER BY d.source_uri"""),
                        dict(kb=self.config.knowledge_base_id),
                    )
                ).mappings()
            ]

    async def eligible_builds(self, builds, filters=None):
        """对冻结构建复核删除、来源范围和已知撤权，不切换到其他构建。"""
        from ai_assistant.knowledge.metadata import validate_filters

        filters = validate_filters(filters)

        # 冻结构建仍需复核文档删除、来源状态和用户筛选范围。
        clauses = [
            "d.kb=:kb",
            "NOT d.deleted",
            "b.id=ANY(CAST(:builds AS text[]))",
            """(NOT EXISTS(SELECT 1 FROM rag_source_documents m WHERE m.kb=d.kb AND m.uri=d.source_uri)
            OR EXISTS(SELECT 1 FROM rag_source_documents m JOIN rag_sources s ON s.kb=m.kb AND s.id=m.source_id
                WHERE m.kb=d.kb AND m.uri=d.source_uri AND s.enabled AND m.state='available'))""",
        ]
        clauses.append("""NOT EXISTS(SELECT 1 FROM rag_source_documents m JOIN rag_sources s
            ON s.kb=m.kb AND s.id=m.source_id WHERE m.kb=d.kb AND m.uri=d.source_uri
            AND s.enabled AND m.state IN ('unavailable','failed'))""")
        params = {"kb": self.config.knowledge_base_id, "builds": builds}
        for key, value in filters.items():
            params[key] = packed(value) if key == "tags" else value
            if key == "document_id":
                clauses.append("d.id=:document_id")
            elif key == "source_id":
                clauses.append("""EXISTS(SELECT 1 FROM rag_source_documents scope_member
                    JOIN rag_sources scope_source ON scope_source.kb=scope_member.kb
                    AND scope_source.id=scope_member.source_id
                    WHERE scope_member.kb=d.kb AND scope_member.uri=d.source_uri
                    AND scope_member.source_id=:source_id AND scope_source.enabled
                    AND scope_member.state='available')""")
            elif key == "tags":
                clauses.append("v.metadata->'tags' @> CAST(:tags AS jsonb)")
            else:
                clauses.append(f"v.metadata->>'{key}'=:{key}")
        async with self.engine.connect() as conn:
            return list(
                (
                    await conn.execute(
                        text(
                            """SELECT b.id FROM rag_builds b JOIN rag_versions v ON v.id=b.version_id
                JOIN rag_documents d ON d.id=v.document_id WHERE """
                            + " AND ".join(clauses)
                        ),
                        params,
                    )
                ).scalars()
            )

    async def cached_revision(self, uri, revision):
        """查找指定远端版本关联的既有快照摘要和路径。"""
        async with self.engine.connect() as conn:
            row = (
                (
                    await conn.execute(
                        text("""SELECT v.digest,v.snapshot FROM rag_documents d JOIN rag_versions v
                ON v.id=d.desired_version WHERE d.kb=:kb AND d.source_uri=:uri
                AND v.metadata->>'remote_revision'=:revision"""),
                        dict(
                            kb=self.config.knowledge_base_id, uri=uri, revision=revision
                        ),
                    )
                )
                .mappings()
                .one_or_none()
            )
            return dict(row) if row else None
