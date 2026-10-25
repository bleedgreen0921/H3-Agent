"""Durable source observations and query-scoped publication sets."""

import hashlib
import json
from uuid import uuid4
from sqlalchemy import text

DDL = (
    """CREATE TABLE IF NOT EXISTS rag_sources (
        kb text NOT NULL,id text NOT NULL,config jsonb NOT NULL,enabled boolean NOT NULL DEFAULT true,
        checked_at timestamptz,error text,PRIMARY KEY(kb,id))""",
    """CREATE TABLE IF NOT EXISTS rag_source_documents (
        kb text NOT NULL,source_id text NOT NULL,uri text NOT NULL,descriptor jsonb NOT NULL,
        state text NOT NULL DEFAULT 'pending',checked_at timestamptz,error text,
        PRIMARY KEY(kb,source_id,uri))""",
    """CREATE TABLE IF NOT EXISTS rag_sync_runs (
        id text PRIMARY KEY,kb text NOT NULL,scope text NOT NULL,state text NOT NULL DEFAULT 'queued',
        request_key text,filters jsonb NOT NULL,builds jsonb NOT NULL DEFAULT '[]',
        errors jsonb NOT NULL DEFAULT '[]',jobs jsonb NOT NULL DEFAULT '[]',
        token text,lease_until timestamptz,created_at timestamptz NOT NULL DEFAULT now(),
        deadline timestamptz NOT NULL,finished_at timestamptz)""",
    """CREATE UNIQUE INDEX IF NOT EXISTS rag_sync_live ON rag_sync_runs(kb,scope)
        WHERE state IN ('queued','running','indexing')""",
    """CREATE TABLE IF NOT EXISTS rag_query_runs (
        kb text NOT NULL,request_key text NOT NULL,scope text NOT NULL,run_id text NOT NULL,
        PRIMARY KEY(kb,request_key,scope))""",
)


def packed(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, default=str)


class SourceStore:
    async def configure_sources(self, sources):
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

    async def add_submission(self, uri, descriptor):
        source = "submitted:" + hashlib.sha256(uri.encode()).hexdigest()[:24]
        kb = self.config.knowledge_base_id
        async with self.engine.begin() as conn:
            await conn.execute(
                text("""INSERT INTO rag_sources(kb,id,config)
                VALUES(:kb,:id,CAST(:c AS jsonb)) ON CONFLICT(kb,id) DO UPDATE SET enabled=true"""),
                dict(kb=kb, id=source, c=packed({"kind": "submitted"})),
            )
            await conn.execute(
                text("""INSERT INTO rag_source_documents(kb,source_id,uri,descriptor)
                VALUES(:kb,:id,:uri,CAST(:d AS jsonb)) ON CONFLICT(kb,source_id,uri)
                DO UPDATE SET descriptor=EXCLUDED.descriptor"""),
                dict(kb=kb, id=source, uri=uri, d=packed(descriptor)),
            )
            await conn.execute(
                text(
                    "UPDATE rag_documents SET deleted=false WHERE kb=:kb AND source_uri=:uri"
                ),
                dict(kb=kb, uri=uri),
            )
        return source

    async def source_list(self):
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

    async def members(self, source=None, *, include_disabled=False):
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

    async def request_sync(self, filters=None, request_key=None):
        filters = filters or {}
        scope = hashlib.sha256(packed(filters).encode()).hexdigest()
        kb = self.config.knowledge_base_id
        async with self.engine.begin() as conn:
            await conn.execute(
                text("SELECT pg_advisory_xact_lock(hashtextextended(:k,0))"),
                dict(k=kb + scope),
            )
            if request_key:
                existing = (
                    await conn.execute(
                        text("""SELECT run_id FROM rag_query_runs
                    WHERE kb=:kb AND request_key=:key AND scope=:scope"""),
                        dict(kb=kb, key=request_key, scope=scope),
                    )
                ).scalar()
                if existing:
                    return existing
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
            if request_key:
                await conn.execute(
                    text("INSERT INTO rag_query_runs VALUES(:kb,:key,:scope,:id)"),
                    dict(kb=kb, key=request_key, scope=scope, id=ident),
                )
            return ident

    async def sync_status(self, ident):
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
        from ai_assistant.knowledge import LeaseLost

        async with self.engine.begin() as conn:
            changed = await conn.execute(
                text("""UPDATE rag_sync_runs SET lease_until=now()+interval '60 seconds'
                WHERE id=:id AND token=:t AND state='running' AND lease_until>now() AND deadline>now()"""),
                dict(id=run["id"], t=run["token"]),
            )
            if changed.rowcount != 1:
                raise LeaseLost("同步任务执行权已过期。")

    async def settle_scan(self, run, jobs, errors):
        async with self.engine.begin() as conn:
            await conn.execute(
                text("""UPDATE rag_sync_runs SET state='indexing',jobs=CAST(:jobs AS jsonb),
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
        async with self.engine.connect() as conn:
            return [
                dict(r)
                for r in (
                    await conn.execute(
                        text("""SELECT d.id,d.source_uri,d.deleted,d.active_build,
                v.metadata FROM rag_documents d LEFT JOIN rag_versions v ON v.id=d.desired_version
                WHERE d.kb=:kb ORDER BY d.source_uri"""),
                        dict(kb=self.config.knowledge_base_id),
                    )
                ).mappings()
            ]

    async def eligible_builds(self, builds, filters=None):
        """对冻结构建复核删除、来源范围和已知撤权，不切换到其他构建。"""
        from ai_assistant.knowledge.metadata import validate_filters

        filters = validate_filters(filters)
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
