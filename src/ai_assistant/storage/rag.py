"""PostgreSQL task leases and fenced publication, separate from vector writes."""
import json
from uuid import uuid4
from sqlalchemy import text
from sqlalchemy.ext.asyncio import create_async_engine
from ai_assistant.knowledge import IngestionError, LeaseLost

DDL = (
    'CREATE EXTENSION IF NOT EXISTS vector',
    '''CREATE TABLE IF NOT EXISTS rag_documents (
      id text PRIMARY KEY, kb text NOT NULL, source_uri text NOT NULL,
      desired_version text, desired_job text, active_build text, deleted boolean NOT NULL DEFAULT false,
      UNIQUE(kb, source_uri))''',
    '''CREATE TABLE IF NOT EXISTS rag_versions (
      id text PRIMARY KEY, document_id text NOT NULL REFERENCES rag_documents(id),
      digest text NOT NULL, config_hash text NOT NULL, snapshot text NOT NULL,
      metadata jsonb NOT NULL, UNIQUE(document_id,digest,config_hash,metadata))''',
    '''CREATE TABLE IF NOT EXISTS rag_jobs (
      id text PRIMARY KEY, version_id text NOT NULL REFERENCES rag_versions(id),
      state text NOT NULL DEFAULT 'queued', attempts integer NOT NULL DEFAULT 0,
      token text, lease_until timestamptz, deadline timestamptz NOT NULL,
      available_at timestamptz NOT NULL DEFAULT clock_timestamp(), error text,
      created_at timestamptz NOT NULL DEFAULT clock_timestamp())''',
    '''CREATE TABLE IF NOT EXISTS rag_builds (
      id text PRIMARY KEY, version_id text NOT NULL REFERENCES rag_versions(id),
      job_id text NOT NULL REFERENCES rag_jobs(id), token text NOT NULL,
      published boolean NOT NULL DEFAULT false, created_at timestamptz NOT NULL DEFAULT clock_timestamp())''',
    '''CREATE TABLE IF NOT EXISTS rag_chunks (
      langchain_id text PRIMARY KEY, content text NOT NULL, embedding vector(1024) NOT NULL,
      langchain_metadata jsonb NOT NULL, content_tsv tsvector NOT NULL DEFAULT ''::tsvector)''',
    'ALTER TABLE rag_builds ADD COLUMN IF NOT EXISTS attempt_number integer',
    'ALTER TABLE rag_builds ADD COLUMN IF NOT EXISTS error_code text',
    'ALTER TABLE rag_builds ADD COLUMN IF NOT EXISTS error_message text',
    'ALTER TABLE rag_builds ADD COLUMN IF NOT EXISTS diagnostic_path text',
    '''WITH ordered AS (SELECT id, row_number() OVER (PARTITION BY job_id ORDER BY created_at,id) AS n
         FROM rag_builds) UPDATE rag_builds b SET attempt_number=ordered.n FROM ordered
         WHERE b.id=ordered.id AND b.attempt_number IS NULL''',
    'CREATE INDEX IF NOT EXISTS rag_chunks_tsv ON rag_chunks USING gin(content_tsv)',
    'CREATE INDEX IF NOT EXISTS rag_chunks_metadata ON rag_chunks USING gin(langchain_metadata)',
    'CREATE INDEX IF NOT EXISTS rag_jobs_claim ON rag_jobs(state,available_at)',
)


class RagStore:
    def __init__(self, config):
        self.config = config
        self.engine = create_async_engine(config.database_url.get_secret_value(), hide_parameters=True)

    async def close(self):
        await self.engine.dispose()

    async def init(self):
        # This is the only schema-mutating entry point.
        async with self.engine.begin() as conn:
            await conn.execute(text('SELECT pg_advisory_xact_lock(741268501)'))
            for statement in DDL:
                await conn.execute(text(statement))
        from langchain_classic.indexes import SQLRecordManager
        manager = SQLRecordManager('rag:init', engine=self.engine, async_mode=True)
        await manager.acreate_schema()

    async def submit(self, *, source_uri, digest, snapshot, metadata, force=False):
        c = self.config
        async with self.engine.begin() as conn:
            doc = str(uuid4())
            await conn.execute(text('''INSERT INTO rag_documents(id,kb,source_uri) VALUES(:id,:kb,:uri)
                ON CONFLICT(kb,source_uri) DO NOTHING'''), dict(id=doc, kb=c.knowledge_base_id, uri=source_uri))
            row = (await conn.execute(text('''SELECT * FROM rag_documents WHERE kb=:kb AND source_uri=:uri FOR UPDATE'''),
                                      dict(kb=c.knowledge_base_id, uri=source_uri))).mappings().one()
            doc = row['id']
            version = (await conn.execute(text('''SELECT id FROM rag_versions WHERE document_id=:doc
                AND digest=:digest AND config_hash=:hash AND metadata=CAST(:metadata AS jsonb)'''),
                dict(doc=doc, digest=digest, hash=c.fingerprint(), metadata=json.dumps(metadata)))).scalar_one_or_none()
            if version is None:
                version = str(uuid4())
                await conn.execute(text('''INSERT INTO rag_versions VALUES(:id,:doc,:digest,:hash,:snapshot,CAST(:metadata AS jsonb))'''),
                    dict(id=version, doc=doc, digest=digest, hash=c.fingerprint(), snapshot=str(snapshot), metadata=json.dumps(metadata)))
            if not force and not row['deleted'] and row['desired_version'] == version:
                job = (await conn.execute(text('''SELECT id FROM rag_jobs WHERE version_id=:v
                    AND state IN ('queued','running','succeeded') ORDER BY created_at DESC LIMIT 1'''), dict(v=version))).scalar_one_or_none()
                if job:
                    return job
            job = str(uuid4())
            await conn.execute(text('''UPDATE rag_documents SET desired_version=:v,desired_job=:job,deleted=false WHERE id=:id'''), dict(v=version, job=job, id=doc))
            await conn.execute(text('''UPDATE rag_jobs SET state='superseded',lease_until=NULL
                WHERE version_id IN (SELECT id FROM rag_versions WHERE document_id=:doc)
                AND state IN ('queued','running')'''), dict(doc=doc))
            await conn.execute(text('''INSERT INTO rag_jobs(id,version_id,deadline)
                VALUES(:id,:v,clock_timestamp()+make_interval(secs=>:seconds))'''), dict(id=job, v=version, seconds=c.deadline_seconds))
            return job

    async def claim(self):
        c = self.config
        async with self.engine.begin() as conn:
            await conn.execute(text('''UPDATE rag_jobs SET state='failed',error='deadline_or_attempt_limit'
                WHERE (state='queued' OR (state='running' AND lease_until<clock_timestamp()))
                AND (deadline<=clock_timestamp() OR attempts>=:max) AND version_id IN
                (SELECT v.id FROM rag_versions v JOIN rag_documents d ON d.id=v.document_id WHERE d.kb=:kb)'''), dict(max=c.max_attempts, kb=c.knowledge_base_id))
            row = (await conn.execute(text('''SELECT j.id FROM rag_jobs j JOIN rag_versions v ON v.id=j.version_id
                JOIN rag_documents d ON d.id=v.document_id WHERE d.kb=:kb AND NOT d.deleted AND d.desired_version=v.id
                AND d.desired_job=j.id AND j.deadline>clock_timestamp() AND j.attempts<:max AND j.available_at<=clock_timestamp()
                AND (j.state='queued' OR (j.state='running' AND j.lease_until<clock_timestamp()))
                ORDER BY j.created_at FOR UPDATE OF j SKIP LOCKED LIMIT 1'''), dict(max=c.max_attempts, kb=c.knowledge_base_id))).first()
            if row is None:
                return None
            token, build = str(uuid4()), str(uuid4())
            job = (await conn.execute(text('''UPDATE rag_jobs SET state='running',attempts=attempts+1,
                token=:token,error=NULL,lease_until=clock_timestamp()+make_interval(secs=>:lease) WHERE id=:id RETURNING *'''),
                dict(token=token, lease=c.lease_seconds, id=row.id))).mappings().one()
            await conn.execute(text('''INSERT INTO rag_builds(id,version_id,job_id,token,attempt_number) VALUES(:b,:v,:j,:t,:attempt)'''),
                               dict(b=build, v=job['version_id'], j=job['id'], t=token, attempt=job['attempts']))
            version = (await conn.execute(text('''SELECT v.*,d.kb,d.source_uri FROM rag_versions v
                JOIN rag_documents d ON d.id=v.document_id WHERE v.id=:v'''), dict(v=job['version_id']))).mappings().one()
            return dict(job) | dict(version) | {'id': job['id'], 'build_id': build, 'version_id': job['version_id']}

    async def heartbeat(self, job):
        async with self.engine.begin() as conn:
            result = await conn.execute(text('''UPDATE rag_jobs SET lease_until=clock_timestamp()+make_interval(secs=>:lease)
                WHERE id=:id AND token=:token AND state='running' AND lease_until>clock_timestamp() AND deadline>clock_timestamp()'''),
                dict(id=job['id'], token=job['token'], lease=self.config.lease_seconds))
            if result.rowcount != 1:
                raise LeaseLost('入库任务已失去执行权。')

    async def publish(self, job):
        async with self.engine.begin() as conn:
            # Lock document before checking desired version; submit/delete use the same row lock.
            await conn.execute(text('SELECT id FROM rag_documents WHERE id=:d FOR UPDATE'), dict(d=job['document_id']))
            valid = (await conn.execute(text('''SELECT j.id FROM rag_jobs j JOIN rag_versions v ON v.id=j.version_id
                JOIN rag_documents d ON d.id=v.document_id
                JOIN rag_builds b ON b.job_id=j.id AND b.version_id=v.id AND b.token=j.token
                WHERE j.id=:id AND j.token=:token AND b.id=:build AND d.id=:doc AND d.kb=:kb
                AND j.state='running' AND j.lease_until>clock_timestamp() AND j.deadline>clock_timestamp()
                AND NOT d.deleted AND d.desired_version=v.id AND d.desired_job=j.id AND v.config_hash=:hash FOR UPDATE OF j'''),
                dict(id=job['id'], token=job['token'], hash=self.config.fingerprint(),
                     build=job['build_id'], doc=job['document_id'], kb=self.config.knowledge_base_id))).first()
            if not valid:
                raise LeaseLost('入库任务已过期或输入版本已变更。')
            present = (await conn.execute(text("""SELECT EXISTS(SELECT 1 FROM rag_chunks
                WHERE langchain_metadata->>'index_version'=:b
                AND langchain_metadata->>'document_id'=:doc
                AND langchain_metadata->>'document_version_id'=:v
                AND langchain_metadata->>'knowledge_base_id'=:kb
                AND content ~ '[^[:space:]]')"""),
                dict(b=job['build_id'], doc=job['document_id'], v=job['version_id'],
                     kb=self.config.knowledge_base_id))).scalar_one()
            if not present:
                raise IngestionError('empty_index', '当前构建没有有效片段，已停止发布。')
            await conn.execute(text('UPDATE rag_builds SET published=true WHERE id=:b AND token=:t'), dict(b=job['build_id'], t=job['token']))
            await conn.execute(text('UPDATE rag_documents SET active_build=:b WHERE id=:d'), dict(b=job['build_id'], d=job['document_id']))
            settled = await conn.execute(text("""UPDATE rag_jobs SET state='succeeded',lease_until=NULL,error=NULL
                WHERE id=:id AND token=:token AND state='running'
                AND lease_until>clock_timestamp() AND deadline>clock_timestamp()"""),
                dict(id=job['id'], token=job['token']))
            if settled.rowcount != 1:
                raise LeaseLost('入库任务在发布期间已过期。')

    async def fail(self, job, *, transient=False, error='ingestion_failed',
                   error_code=None, diagnostic_path=None):
        async with self.engine.begin() as conn:
            result = await conn.execute(text('''UPDATE rag_jobs SET
                state=CASE WHEN :retry AND attempts<:max AND deadline>clock_timestamp() THEN 'queued' ELSE 'failed' END,
                error=:error,available_at=clock_timestamp()+make_interval(secs=>:delay),lease_until=NULL
                WHERE id=:id AND token=:token AND state='running' AND lease_until>clock_timestamp()'''),
                dict(retry=transient, max=self.config.max_attempts, error=error, delay=self.config.retry_seconds,
                     id=job['id'], token=job['token']))
            if result.rowcount == 1:
                await conn.execute(text('''UPDATE rag_builds SET error_code=:code,error_message=:message,
                    diagnostic_path=:path WHERE job_id=:job AND token=:token'''),
                    dict(code=error_code, message=error, path=diagnostic_path,
                         job=job['id'], token=job['token']))

    async def status(self, job_id):
        async with self.engine.connect() as conn:
            row = (await conn.execute(text('''SELECT j.id,j.state,j.attempts,j.error,j.deadline,j.lease_until,
                v.document_id,v.id AS document_version_id,v.metadata->>'source_name' AS source_name,
                b.id AS latest_build_id,p.id AS published_build_id,
                d.active_build AS active_build_id,d.deleted AS document_deleted,
                b.error_code,b.diagnostic_path
                FROM rag_jobs j JOIN rag_versions v ON v.id=j.version_id JOIN rag_documents d ON d.id=v.document_id
                LEFT JOIN LATERAL (SELECT id,error_code,diagnostic_path FROM rag_builds
                    WHERE job_id=j.id ORDER BY attempt_number DESC NULLS LAST,created_at DESC,id DESC LIMIT 1) b ON true
                LEFT JOIN LATERAL (SELECT id FROM rag_builds WHERE job_id=j.id AND published
                    ORDER BY attempt_number DESC NULLS LAST,created_at DESC,id DESC LIMIT 1) p ON true
                WHERE j.id=:id AND d.kb=:kb'''), dict(id=job_id, kb=self.config.knowledge_base_id))).mappings().one_or_none()
            return dict(row) if row else None

    async def active(self, filters=None):
        from ai_assistant.knowledge.metadata import validate_filters
        filters = validate_filters(filters)
        clauses = ['d.kb=:kb', 'NOT d.deleted', 'd.active_build IS NOT NULL']
        params = dict(kb=self.config.knowledge_base_id)
        for key, value in filters.items():
            params[key] = json.dumps(value) if key == 'tags' else value
            if key == 'document_id':
                clauses.append('d.id=:document_id')
            elif key == 'tags':
                clauses.append("v.metadata->'tags' @> CAST(:tags AS jsonb)")
            else:
                # key is selected exclusively from the validated whitelist.
                clauses.append(f"v.metadata->>'{key}'=:{key}")
        async with self.engine.connect() as conn:
            return list((await conn.execute(text(
                'SELECT d.active_build FROM rag_documents d JOIN rag_builds b ON b.id=d.active_build '
                'JOIN rag_versions v ON v.id=b.version_id WHERE ' + ' AND '.join(clauses)), params)).scalars())

    async def delete(self, document_id):
        async with self.engine.begin() as conn:
            count = (await conn.execute(text("""UPDATE rag_documents SET deleted=true WHERE id=:id AND kb=:kb"""),
                dict(id=document_id, kb=self.config.knowledge_base_id))).rowcount
            if count:
                await conn.execute(text("""UPDATE rag_jobs SET state='cancelled',lease_until=NULL
                    WHERE version_id IN (SELECT id FROM rag_versions WHERE document_id=:id)
                    AND state IN ('queued','running')"""), dict(id=document_id))
            return count

    async def version_for_rebuild(self, document_id):
        async with self.engine.connect() as conn:
            row = (await conn.execute(text('''SELECT v.*,d.source_uri FROM rag_documents d JOIN rag_versions v
                ON v.id=d.desired_version WHERE d.id=:id AND d.kb=:kb'''),
                dict(id=document_id, kb=self.config.knowledge_base_id))).mappings().one_or_none()
            return dict(row) if row else None
