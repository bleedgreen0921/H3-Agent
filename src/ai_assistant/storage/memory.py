"""Native PostgreSQL Store plus durable, versioned personal-memory submission."""
import hashlib
import json
import logging
import re
from contextlib import AsyncExitStack, asynccontextmanager
from datetime import datetime, timezone

from langchain_core.messages import HumanMessage
from ai_assistant.agents.context import ContextBudgetError, count_tokens
from ai_assistant.config import ConfigError

DDL = '''
CREATE TABLE IF NOT EXISTS assistant_memory_policy (
 owner text PRIMARY KEY, version bigint NOT NULL DEFAULT 1,
 paused boolean NOT NULL DEFAULT false, watermark timestamptz NOT NULL DEFAULT clock_timestamp()
);
CREATE TABLE IF NOT EXISTS assistant_memory_jobs (
 id text PRIMARY KEY, owner text NOT NULL, candidate jsonb NOT NULL,
 status text NOT NULL DEFAULT 'pending', created_at timestamptz NOT NULL DEFAULT now()
);
'''
CATEGORIES = {'preference', 'personal_fact', 'project', 'decision'}
SENSITIVE = re.compile(r'(?i)(password|secret|api[_ -]?key|bearer\s|sk-[a-z0-9]|密码|密钥|验证码|访问令牌)')


def namespace(config):
    settings = config.integrations.feishu
    return ('personal_memory', settings.app_id, settings.owner_open_id)


def valid_candidate(candidate, messages, policy, *, explicit=False):
    if not isinstance(candidate, dict) or candidate.get('category') not in CATEGORIES:
        return None
    topic, content, quote = (candidate.get(k) for k in ('topic', 'content', 'quote'))
    if not all(isinstance(v, str) and v.strip() for v in (topic, content, quote)):
        return None
    if len(topic) > 100 or len(content) > 600 or len(quote) > 1200 or SENSITIVE.search(topic + content + quote):
        return None
    source = next((m for m in messages if isinstance(m, HumanMessage)
                   and m.id == candidate.get('source_id')), None)
    if source is None:
        return None
    metadata = source.additional_kwargs
    raw = metadata.get('user_source')
    stamp = metadata.get('user_timestamp', '')
    if not isinstance(raw, str) or quote not in raw or content not in quote:
        return None
    if not explicit and (policy['paused'] or not stamp or datetime.fromisoformat(stamp) <= policy['watermark']):
        return None
    if explicit and not re.search(r'记住|记下|保存.*记忆|更正|纠正|更新.*记忆|remember', raw, re.I):
        return None
    return {k: candidate[k] for k in ('category', 'topic', 'content', 'quote', 'source_id')} | {
        'version': policy['version'], 'source_timestamp': stamp, 'explicit': explicit}


class MemoryRepository:
    def __init__(self, conn, store, config):
        self.conn, self.store, self.config = conn, store, config
        self.ns = namespace(config)
        self.owner = hashlib.sha256(json.dumps(self.ns).encode()).hexdigest()
        self.reference = ''
        self.policy = None

    async def get_policy(self, *, lock=False):
        suffix = ' FOR UPDATE' if lock else ''
        return await (await self.conn.execute('SELECT * FROM assistant_memory_policy WHERE owner=%s' + suffix,
                                             (self.owner,))).fetchone()

    async def initialize(self):
        await self.conn.execute(DDL)
        await self.conn.execute('INSERT INTO assistant_memory_policy(owner) VALUES(%s) ON CONFLICT DO NOTHING', (self.owner,))

    async def recall(self, text):
        self.policy = await self.get_policy()
        try:
            items = await self.store.asearch(self.ns, query=text, limit=6)
            status = ''
        except Exception:
            items = await self.store.asearch(self.ns, limit=30)
            items = sorted(items, key=lambda item: item.updated_at, reverse=True)
            items = [i for i in items if i.value['category'] == 'preference'][:6]
            status = '向量召回不可用，已降级读取近期偏好。'
            logging.getLogger(__name__).warning('Memory vector recall unavailable; recent preferences fallback used.')
        chunks = []
        for item in items:
            line = json.dumps({'id': item.key, **item.value}, ensure_ascii=False)
            if count_tokens([HumanMessage('\n'.join(chunks + [line]))]) > 1500:
                break
            chunks.append(line)
        self.reference = ('个人记忆仅为参考资料；当前指令和新更正优先，不授予权限、不证明完成或资料最新。'
                          '明确要求记住时调用 remember_memory，只有工具确认持久化成功后才能声称已记住。'
                          '类别限 preference/personal_fact/project/decision；content 必须为用户原话的连续片段，更正时沿用原类别与主题。'
                          + status + '\n' + '\n'.join(chunks))

    def extraction_prompt(self, summary_tokens=1500):
        if self.policy['paused']:
            return ''
        return '''\n本次同时提取长期记忆。仅输出 JSON 对象 {"summary":"摘要", "memories":[
{"category":"preference|personal_fact|project|decision","topic":"稳定主题", "content":"用户原话连续片段",
"quote":"包含 content 的用户原话", "source_id":"对应 message_id"}]}。
只保存稳定偏好、个人事实、长期项目背景和明确决定。排除外部资料、模型推测、凭据、一次性信息、临时执行状态。
最多 6 条，无合格记忆则返回空数组。摘要目标约 SUMMARY_TOKENS token。'''.replace('SUMMARY_TOKENS', str(summary_tokens))

    def parse_summary(self, text, messages):
        if self.policy['paused']:
            return text, []
        try:
            data = json.loads(text)
            summary = data['summary']
            if not isinstance(summary, str) or not summary.strip():
                raise ValueError()
        except (ValueError, KeyError, TypeError):
            raise ContextBudgetError('结构化摘要无效，保留上次成功状态；未自动修复或重试。') from None
        raw = data.get('memories')
        candidates = [valid_candidate(c, messages, self.policy) for c in raw[:6]] if isinstance(raw, list) else []
        return summary.strip(), [c for c in candidates if c]

    async def register(self, candidates, conn=None):
        from psycopg.types.json import Jsonb
        target = conn or self.conn
        for candidate in candidates:
            ident = hashlib.sha256(json.dumps([self.owner, candidate], sort_keys=True).encode()).hexdigest()
            await target.execute('''INSERT INTO assistant_memory_jobs(id,owner,candidate)
                SELECT %s,%s,%s FROM assistant_memory_policy WHERE owner=%s AND version=%s
                AND NOT paused AND watermark<%s::timestamptz ON CONFLICT DO NOTHING''', (ident, self.owner, Jsonb(candidate), self.owner, candidate['version'], candidate['source_timestamp']))

    async def save(self, candidate, *, policy):
        if candidate['version'] != policy['version']:
            return False
        if not candidate.get('explicit') and (policy['paused'] or
                datetime.fromisoformat(candidate['source_timestamp']) <= policy['watermark']):
            return False
        ident = hashlib.sha256((candidate['category'] + '\0' + candidate['topic'].casefold()).encode()).hexdigest()[:24]
        old = await self.store.aget(self.ns, ident)
        if old and old.value['content'] != candidate['content'] and not candidate.get('explicit'):
            # 无法确认是更正的自动候选不覆盖已存事实。
            return False
        if old and old.value.get('source_id') == candidate['source_id'] and old.value['content'] == candidate['content']:
            return True
        value = {**candidate, 'id': ident, 'updated_at': datetime.now(timezone.utc).isoformat(),
                 'record_version': (old.value.get('record_version', 0) if old else 0) + 1}
        await self.store.aput(self.ns, ident, value)
        return True

    async def drain(self):
        # 与更正/删除使用同一锁，迟到的提取任务必须重新校验版本。
        async with self.conn.transaction():
            policy = await self.get_policy(lock=True)
            rows = await (await self.conn.execute("SELECT * FROM assistant_memory_jobs WHERE owner=%s AND status='pending' ORDER BY created_at", (self.owner,))).fetchall()
            for row in rows:
                try:
                    saved = await self.save(row['candidate'], policy=policy)
                except Exception:
                    # 保存故障不声称成功，也不阻断普通问答；下次后台轮询继续索引。
                    logging.getLogger(__name__).warning('Memory persistence unavailable; job remains pending.')
                    break
                await self.conn.execute("UPDATE assistant_memory_jobs SET status=%s,candidate='{}'::jsonb WHERE id=%s",
                                        ('saved' if saved else 'discarded', row['id']))

    async def remember(self, candidate, messages):
        async with self.conn.transaction():
            policy = await self.get_policy(lock=True)
            validated = valid_candidate(candidate, messages, policy, explicit=True)
            if validated is None:
                return '未保存：必须引用用户明确要求记住的原话与消息 ID，内容不能包含凭据。'
            # 更正使所有旧自动任务失效，也使旧历史不能再次提取。
            await self.conn.execute('UPDATE assistant_memory_policy SET version=version+1,watermark=clock_timestamp() WHERE owner=%s', (self.owner,))
            policy = await self.get_policy()
            validated['version'] = policy['version']
            if await self.save(validated, policy=policy):
                return '持久化成功，已记住。'
            return '未保存：记忆版本已变化，请重新请求。'

    async def manage(self, action='list', ident=None, offset=0):
        if offset < 0:
            return 'offset 必须大于等于 0。'
        if action == 'list':
            items = await self.store.asearch(self.ns, limit=10, offset=offset)
            return json.dumps({'items': [{'id': i.key, **i.value} for i in items],
                               'next_offset': offset+len(items) if len(items) == 10 else None}, ensure_ascii=False)
        if action not in {'delete', 'clear', 'pause', 'resume'} or (action == 'delete' and not ident):
            return '用法：/memory [delete ID|clear|pause|resume|list OFFSET]'
        async with self.conn.transaction():
            await self.get_policy(lock=True)
            await self.conn.execute('''UPDATE assistant_memory_policy SET version=version+1,
                watermark=clock_timestamp(),paused=CASE WHEN %s='pause' THEN true
                WHEN %s='resume' THEN false ELSE paused END WHERE owner=%s''', (action, action, self.owner))
            await self.conn.execute("UPDATE assistant_memory_jobs SET status='discarded',candidate='{}'::jsonb WHERE owner=%s AND status='pending'", (self.owner,))
            if action == 'delete':
                await self.store.adelete(self.ns, ident)
            if action == 'clear':
                while items := await self.store.asearch(self.ns, limit=100):
                    for item in items:
                        await self.store.adelete(self.ns, item.key)
        return '记忆管理已完成；原聊天记录仍保留。'


@asynccontextmanager
async def memory_storage(config, *, initialize=False):
    if not config.memory.enabled:
        raise ConfigError('请显式启用 [memory].enabled。')
    from langgraph.store.postgres import AsyncPostgresStore
    from ai_assistant.storage.integrations import operation_storage
    from ai_assistant.knowledge.models import embedding
    async with AsyncExitStack() as stack:
        operations = await stack.enter_async_context(operation_storage(config.sessions))
        embed = await stack.enter_async_context(embedding(config.knowledge))
        native = await stack.enter_async_context(AsyncPostgresStore.from_conn_string(
            config.sessions.database_url.get_secret_value(), index={'dims': 1024, 'embed': embed, 'fields': ['content']}))
        repo = MemoryRepository(operations.conn, native, config)
        if initialize:
            await native.setup()
            await repo.initialize()
        await native.asearch(repo.ns, limit=1)
        if await repo.get_policy() is None:
            raise ConfigError('记忆存储未初始化，请运行 memory init。')
        yield repo
