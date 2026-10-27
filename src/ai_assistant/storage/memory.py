"""Native Store records and transactional, revision-checked personal memory."""
import asyncio
import hashlib
import json
import logging
import re
from contextlib import AsyncExitStack, asynccontextmanager
from datetime import datetime, timezone
from uuid import uuid4

from langchain_core.messages import HumanMessage
from langgraph.store.base import GetOp, PutOp, SearchOp
from pydantic import ValidationError

from ai_assistant.agents.context import ContextBudgetError, count_tokens
from ai_assistant.config import ConfigError
from ai_assistant.storage.memory_types import (
    FORMAT, AttributeEntry, ProjectEntry, Evidence, MemoryChange, MemoryRecord, MemoryResult, REASONS, applicable,
)

DDL = '''
CREATE TABLE IF NOT EXISTS assistant_memory_policy (
 owner text PRIMARY KEY, version bigint NOT NULL DEFAULT 1,
 paused boolean NOT NULL DEFAULT false, watermark timestamptz NOT NULL DEFAULT clock_timestamp(),
 format integer NOT NULL DEFAULT 3
);
CREATE TABLE IF NOT EXISTS assistant_memory_jobs (
 id text PRIMARY KEY, owner text NOT NULL, candidate jsonb NOT NULL,
 status text NOT NULL DEFAULT 'pending', created_at timestamptz NOT NULL DEFAULT now()
);
'''
SENSITIVE = re.compile(r'(?i)(password|secret|api[_ -]?key|bearer\s|sk-[a-z0-9]|密码|密钥|验证码|访问令牌)')
PROHIBITED = re.compile(r'不要.{0,12}(记|存)|别.{0,8}(记|存)|不许.{0,8}(记|存)|do not (?:save|remember)|don.t remember', re.I)


def namespace(config):
    settings = config.integrations.feishu
    return ('personal_memory', settings.app_id, settings.owner_open_id)


def valid_candidate(candidate, messages, policy, *, explicit=False):
    """证据与当前授权分开校验；自然陈述无需“记住”关键词。"""
    if not isinstance(candidate, dict):
        return None
    protocol = {k: v for k, v in candidate.items()
                if k not in {'version', 'source_timestamp', 'explicit', 'format', 'session_id', 'verified_context', 'naming_evidence', 'naming_candidates'}}
    try:
        change = MemoryChange.model_validate(protocol)
        source = next(m for m in messages if isinstance(m, HumanMessage) and m.id == change.source_id)
        stamp = source.additional_kwargs['user_timestamp']
        timestamp = Evidence(source_id=source.id, quote=change.quote, timestamp=stamp).timestamp
        raw = source.additional_kwargs['user_source']
    except (ValidationError, StopIteration, KeyError, TypeError, ValueError):
        return None
    verified_context = []
    naming_sources = [(source.id, raw, stamp)]
    for reference in change.context_evidence:
        context = next((m for m in messages if isinstance(m, HumanMessage) and m.id == reference.source_id), None)
        if context is None or reference.quote not in context.additional_kwargs.get('user_source', ''):
            return None
        try:
            evidence = Evidence(source_id=context.id, quote=reference.quote, timestamp=context.additional_kwargs['user_timestamp'])
        except (ValidationError, KeyError):
            return None
        if SENSITIVE.search(reference.quote):
            return None
        verified_context.append(evidence.model_dump(mode='json'))
        naming_sources.append((context.id, context.additional_kwargs['user_source'], context.additional_kwargs['user_timestamp']))
    if not isinstance(raw, str) or change.quote not in raw:
        return None
    if re.search(r'临时脚本|一次性脚本|今天.{0,12}写.{0,8}脚本', raw):
        return None
    if re.search(r'别人(?:的|说)|他人(?:的|说)|假设|假如|示例|以下引用', raw) and change.quote != raw:
        return None
    if SENSITIVE.search(change.topic + ''.join(change.values) + change.quote + (change.project_name or '')) or PROHIBITED.search(raw):
        return None
    # 即使显式调用也不能重放旧消息授权；水位在删除、清空、暂停时推进。
    if timestamp <= policy['watermark'] or (policy['paused'] and not explicit):
        return None
    if explicit:
        latest = next((m for m in reversed(messages) if isinstance(m, HumanMessage)
                       and 'user_source' in m.additional_kwargs), None)
        if latest is None or latest.id != source.id:
            return None
    return change.model_dump(mode='json') | {
        'format': FORMAT, 'version': policy['version'], 'source_timestamp': stamp, 'explicit': explicit,
        'verified_context': verified_context,
        'naming_candidates': [name for name in project_mentions(raw) if not SENSITIVE.search(name)],
        'naming_evidence': [Evidence(source_id=ident, quote=change.project_name, timestamp=stamp).model_dump(mode='json')
                            for ident, text, stamp in naming_sources
                            if change.project_name and full_project_name(change.project_name, text)]}


def project_mentions(text):
    mentions = re.findall(r'[\w\u4e00-\u9fff-]+项目', text)
    names = [re.sub(r'^(?:请记住|请记录|请把|关于|讨论|我在|我们在|以后讨论)', '', name) for name in mentions]
    return [name for name in names if name not in {'项目', '我的项目', '这个项目', '那个项目', '该项目', '本项目', '新项目'}]


def full_project_name(name, quote):
    if not name or name not in quote:
        return False
    # 名称证据使用完整消息，不能通过模型截短 quote 绕过正式名称检查。
    mentions = project_mentions(quote)
    return name in mentions or not any(name in mention and name != mention for mention in mentions)


def effective_version(value, history, now):
    """已提交的未来替换生效前继续读取旧版；到期不自动复活旧值。"""
    if value.get('valid_from') and datetime.fromisoformat(value['valid_from']) > now:
        versions = [v for v in history if v['id'] == value['id'] and
                    (v.get('valid_from') is None or datetime.fromisoformat(v['valid_from']) <= now)]
        return max(versions, key=lambda v: v['revision']) if versions else None
    return value


def match_score(query, text):
    if not query.strip():
        return 1
    lowered = text.casefold()
    terms = set(re.findall(r'[a-z0-9_+#.-]+', query.casefold()))
    for run in re.findall(r'[\u4e00-\u9fff]+', query):
        terms.update(run[i:i + 2] for i in range(len(run) - 1))
        if len(run) == 1:
            terms.add(run)
    return sum(len(term) for term in terms if term in lowered)


def result(status, reason, record=None):
    return MemoryResult(status=status, reason=reason, summary=REASONS.get(reason, '记忆操作未完成'), memory_id=record.get('id') if record else None,
                        revision=record.get('revision') if record else None).model_dump_json()


class MemoryRepository:
    def __init__(self, conn, store, config):
        self.conn, self.store, self.config = conn, store, config
        self.ns = namespace(config)
        self.owner = hashlib.sha256(json.dumps(self.ns).encode()).hexdigest()
        self.reference = ''
        self.policy = None
        self.lock = asyncio.Lock()
        self.session_id = None
        self.project_id = None
        self.search_source = None
        self.query = ''

    def area(self, name):
        # 目录、历史、回执不能成为普通正文检索的子命名空间。
        return ('memory_' + name, *self.ns[1:])

    async def get(self, ns, key):
        return (await self.store.abatch([GetOp(ns, key)]))[0]

    async def put(self, ns, key, value, *, index=False):
        await self.store.abatch([PutOp(ns, key, value, index=index)])

    async def scan(self, ns, *, query=None, offset=0, limit=100, filters=None):
        return (await self.store.abatch([SearchOp(ns, filter=filters, query=query, offset=offset, limit=limit)]))[0]

    async def all(self, ns):
        items, offset = [], 0
        while True:
            page = await self.scan(ns, offset=offset)
            items.extend(page)
            if len(page) < 100:
                return items
            offset += len(page)

    async def get_policy(self, *, lock=False):
        suffix = ' FOR UPDATE' if lock else ''
        return await (await self.conn.execute('SELECT * FROM assistant_memory_policy WHERE owner=%s' + suffix,
                                             (self.owner,))).fetchone()

    async def initialize(self, *, reset_existing=False):
        await self.conn.execute(DDL)
        # 旧格式只能显式清理，不能自动迁移正文。
        await self.conn.execute('ALTER TABLE assistant_memory_policy ADD COLUMN IF NOT EXISTS format integer NOT NULL DEFAULT 1')
        await self.conn.execute('INSERT INTO assistant_memory_policy(owner,format) VALUES(%s,%s) ON CONFLICT DO NOTHING', (self.owner, FORMAT))
        if reset_existing:
            self.initialization_receipt = await self.manage('clear')
            return self.initialization_receipt
        policy = await self.get_policy()
        if policy['format'] != FORMAT:
            raise ConfigError('记忆格式需要显式初始化：memory init --reset-existing。')
        self.initialization_receipt = result('unchanged', 'initialized')
        return self.initialization_receipt

    async def search(self, query='', *, scope=None, status=None, offset=0, source_id=None,
                     project_id=None, attribute_id=None, include_global=True, effective_at=None,
                     projects_offset=0, attributes_offset=0, deleted_offset=0):
        if min(offset, projects_offset, attributes_offset, deleted_offset) < 0:
            return result('rejected', 'invalid_offset')
        if scope not in {None, 'global', 'project', 'session'} or status not in {None, 'active', 'inactive'}:
            return result('rejected', 'invalid_filter')
        async with self.lock:
            projects = await self.all(self.area('projects'))
            attributes = await self.all(self.area('attributes'))
            project_map = {i.key: i.value for i in projects}
            attribute_map = {i.key: i.value for i in attributes}
            if project_id and project_id not in project_map:
                return result('needs_clarification', 'project_identity')
            if attribute_id and attribute_id not in attribute_map:
                return result('needs_clarification', 'attribute_identity')
            filters = dict(scope=scope, status=status, project_id=project_id, attribute_id=attribute_id,
                           include_global=include_global, effective_at=effective_at.isoformat() if effective_at else None)

            def in_scope(value):
                return ((scope is None or value['scope'] == scope)
                        and (status is None or value['status'] == status)
                        and (attribute_id is None or value['attribute_id'] == attribute_id)
                        and (project_id is None or value['project_id'] == project_id
                             or (include_global and value['project_id'] is None))
                        and (value['scope'] != 'session' or value['session_id'] == self.session_id)
                        and (effective_at is None or applicable(value, session_id=self.session_id,
                            project_id=project_id or self.project_id, now=effective_at)))

            def score(value):
                project = project_map.get(value.get('project_id'), {})
                attribute = attribute_map.get(value.get('attribute_id'), {})
                text = ' '.join([value.get('content', ''), value.get('topic', ''),
                                 project.get('name', ''), *project.get('aliases', []),
                                 attribute.get('name', ''), *attribute.get('aliases', [])])
                return match_score(query, text)

            matches = []
            history = [i.value for i in await self.all(self.area('history'))] if effective_at else []
            for item in await self.all(self.ns):
                value = effective_version(item.value, history, effective_at) if effective_at else item.value
                if value and value.get('format') == FORMAT and in_scope(value):
                    ranking = score(value)
                    if ranking:
                        matches.append((ranking, item.key, value))
            matches.sort(key=lambda v: (-v[0], v[1]))
            deleted = []
            for barrier in await self.all(self.area('barriers')):
                identity = barrier.value.get('identity')
                if not identity:
                    continue
                aid, pid, record_scope, sid, kind = identity
                marker = dict(attribute_id=aid, project_id=pid, scope=record_scope, session_id=sid,
                              status='inactive', kind=kind)
                # 删除标记不伪装成有效事实；仅使用目录身份匹配，不恢复正文。
                if ((scope is None or scope == record_scope) and (attribute_id is None or aid == attribute_id)
                        and (project_id is None or pid == project_id or (include_global and pid is None))
                        and (record_scope != 'session' or sid == self.session_id) and score(marker)):
                    deleted.append({'memory_id': barrier.key, 'status': 'deleted',
                                    'project_id': pid, 'attribute_id': aid,
                                    'deleted_at': barrier.value['deleted_at']})
            deleted.sort(key=lambda v: v['memory_id'])

            def directory_candidates(items, ident):
                ranked = [(match_score(query, ' '.join([i.value['name'], *i.value['aliases']])), i.key, i.value)
                          for i in items if ident is None or ident == i.key]
                return [v for ranking, _, v in sorted(ranked, key=lambda v: (-v[0], v[1])) if ranking]

            project_values = directory_candidates(projects, project_id)
            attribute_values = directory_candidates(attributes, attribute_id)
            page_status = 'found' if matches else ('deleted' if deleted else 'not_found')
            if source_id:
                self.search_source = source_id
            return json.dumps({'status': page_status, 'summary': REASONS[page_status] + f'；有效候选 {len(matches)} 条，已删除标记 {len(deleted)} 条',
                'items': [v for _, _, v in matches[offset:offset + 20]],
                'deleted': deleted[deleted_offset:deleted_offset + 20],
                'projects': project_values[projects_offset:projects_offset + 20],
                'attributes': attribute_values[attributes_offset:attributes_offset + 20],
                'query': query, 'filters': filters, 'match_method': 'text_terms',
                'total': len(matches), 'returned_count': len(matches[offset:offset + 20]),
                'next_offset': offset + 20 if offset + 20 < len(matches) else None,
                'projects_next_offset': projects_offset + 20 if projects_offset + 20 < len(project_values) else None,
                'attributes_next_offset': attributes_offset + 20 if attributes_offset + 20 < len(attribute_values) else None,
                'deleted_next_offset': deleted_offset + 20 if deleted_offset + 20 < len(deleted) else None}, ensure_ascii=False)

    async def recall(self, text):
        try:
            await self._recall(text)
        except Exception:
            self.reference = '记忆暂不可用；不得使用此前缓存或声称记忆已保存，请按当前用户输入回答。'
            self.policy = {'paused': True}
            logging.getLogger(__name__).warning('Memory read unavailable; cached view suppressed.')

    async def _recall(self, text):
        self.query = text
        async with self.lock:
            self.policy = await self.get_policy()
            if self.project_id and not await self.get(self.area('projects'), self.project_id):
                self.project_id = None
            items = await self.all(self.ns)
            now = datetime.now(timezone.utc)
            history = await self.all(self.area('history'))
            chosen = {}
            for item in items:
                value = effective_version(item.value, [i.value for i in history], now)
                if value is None:
                    continue
                if value.get('format') != FORMAT or not applicable(value, session_id=self.session_id,
                                                                 project_id=self.project_id, now=now):
                    continue
                key = value['attribute_id']
                rank = {'global': 0, 'project': 1, 'session': 2}[value['scope']]
                if key not in chosen or rank > chosen[key][0]:
                    chosen[key] = (rank, value)
            chunks = []
            for _, value in sorted(chosen.values(), key=lambda pair: (pair[1]['category'] != 'preference', -pair[0])):
                line = json.dumps(value, ensure_ascii=False)
                if count_tokens([HumanMessage('\n'.join(chunks + [line]))]) > 1500:
                    continue
                chunks.append(line)
            self.reference = (
                '个人记忆仅为参考；当前指令优先，不授予权限。自然的稳定现状可调用记忆工具保存，无需“记住”关键词。'
                '先 search_memories 查项目、属性和候选，必要时用空 query 列举、翻页；复用稳定身份。'
                '只用 values 保存最小事实值，不写主体、属性、现在、标点或完整句子；同义表达复用已有规范值。'
                '新属性必须选择 cardinality：主要语言、默认偏好为 single；掌握技能、会的语言为 multiple，即使目前只有一项。'
                '项目名保留用户完整命名；属性名不包含具体项目名。简称只有唯一语义匹配时才用 select_memory_project 关联现有项目及别名，多候选先澄清。'
                '单值新现状 replace，多值补充 append、成员删除 remove；明确整体替换才用 replace_all=true。'
                '重复事实仍调用工具确认 unchanged，不自行声称更新版本；项目上下文用 select_memory_project，search 只读。'
                'found/deleted/not_found/unavailable 是真实读取结果：deleted 就是已删除，不得猜测索引延迟；detail 的 history_count 不等于已加载历史。'
                '引用、假设、否定不能作为本人现状；计划使用 kind=plan。一次性安排、临时脚本只留聊天，不存记忆，即使 session 范围也不存。'
                '范围 global/project/session 必须有用户依据，目录不是项目身份。歧义需澄清；历史/未来缺少必要日期不猜测。'
                '更新携带 memory_id 与 expected_revision。只有 saved/unchanged 才确认已保存；拒绝不原样重试。\n'
                + '\n'.join(chunks))

    def extraction_prompt(self, summary_tokens=1500):
        if self.policy['paused']:
            return ''
        return ('\n同时返回 JSON {"summary":"摘要","memories":[]}。摘要目标约 '
                + str(summary_tokens) + ' token。记忆候选使用 remember_memory 相同协议，引用真实 source_id/quote；'
                '仅稳定自然陈述，不提取引用、假设、临时执行、凭据或禁止保存内容。'
                '没有明确属性身份或无法确认新旧关系时 memories 留空，不猜测。最多6条。')

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
        return summary.strip(), [c | {'session_id': getattr(self, 'session_id', None)} for c in candidates if c]

    async def register(self, candidates, conn=None):
        from psycopg.types.json import Jsonb
        target = conn or self.conn
        for candidate in candidates:
            if candidate.get('format') != FORMAT:
                continue
            ident = hashlib.sha256(json.dumps([self.owner, candidate], sort_keys=True).encode()).hexdigest()
            await target.execute('''INSERT INTO assistant_memory_jobs(id,owner,candidate)
                SELECT %s,%s,%s FROM assistant_memory_policy WHERE owner=%s AND version=%s
                AND format=3 AND NOT paused AND watermark<%s::timestamptz ON CONFLICT DO NOTHING''',
                (ident, self.owner, Jsonb(candidate), self.owner, candidate['version'], candidate['source_timestamp']))

    async def directory(self, kind, name, ident=None, *, cardinality=None, evidence=None):
        items = await self.all(self.area(kind))
        if ident:
            return next((i.value for i in items if i.key == ident), None)
        matches = [i.value for i in items if name.casefold() in
                   [v.casefold() for v in [i.value['name'], *i.value['aliases']]]]
        if len(matches) == 1:
            return matches[0]
        if matches:
            return None
        if kind == 'projects':
            if not evidence or not any(full_project_name(name, v['quote']) for v in evidence):
                return None
            source = next(v for v in evidence if full_project_name(name, v['quote']))
            # 目录证据只保留名称片段，不能复制事实正文而使删除后仍可读回旧值。
            naming = Evidence(source_id=source['source_id'], quote=name, timestamp=source['timestamp'])
            value = ProjectEntry(name=name, evidence=[naming]).model_dump(mode='json')
        else:
            if cardinality is None:
                return None
            value = AttributeEntry(name=name, cardinality=cardinality).model_dump(mode='json')
        await self.put(self.area(kind), value['id'], value)
        return value

    async def select_project(self, project_id, *, alias=None, expected_revision=None, source=None):
        async with self.lock, self.conn.transaction():
            policy = await self.get_policy(lock=True)
            if project_id is None:
                self.project_id = None
                return {'status': 'unchanged', 'reason': 'selected', 'summary': '已切换为全局上下文', 'project_id': None}
            item = await self.get(self.area('projects'), project_id)
            if not item:
                return json.loads(result('needs_clarification', 'project_identity'))
            value = item.value
            reason = 'selected'
            if alias and alias not in [value['name'], *value['aliases']]:
                if expected_revision != value['revision']:
                    return json.loads(result('conflict', 'revision_mismatch'))
                try:
                    evidence = Evidence(source_id=source.id, quote=alias,
                                        timestamp=source.additional_kwargs['user_timestamp'])
                    raw = source.additional_kwargs['user_source']
                except (AttributeError, KeyError, ValidationError):
                    return json.loads(result('rejected', 'invalid_current_evidence'))
                if alias not in raw:
                    return json.loads(result('rejected', 'invalid_current_evidence'))
                if not full_project_name(alias, raw):
                    payload = json.loads(result('needs_clarification', 'project_name_evidence'))
                    payload['project_name_candidates'] = project_mentions(raw)
                    return payload
                if (len(alias) > 100 or not alias.strip() or SENSITIVE.search(alias)
                        or PROHIBITED.search(raw) or evidence.timestamp <= policy['watermark']):
                    return json.loads(result('rejected', 'invalid_current_evidence'))
                for other in await self.all(self.area('projects')):
                    if other.key != project_id and alias.casefold() in [v.casefold() for v in [other.value['name'], *other.value['aliases']]]:
                        return json.loads(result('needs_clarification', 'project_identity'))
                value = {**value, 'aliases': sorted(set([*value['aliases'], alias])),
                         'revision': value['revision'] + 1,
                         'evidence': [*value['evidence'], evidence.model_dump(mode='json')]}
                await self.put(self.area('projects'), project_id, value)
                reason = 'alias_registered'
            self.project_id = project_id
            return {'status': 'saved' if reason == 'alias_registered' else 'unchanged', 'reason': reason,
                    'summary': REASONS[reason], 'project_id': project_id, 'project_revision': value['revision']}

    async def save(self, candidate, *, policy):
        if candidate.get('format') != FORMAT or candidate['version'] != policy['version']:
            return result('rejected', 'stale_epoch')
        if datetime.fromisoformat(candidate['source_timestamp']) <= policy['watermark']:
            return result('rejected', 'stale_evidence')
        if policy['paused'] and not candidate.get('explicit'):
            return result('rejected', 'paused')
        data = {k: v for k, v in candidate.items() if k not in
                {'format', 'version', 'source_timestamp', 'explicit', 'session_id', 'verified_context', 'naming_evidence', 'naming_candidates'}}
        change = MemoryChange.model_validate(data)
        digest = hashlib.sha256(json.dumps([self.owner, candidate], sort_keys=True).encode()).hexdigest()
        receipt = await self.get(self.area('receipts'), digest)
        if receipt:
            return MemoryResult.model_validate(receipt.value).model_dump_json()
        old = await self.get(self.ns, change.memory_id) if change.memory_id else None
        if change.action != 'add':
            if not old or old.value['revision'] != change.expected_revision:
                return result('conflict', 'revision_mismatch', old.value if old else None)
            if await self.get(self.area('barriers'), change.memory_id):
                return result('rejected', 'deleted')
        if change.project_name and not change.project_id:
            if not candidate.get('naming_evidence'):
                payload = json.loads(result('needs_clarification', 'project_name_evidence'))
                payload['project_name_candidates'] = candidate.get('naming_candidates', [])
                return json.dumps(payload, ensure_ascii=False)
        project = None
        if change.project_id or change.project_name:
            project = await self.directory('projects', change.project_name or '', change.project_id,
                                           evidence=candidate.get('naming_evidence'))
            if project is None:
                return result('needs_clarification', 'project_identity')
        attribute = await self.directory('attributes', change.topic, change.attribute_id, cardinality=change.cardinality)
        if attribute is None:
            return result('needs_clarification', 'attribute_identity')
        if attribute['cardinality'] != change.cardinality:
            return result('rejected', 'cardinality_mismatch')
        project_id = project['id'] if project else None
        session_id = candidate.get('session_id') if change.scope == 'session' else None
        if change.scope == 'session' and not session_id:
            return result('rejected', 'missing_session')
        identity = (attribute['id'], project_id, change.scope, session_id, change.kind)
        for barrier in await self.all(self.area('barriers')):
            if barrier.value.get('identity') == list(identity) and datetime.fromisoformat(candidate['source_timestamp']) <= datetime.fromisoformat(barrier.value['deleted_at']):
                return result('rejected', 'deleted_evidence')
        if old:
            previous = old.value
            if identity != (previous['attribute_id'], previous['project_id'], previous['scope'], previous['session_id'], previous['kind']):
                return result('rejected', 'identity_mismatch', previous)
        else:
            for item in await self.all(self.ns):
                value = item.value
                if identity == (value['attribute_id'], value['project_id'], value['scope'], value['session_id'], value['kind']):
                    # 明确已结束的历史事实单独保留，不替换现状。
                    if change.valid_until and change.valid_until <= datetime.now(timezone.utc):
                        continue
                    if (change.values == value['values'] and change.kind == value['kind']
                            and change.valid_from == (datetime.fromisoformat(value['valid_from']) if value.get('valid_from') else None)
                            and change.valid_until == (datetime.fromisoformat(value['valid_until']) if value.get('valid_until') else None)
                            and value['status'] == 'active'):
                        response = result('unchanged', 'same_values', value)
                        await self.put(self.area('receipts'), digest, json.loads(response))
                        return response
                    return result('needs_clarification', 'existing_attribute_requires_target', value)
        values = list(old.value['values']) if old else []
        if change.action == 'append':
            values = sorted(set(values) | set(change.values))
        elif change.action == 'remove':
            values = sorted(set(values) - set(change.values))
        else:
            values = change.values
        subject = project['name'] if project else '本人'
        content = f"{subject}的{attribute['name']}：{'、'.join(values)}。"
        now = datetime.now(timezone.utc)
        record = MemoryRecord(id=old.key if old else str(uuid4()),
            revision=old.value['revision'] + 1 if old else 1, epoch=policy['version'],
            category=old.value['category'] if old else change.category, topic=attribute['name'], content=content, values=values, cardinality=attribute['cardinality'],
            attribute_id=attribute['id'], project_id=project_id, scope=change.scope, session_id=session_id,
            kind=change.kind, status='inactive' if change.action == 'invalidate' or (attribute['cardinality'] == 'multiple' and not values) else 'active',
            evidence=[Evidence(source_id=change.source_id, quote=change.quote, timestamp=candidate['source_timestamp']),
                      *(Evidence.model_validate(v) for v in candidate.get('verified_context', []))],
            observed_at=now, valid_from=change.valid_from, valid_until=change.valid_until).model_dump(mode='json')
        if old and all(record[k] == old.value[k] for k in
                       ('values', 'kind', 'status', 'valid_from', 'valid_until')):
            response = result('unchanged', 'same_values', old.value)
        else:
            if old:
                await self.put(self.area('history'), old.key + ':' + str(old.value['revision']), old.value)
            await self.put(self.ns, record['id'], record, index=['content'])
            response = result('saved', 'committed', record)
        await self.put(self.area('receipts'), digest, json.loads(response))
        return response

    async def drain(self):
        async with self.lock, self.conn.transaction():
            policy = await self.get_policy(lock=True)
            rows = await (await self.conn.execute("SELECT * FROM assistant_memory_jobs WHERE owner=%s AND status='pending' ORDER BY created_at", (self.owner,))).fetchall()
            for row in rows:
                # 一个候选一个 savepoint，故障不能提交半条历史。
                async with self.conn.transaction():
                    response = json.loads(await self.save(row['candidate'], policy=policy))
                    await self.conn.execute("UPDATE assistant_memory_jobs SET status=%s,candidate='{}'::jsonb WHERE id=%s",
                                            (response['status'], row['id']))

    async def remember(self, candidate, messages):
        async with self.lock, self.conn.transaction():
            policy = await self.get_policy(lock=True)
            try:
                MemoryChange.model_validate(candidate)
            except ValidationError:
                return result('rejected', 'invalid_change')
            validated = valid_candidate(candidate, messages, policy, explicit=True)
            if validated is None:
                return result('rejected', 'invalid_current_evidence')
            if self.search_source != validated['source_id']:
                return result('rejected', 'search_required')
            validated['session_id'] = self.session_id
            response = await self.save(validated, policy=policy)
        await self.recall(self.query)
        return response

    async def manage(self, action='list', ident=None, offset=0):
        if offset < 0:
            return result('rejected', 'invalid_offset')
        if action == 'list':
            return await self.search(offset=offset)
        if action in {'detail', 'history'}:
            async with self.lock:
                item = await self.get(self.ns, ident) if ident else None
                barrier = await self.get(self.area('barriers'), ident) if ident else None
                status = 'found' if item else ('deleted' if barrier else 'not_found')
                history = sorted([i.value for i in await self.all(self.area('history'))
                                  if i.value['id'] == ident], key=lambda v: v['revision'])
                response = {'status': status, 'summary': REASONS[status], 'memory_id': ident,
                            'current': item.value if item else None, 'history_count': len(history),
                            'history_loaded': action == 'history'}
                if action == 'history':
                    response.update(history=history[offset:offset + 20],
                                    next_offset=offset + 20 if offset + 20 < len(history) else None)
                if barrier:
                    response['deleted_at'] = barrier.value['deleted_at']
                return json.dumps(response, ensure_ascii=False)
        if action not in {'delete', 'clear', 'pause', 'resume'} or (action == 'delete' and not ident):
            return result('rejected', 'invalid_action')
        counts = {}
        async with self.lock, self.conn.transaction():
            await self.get_policy(lock=True)
            if action == 'delete':
                current = await self.get(self.ns, ident)
                if current is None:
                    barrier = await self.get(self.area('barriers'), ident)
                    return result('unchanged', 'deleted' if barrier else 'not_found', {'id': ident})
            if action == 'clear':
                await self.conn.execute('UPDATE assistant_memory_policy SET version=version+1,format=3,watermark=clock_timestamp() WHERE owner=%s', (self.owner,))
            elif action != 'delete':
                await self.conn.execute('''UPDATE assistant_memory_policy SET watermark=clock_timestamp(),
                    paused=CASE WHEN %s='pause' THEN true WHEN %s='resume' THEN false ELSE paused END WHERE owner=%s''', (action, action, self.owner))
            deleted = await self.get(self.ns, ident) if action == 'delete' else None
            if action != 'delete':
                cursor = await self.conn.execute("UPDATE assistant_memory_jobs SET status='discarded',candidate='{}'::jsonb WHERE owner=%s AND candidate<>'{}'::jsonb", (self.owner,))
                counts['candidate_payloads'] = cursor.rowcount
            else:
                rows = await (await self.conn.execute("SELECT id,candidate FROM assistant_memory_jobs WHERE owner=%s AND candidate<>'{}'::jsonb", (self.owner,))).fetchall()
                for row in rows:
                    c = row['candidate']
                    same_attribute = deleted and (c.get('attribute_id') == deleted.value['attribute_id'] or c.get('topic') == deleted.value['topic'])
                    same_scope = deleted and c.get('scope', 'global') == deleted.value['scope'] and c.get('project_id') == deleted.value['project_id']
                    if c.get('memory_id') == ident or (same_attribute and same_scope):
                        await self.conn.execute("UPDATE assistant_memory_jobs SET status='discarded',candidate='{}'::jsonb WHERE id=%s", (row['id'],))
                        counts['candidate_payloads'] = counts.get('candidate_payloads', 0) + 1
            if action in {'delete', 'clear'}:
                for ns in [self.ns, self.area('history'), self.area('receipts')]:
                    for item in await self.all(ns):
                        if action == 'clear' or item.key == ident or item.value.get('id') == ident or item.value.get('memory_id') == ident:
                            await self.put(ns, item.key, None)
                            counts[ns[0]] = counts.get(ns[0], 0) + 1
                if action == 'delete' and deleted:
                    await self.put(self.area('barriers'), ident, {'deleted': True,
                        'identity': [deleted.value[k] for k in ('attribute_id', 'project_id', 'scope', 'session_id', 'kind')] if deleted else None,
                        'deleted_at': datetime.now(timezone.utc).isoformat()})
                elif action == 'clear':
                    for name in ('projects', 'attributes', 'barriers'):
                        for item in await self.all(self.area(name)):
                            await self.put(self.area(name), item.key, None)
                            counts[self.area(name)[0]] = counts.get(self.area(name)[0], 0) + 1
        self.search_source = None
        await self.recall(self.query)
        return json.dumps({'status': 'saved', 'reason': action, 'summary': REASONS[action], 'removed': counts, 'epoch': (await self.get_policy())['version']}, ensure_ascii=False)


@asynccontextmanager
async def memory_storage(config, *, initialize=False, reset_existing=False):
    if not config.memory.enabled:
        raise ConfigError('请显式启用 [memory].enabled。')
    from langgraph.store.postgres import AsyncPostgresStore
    from ai_assistant.storage.integrations import operation_storage
    from ai_assistant.knowledge.models import embedding
    async with AsyncExitStack() as stack:
        operations = await stack.enter_async_context(operation_storage(config.sessions))
        embed = await stack.enter_async_context(embedding(config.knowledge))
        # Store 与策略使用同一连接；只通过公开 abatch 操作，避免后台队列越过事务。
        native = AsyncPostgresStore(operations.conn, index={'dims': 1024, 'embed': embed, 'fields': ['content']})
        repo = MemoryRepository(operations.conn, native, config)
        if initialize:
            await native.setup()
            await repo.initialize(reset_existing=reset_existing)
        try:
            policy = await repo.get_policy()
        except Exception:
            raise ConfigError('记忆存储未初始化，请运行 memory init --reset-existing。') from None
        if policy is None or policy.get('format') != FORMAT:
            raise ConfigError('记忆格式需要初始化，请运行 memory init --reset-existing。')
        yield repo
