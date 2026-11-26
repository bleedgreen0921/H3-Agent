"""私人知识库管理业务：只读查询和同步状态展示。"""
from ai_assistant.knowledge import KnowledgeError


async def sources(config, store):
    """合并数据库来源和配置来源，返回按 ID 排序的展示结果。"""
    rows = {row['id']: row for row in await store.source_list()}
    # 配置来源尚未同步时也可展示，查询不登记来源。
    for name, row in rows.items():
        if name.startswith('submitted:'):
            # 旧单文件来源只保留兼容；不再创建新的 submitted: 来源。
            row['legacy'] = True
        else:
            row['enabled'] = name in config.knowledge.sources
    for name, definition in config.knowledge.sources.items():
        row = rows.setdefault(name, dict(id=name, checked_at=None, error=None))
        row.update(config=definition.model_dump(mode='json'), enabled=True)
    return [rows[name] for name in sorted(rows)]


async def status(store, ident):
    """先查询同步任务，再查询索引任务，并补充同步覆盖信息。"""
    row = await store.sync_status(ident)
    if row is not None:
        counts = await store.sync_status_counts(ident)
        progress = {x['state']: x['n'] for x in counts}
        from datetime import datetime, timezone
        elapsed = ((row.get('finished_at') or datetime.now(timezone.utc))-row['created_at']).total_seconds()
        return dict(row, kind='sync', progress=progress, total=sum(progress.values()),
                    elapsed_seconds=int(elapsed), scan_errors=sum(1 for e in row.get('errors', []) if e.get('reason') not in ('index_not_available', 'deadline_exceeded')), **sync_facts(row))
    row = await store.status(ident)
    if row is None:
        raise KnowledgeError('任务不存在。')
    return dict(row, kind='index')


def sync_facts(row):
    """ready 只表示处理结束；可检索构建与覆盖缺口分别展示。"""
    builds, errors = row.get('builds') or [], row.get('errors') or []
    finished = row['state'] == 'ready'
    coverage = 'pending' if not finished else 'partial' if builds and errors else 'failed' if errors else 'available' if builds else 'empty'
    descriptions = {'pending': '后台处理中', 'partial': '部分可检索，仍有覆盖缺口',
        'failed': '处理结束，全部失败，无已确认可检索构建', 'available': '处理结束，已有可检索构建',
        'empty': '处理结束，范围为空，无可检索构建'}
    return dict(coverage=coverage, processing_finished=finished, searchable_builds=len(builds),
                coverage_gaps=len(errors), summary=descriptions[coverage])


def progress_text(result):
    """将同步状态计数和耗时整理为用户可读文本。"""
    p = result.get('progress', {})
    label = '已发现' if result['state'] in ('queued', 'running') else '总文档数'
    message = (f"{label} {result.get('total', 0)}，已成功 {p.get('succeeded', 0)}，"
               f"处理中 {p.get('running', 0)}，失败 {p.get('failed', 0)}，待处理 {p.get('queued', 0)}")
    for state in sorted(set(p)-{'succeeded','running','failed','queued'}):
        message += f"，{state} {p[state]}"
    return message + f"\n扫描错误 {result.get('scan_errors', 0)}，已用时 {result.get('elapsed_seconds', 0)} 秒"
