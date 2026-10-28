"""一次性清理旧业务数据：固定表清单、无 CASCADE、保留凭据及原始资料。"""
import argparse
import json
from pathlib import Path
import re

from ai_assistant.config import ConfigError, load_config

SESSION_TABLES = (
    'checkpoint_writes', 'checkpoint_blobs', 'checkpoints',
    'assistant_outbox', 'assistant_inbox', 'assistant_chat_commands',
    'assistant_operations', 'assistant_channel_sessions', 'assistant_model_usage',
    'assistant_sessions', 'assistant_memory_jobs', 'assistant_memory_policy',
)
RAG_TABLES = (
    'rag_query_runs', 'rag_sync_runs', 'rag_source_documents', 'rag_sources',
    'rag_chunks', 'rag_builds', 'rag_jobs', 'rag_versions', 'rag_documents',
)


def data_files(config):
    """只枚举约定命名的生成产物；不递归删除配置目录。"""
    if config.knowledge is None:
        return []
    base = config.config_path.parent if config.config_path else Path.cwd()
    settings = config.knowledge
    workspace = (base / config.workspace.root).resolve()
    result = []
    for kind in ('materials_dir', 'diagnostics_dir'):
        root = (base / getattr(settings, kind)).absolute()
        if any(p.is_symlink() for p in (root, *root.parents)):
            raise ConfigError('生成产物目录不允许符号链接。')
        root = root.resolve()
        if root == workspace or root in workspace.parents or root == Path.home():
            raise ConfigError('生成产物目录不能是工作区、主目录或其父目录。')
        if not root.exists():
            continue
        for path in root.rglob('*'):
            if path.is_symlink():
                continue
            relative = path.relative_to(root).as_posix()
            matched = re.fullmatch(r'[0-9a-f]{64}(?:\.[a-z0-9]+)?', relative) if kind == 'materials_dir' else re.fullmatch(r'[0-9a-f]{64}/[0-9a-f-]{36}/[0-9a-f-]{36}\.log', relative)
            if matched and path.is_file() and not any(p.is_symlink() for p in path.parents):
                result.append(path)
    return sorted(set(result))


def database_targets(config):
    targets = {}
    if config.sessions:
        targets.setdefault(config.sessions.database_url.get_secret_value(), set()).update(SESSION_TABLES)
    if config.knowledge:
        url = config.knowledge.database_url.get_secret_value().replace('postgresql+asyncpg://', 'postgresql://')
        targets.setdefault(url, set()).update(RAG_TABLES)
    return targets


def reset(config, *, execute=False, services_stopped=False):
    import psycopg
    from psycopg import sql
    if execute and not services_stopped:
        raise ConfigError('执行清理前须停止接收器、执行器和知识库 worker，并传入 --services-stopped。')
    files = data_files(config)
    reports = []
    # 不输出连接字符串，避免凭据进入日志。
    for number, (url, tables) in enumerate(database_targets(config).items(), 1):
        with psycopg.connect(url, autocommit=True) as conn:
            with conn.transaction():
                conn.execute("SET LOCAL lock_timeout = '3s'")
                if execute:
                    if not conn.execute('SELECT pg_try_advisory_xact_lock(714160004)').fetchone()[0]:
                        raise ConfigError('飞书执行器仍在运行，拒绝清理。')
                existing = [name for name in sorted(tables) if conn.execute('SELECT to_regclass(%s)', (name,)).fetchone()[0]]
                checkpoint_tables = [name for name in existing if name in {'checkpoints', 'checkpoint_blobs', 'checkpoint_writes'}]
                existing = [name for name in existing if name not in checkpoint_tables]
                counts = {name: conn.execute(sql.SQL('SELECT count(*) FROM {}').format(sql.Identifier(name))).fetchone()[0] for name in existing}
                threads = []
                if 'assistant_sessions' in existing:
                    threads = [r[0] for r in conn.execute('SELECT DISTINCT unnest(thread_ids) FROM assistant_sessions').fetchall()]
                for name in checkpoint_tables:
                    counts[name + ':owned_threads'] = conn.execute(sql.SQL('SELECT count(*) FROM {} WHERE thread_id=ANY(%s)').format(sql.Identifier(name)), (threads,)).fetchone()[0]
                filtered = []
                if tables.intersection(SESSION_TABLES):
                    for name in ('store_vectors', 'store'):
                        if conn.execute('SELECT to_regclass(%s)', (name,)).fetchone()[0]:
                            counts[name + ':personal_memory'] = conn.execute(sql.SQL("SELECT count(*) FROM {} WHERE prefix=%s OR prefix LIKE %s").format(sql.Identifier(name)), ('personal_memory', 'personal\\_memory.%')).fetchone()[0]
                            filtered.append((name, 'prefix', ('personal_memory', 'personal\\_memory.%')))
                if tables.intersection(RAG_TABLES) and conn.execute("SELECT to_regclass('upsertion_record')").fetchone()[0]:
                    counts['upsertion_record:rag'] = conn.execute("SELECT count(*) FROM upsertion_record WHERE namespace LIKE 'rag:%'").fetchone()[0]
                    filtered.append(('upsertion_record', 'namespace', ('', 'rag:%')))
                reports.append({'database': number, 'rows': counts})
                if execute:
                    for name in checkpoint_tables:
                        conn.execute(sql.SQL('DELETE FROM {} WHERE thread_id=ANY(%s)').format(sql.Identifier(name)), (threads,))
                    for name, column, values in filtered:
                        conn.execute(sql.SQL('DELETE FROM {} WHERE {}=%s OR {} LIKE %s').format(sql.Identifier(name), sql.Identifier(column), sql.Identifier(column)), values)
                    if existing:
                        # 不使用 CASCADE：出现清单外引用时失败回滚，避免波及其他应用。
                        conn.execute(sql.SQL('TRUNCATE {}').format(sql.SQL(',').join(map(sql.Identifier, existing))))
                    if 'assistant_sessions' in existing and conn.execute("SELECT to_regclass('assistant_usage_schema')").fetchone()[0]:
                        conn.execute('UPDATE assistant_usage_schema SET enabled_at=clock_timestamp()')
    failures = []
    if execute:
        for path in files:
            try:
                # 数据库提交后才删文件；中断可重复扫描并重试。
                if path.is_symlink() or any(p.is_symlink() for p in path.parents):
                    raise OSError
                path.unlink(missing_ok=True)
            except OSError:
                failures.append(str(path))
    return {'executed': execute, 'databases': reports, 'generated_files': [str(p) for p in files], 'file_failures': failures}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default='assistant.toml')
    parser.add_argument('--execute', action='store_true')
    parser.add_argument('--services-stopped', action='store_true')
    args = parser.parse_args(argv)
    try:
        result = reset(load_config(args.config), execute=args.execute, services_stopped=args.services_stopped)
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 2 if result['file_failures'] else 0
    except ConfigError as exc:
        print(f'清理未全部完成：{exc}')
        return 2
    except Exception:
        print('清理未全部完成：请检查服务是否已停止、专用目录及数据库权限；未自动重试。')
        return 2


if __name__ == '__main__':
    raise SystemExit(main())
