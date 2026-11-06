"""一次性重置会话与渠道历史；默认预览，不删除知识库或远端内容。"""
import argparse
import json
from pathlib import Path

from ai_assistant.config import ConfigError, load_config
from ai_assistant.configuration import profile_options, read_toml, resolve_config
from ai_assistant.storage.sessions import lock_key

# 精确列出应用和固定 Saver 拥有的表；禁止前缀匹配与 CASCADE。
TABLES = (
    'assistant_outbox', 'assistant_inbox', 'assistant_chat_commands',
    'assistant_channel_sessions', 'assistant_operations', 'assistant_model_usage',
    'checkpoint_writes', 'checkpoint_blobs', 'checkpoints', 'checkpoint_migrations',
    'assistant_sessions', 'assistant_session_schema',
)
LOCKS = (714160001, 714160002, 714160004, 714160005)
REQUIRED = {
    'assistant_outbox': {'id', 'input_id', 'body', 'status', 'part'},
    'assistant_inbox': {'id', 'binding', 'body', 'status'},
    'assistant_chat_commands': {'input_id', 'binding', 'action'},
    'assistant_channel_sessions': {'binding', 'session_id'},
    'assistant_operations': {'id', 'input_id', 'principal', 'tool', 'args_digest'},
    'assistant_model_usage': {'call_id', 'session_id', 'profile', 'role'},
    'assistant_sessions': {'id', 'workspace', 'thread_ids'},
    'assistant_session_schema': {'singleton', 'version'},
    'checkpoints': {'thread_id', 'checkpoint_ns', 'checkpoint_id', 'checkpoint'},
    'checkpoint_blobs': {'thread_id', 'checkpoint_ns', 'channel', 'version'},
    'checkpoint_writes': {'thread_id', 'checkpoint_ns', 'checkpoint_id', 'task_id'},
    'checkpoint_migrations': {'v'},
}


class ResetError(RuntimeError):
    """只携带维护程序生成的安全摘要。"""


def reset(config, *, execute=False):
    import psycopg
    from psycopg import sql
    from psycopg.rows import dict_row
    if config.sessions is None:
        raise ResetError('未配置会话存储；没有清理目标。')
    # 同一入口的其他方案不得将目标库绑定到其他工作区。
    target_url = config.sessions.database_url.get_secret_value()
    base = config.config_path.parent
    workspace = (base / Path(config.workspace.root).expanduser()).resolve()
    profiles, _ = profile_options(read_toml(config.config_path))
    from psycopg.conninfo import conninfo_to_dict

    def database_identity(url):
        values = conninfo_to_dict(url)
        host = values.get('host', '')
        return ('127.0.0.1' if host == 'localhost' else host,
                values.get('port', '5432'), values.get('dbname'))

    for name in profiles:
        try:
            # 未选方案可以尚未填模型 Key；只读取与删除归属相关的原始配置。
            other, _, _ = resolve_config(config.config_path, name)
            url = other.get('sessions', {}).get('database_url')
            if url and database_identity(url) == database_identity(target_url):
                root = other.get('workspace', {}).get('root', '.')
                if (base / Path(root).expanduser()).resolve() != workspace:
                    raise ResetError('其他方案共享目标库但工作区不同；未删除。')
        except (ConfigError, TypeError, ValueError):
            raise ResetError('存在无法检查的配置方案；无法确认数据库归属。') from None
    with psycopg.connect(target_url, row_factory=dict_row, connect_timeout=5) as conn:
        with conn.transaction():
            conn.execute("SET LOCAL lock_timeout='2s'")
            conn.execute("SET LOCAL statement_timeout='15s'")
            for key in LOCKS:
                if not conn.execute('SELECT pg_try_advisory_xact_lock(%s) AS locked', (key,)).fetchone()['locked']:
                    raise ResetError('服务或初始化仍在运行；请停止并等待清理后重试。')
            # 其他连接可能属于独立 Worker 或另一个使用 Saver 的部署，不能推测所有权。
            others = conn.execute("SELECT count(*) AS n FROM pg_stat_activity WHERE datname=current_database() AND pid<>pg_backend_pid() AND backend_type='client backend'").fetchone()['n']
            if others:
                raise ResetError('目标库仍有其他客户端连接；无法确认独占，未删除。')
            schema = conn.execute('SELECT current_schema() AS name').fetchone()['name']
            found = conn.execute("SELECT table_name,column_name FROM information_schema.columns WHERE table_schema=%s AND table_name=ANY(%s)", (schema, list(TABLES))).fetchall()
            columns = {}
            for row in found:
                columns.setdefault(row['table_name'], set()).add(row['column_name'])
            present = [name for name in TABLES if name in columns]
            for name in present:
                if not REQUIRED[name] <= columns[name]:
                    raise ResetError('目标表不符合已知项目结构；无法确认归属，未删除。')
            if present:
                # 所有权检查和删除共用锁与事务，阻止并发写入制造检查后的新历史。
                conn.execute(sql.SQL('LOCK TABLE {} IN ACCESS EXCLUSIVE MODE NOWAIT').format(
                    sql.SQL(',').join(sql.Identifier(schema, name) for name in present)))
            dependent = conn.execute("""SELECT 1 FROM pg_constraint c
                JOIN pg_class parent ON parent.oid=c.confrelid
                JOIN pg_namespace pn ON pn.oid=parent.relnamespace
                JOIN pg_class child ON child.oid=c.conrelid
                JOIN pg_namespace cn ON cn.oid=child.relnamespace
                WHERE c.contype='f' AND pn.nspname=%s AND parent.relname=ANY(%s)
                  AND NOT (cn.nspname=%s AND child.relname=ANY(%s)) LIMIT 1""",
                (schema, list(TABLES), schema, list(TABLES))).fetchone()
            if dependent:
                raise ResetError('白名单外表依赖会话数据；未扩大删除范围。')
            threads = set()
            if 'assistant_sessions' in present:
                for row in conn.execute('SELECT id,workspace,thread_ids FROM assistant_sessions'):
                    if Path(row['workspace']).resolve() != workspace:
                        raise ResetError('发现其他工作区会话；无法确认归属，未删除。')
                    if not conn.execute('SELECT pg_try_advisory_xact_lock(%s) AS locked', (lock_key(str(row['id'])),)).fetchone()['locked']:
                        raise ResetError('会话仍在使用；未删除。')
                    threads.update(row['thread_ids'])
            for name in ('checkpoints', 'checkpoint_blobs', 'checkpoint_writes'):
                if name in present:
                    rows = conn.execute(sql.SQL('SELECT DISTINCT thread_id FROM {}').format(sql.Identifier(schema, name)))
                    if any(row['thread_id'] not in threads for row in rows):
                        raise ResetError('Saver 中存在归属不明的线程；未删除。')
            counts = {name: conn.execute(sql.SQL('SELECT count(*) AS n FROM {}').format(
                sql.Identifier(schema, name))).fetchone()['n'] for name in present}
            if execute and present:
                # 单条 DROP 默认 RESTRICT；任何视图或未知依赖都会使整个事务回滚。
                conn.execute(sql.SQL('DROP TABLE {}').format(
                    sql.SQL(',').join(sql.Identifier(schema, name) for name in present)))
        return {'mode': 'executed' if execute else 'preview', 'profile': config.profile,
                'tables': counts, 'knowledge': 'untouched', 'remote': 'untouched'}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default='assistant.toml')
    parser.add_argument('--profile')
    parser.add_argument('--execute', action='store_true')
    args = parser.parse_args(argv)
    try:
        result = reset(load_config(args.config, profile=args.profile), execute=args.execute)
    except (ConfigError, ResetError) as error:
        print(str(error))
        return 2
    except Exception:
        print('重置未完成：请检查数据库连接、独占状态及表依赖；未输出凭据，事务内删除已回滚。')
        return 2
    print(json.dumps(result, ensure_ascii=False))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
