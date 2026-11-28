"""v5 -> v6：删除渠道绑定列和旧渠道表。"""
from ai_assistant.storage.migrations.base import Migration, MigrationError

# 只允许删除已知结构的旧渠道表；列不匹配时拒绝，避免误删同名表。
LEGACY_TABLES = {
    'assistant_outbox': {'id', 'input_id', 'body', 'status', 'part'},
    'assistant_inbox': {'id', 'binding', 'body', 'status'},
    'assistant_chat_commands': {'input_id', 'binding', 'action'},
    'assistant_channel_sessions': {'binding', 'session_id'},
}


def _table_exists(conn, schema, name):
    return bool(conn.execute('SELECT to_regclass(%s) AS name',
                             ('%s.%s' % (schema, name),)).fetchone()['name'])


def _legacy_tables(conn, schema):
    """返回存在的旧表；表名相同但列结构不符时拒绝继续。"""
    found = conn.execute(
        """SELECT table_name,column_name FROM information_schema.columns
           WHERE table_schema=%s AND table_name=ANY(%s)""",
        (schema, list(LEGACY_TABLES))).fetchall()
    columns = {}
    for row in found:
        columns.setdefault(row['table_name'], set()).add(row['column_name'])
    present = [name for name in LEGACY_TABLES if name in columns]
    for name in present:
        if not LEGACY_TABLES[name] <= columns[name]:
            raise MigrationError('旧表 %s 结构不符合已知定义；未修改。' % name)
    return present


def _session_columns(conn, schema):
    return {row['column_name'] for row in conn.execute(
        """SELECT column_name FROM information_schema.columns
           WHERE table_schema=%s AND table_name='assistant_sessions'""",
        (schema,)).fetchall()}


def _inspect(conn, schema):
    """读取旧表、旧会话和可定位的输入 ID；只做 SELECT。"""
    present = _legacy_tables(conn, schema)
    has_binding = 'binding' in _session_columns(conn, schema)
    legacy_rows = []
    input_ids = []
    if has_binding:
        legacy_rows = conn.execute(
            """SELECT id,thread_ids,binding FROM assistant_sessions
               WHERE binding IS NOT NULL""").fetchall()
        bindings = [row['binding'] for row in legacy_rows]
        if bindings and 'assistant_inbox' in present:
            input_ids.extend(row['id'] for row in conn.execute(
                'SELECT id FROM assistant_inbox WHERE binding = ANY(%s)',
                (bindings,)).fetchall())
        if legacy_rows and _table_exists(conn, schema, 'assistant_web_messages'):
            input_ids.extend(row['input_id'] for row in conn.execute(
                """SELECT input_id FROM assistant_web_messages
                   WHERE session_id = ANY(%s) AND input_id IS NOT NULL""",
                ([row['id'] for row in legacy_rows],)).fetchall())
    return {
        'present': present,
        'has_binding': has_binding,
        'legacy_rows': legacy_rows,
        'bindings': [row['binding'] for row in legacy_rows],
        'input_ids': input_ids,
    }


def _delete_legacy_sessions(conn, schema, rows, input_ids):
    """删除旧渠道会话及其可定位的派生数据；不清理工作区或远端内容。"""
    from psycopg import sql

    session_ids = [row['id'] for row in rows]
    threads = sorted({thread for row in rows for thread in (row['thread_ids'] or [])})
    if threads:
        for table in ('checkpoint_writes', 'checkpoint_blobs', 'checkpoints'):
            if _table_exists(conn, schema, table):
                conn.execute(
                    sql.SQL('DELETE FROM {} WHERE thread_id = ANY(%s)').format(
                        sql.Identifier(schema, table)), (threads,))
    for table, column in (
            ('assistant_web_messages', 'session_id'),
            ('assistant_attachments', 'session_id'),
            ('assistant_model_usage', 'session_id')):
        if _table_exists(conn, schema, table):
            conn.execute(
                sql.SQL('DELETE FROM {} WHERE {} = ANY(%s)').format(
                    sql.Identifier(schema, table), sql.Identifier(column)),
                (session_ids,))
    if input_ids and _table_exists(conn, schema, 'assistant_operations'):
        conn.execute(
            sql.SQL('DELETE FROM {} WHERE input_id = ANY(%s)').format(
                sql.Identifier(schema, 'assistant_operations')), (input_ids,))
    conn.execute(
        sql.SQL('DELETE FROM {} WHERE id = ANY(%s)').format(
            sql.Identifier(schema, 'assistant_sessions')), (session_ids,))


class V5ToV6(Migration):
    """v5 -> v6：删除 binding、索引和旧渠道表。"""

    def __init__(self):
        super().__init__(
            from_version=5, to_version=6, name='v5_to_v6',
            description='删除渠道绑定列和旧渠道表', destructive=True)

    def preview(self, conn, schema):
        state = _inspect(conn, schema)
        actions = []
        if state['legacy_rows']:
            actions.append('delete_legacy_sessions:%d' % len(state['legacy_rows']))
        if state['present']:
            actions.append('drop_tables:' + ','.join(state['present']))
        if state['has_binding']:
            actions.extend(['drop_index:assistant_sessions_binding',
                            'drop_column:assistant_sessions.binding'])
        return {
            'legacy_sessions': len(state['legacy_rows']),
            'legacy_tables': state['present'],
            'has_binding': state['has_binding'],
            'requires_delete_confirmation': bool(state['legacy_rows']),
            'actions': actions,
        }

    def apply(self, conn, schema, *, delete_legacy_sessions=False):
        from psycopg import sql

        state = _inspect(conn, schema)
        lock_names = ['assistant_sessions', 'assistant_session_schema']
        lock_names.extend(name for name in state['present'] if name not in lock_names)
        conn.execute(sql.SQL('LOCK TABLE {} IN ACCESS EXCLUSIVE MODE NOWAIT').format(
            sql.SQL(',').join(sql.Identifier(schema, name) for name in lock_names)))
        # 锁后重新读取，避免检查与执行之间出现并发写入。
        state = _inspect(conn, schema)
        if state['legacy_rows'] and not delete_legacy_sessions:
            raise MigrationError(
                '发现 %d 个旧渠道会话；确认删除后加 --delete-legacy-sessions 重试。'
                % len(state['legacy_rows']))

        # 先删除旧渠道表，释放它们可能持有的会话外键，再清理旧会话。
        if state['present']:
            conn.execute(sql.SQL('DROP TABLE {}').format(
                sql.SQL(',').join(sql.Identifier(schema, name) for name in state['present'])))
        if state['legacy_rows']:
            _delete_legacy_sessions(conn, schema, state['legacy_rows'], state['input_ids'])
        if state['has_binding']:
            conn.execute(sql.SQL('DROP INDEX IF EXISTS {}').format(
                sql.Identifier(schema, 'assistant_sessions_binding')))
            conn.execute(sql.SQL('ALTER TABLE {} DROP COLUMN IF EXISTS {}').format(
                sql.Identifier(schema, 'assistant_sessions'), sql.Identifier('binding')))
        return {
            'deleted_legacy_sessions': len(state['legacy_rows']),
            'dropped_legacy_tables': state['present'],
            'dropped_binding': state['has_binding'],
        }
