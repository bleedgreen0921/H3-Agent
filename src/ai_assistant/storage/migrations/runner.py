"""显式迁移状态、预检、备份和事务执行。"""
from time import monotonic

from ai_assistant import __version__
from ai_assistant.storage.migrations import MIGRATIONS, pending_migrations
from ai_assistant.storage.migrations.base import (
    MigrationChecksumError, MigrationError,
)
from ai_assistant.storage.migrations.history import (
    BASELINE_CHECKSUM, HISTORY_DDL, record_migration,
)
from ai_assistant.storage.version import SCHEMA_LOCK, SCHEMA_VERSION

MIN_SUPPORTED_VERSION = min(item.from_version for item in MIGRATIONS)


def _connect(settings):
    try:
        import psycopg
        from psycopg.rows import dict_row
        return psycopg.connect(
            settings.database_url.get_secret_value(),
            row_factory=dict_row, connect_timeout=10)
    except ImportError:
        raise MigrationError('缺少 PostgreSQL 会话依赖；请安装 sessions extra。') from None
    except Exception:
        raise MigrationError('无法连接会话数据库；请检查配置和数据库状态。') from None


def _schema_version(conn):
    row = conn.execute(
        'SELECT version FROM assistant_session_schema WHERE singleton=true').fetchone()
    return int(row['version']) if row else None


def _history(conn):
    try:
        return [dict(row) for row in conn.execute(
            '''SELECT version,name,checksum,applied_at,duration_ms,
                      app_version,backup_path,status
               FROM assistant_schema_migrations ORDER BY version''').fetchall()]
    except Exception:
        return []


def _other_clients(conn):
    return conn.execute(
        """SELECT count(*) AS n FROM pg_stat_activity
           WHERE datname=current_database() AND pid<>pg_backend_pid()
             AND backend_type='client backend'""").fetchone()['n']


def _checksum_issues(history):
    """已执行迁移的源码必须与历史记录一致；基线记录除外。"""
    by_version = {item.to_version: item for item in MIGRATIONS}
    issues = []
    for row in history:
        if row.get('checksum') == BASELINE_CHECKSUM:
            continue
        migration = by_version.get(row.get('version'))
        if migration is None:
            issues.append({'version': row.get('version'), 'reason': 'unknown_migration'})
        elif migration.checksum != row.get('checksum'):
            issues.append({'version': row.get('version'), 'reason': 'checksum_mismatch'})
    return issues


def status(settings):
    """只读返回当前版本、待执行迁移、旧数据和历史。"""
    if settings is None:
        raise MigrationError('未配置会话存储；没有迁移目标。')
    with _connect(settings) as conn:
        try:
            conn.execute("SET statement_timeout='5000ms'")
        except Exception:
            pass
        present = conn.execute(
            "SELECT to_regclass('assistant_session_schema') AS name").fetchone()['name']
        if not present:
            return {
                'state': 'uninitialized', 'current_version': None,
                'target_version': SCHEMA_VERSION, 'pending': [],
                'requires_backup': False, 'requires_delete_confirmation': False,
                'legacy_sessions': 0, 'history': [], 'checksum_issues': [],
                'other_clients': _other_clients(conn),
            }
        current = _schema_version(conn)
        if current is None:
            return {
                'state': 'uninitialized', 'current_version': None,
                'target_version': SCHEMA_VERSION, 'pending': [],
                'requires_backup': False, 'requires_delete_confirmation': False,
                'legacy_sessions': 0, 'history': _history(conn),
                'checksum_issues': [], 'other_clients': _other_clients(conn),
            }
        if current < MIN_SUPPORTED_VERSION or current > SCHEMA_VERSION:
            return {
                'state': 'unsupported', 'current_version': current,
                'target_version': SCHEMA_VERSION, 'pending': [],
                'requires_backup': False, 'requires_delete_confirmation': False,
                'legacy_sessions': 0, 'history': _history(conn),
                'checksum_issues': [], 'other_clients': _other_clients(conn),
            }
        try:
            pending = pending_migrations(current)
        except MigrationError:
            return {
                'state': 'unsupported', 'current_version': current,
                'target_version': SCHEMA_VERSION, 'pending': [],
                'requires_backup': False, 'requires_delete_confirmation': False,
                'legacy_sessions': 0, 'history': _history(conn),
                'checksum_issues': [], 'other_clients': _other_clients(conn),
            }

        schema = conn.execute('SELECT current_schema() AS name').fetchone()['name']
        previews = []
        for migration in pending:
            try:
                preview = migration.preview(conn, schema)
            except MigrationError as error:
                preview = {'error': str(error)}
            previews.append({
                'from_version': migration.from_version,
                'to_version': migration.to_version,
                'name': migration.name,
                'description': migration.description,
                'destructive': migration.destructive,
                'checksum': migration.checksum,
                'preview': preview,
            })
        history = _history(conn)
        issues = _checksum_issues(history)
        legacy_sessions = sum(
            item['preview'].get('legacy_sessions', 0)
            for item in previews if isinstance(item['preview'], dict))
        return {
            'state': 'checksum_mismatch' if issues else ('pending' if pending else 'current'),
            'current_version': current,
            'target_version': SCHEMA_VERSION,
            'pending': previews,
            'requires_backup': any(item.destructive for item in pending),
            'requires_delete_confirmation': any(
                item['preview'].get('requires_delete_confirmation', False)
                for item in previews if isinstance(item['preview'], dict)),
            'legacy_sessions': legacy_sessions,
            'history': history,
            'checksum_issues': issues,
            'other_clients': _other_clients(conn),
        }


def migrate(settings, *, execute=False, delete_legacy_sessions=False,
            backup=True, base_dir=None, output=None):
    """预览或执行迁移链；默认预览，破坏性迁移默认要求备份。"""
    current_status = status(settings)
    if current_status['state'] == 'uninitialized':
        raise MigrationError('会话结构尚未初始化；请先执行 init。')
    if current_status['state'] == 'unsupported':
        raise MigrationError('会话结构版本不受支持；请检查数据库。')
    if current_status['state'] == 'checksum_mismatch':
        raise MigrationChecksumError('已执行迁移的校验值与当前代码不一致；未修改。')
    if not current_status['pending']:
        return {**current_status, 'mode': 'already-current', 'backup_path': None}
    if not execute:
        return {**current_status, 'mode': 'preview', 'backup_path': None}
    for item in current_status['pending']:
        preview = item.get('preview') or {}
        if preview.get('error'):
            raise MigrationError(preview['error'])
    if current_status.get('other_clients'):
        raise MigrationError('目标库仍有其他客户端连接；请停止服务后重试。')
    if current_status['requires_delete_confirmation'] and not delete_legacy_sessions:
        raise MigrationError(
            '发现 %d 个旧渠道会话；确认删除后加 --delete-legacy-sessions 重试。'
            % current_status['legacy_sessions'])

    backup_path = None
    if current_status['requires_backup'] and backup:
        from ai_assistant.storage.migrations.backup import create_backup
        backup_path = create_backup(
            settings, base_dir=base_dir, output=output)

    with _connect(settings) as conn:
        with conn.transaction():
            conn.execute("SET LOCAL lock_timeout='2s'")
            conn.execute("SET LOCAL statement_timeout='120s'")
            if not conn.execute(
                    'SELECT pg_try_advisory_xact_lock(%s) AS locked',
                    (SCHEMA_LOCK,)).fetchone()['locked']:
                raise MigrationError('初始化或迁移仍在运行；未修改。')
            if _other_clients(conn):
                raise MigrationError('目标库仍有其他客户端连接；请停止服务后重试。')
            schema = conn.execute('SELECT current_schema() AS name').fetchone()['name']
            current = _schema_version(conn)
            if current != current_status['current_version']:
                raise MigrationError('迁移状态已变化；未修改。')
            pending = pending_migrations(current)
            conn.execute(HISTORY_DDL)
            applied = []
            for migration in pending:
                started = monotonic()
                result = migration.apply(
                    conn, schema, delete_legacy_sessions=delete_legacy_sessions)
                elapsed = (monotonic() - started) * 1000
                record_migration(conn, migration, elapsed, __version__, backup_path)
                conn.execute(
                    'UPDATE assistant_session_schema SET version=%s WHERE singleton=true',
                    (migration.to_version,))
                applied.append({
                    'from_version': migration.from_version,
                    'to_version': migration.to_version,
                    'name': migration.name,
                    'result': result,
                })
            return {
                **current_status, 'mode': 'executed',
                'state': 'current', 'pending': [],
                'current_version': SCHEMA_VERSION,
                'legacy_sessions': 0,
                'requires_delete_confirmation': False,
                'backup_path': backup_path,
                'backup_skipped': bool(current_status['requires_backup'] and not backup),
                'history': _history(conn),
                'applied': applied,
            }
