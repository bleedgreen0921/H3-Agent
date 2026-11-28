"""迁移历史表；只记录已成功提交的迁移。"""

HISTORY_TABLE = 'assistant_schema_migrations'
BASELINE_NAME = 'baseline'
BASELINE_CHECKSUM = 'baseline'

HISTORY_DDL = '''CREATE TABLE IF NOT EXISTS assistant_schema_migrations (
    version integer PRIMARY KEY,
    name text NOT NULL,
    checksum text NOT NULL,
    applied_at timestamptz NOT NULL DEFAULT clock_timestamp(),
    duration_ms integer NOT NULL DEFAULT 0,
    app_version text NOT NULL DEFAULT '',
    backup_path text,
    status text NOT NULL DEFAULT 'applied' CHECK(status = 'applied')
)'''

BASELINE_INSERT = '''INSERT INTO assistant_schema_migrations
    (version,name,checksum,duration_ms,app_version,status)
    VALUES (%s,%s,%s,0,%s,'applied') ON CONFLICT(version) DO NOTHING'''

MIGRATION_INSERT = '''INSERT INTO assistant_schema_migrations
    (version,name,checksum,duration_ms,app_version,backup_path,status)
    VALUES (%s,%s,%s,%s,%s,%s,'applied')'''


def record_migration(conn, migration, duration_ms, app_version, backup_path=None):
    """记录一次成功迁移；同一版本重复插入会由主键拒绝。"""
    conn.execute(MIGRATION_INSERT, (
        migration.to_version, migration.name, migration.checksum,
        int(duration_ms), app_version, backup_path))
