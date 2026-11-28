"""有序迁移注册表；新增迁移只追加，已发布迁移不修改。"""
from ai_assistant.storage.migrations.base import MigrationError
from ai_assistant.storage.migrations.v5_to_v6 import V5ToV6
from ai_assistant.storage.version import SCHEMA_VERSION

MIGRATIONS = (V5ToV6(),)


def pending_migrations(current_version):
    """按 from_version 串起从当前版本到目标版本的迁移链。"""
    if current_version >= SCHEMA_VERSION:
        return []
    by_from = {item.from_version: item for item in MIGRATIONS}
    pending = []
    version = current_version
    while version < SCHEMA_VERSION:
        migration = by_from.get(version)
        if migration is None:
            raise MigrationError(
                '没有从 v%d 到 v%d 的连续迁移；未修改。'
                % (version, SCHEMA_VERSION))
        pending.append(migration)
        version = migration.to_version
    if version != SCHEMA_VERSION:
        raise MigrationError('迁移链未到达目标版本；未修改。')
    return pending
