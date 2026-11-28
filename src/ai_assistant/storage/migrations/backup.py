"""破坏性迁移前的 PostgreSQL 逻辑备份；不把 URL 放进进程参数。"""
from datetime import datetime
import os
from pathlib import Path
import shutil
import subprocess


class BackupError(RuntimeError):
    """备份失败的安全摘要；不包含数据库原始输出或凭据。"""


_ENV_MAP = {
    'host': 'PGHOST', 'port': 'PGPORT', 'dbname': 'PGDATABASE',
    'user': 'PGUSER', 'password': 'PGPASSWORD', 'sslmode': 'PGSSLMODE',
    'connect_timeout': 'PGCONNECT_TIMEOUT',
}


def create_backup(settings, *, base_dir=None, output=None):
    """用 pg_dump 生成自定义格式备份，返回文件路径。"""
    if settings is None:
        raise BackupError('未配置会话存储；没有备份目标。')
    binary = shutil.which('pg_dump')
    if binary is None:
        raise BackupError('未找到 pg_dump；请安装 PostgreSQL 客户端或使用 --no-backup。')

    base = Path(base_dir or Path.cwd())
    if output:
        path = Path(output).expanduser()
        if not path.is_absolute():
            path = base / path
    else:
        path = base / '.data' / 'backups' / (
            'assistant-%s.dump' % datetime.now().strftime('%Y%m%d-%H%M%S'))
    if path.exists():
        raise BackupError('备份文件已存在；请更换 --output。')
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    try:
        path.parent.chmod(0o700)
    except OSError:
        pass

    try:
        from psycopg.conninfo import conninfo_to_dict
        info = conninfo_to_dict(settings.database_url.get_secret_value())
    except Exception:
        raise BackupError('无法解析会话数据库连接；未备份。') from None
    env = os.environ.copy()
    for key, name in _ENV_MAP.items():
        value = info.get(key)
        if value is not None:
            env[name] = str(value)
    env.setdefault('PGCONNECT_TIMEOUT', '10')

    try:
        result = subprocess.run(
            [binary, '--format=custom', '--no-password', '--file', str(path)],
            env=env, capture_output=True, text=True, timeout=600, check=False)
    except (OSError, subprocess.TimeoutExpired):
        path.unlink(missing_ok=True)
        raise BackupError('备份命令未能完成；未执行迁移。') from None
    if result.returncode != 0:
        path.unlink(missing_ok=True)
        raise BackupError('备份失败；未执行迁移。')
    try:
        if not path.is_file() or path.stat().st_size == 0:
            path.unlink(missing_ok=True)
            raise BackupError('备份文件为空；未执行迁移。')
        path.chmod(0o600)
    except OSError:
        path.unlink(missing_ok=True)
        raise BackupError('无法确认备份文件；未执行迁移。') from None
    return str(path)
