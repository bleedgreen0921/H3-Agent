"""迁移定义与稳定校验值；已发布迁移不应修改。"""
from dataclasses import dataclass
import hashlib
import inspect
import sys


class MigrationError(RuntimeError):
    """迁移失败的安全摘要；不包含数据库原始诊断或凭据。"""


class MigrationChecksumError(MigrationError):
    """已执行迁移的源码与历史记录不一致。"""


@dataclass(frozen=True)
class Migration:
    """一个有序、可预览、可执行的数据库结构迁移。"""

    from_version: int
    to_version: int
    name: str
    description: str
    destructive: bool = False

    @property
    def checksum(self):
        """以迁移模块源码作为校验值，防止已执行迁移被静默改写。"""
        try:
            module = sys.modules.get(type(self).__module__)
            source = (inspect.getsource(module) if module is not None
                      else inspect.getsource(type(self)))
        except (OSError, TypeError):
            # 发行包缺少源码时仍保持稳定校验值；此时无法检测源码级改动。
            source = '%s:%s:%s' % (self.from_version, self.to_version, self.name)
        return hashlib.sha256(source.encode('utf-8')).hexdigest()

    def preview(self, conn, schema: str) -> dict:
        """只读检查迁移影响；返回可安全展示的摘要。"""
        raise NotImplementedError

    def apply(self, conn, schema: str, *, delete_legacy_sessions: bool = False) -> dict:
        """在调用方事务内执行结构变更。"""
        raise NotImplementedError
