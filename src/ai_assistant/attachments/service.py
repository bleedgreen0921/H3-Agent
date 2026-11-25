"""附件服务入口；实现位于 storage 层以复用数据库连接约定。"""
from ai_assistant.storage.attachments import (
    AttachmentError,
    AttachmentRecord,
    AttachmentService,
    attachment_store,
    initialize,
)

__all__ = ["AttachmentError", "AttachmentRecord", "AttachmentService",
           "attachment_store", "initialize"]
