"""Native LangChain evidence retrieval; importing this package loads no models."""


class KnowledgeError(RuntimeError):
    """Expected error safe for model-visible reporting."""


class LeaseLost(KnowledgeError):
    """An obsolete worker must never publish."""


class IngestionError(KnowledgeError):
    """Safe failure summary with diagnostics kept out of model-visible messages."""

    def __init__(self, code: str, message: str, diagnostic_text: str = ''):
        super().__init__(message)
        self.code = code
        self.diagnostic_text = diagnostic_text
