"""显式观测：原生 trace、允许列表和标准库日志，不持有业务状态。"""
import asyncio
from contextlib import contextmanager
from contextvars import ContextVar
from datetime import datetime, timezone
import functools
import json
import logging
import os
import re
from time import monotonic
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field


class ObservabilityConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    enabled: bool = False
    capture_content: bool = False
    project: str = Field(default="rag-agent-dev", min_length=1)
    endpoint: str = Field(default="https://api.smith.langchain.com", min_length=1)
    api_key_env: str = Field(default="LANGSMITH_API_KEY", min_length=1)
    workspace_id: str | None = None
    flush_timeout_seconds: float = Field(default=5, gt=0, le=30)


_context = ContextVar("assistant_observation_context", default={})
_service = ContextVar("assistant_observation_service", default=None)
_ALLOWED = frozenset({"input_id", "session_id", "thread_id", "execution_thread_id",
    "operation_id", "outbox_id", "attempt", "part", "status", "phase", "category",
    "elapsed_seconds", "queue_wait_seconds", "preparation_seconds", "agent_seconds",
    "result_commit_seconds", "reply_delivery_seconds", "maintenance_seconds",
    "run_id", "parent_run_id", "profile", "role", "tool", "remote_id", "execution_status",
    "model_first_body_seconds", "input_tokens", "output_tokens", "total_tokens"})


@contextmanager
def bind(**values):
    token = _context.set({**_context.get(), **{k: str(v) if k.endswith("_id") and v is not None else v
                                            for k, v in values.items() if k in _ALLOWED and isinstance(v, (str, int, float, bool, type(None)))}})
    try:
        yield
    finally:
        _context.reset(token)


def event(name, **values):
    service = _service.get()
    if service and service.client and (not name.startswith(("model.", "tool.")) or name == "model.first_body"):
        try:
            from langsmith.run_helpers import get_current_run_tree
            run = get_current_run_tree()
            if run:
                run.add_metadata({k: str(v) if k.endswith("_id") and v is not None else v
                                  for k, v in values.items() if k in _ALLOWED and v is not None})
        except Exception as error:
            service.warning(error)
    logging.getLogger("ai_assistant.observability").info(name,
        extra={"observation": {**_context.get(), **{k: v for k, v in values.items() if k in _ALLOWED}}})


class JsonFormatter(logging.Formatter):
    def format(self, record):
        # 不格式化异常堆栈；业务日志仅使用安全摘要。
        return json.dumps({"time": datetime.fromtimestamp(record.created, timezone.utc).isoformat(),
            "level": record.levelname, "component": record.name, "event": record.getMessage(),
            **_context.get(), **getattr(record, "observation", {})}, ensure_ascii=False, default=str)


def configure_logging():
    logger = logging.getLogger("ai_assistant")
    logger.setLevel(logging.INFO)
    if not logger.handlers:
        logger.addHandler(logging.StreamHandler())
    for handler in logger.handlers:
        handler.setFormatter(JsonFormatter())
    logger.propagate = False


def _sanitize(value, secrets=()):
    if isinstance(value, dict):
        return {str(k): ("[redacted]" if re.search(r"secret|password|authorization|api.key|access.token|refresh.token", str(k), re.I)
                        else _sanitize(v, secrets)) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_sanitize(v, secrets) for v in value]
    if isinstance(value, str):
        for secret in secrets:
            if secret:
                value = value.replace(secret, "[redacted]")
        value = re.sub(r"(?i)bearer\s+[^\s\"']+|\bsk-[A-Za-z0-9_-]+", "[redacted]", value)
    return value


def safe_payload(payload, capture_content=False, secrets=()):
    """最终上传边界：结构模式只保留拓扑与明确允许的指标。"""
    result = {k: v for k, v in payload.items() if k in {
        "id", "trace_id", "parent_run_id", "dotted_order", "name", "run_type",
        "start_time", "end_time", "session_name", "session_id", "reference_example_id"}}
    def usage_counts(value):
        if isinstance(value, dict):
            usage = value.get("usage_metadata") or value.get("token_usage")
            if isinstance(usage, dict):
                return usage
            for nested in value.values():
                found = usage_counts(nested)
                if found:
                    return found
        elif isinstance(value, (list, tuple)):
            for nested in value:
                found = usage_counts(nested)
                if found:
                    return found
        return {}

    outputs = payload.get("outputs")
    is_model = payload.get("run_type") == "llm" or (isinstance(outputs, dict) and "generations" in outputs)
    if is_model:
        usage = usage_counts(outputs)
        for target, names in {"prompt_tokens": ("input_tokens", "prompt_tokens"),
                              "completion_tokens": ("output_tokens", "completion_tokens"),
                              "total_tokens": ("total_tokens",)}.items():
            value = payload.get(target)
            if value is None:
                value = next((usage[n] for n in names if n in usage), None)
            if isinstance(value, int) and not isinstance(value, bool) and value >= 0:
                result[target] = value
    extra = payload.get("extra") or {}
    result["extra"] = {"metadata": {k: v for k, v in extra.get("metadata", {}).items() if k in _ALLOWED and isinstance(v, (str, int, float, bool, type(None)))}}
    if result["extra"]["metadata"].get("session_id"):
        result["extra"]["metadata"]["thread_id"] = result["extra"]["metadata"]["session_id"]
    # 错误可能含工具正文和凭据，始终只保留发生错误这一事实。
    if payload.get("error"):
        result["error"] = "execution_error"
    if capture_content:
        for key in ("inputs", "outputs"):
            if key in payload:
                result[key] = payload[key]
    else:
        for key in ("inputs", "outputs"):
            if key in payload:
                result[key] = {}
    return _sanitize(result, secrets)


def current_service():
    return _service.get()


class ObservationService:
    def __init__(self, settings, secrets=()):
        self.settings = settings
        self.client = None
        self.secrets = tuple(secrets)
        self._warned = False
        self._token = None
        self._tracing = None

    def warning(self, error):
        if not self._warned:
            self._warned = True
            event("trace.upload_failed", category=type(error).__name__)

    async def __aenter__(self):
        if self.settings.enabled:
            from ai_assistant.config import ConfigError
            key = os.environ.get(self.settings.api_key_env)
            if not key:
                raise ConfigError("观测已启用，请设置指定的 LangSmith API Key 环境变量。")
            from langsmith import Client
            owner = self

            class SafeClient(Client):
                # 固定 SDK 版本的 create/update 最终出口均过滤；SDK 的 update 不经过 _run_transform。
                def _create_run(self, payload, **kwargs):
                    try:
                        return super()._create_run(safe_payload(payload, owner.settings.capture_content,
                                                               (*owner.secrets, key)), **kwargs)
                    except Exception as error:
                        owner.warning(error)

                def _update_run(self, payload, **kwargs):
                    try:
                        return super()._update_run(safe_payload(payload, owner.settings.capture_content,
                                                               (*owner.secrets, key)), **kwargs)
                    except Exception as error:
                        owner.warning(error)

                def _insert_runtime_env(self, runs):
                    return None

            self.client = SafeClient(api_key=key, api_url=self.settings.endpoint,
                workspace_id=self.settings.workspace_id, timeout_ms=2000,
                omit_traced_runtime_info=True, tracing_error_callback=self.warning, tracing_sampling_rate=1.0)
        from langsmith import tracing_context
        self._tracing = tracing_context(enabled=False)
        self._tracing.__enter__()
        self._token = _service.set(self)
        return self

    async def __aexit__(self, *exc):
        _service.reset(self._token)
        self._tracing.__exit__(None, None, None)
        if self.client:
            try:
                await asyncio.to_thread(self.client.close, timeout=self.settings.flush_timeout_seconds)
            except Exception as error:
                self.warning(error)


@contextmanager
def span(name, **values):
    """上传故障不能覆盖业务异常；禁用上下文也阻止环境变量隐式上传。"""
    from langsmith import trace, tracing_context
    service = _service.get()
    started = monotonic()
    with bind(**values):
        metadata = dict(_context.get())
        if metadata.get("session_id"):
            metadata["thread_id"] = metadata["session_id"]
        enabled = bool(service and service.client)
        with tracing_context(enabled=enabled, client=service.client if enabled else None,
                             project_name=service.settings.project if enabled else None,
                             metadata=metadata):
            run = None
            if enabled:
                try:
                    run = trace(name, inputs={}, metadata=metadata, client=service.client,
                                project_name=service.settings.project)
                    tree = run.__enter__()
                except Exception as error:
                    service.warning(error)
                    run = None
            identity = {"run_id": str(tree.id), "parent_run_id": str(tree.parent_run_id) if tree.parent_run_id else None} if run is not None else {}
            event(name + ".start", **identity)
            category = None
            try:
                yield
            except BaseException as error:
                category = type(error).__name__
                raise
            finally:
                event(name + ".end", elapsed_seconds=monotonic() - started,
                      execution_status="failed" if category else "complete", category=category, **identity)
                if run is not None:
                    try:
                        if category:
                            tree.end(error=category)
                        run.__exit__(None, None, None)
                    except Exception as error:
                        service.warning(error)


def observed(name, identity=None, *, root=False):
    def decorate(fn):
        @functools.wraps(fn)
        async def wrapper(*args, **kwargs):
            values = identity(*args, **kwargs) if identity else {}
            if root:
                from langsmith import tracing_context
                with tracing_context(parent=False):
                    with span(name, **values):
                        return await fn(*args, **kwargs)
            else:
                with span(name, **values):
                    return await fn(*args, **kwargs)
        return wrapper
    return decorate


def add_commands(commands):
    actions = commands.add_parser("observability", help="观测存储初始化与模拟连接检查").add_subparsers(dest="action", required=True)
    for name in ("init", "check"):
        actions.add_parser(name).add_argument("--config", default="assistant.toml")


async def execute(config, args):
    from ai_assistant.config import ConfigError
    if args.action == "init":
        if config.sessions is None:
            raise ConfigError("请先配置 sessions。")
        from ai_assistant.storage.sessions import session_storage
        from ai_assistant.storage.usage import initialize_observability
        async with session_storage(config.sessions) as repo:
            await initialize_observability(repo.conn)
        print("观测用量关联字段已初始化；历史记录未回填。")
        return 0
    if not config.observability.enabled:
        raise ConfigError("请先启用 observability。")
    try:
        async with ObservationService(config.observability, configuration_secrets(config)) as service:
            from langsmith import trace, tracing_context
            identity = uuid4()
            with tracing_context(enabled=True, client=service.client, project_name=config.observability.project):
                with trace("observability.check", run_id=identity, inputs={}, client=service.client,
                           project_name=config.observability.project):
                    pass
            await asyncio.to_thread(service.client.flush, timeout=config.observability.flush_timeout_seconds)
            for attempt in range(5):
                try:
                    await asyncio.to_thread(service.client.read_run, identity)
                    print("模拟 trace 已上传并确认可读取；未调用模型或飞书。")
                    return 0
                except Exception:
                    if attempt == 4:
                        raise
                    await asyncio.sleep(0.5)
    except Exception:
        print("观测检查失败：请检查 LangSmith 目标、网络和 API Key。")
        return 2

def configuration_secrets(config):
    from pydantic import BaseModel, SecretStr
    def collect(value):
        if isinstance(value, SecretStr):
            secret = value.get_secret_value()
            yield secret
            # 数据库 URL 是整体 SecretStr，正文中的独立密码也必须过滤。
            from urllib.parse import urlsplit, unquote
            try:
                password = urlsplit(secret).password
                if password:
                    yield unquote(password)
            except ValueError:
                pass
        elif isinstance(value, BaseModel):
            for name in type(value).model_fields:
                yield from collect(getattr(value, name))
        elif isinstance(value, dict):
            for item in value.values():
                yield from collect(item)
    return tuple(collect(config))

