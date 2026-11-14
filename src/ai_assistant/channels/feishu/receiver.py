"""飞书接收器、Node 子进程与服务资源监督。"""
import asyncio
from contextlib import AsyncExitStack, asynccontextmanager, suppress
import json
import logging
from pathlib import Path

import httpx

from ai_assistant.config import ConfigError
from ai_assistant.executables import executable, executable_env
from ai_assistant.integrations.mcp import mcp_resources
from ai_assistant.integrations.feishu_files import FeishuAPI
from ai_assistant.observability import event, ObservabilityConfig
from ai_assistant.storage.integrations import operation_storage
from ai_assistant.tools.context import tool_context
from ai_assistant.channels.feishu.worker import BotWorker, ProcessResources
from ai_assistant.channels.feishu.delivery import FeishuOutboxSender
from ai_assistant.channels.feishu.commands import default_registry

logger = logging.getLogger(__name__)


@asynccontextmanager
async def optional_knowledge_worker(config):
    """同进程启动一次；知识故障不参与聊天服务退出判定。"""
    async with AsyncExitStack() as stack:
        # 知识后台任务是可选资源，启动失败时尽量保持聊天服务可用。
        try:
            from ai_assistant.knowledge.sync import background_worker
            task = await stack.enter_async_context(background_worker(config))
        except Exception as error:
            # 会话/清理错误仍必须传播；知识数据库启动错误单独隔离。
            from ai_assistant.storage.sessions import SessionError
            from langgraph.errors import GraphBubbleUp
            if (isinstance(error, (SessionError, GraphBubbleUp))
                    or asyncio.current_task().cancelling()):
                raise
            logger.error('知识 Worker 启动失败；聊天继续，队列尚未确认处理，不自动重启。category=%s',
                         type(error).__name__)
            task = None

        # 后台任务意外结束只记录状态，不自动重启或重放队列。
        def stopped(completed):
            error = None if completed.cancelled() else completed.exception()
            logger.error('知识 Worker 已退出；聊天继续，队列尚未确认处理，不自动重启。state=%s category=%s',
                         'cancelled' if completed.cancelled() else 'failed' if error else 'ended',
                         type(error).__name__ if error else 'none')

        if task is not None:
            task.add_done_callback(stopped)
        try:
            yield task
        finally:
            if task is not None:
                # 正常关闭由 background_worker 等待任务清理并关闭连接，不记作意外退出。
                task.remove_done_callback(stopped)


async def serve(config):
    """初始化日志与观测资源，再启动飞书服务。"""
    from ai_assistant.observability import ObservationService, configure_logging, configuration_secrets
    configure_logging()
    async with ObservationService(getattr(config, "observability", ObservabilityConfig()), secrets=configuration_secrets(config)):
        await _serve(config)


async def _serve(config):
    """校验依赖、装配资源并启动接收、处理和投递循环。"""
    # 未显式启用渠道或缺少必要配置时，不连接外部服务。
    if not config.channels.feishu.enabled or config.integrations.feishu is None or config.sessions is None:
        raise ConfigError('Enable [channels.feishu] and configure Feishu and sessions.')
    from ai_assistant.integrations.mcp import mcp_resources
    from ai_assistant.integrations.feishu_files import FeishuAPI
    base = config.config_path.parent if config.config_path else Path.cwd()
    project = (base / config.integrations.feishu.node_project).resolve()
    receiver = project / 'receiver.mjs'

    # 只检查已准备好的 Node 接收器和依赖；启动时不自动安装。
    if not receiver.is_file() or not (project / 'node_modules/pg/package.json').is_file():
        raise ConfigError('Deploy the pinned Feishu Node project with npm ci.')
    async with AsyncExitStack() as stack:
        store = await stack.enter_async_context(operation_storage(config.sessions))
        from ai_assistant.storage.locks import try_advisory_lock

        # 数据库锁限制同一存储上的飞书执行器同时运行。
        locked = await try_advisory_lock(store.conn, 714160004, wait_seconds=5)
        if not locked:
            raise ConfigError('Another Feishu executor is already running; wait for the previous service to exit or check stale database connections.')

        # 投递使用独立连接，避免慢发送阻塞消息处理。
        delivery_store = await stack.enter_async_context(operation_storage(config.sessions))
        # Fail before starting reception if migration has not been explicitly run.
        await store.conn.execute('SELECT last_input_id FROM assistant_sessions LIMIT 0')
        await store.conn.execute('SELECT binding,title FROM assistant_sessions LIMIT 0')
        await store.conn.execute('SELECT input_id FROM assistant_chat_commands LIMIT 0')
        await store.conn.execute('SELECT id FROM assistant_inbox LIMIT 0')
        await store.conn.execute('SELECT part,next_attempt_at FROM assistant_outbox LIMIT 0')

        # 按配置装配知识后台任务、工具上下文、MCP 和飞书客户端。
        if config.knowledge is not None:
            await stack.enter_async_context(optional_knowledge_worker(config))
        context = await stack.enter_async_context(tool_context(config))
        directory = await stack.enter_async_context(mcp_resources(config, context.workspace))
        resources = ProcessResources(directory, context.services.get('skills'), {}, default_registry())
        client = await stack.enter_async_context(httpx.AsyncClient(timeout=30, trust_env=False))
        api = FeishuAPI(config.integrations.feishu, client)
        from ai_assistant.channels.feishu.delivery import FeishuOutboxSender
        bot = BotWorker(config, store, resources)
        bot.api = api
        sender = FeishuOutboxSender(config.channels.feishu, delivery_store, api)

        # 恢复只登记回复；完成之后才允许接收和发送新消息。
        await sender.recover()
        await bot.recover()

        # Node 子进程接收飞书事件；退出栈保证后续按顺序关闭它。
        binary = executable('node')
        process = await asyncio.create_subprocess_exec(binary, str(receiver), cwd=str(project),
            stdin=asyncio.subprocess.PIPE, stdout=asyncio.subprocess.DEVNULL, stderr=asyncio.subprocess.PIPE,
            env=executable_env(binary))
        stack.push_async_callback(_stop_receiver, process)

        # 通过标准输入传递接收器配置，避免将凭据放入命令行参数。
        payload = dict(final_delivery_timeout_seconds=config.channels.feishu.final_delivery_timeout_seconds, app_id=config.integrations.feishu.app_id,
            app_secret=config.integrations.feishu.app_secret.get_secret_value(),
            owner_open_id=config.integrations.feishu.owner_open_id,
            database_url=config.sessions.database_url.get_secret_value())
        process.stdin.write((json.dumps(payload) + '\n').encode())
        await process.stdin.drain()
        await _run_workers(bot, sender, process)


async def _stop_receiver(process):
    """接收器退出必须等待，随后才能释放服务资源和单实例锁。"""
    if process.returncode is None:
        # 先请求正常退出；超时后终止进程并等待回收。
        with suppress(ProcessLookupError):
            process.terminate()
        try:
            await asyncio.wait_for(process.wait(), 10)
        except TimeoutError:
            with suppress(ProcessLookupError):
                process.kill()
            await process.wait()


async def _run_workers(bot, sender, process):
    """统一监督主任务，等待取消清理后才允许资源栈关闭。"""
    from ai_assistant.cancellation import _drain_worker

    # 并发监督消息处理、回复投递、接收器退出及其安全日志。
    worker = asyncio.create_task(bot.run())
    delivery = asyncio.create_task(sender.run())
    receiver_wait = asyncio.create_task(process.wait())
    tasks = [worker, delivery, receiver_wait]
    if getattr(process, 'stderr', None) is not None:
        tasks.append(asyncio.create_task(_forward_receiver_logs(process.stderr)))
    try:
        done, _ = await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
        # 先传播任务原始异常；正常意外结束也必须关闭整个服务。
        for task in tasks:
            if task in done:
                await task
        raise ConfigError('飞书接收器、执行器或投递器意外停止，请检查服务日志与操作回执。')
    finally:
        # 任一受监督任务结束时，停止其余任务并等待取消清理。
        if process.returncode is None:
            with suppress(ProcessLookupError):
                process.terminate()
        for task in tasks:
            task.cancel()
        for task in tasks:
            await _drain_worker(task)


async def _forward_receiver_logs(stream):
    """只转发接收器约定的安全事件，丢弃 SDK 原始异常和非结构化输出。"""
    while line := await stream.readline():
        try:
            record = json.loads(line)

            # 仅保留约定事件及有限字段，避免转发原始日志和凭据。
            if record.get('event') not in {'inbox.persisted', 'inbox.failed'}:
                continue
            event(record['event'], input_id=record.get('input_id'),
                  status=record.get('status'), category=record.get('category'),
                  elapsed_seconds=record.get('elapsed_seconds'))
        except (ValueError, AttributeError, TypeError):
            continue
