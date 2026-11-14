"""回复完成后的会话记忆维护；辅助任务失败不改写业务结果。"""
import asyncio
import logging
from time import monotonic

from ai_assistant.storage.sessions import SessionStorageError
from ai_assistant.observability import observed, event as observation_event

logger = logging.getLogger(__name__)


@observed("session.maintenance", lambda session: {"session_id": session.session_id,
    "execution_thread_id": session._thread, "input_id": session._maintenance[0] if session._maintenance else None}, root=True)
async def maintain_session_memory(session):
    """入口成功处理最终回复后，在同一任务、会话锁内整理；不改写回答。"""
    maintenance_started = monotonic()
    pending = getattr(session, '_maintenance', None)
    session._maintenance = None
    if pending is None or session._storage_lost or session._closed:
        observation_event('maintenance.skipped', status='skipped')
        return
    async with session._turn_lock:
        # 回复处理后在会话锁内检查更新间隔和摘要调用预算。
        input_id, context, config, observer = pending
        session._active_task = asyncio.current_task()
        try:
            if session.repository is not None:
                await session.repository.check()
            from ai_assistant.runtime.history import latest_state
            state = await latest_state(session.agent, session._thread)
            every = getattr(session.agent, 'memory_every_turns', 0)
            summarizer = getattr(session.agent, 'session_memory', None)
            completed = state.values.get('completed_turns', 0)
            from ai_assistant.agents.session_state import SessionState
            covered = SessionState.from_values(state.values).memory.covered_turns
            if (state.values.get('turn_result', {}).get('input_id') != input_id
                    or state.values['turn_result']['status'] != 'complete'
                    or not every or not summarizer
                    or completed - covered < every
                    or observer.usage.summary_model_calls >= session.settings.summary_call_limit):
                observation_event('maintenance.skipped', status='skipped')
                return
            update = await summarizer.periodic_update(state.values, context, config)
            if update:
                await session.agent.aupdate_state(state.config, update)
            observation_event('maintenance.finished', status='complete', maintenance_seconds=monotonic() - maintenance_started)
        except Exception as error:
            if isinstance(error, SessionStorageError):
                session._storage_lost = True
            observation_event('maintenance.failed', status='failed', category=type(error).__name__, maintenance_seconds=monotonic() - maintenance_started)
            logger.warning('Turn auxiliary failure: input_id=%s phase=session_memory category=%s', input_id, type(error).__name__)
        finally:
            session._active_task = None

