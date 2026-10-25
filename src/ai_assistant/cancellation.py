"""Shared cancellation cleanup, independent of tools and platform policies."""

import asyncio


async def _drain_worker(worker):
    """Join a cancelled call's thread even if its owner is cancelled again."""
    while not worker.done():
        try:
            await asyncio.shield(worker)
        except asyncio.CancelledError:
            continue
        except Exception:
            break
    if worker.done() and not worker.cancelled():
        # Retrieve any exception so cancellation does not leave an unhandled task.
        worker.exception()
