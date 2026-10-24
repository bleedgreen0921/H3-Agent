"""Local MCP fixture; no credentials or external endpoints."""
import asyncio
import os
import sys
from fastmcp import FastMCP, Context

server = FastMCP('office-fixture')


@server.tool
async def echo(value: str = 'fixture') -> str:
    """Return local fixture text."""
    return value


@server.tool
async def environment() -> dict:
    """Verify that unrelated credentials are not inherited by subprocesses."""
    return {'leaked': 'ASSISTANT_TEST_MODEL_SECRET' in os.environ}


@server.tool
async def wait() -> str:
    """Wait long enough to exercise caller cancellation."""
    await asyncio.sleep(60)
    return 'done'


@server.tool
async def ask(ctx: Context) -> str:
    """Exercise elicitation without interrupting the native Agent."""
    result = await ctx.elicit('Need login', response_type=bool)
    return result.action


if __name__ == '__main__':
    if len(sys.argv) > 1:
        server.run(transport='http', host='127.0.0.1', port=int(sys.argv[1]), show_banner=False)
    else:
        server.run(show_banner=False)
