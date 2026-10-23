"""POSIX asynchronous chat console; no background stdin threads."""
import asyncio
import os
import signal
import select
import sys

from langchain_core.messages import ToolMessage
from ai_assistant.runtime import assistant_session, final_text
from ai_assistant.config import ConfigError
from ai_assistant.storage.sessions import SessionError, SessionStorageError
from ai_assistant.agents.context import ContextBudgetError, SummaryCallLimitError

HELP = '/help 帮助；/new 新会话；/exit 退出。每行输入一个请求。'


class ConsoleInput:
    def __init__(self):
        self.queue = asyncio.Queue()
        self.buffer = b''
        self.eof = False
        self.handle = None

    def poll(self):
        if select.select([sys.stdin.fileno()], [], [], 0)[0]:
            self.ready()
        if not self.eof:
            self.handle = asyncio.get_running_loop().call_later(.01, self.poll)

    def ready(self):
        data = os.read(sys.stdin.fileno(), 4096)
        if not data:
            self.eof = True
            if self.buffer:
                self.queue.put_nowait(self.buffer.decode('utf-8', errors='replace'))
                self.buffer = b''
            self.queue.put_nowait(None)
            asyncio.get_running_loop().remove_reader(sys.stdin.fileno())
            return
        self.buffer += data
        while b'\n' in self.buffer:
            line, self.buffer = self.buffer.split(b'\n', 1)
            self.queue.put_nowait(line.decode('utf-8', errors='replace').rstrip('\r'))


def show_event(event):
    label = {'on_tool_start': '开始', 'on_tool_end': '完成', 'on_tool_error': '失败'}[event['event']]
    output = event.get('data', {}).get('output')
    if isinstance(output, ToolMessage) and output.status == 'error':
        label = '失败'
    print(f"[{event['run_id']}] {event['name']}：{label}", flush=True)


def show_turn(turn):
    print(final_text(turn.messages), flush=True)
    if turn.sources:
        from ai_assistant.knowledge.metadata import location
        print('本轮检索来源（实际检索结果）：')
        for doc in turn.sources:
            m = doc.metadata
            print(f"- {doc.id} | {m.get('source_name', '')} | "
                  f"{m.get('document_version_id', '')} | {location(m)}")

    usage = turn.usage
    print(f'调用：模型 {usage.main_model_calls}，摘要 {usage.summary_model_calls}，工具 {usage.tool_calls}；'
          f'未报告用量的模型调用 {usage.unreported_model_calls}')
    for profile, tokens in usage.token_usage.items():
        print(f'用量 {profile}：' + ', '.join(f'{k}={v}' for k, v in tokens.items()))


async def chat(config, model_name=None, *, session_id=None):
    if config.sessions is None or config.context is None:
        raise ConfigError('chat requires [sessions], [context] and model max_input_tokens; run session init first.')
    async with assistant_session(config, model_name, session_id=session_id) as session:
        print(f'会话：{session.session_id}')
        print(f'模型：{session.model_name}；工作区：{session.context.workspace.root}')
        print('工具：' + ', '.join(tool.name for tool in session.tools))
        print(HELP, flush=True)
        loop = asyncio.get_running_loop()
        console = ConsoleInput()
        active = None
        interrupted = False
        idle = asyncio.Event()

        def interrupt():
            nonlocal interrupted
            if active is not None:
                if not interrupted and not active.cancelling():
                    interrupted = True
                    active.cancel()
            else:
                idle.set()

        old_handler = signal.getsignal(signal.SIGINT)
        loop.add_signal_handler(signal.SIGINT, interrupt)
        try:
            try:
                loop.add_reader(sys.stdin.fileno(), console.ready)
            except PermissionError:
                # epoll cannot register regular files or /dev/null. Read only
                # when POSIX select reports readiness; still no input thread.
                console.handle = loop.call_soon(console.poll)
            while True:
                print('> ', end='', flush=True)
                read = asyncio.create_task(console.queue.get())
                stop = asyncio.create_task(idle.wait())
                try:
                    await asyncio.wait((read, stop), return_when=asyncio.FIRST_COMPLETED)
                    if idle.is_set():
                        return 130
                    line = read.result()
                finally:
                    for task in (read, stop):
                        if not task.done():
                            task.cancel()
                    await asyncio.gather(read, stop, return_exceptions=True)
                if line is None or line.strip() == '/exit':
                    return 0
                line = line.strip()
                if not line:
                    continue
                if line == '/help':
                    print(HELP)
                    continue
                if line == '/new':
                    try:
                        await session.new_conversation()
                    except SessionError as exc:
                        print(f'新会话创建失败：{exc}')
                        return 2
                    print(f'已开始新会话：{session.session_id}')
                    continue
                interrupted = False
                active = asyncio.create_task(session.run_turn(line, on_event=show_event))
                try:
                    turn = await active
                    show_turn(turn)
                except asyncio.CancelledError:
                    print('本轮已取消；已发生的文件或命令效果不会回滚。')
                except TimeoutError:
                    print('本轮超时，未完成；已发生的文件或命令效果不会回滚。')
                except SessionStorageError as exc:
                    print(f'本轮未完成：{exc}；已发生的文件或命令效果不会回滚。')
                    return 2
                except ContextBudgetError:
                    print('本轮上下文超出预算或无法安全压缩，未完成；请缩小输入或 /new。已发生的文件或命令效果不会回滚。')
                except Exception as exc:
                    from langchain.agents.middleware.model_call_limit import ModelCallLimitExceededError
                    from langchain.agents.middleware.tool_call_limit import ToolCallLimitExceededError
                    detail = '调用次数达到上限' if isinstance(exc, (ModelCallLimitExceededError, ToolCallLimitExceededError, SummaryCallLimitError)) else '执行失败'
                    print(f'本轮{detail}，未完成；已发生的文件或命令效果不会回滚。')
                finally:
                    active = None
        finally:
            loop.remove_reader(sys.stdin.fileno())
            if console.handle is not None:
                console.handle.cancel()
            loop.remove_signal_handler(signal.SIGINT)
            signal.signal(signal.SIGINT, old_handler)
