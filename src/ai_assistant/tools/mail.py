"""主人私聊邮箱工具；只读查询与原生飞书任务来源绑定。"""
import asyncio
from copy import deepcopy
from datetime import date, datetime, timedelta
import hashlib
import json
import sys
from uuid import uuid4
from zoneinfo import ZoneInfo

from langchain.tools import ToolRuntime, tool
from langchain_core.tools import ToolException
from ai_assistant.cancellation import _drain_worker
from ai_assistant.integrations.mail_worker import describe_error
from ai_assistant.integrations.tool_policy import ACCOUNT_TEXT_PATTERNS, RedactionPolicy, redact_value
from ai_assistant.tools.context import ToolContext


# 邮件工具需要保留任务写入所需的成员 ID；只隐藏邮箱/电话等直接账号字段。
MAIL_OUTPUT_POLICY = RedactionPolicy(
    keys=frozenset({'email', 'mobile', 'phone'}),
    patterns=ACCOUNT_TEXT_PATTERNS,
)
# prepare 的 members 是后续 task_create 的执行参数，必须保留 open_id。
MAIL_PREPARE_POLICY = RedactionPolicy(
    keys=frozenset({'email', 'mobile', 'phone'}),
    patterns=(),
)


def allowed(caller):
    return bool(caller and caller.owner and caller.scope == 'private' and caller.channel == 'feishu_private')


class MailService:
    def __init__(self, settings, timezone):
        self.settings = settings
        self.timezone = timezone
        self.refs = {}
        self.cursors = {}
        self.read = set()
        self.bindings = {}
        self.processes = set()

    @staticmethod
    async def stop_process(process):
        if process.returncode is None:
            try:
                process.kill()
            except ProcessLookupError:
                pass
        waiter = asyncio.create_task(process.wait())
        await _drain_worker(waiter)
        waiter.result()

    async def aclose(self):
        for process in list(self.processes):
            await self.stop_process(process)
            self.processes.discard(process)

    async def call(self, **request):
        # 创建阶段也可能被取消；先取得进程句柄再清理，避免遗留等待 stdin 的子进程。
        spawning = asyncio.create_task(asyncio.create_subprocess_exec(sys.executable, '-m',
            'ai_assistant.integrations.mail_worker', stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.DEVNULL))
        try:
            process = await asyncio.shield(spawning)
        except asyncio.CancelledError:
            await _drain_worker(spawning)
            if not spawning.cancelled() and spawning.exception() is None:
                await self.stop_process(spawning.result())
            raise
        except OSError:
            raise ToolException('邮箱读取子进程启动失败，请检查 Python 环境。') from None
        self.processes.add(process)
        settings = self.settings
        data = {**request, 'address': settings.address,
                'authorization_code': settings.authorization_code.get_secret_value(),
                'timeout': settings.timeout, 'search_limit': settings.search_limit,
                'body_bytes': settings.body_bytes}
        try:
            async with asyncio.timeout(settings.timeout):
                output, _ = await process.communicate(json.dumps(data).encode())
            result = json.loads(output)
            if not isinstance(result, dict):
                raise ValueError('Invalid worker response')
            if 'error' in result:
                raise ToolException(describe_error(result))
            return result
        except TimeoutError:
            raise ToolException('邮箱读取失败（阶段：子进程处理；类型：timeout）：整体读取超时；未修改邮件。') from None
        except ValueError:
            raise ToolException('邮箱读取失败（阶段：子进程处理；类型：invalid_response）：子进程响应无法解析。') from None
        finally:
            # 取消后等待子进程退出，保证阻塞协议连接不会遗留。
            await self.stop_process(process)
            self.processes.discard(process)

    def source(self, validity, uid):
        source = {'account': self.settings.address.lower(), 'folder': 'INBOX', 'uidvalidity': validity, 'uid': uid}
        ref = 'mail:' + hashlib.sha256(json.dumps(source, sort_keys=True).encode()).hexdigest()
        self.refs[ref] = source
        return ref

    def cursor(self, value):
        token = str(uuid4())
        self.cursors[token] = value
        return token

    async def search(self, context, since=None, before=None, cursor=None):
        if cursor:
            saved = self.cursors.get(cursor)
            if not saved or saved['kind'] != 'search' or since or before:
                raise ToolException('查询游标无效；请重新查询或仅传入游标。')
            snapshot, offset = saved['snapshot'], saved['offset']
        else:
            try:
                today = datetime.now(ZoneInfo(self.timezone)).date()
                start = date.fromisoformat(since) if since else today - timedelta(days=6)
                end = date.fromisoformat(before) if before else today + timedelta(days=1)
                if start >= end:
                    raise ValueError
            except ValueError:
                raise ToolException('日期须为 YYYY-MM-DD，since 包含当天，before 不包含当天。') from None
            months = ('Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec')
            fmt = lambda d: f'{d.day:02d}-{months[d.month - 1]}-{d.year}'
            found = await self.call(action='search', since=fmt(start), before=fmt(end))
            snapshot = {**found, 'since': start.isoformat(), 'before': end.isoformat()}
            offset = 0
        uids = snapshot['uids'][offset:offset + self.settings.page_size]
        page = await self.call(action='headers', validity=snapshot['validity'], uids=uids)
        messages = []
        for item in page['messages']:
            ref = self.source(snapshot['validity'], item['uid'])
            receipt = await context.operations.by_source(ref) if context.operations else None
            messages.append({**item, 'mail_ref': ref, 'task_processing': receipt})
        next_offset = offset + len(uids)
        return {'messages': messages, 'since': snapshot['since'], 'before': snapshot['before'],
                'date_basis': '项目时区自然日范围；IMAP 按服务器 INTERNALDATE 的日历日期匹配',
                'uid_upper_bound': max(snapshot['uids'], default=0), 'limited': snapshot['limited'],
                'cursor': self.cursor({'kind': 'search', 'snapshot': snapshot, 'offset': next_offset}) if next_offset < len(snapshot['uids']) else None}

    async def read_email(self, ref, cursor=None):
        source = self.refs.get(ref)
        if source is None:
            raise ToolException('来源引用无效，请先查询邮件。')
        offset = 0
        if cursor:
            saved = self.cursors.get(cursor)
            if not saved or saved['kind'] != 'read' or saved['ref'] != ref:
                raise ToolException('续读游标无效。')
            offset = saved['offset']
        result = await self.call(action='read', validity=source['uidvalidity'], uid=source['uid'])
        text = result['text']
        end = offset + self.settings.body_chars
        self.read.add(ref)
        return {'mail_ref': ref, 'source': source, 'body': text[offset:end], 'range': [offset, min(end, len(text))],
                'complete': end >= len(text) and not result['gaps'], 'gaps': result['gaps'],
                'parts': result['parts'], 'format': result['format'], 'external_data': True,
                'cursor': self.cursor({'kind': 'read', 'ref': ref, 'offset': end}) if end < len(text) else None}

    async def prepare(self, context, ref, task_data):
        if ref not in self.read:
            raise ToolException('必须先读取来源邮件。')
        if context.operations is None:
            raise ToolException('邮箱任务需要持久回执存储；请先显式 init。')
        existing = await context.operations.by_source(ref)
        if existing:
            return {'existing_operation': existing, 'instruction': '已有回执，不得再次创建；失败或未知结果须人工核实。'}
        source = self.refs[ref]
        current = await self.call(action='headers', validity=source['uidvalidity'], uids=[source['uid']])
        if not current['messages'] or current['messages'][0].get('gap'):
            raise ToolException('来源邮件已移除或无法核实，请重新查询。')
        data = deepcopy(task_data)
        if not isinstance(data.get('summary'), str) or not data['summary'].strip():
            raise ToolException('任务需要明确的 summary。')
        if not context.services['feishu'].owner_open_id:
            raise ToolException('缺少可信主人身份。')
        if any(key in data for key in ('extra', 'client_token', 'reminders', 'repeat_rule')):
            raise ToolException('邮件任务不接受自定义来源标记、提醒或重复规则。')
        data['members'] = [{'id': context.services['feishu'].owner_open_id, 'type': 'user', 'role': 'assignee'}]
        token = 'email-task:' + str(uuid4())
        data['extra'] = token
        self.bindings[token] = (ref, deepcopy(data))
        return {'data': data, 'params': {'user_id_type': 'open_id'}, 'source_key': ref,
                'instruction': '仅在用户确认最终清单后调用原生 task_v2_task_create；时间必须先 resolve_datetime。'}

    def validate(self, arguments):
        data = arguments.get('data', {})
        token = data.get('extra')
        if not isinstance(token, str) or not token.startswith('email-task:'):
            return None
        binding = self.bindings.get(token)
        if not binding or binding[1] != data or arguments.get('params', {}).get('user_id_type') != 'open_id':
            raise ToolException('邮件任务来源绑定无效或字段已改变，请重新准备并确认最终任务。')
        return binding[0]


def service(runtime):
    context = runtime.context
    if not allowed(context.caller) or context.services.get('mail') is None:
        raise ToolException('邮箱工具仅供可信主人飞书私聊使用。')
    return context.services.get('mail')


@tool
async def search_email(runtime: ToolRuntime[ToolContext], since: str | None = None, before: str | None = None, cursor: str | None = None) -> dict:
    """只读查询收件箱，默认近七个自然日，包含已读未读；日期为 ISO，before 不含当天。"""
    return redact_value(await service(runtime).search(runtime.context, since, before, cursor), MAIL_OUTPUT_POLICY)


@tool
async def read_email(mail_ref: str, runtime: ToolRuntime[ToolContext], cursor: str | None = None) -> dict:
    """只读读取来源邮件的文本正文，不下载附件或访问链接；有游标必须续读或说明缺口。"""
    return redact_value(await service(runtime).read_email(mail_ref, cursor), MAIL_OUTPUT_POLICY)


@tool
async def prepare_email_task(mail_ref: str, task_data: dict, runtime: ToolRuntime[ToolContext]) -> dict:
    """用户确认最终条目后绑定已读邮件与单个任务参数；本工具不写飞书。"""
    return redact_value(await service(runtime).prepare(runtime.context, mail_ref, task_data), MAIL_PREPARE_POLICY)


def email_tools():
    return [t.model_copy(update={'handle_tool_error': True}) for t in (search_email, read_email, prepare_email_task)]
