"""Local native loaders in a disposable process; explicit loader granularity."""
import asyncio
import json
import os
from pathlib import Path
import tempfile
from langchain_core.documents import Document
from ai_assistant.knowledge import IngestionError
from ai_assistant.knowledge.diagnostics import MAX_BYTES, bounded


def require_content(documents, code, message):
    """过滤空白文档；没有有效正文时返回可诊断的入库错误。"""
    docs = list(documents)
    valid = [doc for doc in docs if doc.page_content.strip()]
    if not valid:
        raise IngestionError(code, message, f'input_count={len(docs)}; nonempty_count=0')
    return valid




# 子进程错误码对应的用户可读摘要，详细信息单独进入诊断。
MESSAGES = {
    'parser_dependency_error': '本地解析依赖缺失或不兼容，请查看诊断并安装对应依赖。',
    'parser_resource_missing': '本地解析资源缺失，请准备 NLTK、OCR 或其他所需离线资源。',
    'parser_network_blocked': '本地解析器尝试联网，已阻止；请使用完整的离线资源。',
    'parser_timeout': '本地解析超时。',
    'parser_process_failed': '本地解析子进程异常退出，请查看诊断。',
    'parser_protocol_error': '本地解析结果文件无效，请查看诊断。',
    'pdf_encrypted': 'PDF 已加密，暂不支持读取。',
    'docx_invalid': 'DOCX 已损坏、加密或不是有效的 Word 文档包。',
    'docx_too_large': 'DOCX 解压总量超过 100 MiB。',
    'markdown_invalid': 'Markdown 必须是 UTF-8 文本，不能是二进制附件。',
    'pdf_invalid': 'PDF 已损坏或格式无效。',
    'text_encoding': '文本必须是 UTF-8 或带 BOM 的 UTF-16。',
    'text_binary': '文件包含二进制内容，不能按文本读取。',
    'json_invalid': 'JSON 文件语法无效。',
    'csv_invalid': 'CSV/TSV 记录格式无效或字段过大。',
    'attachment_format_unsupported': '不支持的附件格式。',
    'parser_failed': '本地解析失败，请查看诊断中的具体原因。',
}


class ParserResourceMissing(RuntimeError):
    pass


class ParserNetworkBlocked(RuntimeError):
    pass


def classify_parser_error(error):
    """将常见解析异常归类，其他异常统一视为解析失败。"""
    if isinstance(error, ParserResourceMissing) or type(error) is LookupError:
        return 'parser_resource_missing'
    if isinstance(error, ParserNetworkBlocked):
        return 'parser_network_blocked'
    if isinstance(error, ImportError):
        return 'parser_dependency_error'
    return 'parser_failed'


def parser_diagnostic(detail, stderr):
    """限制异常详情与标准错误的总长度，避免诊断无限增长。"""
    # Keep the exception tail even when a parser floods stderr.
    detail = bounded(detail, MAX_BYTES // 2)
    prefix = detail + '\nstderr:\n'
    return prefix + bounded(stderr, MAX_BYTES - len(prefix.encode('utf-8')))


async def _stderr_tail(stream):
    """并发读取并保留受限长度的标准错误尾部。"""
    tail = bytearray()
    truncated = False
    while data := await stream.read(8192):
        tail.extend(data)
        if len(tail) > MAX_BYTES:
            del tail[:-MAX_BYTES]
            truncated = True
    text = tail.decode('utf-8', errors='replace')
    return bounded(('[stderr 已截断]\n' if truncated else '') + text)


async def _stop_parser(process):
    """终止解析进程组并等待子进程退出。"""
    import signal
    # Kill the group even if its leader exited, so descendants cannot hold stderr open.
    try:
        os.killpg(process.pid, signal.SIGKILL)
    except ProcessLookupError:
        pass
    await process.wait()


async def parse_snapshot(path: Path, *, timeout: float, attachment=False,
                         granularity: str | None = None):
    """启动解析子进程并读取结果；超时或取消时等待进程组清理。"""
    if granularity is None:
        granularity = 'read' if attachment else 'ingest'
    if granularity not in {'read', 'ingest'}:
        raise ValueError('granularity must be read or ingest')
    # 独立解释器隔离解析资源，并允许超时终止。
    with tempfile.TemporaryDirectory(prefix='rag-parse-') as directory:
        output = Path(directory) / 'documents.json'
        env = dict(os.environ, HF_HUB_OFFLINE='1',
                   TRANSFORMERS_OFFLINE='1', UNSTRUCTURED_DOWNLOAD_NLTK='False')
        process = await asyncio.create_subprocess_exec(
            __import__('sys').executable, '-m', 'ai_assistant.knowledge.loading', str(path), str(output), granularity,
            env=env, stdout=asyncio.subprocess.DEVNULL, stderr=asyncio.subprocess.PIPE,
            start_new_session=True)

        # 同时消费 stderr，避免管道写满造成子进程阻塞。
        stderr_task = asyncio.create_task(_stderr_tail(process.stderr))
        try:
            try:
                async with asyncio.timeout(timeout):
                    code = await process.wait()
                    stderr = await asyncio.shield(stderr_task)
            except TimeoutError:
                await _stop_parser(process)
                stderr = await stderr_task
                raise IngestionError('parser_timeout', MESSAGES['parser_timeout'], stderr) from None
            if code:
                # Normal child failures have a structured envelope; signals/crashes may not.
                error_code, detail = 'parser_process_failed', f'exit_code={code}\n'
                try:
                    payload = json.loads(output.read_text(encoding='utf-8'))
                    if payload.get('status') == 'error' and payload.get('code') in MESSAGES:
                        error_code = payload['code']
                        detail += str(payload.get('diagnostic', ''))
                except (OSError, ValueError, AttributeError):
                    pass
                raise IngestionError(error_code, MESSAGES[error_code], parser_diagnostic(detail, stderr))

            # 只接受约定的结果信封，再恢复为 LangChain Document。
            try:
                payload = json.loads(output.read_text(encoding='utf-8'))
                if payload['status'] != 'ok' or not isinstance(payload['documents'], list):
                    raise ValueError('Invalid parser result envelope')
                return [Document(**item) for item in payload['documents']]
            except (OSError, ValueError, KeyError, TypeError) as exc:
                raise IngestionError('parser_protocol_error', MESSAGES['parser_protocol_error'],
                                     parser_diagnostic(f'{type(exc).__name__}: {exc}', stderr)) from None
        finally:
            await _stop_parser(process)
            # Drain the concurrently consumed pipe before removing temporary files.
            await stderr_task


def _run_local_loader(path):
    """兼容旧入口；统一解析器的 ingest 粒度。"""
    from ai_assistant.documents.parsing import parse_document
    return parse_document(path, granularity='ingest')


def _main():
    """子进程入口：解析后将成功或失败信封写入结果文件。"""
    import sys
    import traceback
    try:
        mode = sys.argv[3] if len(sys.argv) > 3 else 'ingest'
        if mode == 'attachment':
            mode = 'read'
        if mode not in {'read', 'ingest'}:
            mode = 'ingest'
        if mode == 'read':
            from ai_assistant.documents.parsing import parse_document
            docs = parse_document(sys.argv[1], granularity='read')
        else:
            docs = _run_local_loader(sys.argv[1])

        # 使用结构化结果和退出码供父进程判断，不输出文档正文到日志。
        payload = dict(status='ok', documents=[
            dict(page_content=d.page_content, metadata=d.metadata) for d in docs])
        code = 0
    except Exception as exc:
        payload = dict(status='error', code=exc.code if isinstance(exc, IngestionError) else classify_parser_error(exc), diagnostic=bounded(traceback.format_exc()))
        code = 1
    Path(sys.argv[2]).write_text(json.dumps(payload, ensure_ascii=False, default=str), encoding='utf-8')
    raise SystemExit(code)


if __name__ == '__main__':
    _main()
