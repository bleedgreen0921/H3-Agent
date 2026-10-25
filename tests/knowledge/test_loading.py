import pytest
pytest.importorskip("langchain_postgres")
import asyncio
import sys
from unittest.mock import patch
import pytest
from langchain_core.documents import Document
from langchain_core.document_loaders import BaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from ai_assistant.knowledge import KnowledgeError
from ai_assistant.knowledge.loading import prepare_documents, parse_snapshot

IDENTITY = dict(knowledge_base_id='kb', document_id='d', document_version_id='v',
                source_uri='file:///a', source_name='a', media_type='text/plain',
                parse_config_fingerprint='f', index_version='b')


class Loader(BaseLoader):
    def lazy_load(self):
        yield Document(page_content='abcdef\nghijkl', metadata={'line': 0})


def test_native_loader_granularity_and_metadata_survive_splitting():
    splitter = RecursiveCharacterTextSplitter(chunk_size=5, chunk_overlap=1, add_start_index=True)
    chunks = prepare_documents(Loader(), identity=IDENTITY, parser='declared', granularity='document',
        splitter=splitter, mapping={'line_start': 'line'}, zero_based=('line',))
    assert len(chunks) > 1
    assert all(c.metadata['line_start'] == 1 and c.metadata['document_id'] == 'd' for c in chunks)
    assert all(c.metadata['parser_metadata'] == {'line': 0} for c in chunks)
    unchanged = prepare_documents(Loader(), identity=IDENTITY, parser='declared', granularity='chunks', splitter=splitter)
    assert len(unchanged) == 1 and unchanged[0].page_content == 'abcdef\nghijkl'
    with pytest.raises(ValueError):
        prepare_documents(Loader(), identity=IDENTITY, parser='declared', granularity='unknown', splitter=splitter)


@pytest.mark.parametrize('cancel', [False, True])
def test_parser_process_is_reaped_on_timeout_or_cancellation(tmp_path, cancel):
    async def scenario():
        original = asyncio.create_subprocess_exec
        processes = []
        async def slow(*args, **kwargs):
            assert kwargs['env']['HF_HUB_OFFLINE'] == '1'
            process = await original(sys.executable, '-c', 'import time; time.sleep(30)', **kwargs)
            processes.append(process)
            return process
        with patch('ai_assistant.knowledge.loading.asyncio.create_subprocess_exec', slow):
            task = asyncio.create_task(parse_snapshot(tmp_path/'unused', timeout=0.2 if not cancel else 10))
            if cancel:
                await asyncio.sleep(0.1)
                task.cancel()
                with pytest.raises(asyncio.CancelledError):
                    await task
            else:
                with pytest.raises(KnowledgeError, match='解析超时'):
                    await task
        assert processes and processes[0].returncode is not None
    asyncio.run(scenario())


@pytest.mark.parametrize('documents', [[], [Document(page_content=' \t\n\u3000')]])
def test_empty_document_fails_before_loading_model(tmp_path, documents):
    from ai_assistant.knowledge import IngestionError
    from ai_assistant.knowledge.ingestion import build
    from hashlib import sha256
    from types import SimpleNamespace
    from unittest.mock import AsyncMock, Mock
    path = tmp_path/'input'
    path.write_bytes(b'snapshot')
    config = SimpleNamespace(fingerprint=lambda: 'config', parse_timeout=10)
    job = dict(config_hash='config', snapshot=path, digest=sha256(b'snapshot').hexdigest())
    model = Mock(side_effect=AssertionError('must not load model'))
    with patch('ai_assistant.knowledge.ingestion.parse_snapshot', AsyncMock(return_value=documents)), \
         patch('ai_assistant.knowledge.ingestion.embedding', model):
        with pytest.raises(IngestionError) as caught:
            asyncio.run(build(SimpleNamespace(config=config), job))
    assert caught.value.code == 'empty_document'
    model.assert_not_called()


def test_blank_documents_and_chunks_are_ignored_without_rewriting():
    class Mixed(BaseLoader):
        def lazy_load(self):
            yield Document(page_content=' \n')
            yield Document(page_content='  正文\n保留  ', metadata={'page_number': 2})
    class Splitter:
        def split_documents(self, docs):
            assert len(docs) == 1 and docs[0].page_content == '  正文\n保留  '
            return [Document(page_content='\t'), *docs]
    docs = prepare_documents(Mixed(), identity=IDENTITY, parser='unstructured', granularity='elements', splitter=Splitter())
    assert len(docs) == 1 and docs[0].page_content == '  正文\n保留  '
    assert docs[0].metadata['chunk_index'] == 0 and docs[0].metadata['page_start'] == 2
    Splitter.split_documents = lambda self, docs: []
    with pytest.raises(KnowledgeError) as caught:
        prepare_documents(Mixed(), identity=IDENTITY, parser='unstructured', granularity='elements', splitter=Splitter())
    assert caught.value.code == 'empty_chunks'


@pytest.mark.parametrize('mode,expected', [
    ('dependency', 'parser_dependency_error'), ('resource', 'parser_resource_missing'),
    ('network', 'parser_network_blocked'), ('unknown', 'parser_failed'),
    ('crash', 'parser_process_failed'), ('invalid', 'parser_protocol_error'),
    ('missing', 'parser_protocol_error')])
def test_parser_failure_envelopes_and_bounded_stderr(tmp_path, mode, expected):
    import json
    from ai_assistant.knowledge import IngestionError
    from ai_assistant.knowledge.diagnostics import MAX_BYTES
    async def scenario():
        original = asyncio.create_subprocess_exec
        async def child(*args, **kwargs):
            payload = dict(status='error', code=expected, diagnostic='private traceback')
            script = 'import pathlib,sys; sys.stderr.write("x"*200000+"STDERR_END"); '
            if mode not in ('crash', 'missing'):
                value = 'not-json' if mode == 'invalid' else json.dumps(payload)
                script += f'pathlib.Path({args[-1]!r}).write_text({value!r}); '
            script += f'sys.exit({1 if mode not in ("invalid", "missing") else 0})'
            return await original(sys.executable, '-c', script, **kwargs)
        with patch('ai_assistant.knowledge.loading.asyncio.create_subprocess_exec', child):
            with pytest.raises(IngestionError) as caught:
                await parse_snapshot(tmp_path/'unused', timeout=10)
        assert caught.value.code == expected
        assert len(caught.value.diagnostic_text.encode()) <= MAX_BYTES
        assert 'STDERR_END' in caught.value.diagnostic_text
        assert 'private traceback' not in str(caught.value)
        if mode in ('dependency', 'resource', 'network', 'unknown'):
            assert 'private traceback' in caught.value.diagnostic_text
    asyncio.run(scenario())


def test_child_error_classification_and_structured_output(tmp_path):
    import json
    import sys
    from ai_assistant.knowledge.loading import _main, ParserResourceMissing, ParserNetworkBlocked
    from ai_assistant.knowledge.loading import classify_parser_error
    for error, code in [(ModuleNotFoundError('missing parser'), 'parser_dependency_error'),
                        (LookupError('missing punkt'), 'parser_resource_missing'),
                        (ParserResourceMissing('offline resources'), 'parser_resource_missing'),
                        (ParserNetworkBlocked('network'), 'parser_network_blocked'),
                        (ValueError('bad document'), 'parser_failed'),
                        (KeyError('unexpected field'), 'parser_failed')]:
        assert classify_parser_error(error) == code
        output = tmp_path/'result.json'
        with patch.object(sys, 'argv', ['loader', 'input', str(output)]), \
             patch('ai_assistant.knowledge.loading._run_local_loader', side_effect=error), \
             patch.object(sys, 'platform', 'test'):
            with pytest.raises(SystemExit) as caught:
                _main()
        assert caught.value.code == 1
        payload = json.loads(output.read_text())
        assert payload['code'] == code and type(error).__name__ in payload['diagnostic']
