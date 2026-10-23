import pytest
pytest.importorskip("langchain_postgres")
import asyncio
from pathlib import Path
import pytest
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from ai_assistant.knowledge.metadata import normalize, location, validate_filters, terms
from ai_assistant.knowledge.retrieval import create_knowledge_tool, render_evidence
from ai_assistant.knowledge.ingestion import snapshot
from ai_assistant.workspace.files import ReadOnlyWorkspace, WorkspaceError

IDENTITY = dict(knowledge_base_id='kb', document_id='d', document_version_id='v',
                source_uri='file:///sample.pdf', source_name='sample.pdf', media_type='application/pdf',
                parse_config_fingerprint='hash', index_version='b', tags=['中文'])


def test_explicit_semantic_mapping():
    a = normalize(Document(page_content='a\n b', metadata={'page_number': 2}), identity=IDENTITY, parser='unstructured')
    b = normalize(Document(page_content='different', metadata={'page': 1}), identity=IDENTITY,
                  parser='pypdf', mapping={'page_start': 'page', 'page_end': 'page'}, zero_based=('page',))
    assert a.metadata['page_start'] == b.metadata['page_start'] == 2
    assert a.page_content == 'a\n b'
    assert a.metadata['parser_metadata'] == {'page_number': 2}
    missing = normalize(Document(page_content='x', metadata={'page': 7}), identity=IDENTITY, parser='other')
    assert location(missing.metadata) == '未提供精确定位'
    assert 'page_start' not in missing.metadata
    with pytest.raises(ValueError):
        normalize(Document(page_content='', metadata={'page_number': '2'}), identity=IDENTITY, parser='unstructured')
    with pytest.raises(ValueError):
        normalize(Document(page_content=''), identity=IDENTITY | {'tags': 'tag'}, parser='other')


def test_filter_whitelist():
    with pytest.raises(ValueError):
        validate_filters({'index_version': 'escape'})
    with pytest.raises(ValueError):
        validate_filters({'tags': 'wrong'})
    assert validate_filters({'tags': ['x']}) == {'tags': ['x']}
    assert '检索' in terms('中文检索 search_knowledge HTTPServer')
    assert 'search_knowledge' in terms('中文检索 search_knowledge HTTPServer')


class Evidence(BaseRetriever):
    def _get_relevant_documents(self, query, *, run_manager):
        return [Document(id='chunk-1', page_content='文' * 25000, metadata=IDENTITY)]


def test_native_tool_artifact_and_display_budget():
    tool = create_knowledge_tool(Evidence())
    for result in [tool.invoke({'type': 'tool_call', 'id': 'call1', 'name': tool.name, 'args': {'query': 'x'}}),
                   asyncio.run(tool.ainvoke({'type': 'tool_call', 'id': 'call2', 'name': tool.name, 'args': {'query': 'x'}}))]:
        assert len(result.content) <= 20000
        assert '展示已截断' in result.content
        assert result.artifact[0].id == 'chunk-1'
        assert len(result.artifact[0].page_content) == 25000
        assert result.artifact[0].metadata == IDENTITY


def test_immutable_binary_snapshot(tmp_path):
    workspace = tmp_path / 'workspace'
    workspace.mkdir()
    source = workspace / 'a.pdf'
    source.write_bytes(b'%PDF\x00test')
    policy = ReadOnlyWorkspace(workspace)
    uri, digest, path, name = snapshot(policy, 'a.pdf', tmp_path / 'managed')
    assert uri == source.as_uri() and name == 'a.pdf'
    source.write_bytes(b'changed')
    assert path.read_bytes() == b'%PDF\x00test'
    assert snapshot(policy, 'a.pdf', tmp_path / 'managed')[1] != digest
    with pytest.raises(WorkspaceError):
        snapshot(policy, '../secret', tmp_path / 'managed')
    (workspace / 'link').symlink_to(source)
    with pytest.raises(WorkspaceError):
        snapshot(policy, 'link', tmp_path / 'managed')
