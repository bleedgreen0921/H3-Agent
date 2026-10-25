import pytest
pytest.importorskip("langchain_postgres")
import asyncio
from contextlib import asynccontextmanager
from unittest.mock import patch
from types import SimpleNamespace
import pytest
from langchain_core.documents import Document
from sqlalchemy.exc import OperationalError
from ai_assistant.knowledge import KnowledgeError
from ai_assistant.knowledge.retrieval import SnapshotRetriever


def test_concurrent_queries_have_independent_hybrid_configs_and_shared_scope():
    async def scenario():
        configs = []
        class Store:
            def __init__(self, settings): pass
            async def active(self, filters):
                assert filters == {'tags': ['中文']}
                return ['fixed-build']
            async def close(self): pass
        class Engine:
            async def close(self): pass
        class Vectors:
            async def asimilarity_search(self, query, **kwargs):
                configs.append(kwargs['hybrid_search_config'])
                await asyncio.sleep(0.01)
                assert kwargs['filter'] == {'$and': [{'knowledge_base_id': 'kb'}, {'index_version': {'$in': ['fixed-build']}}]}
                assert kwargs['hybrid_search_config'].fts_query != ''
                return [Document(page_content=query)]
        async def vectors(*args): return Engine(), Vectors()
        @asynccontextmanager
        async def embeddings(c):
            yield object()
        settings = SimpleNamespace(knowledge_base_id='kb', vector_k=20, keyword_k=20)
        retriever = SnapshotRetriever(settings=settings, filters={'tags': ['中文']})
        with patch('ai_assistant.knowledge.retrieval.RagStore', Store), \
             patch('ai_assistant.knowledge.retrieval.embedding', embeddings), \
             patch('ai_assistant.knowledge.retrieval.vector_store', vectors):
            results = await asyncio.gather(retriever.ainvoke('中文检索'), retriever.ainvoke('another_query'))
        assert results[0][0].page_content == '中文检索'
        assert configs[0] is not configs[1]
        assert configs[0].fusion_function_parameters is not configs[1].fusion_function_parameters
        assert configs[0].fts_query != configs[1].fts_query
    asyncio.run(scenario())


def test_database_error_does_not_become_empty_result():
    class BrokenStore:
        def __init__(self, settings): pass
        async def active(self, filters):
            raise OperationalError('SELECT', {}, Exception('private connection details'))
        async def close(self): pass
    with patch('ai_assistant.knowledge.retrieval.RagStore', BrokenStore):
        with pytest.raises(KnowledgeError, match='数据库检索失败'):
            asyncio.run(SnapshotRetriever(settings=object()).ainvoke('q'))
