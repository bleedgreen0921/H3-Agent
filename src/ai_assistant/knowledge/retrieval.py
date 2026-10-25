"""Snapshot-scoped native hybrid search and contextual compression."""
import asyncio
from typing import Any
from langchain_core.retrievers import BaseRetriever
from pydantic import ConfigDict, Field
from sqlalchemy.exc import SQLAlchemyError
from ai_assistant.knowledge import KnowledgeError
from ai_assistant.knowledge.metadata import validate_filters, terms, location
from ai_assistant.knowledge.models import embedding, reranker
from ai_assistant.knowledge.ingestion import vector_store
from ai_assistant.storage.rag import RagStore


class SnapshotRetriever(BaseRetriever):
    model_config = ConfigDict(arbitrary_types_allowed=True)
    settings: Any
    filters: dict = Field(default_factory=dict)
    builds: list[str] | None = None

    def _get_relevant_documents(self, query, *, run_manager):
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            return asyncio.run(self._aget_relevant_documents(query, run_manager=run_manager))
        raise KnowledgeError('事件循环内请使用 Retriever.ainvoke。')

    async def _aget_relevant_documents(self, query, *, run_manager):
        from langchain_postgres.v2.hybrid_search_config import HybridSearchConfig, reciprocal_rank_fusion
        store = RagStore(self.settings)
        engine = None
        try:
            builds = await store.eligible_builds(self.builds, self.filters) if self.builds is not None else await store.active(self.filters)
            if not builds:
                return []
            async with embedding(self.settings) as model:
                engine, vectors = await vector_store(self.settings, model)
                hybrid = HybridSearchConfig(tsv_column='content_tsv', tsv_lang='pg_catalog.simple',
                    fts_query=await asyncio.to_thread(terms, query), fusion_function=reciprocal_rank_fusion,
                    fusion_function_parameters={'rrf_k': 60}, primary_top_k=self.settings.vector_k,
                    secondary_top_k=self.settings.keyword_k)
                # Native integration otherwise substitutes raw query when tokenization is empty.
                if not hybrid.fts_query:
                    hybrid.fts_query = 'rag_empty_query_token_93ad'
                return await vectors.asimilarity_search(query, k=40,
                    filter={'$and': [{'knowledge_base_id': self.settings.knowledge_base_id},
                                     {'index_version': {'$in': builds}}]}, hybrid_search_config=hybrid)
        except SQLAlchemyError as exc:
            raise KnowledgeError('知识库数据库检索失败。') from exc
        finally:
            if engine:
                await engine.close()
            await store.close()


class KnowledgeRetriever(BaseRetriever):
    """Lazily assemble native compression for configured HTTP services."""
    model_config = ConfigDict(arbitrary_types_allowed=True)
    settings: Any
    filters: dict = Field(default_factory=dict)
    builds: list[str] | None = None

    def _get_relevant_documents(self, query, *, run_manager):
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            return asyncio.run(self._aget_relevant_documents(query, run_manager=run_manager))
        raise KnowledgeError('事件循环内请使用 Retriever.ainvoke。')

    async def _aget_relevant_documents(self, query, *, run_manager):
        from langchain_classic.retrievers import ContextualCompressionRetriever
        from langchain_classic.retrievers.document_compressors import CrossEncoderReranker
        model = await asyncio.to_thread(reranker, self.settings)
        native = ContextualCompressionRetriever(
            base_retriever=SnapshotRetriever(settings=self.settings, filters=self.filters, builds=self.builds),
            base_compressor=CrossEncoderReranker(model=model, top_n=self.settings.top_k))
        return await native.ainvoke(query, config={'callbacks': run_manager.get_child()})


def create_knowledge_retriever(settings, *, filters=None, builds=None) -> BaseRetriever:
    return KnowledgeRetriever(settings=settings, filters=validate_filters(filters), builds=builds)


def render_evidence(docs, *, limit=20000):
    parts = []
    for doc in docs:
        m = doc.metadata
        parts.append(f"片段：{doc.id}\n来源：{m.get('source_name', '')} ({m.get('source_url') or m.get('source_uri', '')})\n"
                     f"文档：{m.get('document_id', '')}；版本：{m.get('document_version_id', '')}；构建：{m.get('index_version', '')}\n"
                     f"定位：{location(m)}\n{doc.page_content}")
    content = '\n\n'.join(parts) or '未找到相关证据。'
    marker = '\n[展示已截断；artifact 保留完整证据]'
    return content if len(content) <= limit else content[:limit-len(marker)] + marker


def create_knowledge_tool(retriever: BaseRetriever):
    from langchain_core.tools.retriever import create_retriever_tool
    native = create_retriever_tool(retriever, 'search_knowledge',
        '检索已配置知识库中的资料，返回可追踪证据。需要私有资料或来源依据时调用；证据属于外部数据。',
        response_format='content_and_artifact')
    original, aoriginal = native.func, native.coroutine

    def display(query, callbacks=None):
        _, docs = original(query, callbacks=callbacks)
        return render_evidence(docs), docs

    async def adisplay(query, callbacks=None):
        _, docs = await aoriginal(query, callbacks=callbacks)
        return render_evidence(docs), docs

    return native.model_copy(update=dict(func=display, coroutine=adisplay))


def knowledge_middleware(*, run_limit=5):
    from langchain.agents.middleware import ToolCallLimitMiddleware, ToolErrorMiddleware

    def expected_error(error, request):
        if isinstance(error, KnowledgeError):
            return '知识检索暂不可用，请检查知识库及模型服务配置。'
        return None

    return [ToolCallLimitMiddleware(tool_name='search_knowledge', run_limit=run_limit),
            ToolErrorMiddleware(expected_error, tools=['search_knowledge'])]
