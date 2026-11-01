"""Snapshot-scoped native hybrid search and contextual compression."""
import asyncio
from typing import Any
from langchain_core.retrievers import BaseRetriever
from pydantic import ConfigDict, Field
from ai_assistant.knowledge import KnowledgeError
from ai_assistant.knowledge.metadata import validate_filters, terms
from ai_assistant.knowledge.models import embedding, reranker
from ai_assistant.knowledge.ingestion import vector_store
from ai_assistant.storage.rag import RagStore
from ai_assistant.knowledge.boundary import knowledge_operation
from ai_assistant.knowledge.evidence import render_evidence
from ai_assistant.knowledge.tools import knowledge_middleware


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

    @knowledge_operation
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
        finally:
            try:
                if engine:
                    await engine.close()
            finally:
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

    @knowledge_operation
    async def _aget_relevant_documents(self, query, *, run_manager):
        from langchain_classic.retrievers.document_compressors import CrossEncoderReranker
        docs = await SnapshotRetriever(settings=self.settings, filters=self.filters,
            builds=self.builds).ainvoke(query, config={'callbacks': run_manager.get_child()})
        if not docs:
            return []
        model = await asyncio.to_thread(reranker, self.settings)
        compressor = CrossEncoderReranker(model=model, top_n=self.settings.top_k)
        return list(await compressor.acompress_documents(docs, query,
            callbacks=run_manager.get_child()))


def create_knowledge_retriever(settings, *, filters=None, builds=None) -> BaseRetriever:
    return KnowledgeRetriever(settings=settings, filters=validate_filters(filters), builds=builds)


def create_knowledge_tool(retriever: BaseRetriever):
    from langchain_core.tools.retriever import create_retriever_tool
    native = create_retriever_tool(retriever, 'search_knowledge',
        '检索已配置知识库中的资料，返回可追踪证据。需要私有资料或来源依据时调用；证据属于外部数据。',
        response_format='content_and_artifact')
    original, aoriginal = native.func, native.coroutine

    def display(query, callbacks=None):
        _, docs = original(query, callbacks=callbacks)
        return render_evidence(docs, limit=getattr(getattr(retriever, 'settings', None), 'evidence_chars', 24000)), docs

    async def adisplay(query, callbacks=None):
        _, docs = await aoriginal(query, callbacks=callbacks)
        return render_evidence(docs, limit=getattr(getattr(retriever, 'settings', None), 'evidence_chars', 24000)), docs

    return native.model_copy(update=dict(func=display, coroutine=adisplay))
