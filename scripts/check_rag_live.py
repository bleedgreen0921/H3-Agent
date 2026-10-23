"""Explicit local E2E check. Uses a dedicated configured knowledge base; no LLM API.

python scripts/check_rag_live.py --config rag-check.toml --sample sample.csv --query 星河发布日
Run `python -m ai_assistant rag init --config rag-check.toml` first.
"""
import argparse
import asyncio
from pathlib import Path
from ai_assistant.config import load_config
from ai_assistant.knowledge.ingestion import submit, run_once
from ai_assistant.knowledge.retrieval import create_knowledge_retriever, render_evidence
from ai_assistant.storage.rag import RagStore
from ai_assistant.workspace.files import ReadOnlyWorkspace


async def check(args):
    config = load_config(args.config)
    c = config.knowledge
    if c is None or not c.knowledge_base_id.startswith('rag-check-'):
        raise ValueError('Use a dedicated knowledge_base_id beginning with rag-check-')
    base = config.config_path.parent
    c = c.model_copy(update={key: (base / getattr(c, key)).resolve()
                            for key in ('materials_dir', 'diagnostics_dir', 'embedding_model', 'reranker_model')})
    workspace = ReadOnlyWorkspace((base / config.workspace.root).resolve(),
        exclude_patterns=config.workspace.exclude_patterns, config_path=config.config_path)
    store = RagStore(c)
    try:
        job = await submit(store, workspace, args.sample, tags=['rag-e2e'])
        duplicate = await submit(store, workspace, args.sample, tags=['rag-e2e'])
        assert duplicate == job, 'Duplicate submission did not reuse task'
        print('任务：', job, flush=True)
        while (status := await store.status(job))['state'] in ('queued', 'running'):
            await run_once(store)
            await asyncio.sleep(0.1)
        assert status['state'] == 'succeeded', status
        docs = await create_knowledge_retriever(c, filters={'tags': ['rag-e2e']}).ainvoke(args.query)
        assert docs and all('rag-e2e' in doc.metadata['tags'] for doc in docs)
        assert all(doc.id and doc.metadata['document_version_id'] for doc in docs)
        assert status['document_id'] == docs[0].metadata['document_id']
        assert status['document_version_id'] == docs[0].metadata['document_version_id']
        assert status['published_build_id'] == status['active_build_id'] == status['latest_build_id']
        assert not await create_knowledge_retriever(c, filters={'source_name': 'nonexistent-e2e-source'}).ainvoke(args.query)
        print(render_evidence(docs), flush=True)
        print('通过：不可变入库、任务发布、本地混合检索与重排、来源追踪及过滤。', flush=True)
    finally:
        await store.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True)
    parser.add_argument('--sample', required=True, help='Existing sample relative to configured workspace')
    parser.add_argument('--query', required=True)
    asyncio.run(check(parser.parse_args()))
