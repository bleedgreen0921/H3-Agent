"""Reject invalid local embeddings before any vector database call."""

from types import SimpleNamespace
import pytest
from ai_assistant.knowledge.models import ServiceEmbeddings
from ai_assistant.knowledge import IngestionError


@pytest.mark.parametrize(
    "bad",
    [
        [1.0] * 1023,
        [0.0] * 1024,
        [float("nan")] + [1.0] * 1023,
        [float("inf")] + [1.0] * 1023,
    ],
)
def test_invalid_query_and_document_vectors_are_rejected(bad):
    model = ServiceEmbeddings(
        SimpleNamespace(
            embed_query=lambda text: bad, embed_documents=lambda texts: [bad]
        )
    )
    for invoke in (
        lambda: model.embed_query("fixture"),
        lambda: model.embed_documents(["fixture"]),
    ):
        with pytest.raises(IngestionError) as caught:
            invoke()
        assert caught.value.code == "embedding_invalid_output"


def test_valid_vectors_remain_unchanged():
    v = [1.0] + [0.0] * 1023
    model = ServiceEmbeddings(
        SimpleNamespace(embed_query=lambda text: v, embed_documents=lambda texts: [v])
    )
    assert model.embed_query("fixture") == v
    assert model.embed_documents(["fixture"]) == [v]


def test_embedding_service_index_order_and_rejection():
    from ai_assistant.knowledge.models import _ordered_embeddings
    from ai_assistant.knowledge import KnowledgeError

    rows = [{"index": 1, "embedding": [2.0]}, {"index": 0, "embedding": [1.0]}]
    assert _ordered_embeddings({"data": rows}, ["a", "b"])["data"][0]["embedding"] == [
        1.0
    ]
    for rows in [
        [{"index": 0}, {"index": 0}],
        [{"index": 0}],
        [{"index": -1}, {"index": 1}],
        [{"index": False}, {"index": 1}],
    ]:
        with pytest.raises(KnowledgeError):
            _ordered_embeddings({"data": rows}, ["a", "b"])


@pytest.mark.parametrize(
    "payload",
    [
        {"error": "unexpected endpoint"},
        {"results": []},
        {
            "results": [
                {"index": 0, "relevance_score": 1},
                {"index": 0, "relevance_score": 2},
            ]
        },
        {
            "results": [
                {"index": 0, "relevance_score": 1},
                {"index": 2, "relevance_score": 2},
            ]
        },
        {
            "results": [
                {"index": 0, "relevance_score": 1},
                {"index": 1, "relevance_score": float("nan")},
            ]
        },
    ],
)
def test_rerank_service_rejects_bad_response(monkeypatch, payload):
    import httpx
    from ai_assistant.knowledge.config import KnowledgeConfig
    from ai_assistant.knowledge.models import reranker
    from ai_assistant.knowledge import KnowledgeError

    client_type = httpx.Client
    monkeypatch.setattr(
        httpx,
        "Client",
        lambda **kwargs: client_type(
            transport=httpx.MockTransport(
                lambda request: httpx.Response(200, json=payload)
            ),
            **kwargs,
        ),
    )
    c = KnowledgeConfig(database_url="postgresql+asyncpg://user@localhost/test")
    with pytest.raises(KnowledgeError):
        reranker(c).score([("q", "a"), ("q", "b")])


def test_rerank_service_restores_input_order_and_raw_scores(monkeypatch):
    import httpx
    from ai_assistant.knowledge.config import KnowledgeConfig
    from ai_assistant.knowledge.models import reranker

    client_type = httpx.Client
    monkeypatch.setattr(
        httpx,
        "Client",
        lambda **kwargs: client_type(
            transport=httpx.MockTransport(
                lambda request: httpx.Response(
                    200,
                    json={
                        "results": [
                            {"index": 1, "relevance_score": 7.2},
                            {"index": 0, "relevance_score": -11.0},
                        ]
                    },
                )
            ),
            **kwargs,
        ),
    )
    c = KnowledgeConfig(database_url="postgresql+asyncpg://user@localhost/test")
    assert reranker(c).score([("q", "a"), ("q", "b")]) == [-11.0, 7.2]


def test_fingerprint_tracks_semantics_not_service_location():
    from ai_assistant.knowledge.config import KnowledgeConfig

    c = KnowledgeConfig(database_url="postgresql+asyncpg://user@localhost/test")
    assert (
        c.model_copy(
            update={
                "embedding": c.embedding.model_copy(
                    update={"base_url": "http://other/v1"}
                )
            }
        ).fingerprint()
        == c.fingerprint()
    )
    for update in [
        {"chunk_size": 600},
        {"query_instruction": "different"},
        {"embedding": c.embedding.model_copy(update={"model": "other"})},
    ]:
        assert c.model_copy(update=update).fingerprint() != c.fingerprint()


def test_native_embedding_http_raw_text_batch_and_query_instruction(monkeypatch):
    import asyncio
    import json
    import httpx
    from ai_assistant.knowledge.config import KnowledgeConfig
    from ai_assistant.knowledge.models import embedding

    requests = []

    def handle(request):
        payload = json.loads(request.content)
        requests.append(payload)
        assert request.extensions["timeout"]["read"] == 60
        assert all(isinstance(t, str) for t in payload["input"])
        rows = [
            {
                "object": "embedding",
                "index": i,
                "embedding": [float(i + 1)] + [0.0] * 1023,
            }
            for i in range(len(payload["input"]))
        ]
        return httpx.Response(
            200,
            json={
                "object": "list",
                "data": list(reversed(rows)),
                "model": "test",
                "usage": {"prompt_tokens": 1, "total_tokens": 1},
            },
        )

    sync_type, async_type = httpx.Client, httpx.AsyncClient
    __import__("langchain_openai")  # 在替换 HTTP 客户端前加载原生 SDK。

    class MockClient(sync_type):
        def __init__(self, **kwargs):
            super().__init__(transport=httpx.MockTransport(handle), **kwargs)

    class MockAsyncClient(async_type):
        def __init__(self, **kwargs):
            super().__init__(transport=httpx.MockTransport(handle), **kwargs)

    monkeypatch.setattr(httpx, "Client", MockClient)
    monkeypatch.setattr(httpx, "AsyncClient", MockAsyncClient)

    async def scenario():
        config = KnowledgeConfig(
            database_url="postgresql+asyncpg://user@localhost/test"
        )
        async with embedding(config) as model:
            docs = await model.aembed_documents(["first", "second"])
            assert [v[0] for v in docs] == [1.0, 2.0]
            await model.aembed_query("question")
            assert model.embed_documents(["first"])[0][0] == 1.0
        assert requests[0]["input"] == ["first", "second"]
        assert requests[1]["input"] == [
            "Instruct: " + config.query_instruction + "\nQuery: question"
        ]

    asyncio.run(scenario())
