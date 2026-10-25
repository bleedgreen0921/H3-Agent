"""Thin HTTP adapters; model loading and hardware selection belong to the service."""

import math
from types import SimpleNamespace
from contextlib import asynccontextmanager

import httpx
from langchain_core.embeddings import Embeddings
from ai_assistant.knowledge import KnowledgeError, IngestionError


class ServiceEmbeddings(Embeddings):
    def __init__(self, native, instruction=""):
        self.native = native
        self.instruction = instruction

    @staticmethod
    def validate(vector):
        if (
            len(vector) != 1024
            or not all(type(v) in (int, float) and math.isfinite(v) for v in vector)
            or not any(vector)
        ):
            raise IngestionError(
                "embedding_invalid_output",
                "Embedding 输出无效：要求 1024 维、有限且非零的向量；已停止入库。",
            )
        return vector

    def _documents(self, vectors, count):
        if len(vectors) != count:
            raise KnowledgeError("Embedding 返回数量与输入不一致。")
        return [self.validate(v) for v in vectors]

    def _query(self, text):
        return (
            f"Instruct: {self.instruction}\nQuery: {text}" if self.instruction else text
        )

    def embed_documents(self, texts):
        try:
            return self._documents(self.native.embed_documents(texts), len(texts))
        except KnowledgeError:
            raise
        except Exception:
            raise KnowledgeError(
                "Embedding 服务请求失败，请检查地址、模型和响应。"
            ) from None

    def embed_query(self, text):
        try:
            return self.validate(self.native.embed_query(self._query(text)))
        except KnowledgeError:
            raise
        except Exception:
            raise KnowledgeError(
                "查询编码服务请求失败，请检查地址、模型和响应。"
            ) from None

    async def aembed_documents(self, texts):
        try:
            return self._documents(
                await self.native.aembed_documents(texts), len(texts)
            )
        except KnowledgeError:
            raise
        except Exception:
            raise KnowledgeError(
                "Embedding 服务请求失败，请检查地址、模型和响应。"
            ) from None

    async def aembed_query(self, text):
        try:
            return self.validate(await self.native.aembed_query(self._query(text)))
        except KnowledgeError:
            raise
        except Exception:
            raise KnowledgeError(
                "查询编码服务请求失败，请检查地址、模型和响应。"
            ) from None


def _ordered_embeddings(response, inputs):
    """校验服务批量索引，避免错误顺序或重复索引污染知识库。"""
    try:
        payload = response if isinstance(response, dict) else response.model_dump()
        rows = payload["data"]
        count = 1 if isinstance(inputs, str) else len(inputs)
        indices = [row["index"] for row in rows]
        if (
            payload.get("error")
            or len(rows) != count
            or any(type(i) is not int for i in indices)
            or set(indices) != set(range(count))
        ):
            raise ValueError("Invalid embedding indices")
        payload["data"] = sorted(rows, key=lambda row: row["index"])
        return payload
    except (KeyError, TypeError, ValueError, AttributeError):
        raise KnowledgeError("Embedding 响应数量或索引无效；未提交向量。") from None


@asynccontextmanager
async def embedding(config):
    """按操作管理原生客户端，连接在成功、异常和取消时均关闭。"""
    from langchain_openai import OpenAIEmbeddings

    service = config.embedding
    with httpx.Client(timeout=service.timeout, trust_env=False) as sync_client:
        async with httpx.AsyncClient(
            timeout=service.timeout, trust_env=False
        ) as async_client:
            native = OpenAIEmbeddings(
                model=service.model,
                base_url=service.base_url,
                api_key=service.api_key.get_secret_value(),
                chunk_size=service.batch_size,
                max_retries=0,
                request_timeout=service.timeout,
                check_embedding_ctx_length=False,
                http_client=sync_client,
                http_async_client=async_client,
            )
            sync_api, async_api = native.client, native.async_client

            def create(**kwargs):
                return _ordered_embeddings(sync_api.create(**kwargs), kwargs["input"])

            async def acreate(**kwargs):
                return _ordered_embeddings(
                    await async_api.create(**kwargs), kwargs["input"]
                )

            native.client = SimpleNamespace(create=create)
            native.async_client = SimpleNamespace(create=acreate)
            yield ServiceEmbeddings(native, config.query_instruction)


def reranker(config):
    from langchain_community.cross_encoders.base import BaseCrossEncoder

    class ServiceCrossEncoder(BaseCrossEncoder):
        def score(self, pairs):
            if not pairs:
                return []
            service = config.reranker
            scores = []
            try:
                with httpx.Client(timeout=service.timeout, trust_env=False) as client:
                    # 一个请求只包含一个 query；按连续相同 query 分组并保留输入顺序。
                    from itertools import groupby

                    for query, group in groupby(pairs, key=lambda pair: pair[0]):
                        documents = [pair[1] for pair in group]
                        for start in range(0, len(documents), service.batch_size):
                            batch = documents[start : start + service.batch_size]
                            response = client.post(
                                service.base_url + "/rerank",
                                headers={
                                    "Authorization": "Bearer "
                                    + service.api_key.get_secret_value()
                                },
                                json={
                                    "model": service.model,
                                    "query": query,
                                    "documents": batch,
                                },
                            )
                            response.raise_for_status()
                            payload = response.json()
                            if not isinstance(payload, dict) or payload.get("error"):
                                raise ValueError("Invalid rerank response")
                            rows = payload["results"]
                            ordered = [None] * len(batch)
                            if len(rows) != len(batch):
                                raise ValueError("Missing scores")
                            for row in rows:
                                i, value = row["index"], row["relevance_score"]
                                if (
                                    type(i) is not int
                                    or not 0 <= i < len(batch)
                                    or ordered[i] is not None
                                    or type(value) not in (int, float)
                                    or not math.isfinite(value)
                                ):
                                    raise ValueError("Invalid score or index")
                                ordered[i] = value
                            scores.extend(ordered)
                return scores
            except (httpx.HTTPError, ValueError, KeyError, TypeError):
                raise KnowledgeError(
                    "重排服务失败：检查地址、模型及评分数量、索引和数值。"
                ) from None

    return ServiceCrossEncoder()
