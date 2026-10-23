"""Process-local model reuse and serialized, off-loop inference."""
import threading
from functools import lru_cache
from langchain_core.embeddings import Embeddings
from ai_assistant.knowledge import KnowledgeError

GPU_LOCK = threading.RLock()


def check_device(config, *, reranker=False):
    try:
        import torch
    except ImportError as exc:
        raise KnowledgeError('未安装本地模型依赖；请安装 RAG 可选依赖。') from exc
    paths = [config.embedding_model] + ([config.reranker_model] if reranker else [])
    if any(not path.is_dir() for path in paths):
        raise KnowledgeError('本地模型目录不存在；不会自动下载。')
    if config.device.startswith('cuda') and not torch.cuda.is_available():
        raise KnowledgeError('CUDA 不可用；请安装 CUDA PyTorch 或显式配置 device="cpu"。')


class SerializedEmbeddings(Embeddings):
    def __init__(self, native):
        self.native = native

    def embed_documents(self, texts):
        with GPU_LOCK:
            try:
                return self.native.embed_documents(texts)
            except RuntimeError as exc:
                raise KnowledgeError('Embedding 推理失败，请检查设备及显存。') from exc

    def embed_query(self, text):
        with GPU_LOCK:
            try:
                return self.native.embed_query(text)
            except RuntimeError as exc:
                raise KnowledgeError('查询编码失败，请检查设备及显存。') from exc


@lru_cache(maxsize=4)
def _embedding(path, device, batch):
    from langchain_huggingface import HuggingFaceEmbeddings
    with GPU_LOCK:
        native = HuggingFaceEmbeddings(
            model_name=path,
            model_kwargs=dict(device=device, local_files_only=True, trust_remote_code=False),
            encode_kwargs=dict(prompt_name='document', normalize_embeddings=True, batch_size=batch),
            query_encode_kwargs=dict(prompt_name='query', normalize_embeddings=True, batch_size=batch),
        )
        if native._client.get_sentence_embedding_dimension() != 1024:
            raise KnowledgeError('Embedding 模型必须输出 1024 维向量。')
        if not {'query', 'document'} <= native._client.prompts.keys():
            raise KnowledgeError('本地模型缺少 query/document 提示。')
        return SerializedEmbeddings(native)


def embedding(config):
    check_device(config)
    try:
        return _embedding(str(config.embedding_model), config.device, config.embedding_batch)
    except KnowledgeError:
        raise
    except (ImportError, OSError, RuntimeError, ValueError) as exc:
        raise KnowledgeError('本地 Embedding 加载失败；不会自动下载或回退设备。') from exc


@lru_cache(maxsize=4)
def _reranker(path, device, batch):
    from langchain_community.cross_encoders import HuggingFaceCrossEncoder
    # Native integration exposes no predict kwargs; keep only batching in a thin subclass.
    class BatchedCrossEncoder(HuggingFaceCrossEncoder):
        def score(self, pairs):
            with GPU_LOCK:
                try:
                    return self.client.predict(pairs, batch_size=batch, show_progress_bar=False)
                except RuntimeError as exc:
                    raise KnowledgeError('重排失败，请检查设备及显存。') from exc
    with GPU_LOCK:
        return BatchedCrossEncoder(model_name=path, model_kwargs=dict(
            device=device, local_files_only=True, trust_remote_code=False, max_length=1024))


def reranker(config):
    check_device(config, reranker=True)
    try:
        return _reranker(str(config.reranker_model), config.device, config.reranker_batch)
    except (ImportError, OSError, RuntimeError, ValueError) as exc:
        raise KnowledgeError('本地重排模型加载失败；不会自动下载或回退设备。') from exc
