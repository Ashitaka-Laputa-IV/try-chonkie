"""RAG 示例的公共配置与小工具。

本机在 http://127.0.0.1:9931 运行了一个 OpenAI 兼容的 embedding 服务
（llama.cpp + bge-m3，1024 维）。chonkie 1.7.0 自带的 OpenAIEmbeddings
基于 catsu，暂不支持自定义 base_url，所以这里用 openai SDK
实现一个最小的 BaseEmbeddings，接入任意 OpenAI 兼容服务。

其他 RAG 示例都会从本文件导入配置和工具函数。
"""

import os

import numpy as np
from chonkie import BaseEmbeddings
from chonkie.tokenizer import Tokenizer

EMBEDDING_BASE_URL = os.getenv("EMBEDDING_BASE_URL", "http://127.0.0.1:9931/v1")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "bge-m3")
EMBEDDING_API_KEY = os.getenv("EMBEDDING_API_KEY", "dummy")  # 本地服务不校验 key

DOCUMENTS = [
    """Chonkie 是一个轻量级的文本切分库，专门为 RAG 的文本预处理设计。它提供 TokenChunker、SentenceChunker、RecursiveChunker 等多种切分器，还支持 refinery 给 chunk 补充上下文。Chonkie 的 Pipeline 把读取、切分、精炼和导出串成一条链，使用起来非常方便。""",
    """RAG（Retrieval-Augmented Generation，检索增强生成）是一种把检索和生成结合的技术。它先把文档切成 chunk 并转成向量存入向量库。用户提问时，系统先检索最相关的 chunk，再把它们和问题一起交给大模型生成回答。这样可以减少幻觉，并让回答有据可查。""",
    """向量检索的核心是 embedding 和相似度计算。embedding 把文本映射成高维向量，语义相近的文本在向量空间里距离更近。最常用的相似度是余弦相似度，它等于两个向量点积除以模长乘积。检索时把问题也转成向量，然后找出与它最相似的 top-k 个 chunk。""",
    """切分参数直接影响检索质量。chunk_size 太小会丢失上下文，太大又会引入无关信息。chunk_overlap 可以在相邻 chunk 之间保留重叠内容，缓解句子被切断的问题。实际项目中通常先用几百个 token 的 chunk_size 做基线，再根据检索效果调整。""",
]


class OpenAICompatibleEmbeddings(BaseEmbeddings):
    """把任意 OpenAI 兼容的 embedding 服务接入 Chonkie。

    只需要实现 BaseEmbeddings 的三个抽象成员：embed / dimension / get_tokenizer。
    """

    def __init__(
        self,
        model: str = EMBEDDING_MODEL,
        base_url: str = EMBEDDING_BASE_URL,
        api_key: str = EMBEDDING_API_KEY,
    ) -> None:
        super().__init__()
        from openai import OpenAI

        self.model = model
        self.client = OpenAI(base_url=base_url, api_key=api_key)
        self._dimension: int | None = None

    def embed(self, text: str) -> np.ndarray:
        """把单条文本转成向量。"""
        response = self.client.embeddings.create(model=self.model, input=text)
        return np.array(response.data[0].embedding, dtype=np.float32)

    def embed_batch(self, texts: list[str]) -> list[np.ndarray]:
        """批量转成向量，一次请求完成。"""
        response = self.client.embeddings.create(model=self.model, input=texts)
        return [np.array(item.embedding, dtype=np.float32) for item in response.data]

    @property
    def dimension(self) -> int:
        """向量维度，首次访问时探测一次。"""
        if self._dimension is None:
            self._dimension = len(self.embed("维度探测"))
        return self._dimension

    def get_tokenizer(self) -> Tokenizer:
        return Tokenizer("character")


_embeddings: OpenAICompatibleEmbeddings | None = None


def get_embeddings() -> OpenAICompatibleEmbeddings:
    """复用同一个 embedding 客户端。"""
    global _embeddings
    if _embeddings is None:
        _embeddings = OpenAICompatibleEmbeddings()
    return _embeddings


def load_chunks():
    """把示例文档切成 chunk，并标注来源。"""
    from chonkie import RecursiveChunker

    chunker = RecursiveChunker(chunk_size=150)
    chunks = []
    for doc_id, text in enumerate(DOCUMENTS, start=1):
        for chunk in chunker(text):
            chunk.metadata["source"] = f"doc_{doc_id}"
            chunks.append(chunk)
    return chunks


def build_index() -> tuple[list, np.ndarray]:
    """切分 + 向量化，返回 (chunks, 向量矩阵)。"""
    chunks = load_chunks()
    vectors = np.array(get_embeddings().embed_batch([c.text for c in chunks]))
    return chunks, vectors


def search(query: str, chunks: list, vectors: np.ndarray, top_k: int = 3):
    """余弦相似度检索，返回 [(chunk, score), ...]。"""
    query_vector = get_embeddings().embed(query)
    scores = vectors @ query_vector / (
        np.linalg.norm(vectors, axis=1) * np.linalg.norm(query_vector) + 1e-10
    )
    top_indices = np.argsort(scores)[::-1][:top_k]
    return [(chunks[i], float(scores[i])) for i in top_indices]
