"""RAG 第 1 步：把 chunk 变成向量（embedding）。

检索的前提是"文本之间能比较相似度"，而 embedding 就是把文本映射成高维向量。
Chonkie 的 EmbeddingsRefinery 会给每个 chunk 写入 embedding 字段。

运行：
    uv run python src/try_chonkie/rag_embed.py

依赖：本机 9931 端口的 OpenAI 兼容 embedding 服务（bge-m3），配置见 rag_common.py。
"""

from chonkie import EmbeddingsRefinery
from rich import print

from try_chonkie.rag_common import get_embeddings, load_chunks

chunks = load_chunks()
print(f"切出 {len(chunks)} 个 chunk")

# EmbeddingsRefinery 会原地写入 chunk.embedding
refinery = EmbeddingsRefinery(embedding_model=get_embeddings())
chunks = refinery.refine(chunks)

print(f"向量维度：{get_embeddings().dimension}\n")
for i, chunk in enumerate(chunks, start=1):
    print(f"  {i}. ({chunk.metadata.get('source')}) {chunk.text[:40]!r}...")
    print(f"     embedding.shape = {chunk.embedding.shape}")
