"""RAG 第 2 步：用余弦相似度做向量检索（不依赖向量库）。

检索过程只有三步：
1. 把问题也转成向量
2. 计算问题和每个 chunk 的余弦相似度
3. 取分数最高的 top-k

余弦相似度 = 两个向量的点积 /（模长乘积），取值 -1 ~ 1，越接近 1 越相似。

运行：
    uv run python src/try_chonkie/rag_search.py
"""

from rich import print

from try_chonkie.rag_common import build_index, search

chunks, vectors = build_index()
print(f"索引就绪：{len(chunks)} 个 chunk，向量矩阵 shape={vectors.shape}\n")

questions = [
    "Chonkie 有哪些 chunker？",
    "RAG 怎么减少幻觉？",
    "余弦相似度怎么计算？",
    "chunk_size 应该设多大？",
]

for question in questions:
    print(f"[bold]问题：{question}[/bold]")
    for chunk, score in search(question, chunks, vectors, top_k=2):
        preview = chunk.text.replace("\n", " ")[:48]
        print(f"  {score:.4f}  ({chunk.metadata.get('source')}) {preview}...")
    print()
