"""入门 5：三个最常用的 chunker，对比切分方式的差异。

- TokenChunker     按固定 token 数切分，长度可控，但可能在句子中间断开
- SentenceChunker  按句子边界切分，语义更完整
- RecursiveChunker 按 段落 -> 句子 -> 单词 逐层递归切分，兼顾结构与长度

运行：
    uv run python src/try_chonkie/chunkers.py

提示：默认按英文标点识别句子。处理中文时，可以给 SentenceChunker 传入
delim=["。", "！", "？", "\n"]。
"""

from chonkie import RecursiveChunker, SentenceChunker, TokenChunker
from rich import print

TEXT = """Retrieval-Augmented Generation (RAG) combines a retriever with a language model. The retriever finds relevant passages from a knowledge base, and the model answers questions using those passages.

Chunking is the first step of every RAG pipeline. A good chunk should be self-contained, reconstructable, and large enough to be meaningful. Bad chunks lead to bad retrieval, which leads to bad answers.

Chonkie focuses on making this step fast, simple and robust. It ships with several chunkers, each tuned for a different content type and use case. Swapping one chunker for another only takes a single line of code."""

# 三种 chunker 用相同的 chunk_size，方便对比
chunkers = {
    "TokenChunker": TokenChunker(chunk_size=120, chunk_overlap=20),
    "SentenceChunker": SentenceChunker(chunk_size=120),
    "RecursiveChunker": RecursiveChunker(chunk_size=120),
}

for name, chunker in chunkers.items():
    chunks = chunker(TEXT)
    print(f"[bold]{name}[/bold]：{len(chunks)} 个 chunk")
    for i, chunk in enumerate(chunks, start=1):
        preview = " ".join(chunk.text.split())  # 折叠换行，方便单行预览
        print(f"  {i}. ({chunk.token_count} tokens) {preview[:60]}...")
    print()
