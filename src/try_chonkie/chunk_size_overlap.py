"""入门 3：chunk_size 和 chunk_overlap 到底影响什么。

- chunk_size    : 每个 chunk 的大小上限（单位取决于 tokenizer，默认按字符）
- chunk_overlap : 相邻 chunk 重叠多少，用来缓解"句子正好被切断"

用同一段文本跑三组参数，观察 chunk 数量、内容和位置的变化。

运行：
    uv run python src/try_chonkie/chunk_size_overlap.py
"""

from chonkie import TokenChunker
from rich import print

TEXT = """Chonkie is a lightweight ingestion library for RAG pipelines. It turns long documents into small, meaningful chunks before they are embedded and stored. Chunking matters because every model has a limited context window. Well-formed chunks keep retrieval precise and reduce hallucinations."""

configs = [
    ("chunk_size=80,  chunk_overlap=0", 80, 0),
    ("chunk_size=80,  chunk_overlap=20", 80, 20),
    ("chunk_size=160, chunk_overlap=0", 160, 0),
]

for label, size, overlap in configs:
    chunker = TokenChunker(chunk_size=size, chunk_overlap=overlap)
    chunks = chunker(TEXT)
    print(f"[bold]{label}[/bold] -> {len(chunks)} 个 chunk")
    for i, chunk in enumerate(chunks, start=1):
        preview = " ".join(chunk.text.split())[:55]
        print(f"  {i}. chars=[{chunk.start_index:>3}, {chunk.end_index:>3})  {preview}...")
    print()

# 观察重点：
# 1. chunk_size 越小，chunk 越多，单块携带的上下文越少
# 2. chunk_overlap > 0 时，相邻 chunk 的 start_index 间隔会变小（内容有重叠）
# 3. overlap 不能大于等于 chunk_size，否则会报错
