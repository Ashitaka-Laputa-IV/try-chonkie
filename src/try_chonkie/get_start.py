"""入门 1：最小示例 —— TokenChunker。

TokenChunker 按固定 token 数切分，是最简单、最可控的 chunker，
也是理解 Chonkie 工作方式的最佳起点。

运行：
    uv run python src/try_chonkie/get_start.py
"""

from chonkie import TokenChunker
from rich import print

# 1. 创建 chunker
#    chunk_size    : 每个 chunk 最多 120 个 token
#    chunk_overlap : 相邻 chunk 重叠 20 个 token，避免边界处信息被切断
#    tokenizer     : 默认 "character"（按字符计数）；真实场景常换成 "gpt2" 等
chunker = TokenChunker(chunk_size=120, chunk_overlap=20)

# 2. 待切分的文本
text = """Chonkie is a lightweight ingestion library for RAG pipelines. It turns long documents into small, meaningful chunks. Chonkie ships with several chunkers, including TokenChunker, SentenceChunker and RecursiveChunker. They all share the same interface, so swapping one for another only takes a single line."""

# 3. 执行切分：chunker(text) 等价于 chunker.chunk(text)
chunks = chunker(text)

# 4. 每个元素都是一个 Chunk 对象
#    常用字段：text / token_count / start_index / end_index / context / embedding
for i, chunk in enumerate(chunks, start=1):
    print(
        f"[bold cyan]Chunk {i}[/bold cyan]  "
        f"tokens={chunk.token_count}  chars=[{chunk.start_index}, {chunk.end_index})"
    )
    print(f"  {chunk.text!r}")
