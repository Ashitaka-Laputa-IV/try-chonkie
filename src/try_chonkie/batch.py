"""入门 4：批量处理 —— 一次切分多条文本。

真实场景很少只有一段文本。所有 chunker 都提供 chunk_batch，
返回结构是 list[list[Chunk]]：外层对应每条输入，内层是它切出的 chunks。

运行：
    uv run python src/try_chonkie/batch.py
"""

from chonkie import TokenChunker
from rich import print

documents = [
    "RAG combines retrieval with generation. A retriever finds relevant passages, and the model answers questions using them.",
    "Chunking breaks long documents into smaller pieces. Each piece should keep a single idea so retrieval stays precise.",
    "Chonkie makes chunking simple. It ships with several chunkers and a unified interface, so you can swap strategies in one line.",
]

chunker = TokenChunker(chunk_size=80, chunk_overlap=10)

# 方式一：chunk_batch；show_progress_bar=False 关闭进度条，让输出更干净
batch = chunker.chunk_batch(documents, show_progress_bar=False)

# 方式二：直接调用，效果相同（默认带进度条）
# batch = chunker(documents)

for doc_index, chunks in enumerate(batch, start=1):
    print(f"[bold]文档 {doc_index}[/bold]：{len(chunks)} 个 chunk")
    for chunk in chunks:
        preview = chunk.text.replace("\n", " ")[:55]
        print(f"  - ({chunk.token_count} tokens) {preview}...")
    print()

total = sum(len(chunks) for chunks in batch)
print(f"合计：{len(documents)} 条文本，{total} 个 chunk")
