"""入门 8：可视化对比 —— 直观看到不同 chunker 的切分边界。

Visualizer 会在终端给每个 chunk 上不同颜色，适合调参时快速对比。
它需要额外的可视化依赖：

    uv add "chonkie[viz]"

运行：
    uv run python src/try_chonkie/visualize.py
"""

from chonkie import RecursiveChunker, SentenceChunker, TokenChunker, Visualizer
from rich import print

TEXT = """Chonkie is a lightweight ingestion library for RAG pipelines. It turns long documents into small, meaningful chunks. Chunking matters because every model has a limited context window. Well-formed chunks keep retrieval precise and reduce hallucinations. Chonkie ships with several chunkers, each tuned for a different content type."""

viz = Visualizer(theme="pastel")

print("[bold]TokenChunker：固定长度，可能在句子中间断开[/bold]")
viz.print(TokenChunker(chunk_size=100)(TEXT))

print("\n[bold]SentenceChunker：按句子边界切分[/bold]")
viz.print(SentenceChunker(chunk_size=100)(TEXT))

print("\n[bold]RecursiveChunker：先按段落，再按句子，逐层递归[/bold]")
viz.print(RecursiveChunker(chunk_size=100)(TEXT))

# 想把结果保存成网页对比，可以：
# viz.save("chunks.html", chunks)
