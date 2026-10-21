"""入门 7：Refinery —— 切分之后，再给 chunk 补充信息。

OverlapRefinery 会把相邻 chunk 的文本补充到当前 chunk，缓解
"关键信息正好被切在边界上"的问题。补充的内容写在 context 字段里；
默认 merge=True，还会直接拼到 text 末尾，拿去 embedding 即可。

参数：
    context_size : 补充多少 token（也可以传比例，例如 0.25）
    method       : "suffix" 补后文 / "prefix" 补前文 / "justified" 两侧对齐
    merge        : True 把 context 直接拼进 text；False 只写 context 字段
    inplace      : True 直接修改传入的 chunks；False 返回新列表，便于对比

运行：
    uv run python src/try_chonkie/refinery.py
"""

from chonkie import OverlapRefinery, SentenceChunker
from rich import print

TEXT = """Chonkie is a lightweight ingestion library for RAG pipelines. It turns long documents into small, meaningful chunks.

Chunking matters because every model has a limited context window. Well-formed chunks keep retrieval precise and reduce hallucinations.

Chonkie ships with several chunkers and refineries. Refineries add extra context to chunks after splitting, so a chunk can stand on its own during retrieval."""

chunker = SentenceChunker(chunk_size=150)
chunks = chunker(TEXT)

print(f"切出 {len(chunks)} 个 chunk，此时 context 默认为 None：{chunks[0].context}\n")

# inplace=False：不修改原始 chunks，方便前后对比
refinery = OverlapRefinery(context_size=40, method="suffix", inplace=False)
refined = refinery.refine(chunks)

for i, (before, after) in enumerate(zip(chunks, refined), start=1):
    print(f"[bold cyan]Chunk {i}[/bold cyan]")
    print(f"  切分后 text : {before.text!r}")
    print(f"  补充 context: {after.context!r}")
    print(f"  refine 后 text: {after.text!r}")
    print()

# 如果只想保留原始 text，把补充内容和 text 分开管理：
# OverlapRefinery(context_size=40, merge=False).refine(chunks)
