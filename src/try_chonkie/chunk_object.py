"""入门 2：认识 Chunk 对象 —— 切分结果里到底有什么。

chunker 返回的是一个 Chunk 列表。每个 Chunk 除了 text，还带着位置、token 数等
信息，后续做检索、溯源、去重都要用到这些字段。

运行：
    uv run python src/try_chonkie/chunk_object.py
"""

from chonkie import SentenceChunker
from rich import print

TEXT = """Chonkie is a lightweight ingestion library for RAG pipelines. It turns long documents into small, meaningful chunks.

Chunking matters because every model has a limited context window. Well-formed chunks keep retrieval precise and reduce hallucinations."""

chunker = SentenceChunker(chunk_size=120)
chunks = chunker(TEXT)

first = chunks[0]
print("[bold]第一个 Chunk 的字段[/bold]")
print(f"  id          : {first.id}")
print(f"  text        : {first.text!r}")
print(f"  start_index : {first.start_index}   # 在原文中的起始位置")
print(f"  end_index   : {first.end_index}   # 在原文中的结束位置")
print(f"  token_count : {first.token_count}")
print(f"  context     : {first.context!r}   # 由 refinery 补充，这里还是 None")
print(f"  embedding   : {first.embedding!r}   # 由 embedding refinery 补充，这里还是 None")

print("\n[bold]Chunk 可以当字符串用[/bold]")
print(f"  len(chunk)  = {len(first)}")
print(f"  chunk[:40]  = {first[:40]!r}")

print("\n[bold]转成字典，方便存 JSON[/bold]")
print(first.to_dict())

print("\n[bold]所有 chunk 一览[/bold]")
for i, chunk in enumerate(chunks, start=1):
    preview = chunk.text.replace("\n", " ")[:50]
    print(f"  {i}. chars=[{chunk.start_index:>3}, {chunk.end_index:>3})  {preview}...")
