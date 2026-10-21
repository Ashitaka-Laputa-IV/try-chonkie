"""入门 4：Pipeline —— 用一条链串起整个处理流程。

Chonkie 把 RAG 前处理抽象成 CHOMP 架构：
    Fetcher -> Chef -> Chunker -> Refinery -> Porter / Handshake
Pipeline 会按这个顺序自动编排组件，你只需要链式调用。

本例：直接给出文本 -> 递归切分 -> 补充重叠上下文 -> 导出 JSONL。

运行：
    uv run python src/try_chonkie/pipeline.py
"""

from pathlib import Path

from chonkie import Pipeline
from rich import print

OUTPUT = Path(__file__).with_name("chunks.jsonl")

TEXT = """Chonkie is a lightweight ingestion library for RAG pipelines. It turns long documents into small, meaningful chunks.

Chunking matters because every model has a limited context window. Well-formed chunks keep retrieval precise and reduce hallucinations.

Chonkie ships with several chunkers and refineries. Refineries add extra context to chunks after splitting, so a chunk can stand on its own during retrieval."""

doc = (
    Pipeline()
    .chunk_with("recursive", chunk_size=150)  # 必须：选择 chunker
    .refine_with("overlap", context_size=40)  # 可选：补充重叠上下文
    .export_with("json", file=str(OUTPUT))    # 可选：导出为 JSONL
    .run(texts=TEXT)                          # 直接传入文本
)

print(f"共 {len(doc.chunks)} 个 chunk，已导出到 {OUTPUT}")
print(f"第一行预览：{OUTPUT.read_text(encoding='utf-8').splitlines()[0][:120]}...")

# 也可以从文件读取（Fetcher + Chef）：
# doc = (
#     Pipeline()
#     .fetch_from("file", path="document.txt")
#     .process_with("text")
#     .chunk_with("recursive", chunk_size=512)
#     .run()
# )
