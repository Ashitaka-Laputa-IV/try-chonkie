"""入门 10：从文件到 JSON —— 一条完整的小流程。

前面都是直接传字符串。真实项目里文本通常来自文件，
Pipeline 可以把 读取 -> 预处理 -> 切分 -> 精炼 -> 导出 串成一条链：

    Fetcher -> Chef -> Chunker -> Refinery -> Porter

运行：
    uv run python src/try_chonkie/from_file.py

输入：同目录的 sample.txt
输出：同目录的 chunks.jsonl（每行一个 chunk 的 JSON）
"""

from pathlib import Path

from chonkie import Pipeline
from rich import print

HERE = Path(__file__).parent
INPUT = HERE / "sample.txt"
OUTPUT = HERE / "chunks.jsonl"

doc = (
    Pipeline()
    .fetch_from("file", path=str(INPUT))               # 1. 读取文件
    .process_with("text")                              # 2. 预处理成 Document
    .chunk_with("recursive", chunk_size=120)           # 3. 切分
    .refine_with("overlap", context_size=30)           # 4. 补充上下文
    .export_with("json", file=str(OUTPUT))             # 5. 导出 JSONL
    .run()
)

print(f"读取 {INPUT.name}：{len(doc.content)} 个字符")
print(f"切出 {len(doc.chunks)} 个 chunk，已导出到 {OUTPUT.name}\n")

for i, chunk in enumerate(doc.chunks, start=1):
    print(f"  {i}. ({chunk.token_count} tokens) {chunk.text[:60]!r}...")
