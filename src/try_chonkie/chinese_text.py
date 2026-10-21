"""入门 6：中文文本怎么切。

Chonkie 默认按英文标点（". "、"! "、"?"）和换行断句，
不认识中文的"。"、"！"、"？"，中文句子很容易被切碎。

解决方式：给 SentenceChunker 显式传入中文分隔符。
    delim         : 句子分隔符列表
    include_delim : "prev" 把分隔符留在前一句，"next" 放到后一句

运行：
    uv run python src/try_chonkie/chinese_text.py
"""

from chonkie import SentenceChunker
from rich import print

TEXT = """Chonkie 是一个轻量级文本切分库。它可以帮助你构建 RAG 的文本预处理流程！切分质量直接影响检索效果，因此不能随便切。

好的 chunk 应该自成一体、保留完整语义。中文句子以句号、感叹号、问号结尾，不能用英文的 ". " 来识别。"""

print("[bold]默认分隔符：只认英文标点和换行[/bold]")
for i, chunk in enumerate(SentenceChunker(chunk_size=40)(TEXT), start=1):
    print(f"  {i}. {chunk.text!r}")

print("\n[bold]中文分隔符：按中文句子边界切分[/bold]")
chinese_chunker = SentenceChunker(
    chunk_size=40,
    delim=["。", "！", "？", "\n"],
    include_delim="prev",
)
for i, chunk in enumerate(chinese_chunker(TEXT), start=1):
    print(f"  {i}. {chunk.text!r}")
