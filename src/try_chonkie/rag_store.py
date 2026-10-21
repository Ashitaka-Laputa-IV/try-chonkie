"""RAG 第 3 步：把向量存进 Chroma，用向量库检索。

上一课用 numpy 手算余弦相似度；真实项目通常交给向量库负责存储和检索。
Chonkie 的 ChromaHandshake 负责写入，查询用 handshake.search()。

运行：
    uv run python src/try_chonkie/rag_store.py

数据落在同目录的 chroma_db/ 下（已加入 .gitignore），重复运行会覆盖同一批 chunk。
"""

from pathlib import Path

from chonkie import ChromaHandshake
from rich import print

from try_chonkie.rag_common import get_embeddings, load_chunks

DB_PATH = Path(__file__).parent / "chroma_db"

chunks = load_chunks()
handshake = ChromaHandshake(
    collection_name="rag_demo",  # Chroma 要求集合名至少 3 个字符
    embedding_model=get_embeddings(),
    path=str(DB_PATH),
)
handshake.write(chunks)
print(f"已写入 {len(chunks)} 个 chunk 到 Chroma：{DB_PATH}\n")

for question in ["Chonkie 有哪些 chunker？", "RAG 怎么减少幻觉？"]:
    print(f"[bold]问题：{question}[/bold]")
    for item in handshake.search(query=question, limit=2):
        preview = item["text"].replace("\n", " ")[:48]
        print(f"  {item['score']:.4f}  ({item['source']}) {preview}...")
    print()
