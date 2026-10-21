"""RAG 第 4 步：检索 + 拼 prompt + 生成回答。

这是 RAG 的完整闭环：
    问题 -> 检索 top-k chunk -> 把 chunk 作为资料放进 prompt -> LLM 回答

本机 9931 端口只提供 bge-m3（embedding 模型），没有 chat 模型，
所以默认只打印组装好的 prompt。配置任一 OpenAI 兼容的 LLM 服务后即可生成：

    export RAG_LLM_BASE_URL=http://127.0.0.1:端口/v1
    export RAG_LLM_API_KEY=dummy
    export RAG_LLM_MODEL=模型名

运行：
    uv run python src/try_chonkie/rag_answer.py
"""

import os

from rich import print

from rag_common import build_index, search

QUESTION = "Chonkie 是什么？它对 RAG 有什么帮助？"

chunks, vectors = build_index()
results = search(QUESTION, chunks, vectors, top_k=3)

# 把检索结果编号后拼进 prompt，方便模型引用来源
context = "\n\n".join(
    f"[{i}] {chunk.text}" for i, (chunk, _) in enumerate(results, start=1)
)

prompt = f"""你是一个严谨的 RAG 助手。只根据下面的资料回答问题；
资料里没有答案时，直接说"资料中没有相关信息"，不要编造。

资料：
{context}

问题：{QUESTION}
回答："""

print("[bold]检索到的 chunk：[/bold]")
for chunk, score in results:
    print(f"  {score:.4f}  ({chunk.metadata.get('source')}) {chunk.text[:40]!r}...")

print("\n[bold]组装好的 prompt：[/bold]")
print("-" * 60)
print(prompt)
print("-" * 60)

base_url = os.getenv("RAG_LLM_BASE_URL")
api_key = os.getenv("RAG_LLM_API_KEY")
model = os.getenv("RAG_LLM_MODEL")

if not (base_url and api_key and model):
    print("\n未配置 RAG_LLM_BASE_URL / RAG_LLM_API_KEY / RAG_LLM_MODEL，跳过生成。")
else:
    from openai import OpenAI

    client = OpenAI(base_url=base_url, api_key=api_key)
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
    )
    print("\n[bold]模型回答：[/bold]")
    print(response.choices[0].message.content)
