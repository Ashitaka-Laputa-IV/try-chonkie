# try-chonkie

[Chonkie](https://docs.chonkie.ai/) 入门学习项目：用尽量少的代码跑通一条 RAG 前处理链路 —— 文本切分（chunking）→ 上下文补充（refinery）→ 流水线编排（pipeline）。

## 环境要求

- Python >= 3.13
- [uv](https://docs.astral.sh/uv/)

## 快速开始

```bash
uv sync
uv run python src/try_chonkie/get_start.py
```

## 示例索引

示例按学习顺序排列，每个文件都可以独立运行。统一运行方式：

```bash
uv run python src/try_chonkie/get_start.py
```

### 基础操作

| # | 文件 | 学习内容 |
| :-: | --- | --- |
| 1 | [`get_start.py`](src/try_chonkie/get_start.py) | 最小示例：创建 chunker、切分文本、读取 chunk |
| 2 | [`chunk_object.py`](src/try_chonkie/chunk_object.py) | Chunk 对象有哪些字段、如何转成字典 |
| 3 | [`chunk_size_overlap.py`](src/try_chonkie/chunk_size_overlap.py) | `chunk_size` / `chunk_overlap` 对切分结果的影响 |
| 4 | [`batch.py`](src/try_chonkie/batch.py) | 用 `chunk_batch` 一次处理多条文本 |

### 切分策略

| # | 文件 | 学习内容 |
| :-: | --- | --- |
| 5 | [`chunkers.py`](src/try_chonkie/chunkers.py) | Token / Sentence / Recursive 三种策略对比 |
| 6 | [`chinese_text.py`](src/try_chonkie/chinese_text.py) | 中文文本按句子边界切分 |

### 组合与输出

| # | 文件 | 学习内容 |
| :-: | --- | --- |
| 7 | [`refinery.py`](src/try_chonkie/refinery.py) | 用 `OverlapRefinery` 给 chunk 补充相邻上下文 |
| 8 | [`visualize.py`](src/try_chonkie/visualize.py) | 终端彩色可视化，直观对比切分边界 |
| 9 | [`pipeline.py`](src/try_chonkie/pipeline.py) | 用链式 Pipeline 串联切分、精炼、导出 |
| 10 | [`from_file.py`](src/try_chonkie/from_file.py) | 从 `sample.txt` 读取，切分后导出 `chunks.jsonl` |

### RAG 实战（需本地 embedding 服务）

| # | 文件 | 学习内容 |
| :-: | --- | --- |
| 11 | [`rag_common.py`](src/try_chonkie/rag_common.py) | RAG 共享配置：自定义 embedding 接入、切分与检索工具 |
| 12 | [`rag_embed.py`](src/try_chonkie/rag_embed.py) | `EmbeddingsRefinery` 给 chunk 生成 1024 维向量 |
| 13 | [`rag_search.py`](src/try_chonkie/rag_search.py) | 纯 numpy 余弦相似度检索 top-k |
| 14 | [`rag_store.py`](src/try_chonkie/rag_store.py) | `ChromaHandshake` 写入 Chroma 并检索 |
| 15 | [`rag_answer.py`](src/try_chonkie/rag_answer.py) | 检索结果拼 prompt，调用 LLM 回答（可选） |

`rag_common.py` 默认连接本机的 OpenAI 兼容 embedding 服务（llama.cpp + bge-m3），可用环境变量覆盖：

| 配置项 | 默认值 | 环境变量 |
| --- | --- | --- |
| 地址 | `http://127.0.0.1:9931/v1` | `EMBEDDING_BASE_URL` |
| 模型 | `bge-m3` | `EMBEDDING_MODEL` |
| API Key | `dummy`（本地服务不校验） | `EMBEDDING_API_KEY` |

`rag_answer.py` 的生成环节需要另配 OpenAI 兼容 LLM：`RAG_LLM_BASE_URL`、`RAG_LLM_API_KEY`、`RAG_LLM_MODEL`；未配置时只打印组装好的 prompt。

按顺序运行全部示例：

```bash
for f in src/try_chonkie/*.py; do
  echo "===== $f ====="
  uv run python "$f"
done
```

运行 `pipeline.py` 和 `from_file.py` 会生成 `src/try_chonkie/chunks.jsonl`，`rag_store.py` 会生成 `src/try_chonkie/chroma_db/`，两者都已加入 `.gitignore`。

## 核心概念

- **Chunk**：切分结果。常用字段为 `text`、`token_count`、`start_index`、`end_index`、`context`、`embedding`，支持 `len()`、切片和 `to_dict()`。
- **Chunker**：把长文本切成 chunk。所有 chunker 接口一致（`chunk(text)`、`chunk_batch(texts)`、可调用 `chunker(text)`），更换策略只需改一行。
- **Document**：`Pipeline` 读取文件后返回的对象，`doc.content` 是原文，`doc.chunks` 是切分结果。
- **Refinery**：切分后的加工。`OverlapRefinery` 默认 `merge=True`，会把补充的上下文直接拼进 `text`；传 `merge=False` 则只写入 `context` 字段。
- **Pipeline**：按 CHOMP 架构（Fetcher → Chef → Chunker → Refinery → Porter/Handshake）自动编排组件，链式调用即可。

## 说明

- 示例基于 `chonkie 1.7.0` 验证。
- `visualize.py` 需要可视化依赖（`uv add "chonkie[viz]"`）。
- `SemanticChunker` 需要额外依赖（`chonkie[model2vec]` 或 `chonkie[st]`），未包含在入门示例中。
- `SentenceChunker` 默认按英文标点断句，中文处理见 `chinese_text.py`。

## 参考

- [Chonkie 官方文档](https://docs.chonkie.ai/)
- [Chonkie Get Started](https://docs.chonkie.ai/oss/quick-start)
