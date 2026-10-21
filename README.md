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

示例按学习顺序排列，每个文件都可以独立运行：

| # | 文件 | 学习内容 | 关键 API |
| :-: | --- | --- | --- |
| 1 | [`get_start.py`](src/try_chonkie/get_start.py) | 最小可运行示例：创建 chunker、切分文本、读取 `Chunk` 字段 | `TokenChunker` |
| 2 | [`chunkers.py`](src/try_chonkie/chunkers.py) | 对比三个常用 chunker 的切分方式 | `TokenChunker`、`SentenceChunker`、`RecursiveChunker` |
| 3 | [`refinery.py`](src/try_chonkie/refinery.py) | 给 chunk 补充相邻上下文，缓解边界信息被切断 | `OverlapRefinery` |
| 4 | [`pipeline.py`](src/try_chonkie/pipeline.py) | 用链式 Pipeline 串联切分、精炼、导出 | `Pipeline`、`JSONPorter` |

运行全部示例：

```bash
uv run python src/try_chonkie/get_start.py
uv run python src/try_chonkie/chunkers.py
uv run python src/try_chonkie/refinery.py
uv run python src/try_chonkie/pipeline.py
```

运行 `pipeline.py` 会在 `src/try_chonkie/` 下生成 `chunks.jsonl`，该文件已加入 `.gitignore`。

## 核心概念

- **Chunk**：切分结果。常用字段为 `text`、`token_count`、`start_index`、`end_index`、`context`、`embedding`。
- **Chunker**：把长文本切成 chunk。所有 chunker 接口一致（`chunk(text)`、`chunk_batch(texts)`、可调用 `chunker(text)`），更换策略只需改一行。
- **Refinery**：切分后的加工。`OverlapRefinery` 默认 `merge=True`，会把补充的上下文直接拼进 `text`；传 `merge=False` 则只写入 `context` 字段。
- **Pipeline**：按 CHOMP 架构（Fetcher → Chef → Chunker → Refinery → Porter/Handshake）自动编排组件，链式调用即可。

## 说明

- 示例基于 `chonkie 1.7.0` 验证。
- `SemanticChunker` 需要额外依赖（`chonkie[model2vec]` 或 `chonkie[st]`），未包含在入门示例中。
- `SentenceChunker` 默认按英文标点断句，处理中文可传入 `delim=["。", "！", "？", "\n"]`。

## 参考

- [Chonkie 官方文档](https://docs.chonkie.ai/)
- [Chonkie Get Started](https://docs.chonkie.ai/oss/quick-start)
