# 评测集

用一个可复现的小语料，把「感觉更准了」变成可比较的数字。

## 目录

```
evals/
  corpus/     5 篇 Markdown（BM25 / RRF / 切块 / 引用 / 嵌入向量），约 8 千字
  qa.jsonl    35 条用例：33 条可回答 + 2 条应拒答
```

`qa.jsonl` 每行一条用例：

| 字段 | 说明 |
| --- | --- |
| `id` | 用例编号，报告里的 `misses` 直接引用它 |
| `question` | 提问原文 |
| `gold` | 金标来源，`{"file": "01-bm25.md", "page": 1}`；page 对 Markdown 恒为 1，PDF 为真实页码 |
| `expect` | `answer`（应给出答案）或 `refusal`（文档中没有依据，应拒答） |
| `note` | 这条用例想测什么 |

`gold` 刻意标到「文件 + 页码」而非 chunk id：chunk 会随切块参数变化，金标不应该跟着变。
召回判定沿用 `citations._normalize`，所以 `data/论文.pdf` 与 `/abs/path/论文.pdf` 视为同一来源。

## 指标

| 指标 | 定义 | 需要生成模型 |
| --- | --- | --- |
| `Recall@k` | 金标来源被召回的比例，按用例平均 | 否 |
| `Hit@k` | 至少命中一个金标来源的用例占比 | 否 |
| `MRR` | 首个命中排名的倒数，按用例平均 | 否 |
| `citation precision` | 答案引用的来源里落在金标内的比例 | 是 |
| `refusal accuracy` | 应拒答的拒答了、应回答的没拒答 | 是 |
| `fabricated citations` | 引用了本次证据之外来源的次数 | 是 |

检索三项无需 LLM，CI 就只跑这三项；答案三项在没有可达生成器时显式报 `skipped`，不会悄悄记 0。

## 用法

```bash
# 检索评测（无需 Ollama，用 hash 嵌入 + 内存库，CI 同款）
python -m marcusrag eval --embedder hash --store memory --data-dir evals/corpus \
  --search-type hybrid --rerank lexical --k 3

# 真实语义嵌入 + 持久化 Chroma
python -m marcusrag eval --embedder hf --embedding-model BAAI/bge-small-zh-v1.5 \
  --store chroma --data-dir evals/corpus --search-type hybrid --rerank lexical --k 3

# 端到端（含答案质量），需要本地 Ollama
python -m marcusrag eval --data-dir evals/corpus --end-to-end --output report.json
```

`eval` 默认重建索引（`rebuild=True`），保证同一份语料、同一份金标下的数字可复现，
也顺带绕过损坏的索引目录。参数组合可扫描：

```bash
for w in 0.3 0.5 0.7 0.9; do
  python -m marcusrag eval --embedder hf --store chroma --data-dir evals/corpus \
    --search-type hybrid --rerank lexical --rerank-base-weight $w --k 3
done
```

## 已知局限

- 语料是 5 篇短文档、35 条用例，只能挡住明显的召回回归，不能代替真实业务语料。
- Markdown 语料没有页码维度，页码相关缺陷需要 PDF 语料才测得出。
- 答案三项依赖生成模型，模型换了指标不可直接横向比较。
