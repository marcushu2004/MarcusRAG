# MarcusRAG

基于本地文档的**证据优先**检索增强问答（RAG）系统：把 PDF / TXT / Markdown 切块入库，
检索后只依据召回到的片段作答，并强制校验答案里的引用是否真实存在于证据中。

- 检索：混合检索（向量 + BM25，RRF 融合）/ MMR / 纯相似度，可选词法或 cross-encoder 重排
- 向量库：Chroma（持久化）/ 内存 JSON（零依赖演示）
- 嵌入：sentence-transformers（`all-MiniLM-L6-v2` 等）/ 哈希嵌入（离线兜底）
- 生成：本地 Ollama（默认 `qwen2.5:3b`），通过 HTTP API 调用，无需 LangChain provider 包
- 评测：内置 35 条用例的评测集，量化 Recall@k / Hit@k / MRR / 引用精确率，CI 带门禁

> **v0.2.0 重构说明**：原 `src/` 包与 `loaders.py` 存在循环导入、多文件索引丢失、重复写入
> 等问题，已整体重写为 `marcusrag/` 包并提供 CLI。旧入口 `python main.py` 仍然可用。

---

## 快速开始

```bash
# 1. 最小依赖（纯 Python，无 torch）
pip install -r requirements.txt

# 2. 放入文档
mkdir -p data/pdfs && cp your.pdf data/pdfs/

# 3. 建立索引
python -m marcusrag index --profile zh --chunk-size 500 --chunk-overlap 80

# 4. 提问（--show-context 看证据，--color 高亮命中的查询词）
python -m marcusrag ask "这篇论文的核心贡献是什么？" --show-context --color

# 5. 查看配置与索引状态
python -m marcusrag info
```

完整质量栈（真实语义嵌入 + Chroma）：

```bash
pip install -r requirements.txt -r requirements-full.txt
ollama serve &
ollama pull qwen2.5:3b
python -m marcusrag index
```

零依赖端到端自检（不需要 GPU、Chroma 或 Ollama）：

```bash
python -m marcusrag index --embedder hash --store memory --llm-backend fake --data-dir data/pdfs
python -m marcusrag ask "问题" --embedder hash --store memory --llm-backend fake
python -m unittest discover -s tests -v
```

---

## 架构

```
marcusrag/
  config.py       配置（YAML / CLI 覆盖），含 chunk_size 与模型 token 上限的对齐告警
  documents.py    PDF / TXT / MD 加载，逐文件累积（修复"只索引最后一个文件"）
  chunking.py     语言感知的递归切块，校验 overlap < size
  embeddings.py   HfEmbedder（延迟导入 torch）/ HashEmbedder（离线）
  vectorstore.py  ChromaStore（持久化）/ MemoryStore（JSON）
  bm25.py         纯 Python BM25 索引，中文按字二元组切分
  retrieval.py    混合检索（RRF）/ MMR / 相似度，串接可插拔重排
  rerank.py       LexicalReranker（idf 覆盖率 + 邻近度，无模型）/ CrossEncoderReranker
  prompts.py      指令与证据隔离（system 消息 + <context> 包裹）
  citations.py    [文件:页码] 解析、真实性校验、答案引用回链与证据高亮
  llm.py          Ollama HTTP 后端 / Fake 后端
  pipeline.py     ingest 与问答编排（依赖全部可注入）
  evaluate.py     JSONL 评测集执行器与指标计算
  cli.py          index / ask / eval / info 四个子命令
```

数据流：

```
documents → chunks → embeddings → store
                                     ↓
question → embeddings → 候选召回(fetch_k)：向量 ⊕ BM25 → RRF 融合
                                     ↓
                       rerank(词法 / cross-encoder) → top_k
                                     ↓
                        prompt（证据隔离）→ LLM → answer
                                     ↓
                       citation 校验 + 回链高亮 → answer + sources
```

索引目录按嵌入模型隔离（`index/<model>/<store>/`），切换模型不会导致维度不匹配崩溃；
`manifest.json` 记录文件指纹，重复执行 `index` 为幂等操作。

---

## 与旧版的差异（破坏性变更）

| 旧 | 新 |
| --- | --- |
| `python main.py` 内嵌全部逻辑 | `main.py` 仅转发到 `marcusrag.cli` |
| `src/*.py` + `loaders.py` | 已删除，功能拆分进 `marcusrag/` 各模块 |
| 硬编码路径与参数 | `config.yaml` + CLI 覆盖 |
| 循环导入 `main → src → main` | 单向依赖 `cli → pipeline → 组件` |
| 只索引最后一个 PDF | 所有文件累积索引 |
| 重复运行产生重复向量 | 确定性 chunk id + manifest 增量 |
| 答案不打印 | CLI 打印答案、来源与未验证引用 |
| 只有向量相似度 | 混合检索（向量 + BM25 + RRF）与可插拔重排 |
| 无法判断"改好了没有" | 35 条用例的评测集 + CI 门禁 |

---

## 配置

复制 `config.example.yaml` 为 `config.yaml`。关键项：

| 键 | 说明 |
| --- | --- |
| `data_dir` | 输入文档目录 |
| `embedding_model` | 嵌入模型；`marcusrag info` 会提示 chunk 是否超出模型 token 上限 |
| `chunk_size` / `chunk_overlap` | 默认按 `profile`（en/zh）取值，必须满足 overlap < size |
| `store` | `chroma`（持久化）或 `memory`（JSON，演示/测试） |
| `search_type` | `hybrid`（默认，向量 + BM25 经 RRF 融合）、`mmr` 或 `similarity` |
| `hybrid_vector_weight` / `hybrid_bm25_weight` / `rrf_k` | 混合检索的通道权重与 RRF 平滑常数 |
| `rerank` | `lexical`（默认，无模型）/ `cross-encoder` / `none` |
| `rerank_base_weight` | 词法重排保留首阶段分数的比例，1.0 等价于不改序 |
| `top_k` / `fetch_k` | 最终证据条数 / 候选召回条数 |
| `refuse_when_empty` | 无召回时直接回答"文档中未找到相关依据" |
| `verify_citations` | 校验答案引用是否出现在本次证据中，编造引用会被标记 |

---

## 评测

`evals/` 下是一套可复现的小语料（5 篇 Markdown）与 35 条用例，详见 [evals/README.md](evals/README.md)。

```bash
# 检索评测（无需 Ollama）
python -m marcusrag eval --data-dir evals/corpus --embedder hash --store memory \
  --search-type hybrid --rerank lexical --k 3

# 端到端（含答案质量），需要本地 Ollama
python -m marcusrag eval --data-dir evals/corpus --end-to-end --output report.json
```

本机实测（30 条可回答用例，`k=3`，`BAAI/bge-small-zh-v1.5` + Chroma，chunk 320/64）：

| 配置 | Recall@3 | Hit@3 | MRR |
| --- | --- | --- | --- |
| similarity / 无重排 | 1.000 | 1.000 | 0.967 |
| mmr / 无重排 | 0.967 | 0.967 | 0.950 |
| hybrid / 无重排 | 1.000 | 1.000 | **0.983** |
| hybrid + lexical（base_weight=0.3） | 1.000 | 1.000 | 0.950 |
| hybrid + lexical（base_weight=0.5） | 1.000 | 1.000 | 0.967 |
| hybrid + lexical（base_weight=0.7，默认） | 1.000 | 1.000 | **0.983** |
| hybrid + lexical（base_weight=0.9） | 1.000 | 1.000 | **0.983** |
| hybrid + cross-encoder（bge-reranker-base） | 1.000 | 1.000 | **0.983** |

`hash` 嵌入 + 内存库（CI 同款配置）下三项均为 1.000，作为回归门禁。

读法（诚实版）：

- **收益主要来自混合检索**：`hybrid` 比 `mmr` 在 Recall 上多捞回一条，MRR 也从 0.950 升到 0.983。
- **重排在这套语料上没有测出增益**：`base_weight=0.7/0.9` 与"不重排"打平；`0.3` 这种强词法覆盖反而把正确片段往下压。cross-encoder 同样只是打平，却要多付 1.1 GB 模型和推理时间。
- 因此默认走 `lexical` + `base_weight=0.7`（近乎保守调整），需要更强语义重排时再切换 cross-encoder；真实长文档上重排的价值需要用生产语料复测，不要照搬这张表。

---

## 测试

```bash
python -m unittest discover -s tests -v
```

37 条用例，覆盖的回归点：多文件全部入库、索引幂等、增量跳过、空索引拒答、引用校验（含伪造引用）、
prompt 证据隔离、chunk 参数校验、BM25 中文切分、RRF 融合顺序、重排的 base 分数融合与边界（常量分数、
缺 base、权重 0/1）。CI 在 3.10 / 3.11 / 3.12 上跑全部单测，并用 hash + 内存库执行评测门禁
（Recall@3 ≥ 0.90、MRR ≥ 0.85、编造引用为 0）。

---

## 后续路线

- S3：多轮对话记忆、PDF 语料上的页码级引用评测、批量问答回归脚本（把评测集接到真实业务文档）

---

## License & Attribution

本项目以 MIT License 发布，见 [LICENSE](LICENSE)。

原 README 中的第三方声明在此保留：

- 生成模型经 [Ollama](https://ollama.com/) 调用，模型权重遵循其各自许可（如 Qwen 系列遵循 Qwen License）。
- 若用 AlphaGo / 强化学习相关论文作为测试语料，请注意其原文版权归属，仅作本地检索实验使用，不随本仓库分发。
