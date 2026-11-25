# MarcusRAG

基于本地文档的**证据优先**检索增强问答（RAG）系统：把 PDF / TXT / Markdown 切块入库，
检索后只依据召回到的片段作答，并强制校验答案里的引用是否真实存在于证据中。

- 向量库：Chroma（持久化）/ 内存 JSON（零依赖演示）
- 嵌入：sentence-transformers（`all-MiniLM-L6-v2` 等）/ 哈希嵌入（离线兜底）
- 生成：本地 Ollama（默认 `qwen2.5:3b`），通过 HTTP API 调用，无需 LangChain provider 包

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

# 4. 提问
python -m marcusrag ask "这篇论文的核心贡献是什么？"

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
  retrieval.py    相似度或 MMR 重排
  prompts.py      指令与证据隔离（system 消息 + <context> 包裹）
  citations.py    [文件:页码] 解析与真实性校验
  llm.py          Ollama HTTP 后端 / Fake 后端
  pipeline.py     ingest 与问答编排（依赖全部可注入）
  cli.py          index / ask / info 三个子命令
```

数据流：

```
documents → chunks → embeddings → store
                                     ↓
question → embeddings → retrieve(fetch_k) → MMR(top_k) → prompt → LLM → answer
                                                          ↓
                                             citation verification → answer + sources
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

---

## 配置

复制 `config.example.yaml` 为 `config.yaml`。关键项：

| 键 | 说明 |
| --- | --- |
| `data_dir` | 输入文档目录 |
| `embedding_model` | 嵌入模型；`marcusrag info` 会提示 chunk 是否超出模型 token 上限 |
| `chunk_size` / `chunk_overlap` | 默认按 `profile`（en/zh）取值，必须满足 overlap < size |
| `store` | `chroma`（持久化）或 `memory`（JSON，演示/测试） |
| `search_type` | `mmr`（默认，去冗余）或 `similarity` |
| `top_k` / `fetch_k` | 最终证据条数 / 候选召回条数 |
| `refuse_when_empty` | 无召回时直接回答"文档中未找到相关依据" |
| `verify_citations` | 校验答案引用是否出现在本次证据中，编造引用会被标记 |

---

## 测试

```bash
python -m unittest discover -s tests -v
```

覆盖的回归点：多文件全部入库、索引幂等、增量跳过、空索引拒答、引用校验（含伪造引用）、
prompt 证据隔离、chunk 参数校验。

---

## 后续路线

- S2：评测集（Recall@k / MRR / 引用正确率）+ 混合检索（BM25 + 向量）+ rerank
- S3：多轮对话记忆、引用定位高亮、批量问答回归脚本

---

## License & Attribution

本项目以 MIT License 发布，见 [LICENSE](LICENSE)。

原 README 中的第三方声明在此保留：

- 生成模型经 [Ollama](https://ollama.com/) 调用，模型权重遵循其各自许可（如 Qwen 系列遵循 Qwen License）。
- 若用 AlphaGo / 强化学习相关论文作为测试语料，请注意其原文版权归属，仅作本地检索实验使用，不随本仓库分发。
