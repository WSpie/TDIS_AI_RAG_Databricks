# 数据使用与归属（Data Usage & Ownership）

> 本文档梳理 TDIS RAG 系统涉及的全部数据资产：来源、存储位置、归属层级、血缘关系，
> 以及每个资产在代码中的硬编码引用位置。目的是为**重构 / 迁移**提供完整底图。
>
> 源数据性质：**公开灾害报告**（public disaster reports）。检索文本会发送给外部 LLM
> （OpenAI / Databricks 托管模型），由于内容公开，不涉及数据外流合规问题。

---

## 1. 数据血缘全景

```
【PROD 数据湖】 Azure ADLS Gen2  —— 本项目只读
abfss://tdis-data-bronze@tdisproddatalakehouse.dfs.core.windows.net/RAG_files/
   ├── optimized_chunks/*/chunks.jsonl                ← 已切分文本块
   └── optimized_embeddings/DMRetriever-33M/*/         ← 预计算向量
         ├── embeddings.npy
         └── chunk_ids.json
              │
              │  0_preprocess.ipynb   (读 PROD → 写 DEV)
              ▼
【DEV Unity Catalog】 tdis_dev_data_catalog.tdir.*
   ├── optimized_chunks_text                  ← 仅文本
   ├── optimized_embeddings_dmretriever33m    ← 仅向量
   └── optimized_rag_chunks  ★核心宽表★       ← text + embedding (按 chunk_id inner join)
              │
       ┌──────┴───────────────────────────┐
       ▼                                   ▼
 [向量检索路]                          [关键词检索路]  2_keyword_search_BM25.ipynb
 optimized_rag_chunks_vs               由 optimized_rag_chunks 派生（tokenize→倒排）:
 (Vector Search 索引)                   ├ optimized_kw_postings    倒排表 term→chunk(tf)
                                        ├ optimized_kw_df          document frequency
                                        ├ optimized_kw_doc_stats   每 chunk 文档长度
                                        └ optimized_kw_meta        N, avgdl
       └──────┬───────────────────────────┘
              ▼  dbx_hybrid_search.py  (RRF 融合, rrf_k=20)
        top_n（默认 20）候选
              ▼  dbx_rerank.py  (cross-encoder rerank)
        top_k（默认 10）chunks（拼接为 CONTEXT）
              ▼  LLMs_call.py  (build_messages → 生成)
   ┌──────────┴──────────┐
   ▼                     ▼
 OpenAI API         Databricks Serving Endpoints
 (config.yaml        (adb-3300405005568038.18.azuredatabricks.net,
  gpt_api_key)        DATABRICKS_TOKEN)
              ▼
   答案 →（评测时）qa_dict_eval 表
```

---

## 1b. 仓库结构（重构后）

```
.
├── query_nbk.ipynb     # 查询流程：retrieve → RRF → rerank → LLM →（可选）批量评测
├── setup_nbk.ipynb     # 一次性构建：下载模型 → 预处理 → 向量索引 → BM25 建表
├── scripts/            # 可复用 .py 模块
│   ├── dbx_vector_search.py        # 向量检索
│   ├── dbx_keyword_search_bm25.py  # BM25 关键词检索
│   ├── dbx_hybrid_search.py        # RRF 融合
│   ├── dbx_rerank.py               # cross-encoder rerank（新增）
│   ├── LLMs_call.py                # LLM 调用（OpenAI / Databricks）
│   └── rag_pipeline.py             # 端到端编排 rag_pipe_main（新增）
├── archives/           # 旧 notebook + 冗余脚本（dbx_retrieve.py 已被 dbx_vector_search.py 取代）
└── DATA.md
```

> 全程只使用 `optimized_*` 家族数据；旧索引 `rag_chunks_vs` 等不再被引用（仅存于 archives/）。

---

## 2. 数据资产清单与归属

| # | 资产 | 位置 / FQN | 归属层 | 产生方 / 拥有方 | 可重建 |
|---|------|-----------|--------|----------------|:-----:|
| 1 | 原始 chunks + embeddings | `abfss://tdis-data-bronze@tdisproddatalakehouse.dfs.core.windows.net/RAG_files/` | **PROD 数据湖 (bronze)** | 上游 TDIS 生产管线，本项目**只读** | — (源) |
| 2 | 嵌入模型 DMRetriever-33M | UC Volume `/Volumes/tdis_dev_data_catalog/tdir/tdir/models/DMRetriever-33M` | DEV（外部下载） | HuggingFace `DMIR01/DMRetriever-33M`，第三方 | ✅ 重新下载 |
| 2b | Rerank 模型（cross-encoder） | UC Volume `/Volumes/tdis_dev_data_catalog/tdir/tdir/models/ms-marco-MiniLM-L-6-v2` | DEV（外部下载） | HuggingFace `cross-encoder/ms-marco-MiniLM-L-6-v2`，第三方 | ✅ 重新下载 |
| 3a | 文本表 | `tdis_dev_data_catalog.tdir.optimized_chunks_text` | DEV Catalog | `0_preprocess` 派生 | ✅ |
| 3b | 向量表 | `tdis_dev_data_catalog.tdir.optimized_embeddings_dmretriever33m` | DEV Catalog | `0_preprocess` 派生 | ✅ |
| 3c | **核心宽表** | `tdis_dev_data_catalog.tdir.optimized_rag_chunks` | DEV Catalog | `0_preprocess` 派生（3a ⋈ 3b） | ✅ |
| 4 | 向量索引 | `tdis_dev_data_catalog.tdir.optimized_rag_chunks_vs` | DEV Catalog | Vector Search（基于 3c） | ✅ |
| 5a | BM25 倒排表 | `tdis_dev_data_catalog.tdir.optimized_kw_postings` | DEV Catalog | `2_keyword_search_BM25` 派生 | ✅ |
| 5b | BM25 df 表 | `tdis_dev_data_catalog.tdir.optimized_kw_df` | DEV Catalog | `2_keyword_search_BM25` 派生 | ✅ |
| 5c | BM25 doc_stats | `tdis_dev_data_catalog.tdir.optimized_kw_doc_stats` | DEV Catalog | `2_keyword_search_BM25` 派生 | ✅ |
| 5d | BM25 meta | `tdis_dev_data_catalog.tdir.optimized_kw_meta` | DEV Catalog | `2_keyword_search_BM25` 派生 | ✅ |
| 6 | 评测结果 | `tdis_dev_data_catalog.tdir.qa_dict_eval` | DEV Catalog | `6_RAG_test_eval` 输出 | ✅ |
| 7 | 生成 LLM | OpenAI / Databricks serving endpoints | 外部服务 | 凭据存 `config.yaml`（已 gitignore） | — |

**Source of truth**：只有 **#1（PROD 湖 `RAG_files/`）** 是权威数据源。#3–#6 全部是可从 #1 一键重跑生成的派生资产。

---

## 3. 硬编码引用位置（迁移时必须修改的清单）

重构后，运行时引用集中在 `scripts/` + 两个 notebook（旧引用全部归档到 `archives/`，不再生效）。下表给出活跃代码中每个标识符的位置，供迁移时全量替换。

### 3.1 Catalog / Schema 前缀 `tdis_dev_data_catalog.tdir`

| 文件 | 说明 |
|------|------|
| `setup_nbk.ipynb` | 写入 3a/3b/3c、5a–5d；向量索引 FQN 说明 |
| `query_nbk.ipynb` | 评测写入 `qa_dict_eval` |
| `scripts/dbx_keyword_search_bm25.py` | 模块顶部 `CHUNKS/POST/DFT/DST/META` 常量 |
| `scripts/dbx_vector_search.py` | `MODEL_DIR` |
| `scripts/dbx_rerank.py` | `RERANKER_DIR` |
| `scripts/dbx_hybrid_search.py` | `DEFAULT_INDEX_FQN`、`DEFAULT_CHUNKS_TABLE` |

### 3.2 PROD 源路径

- `setup_nbk.ipynb` → `base = "abfss://tdis-data-bronze@tdisproddatalakehouse.dfs.core.windows.net/RAG_files/"`

### 3.3 模型 Volume 路径

- `scripts/dbx_vector_search.py`、`setup_nbk.ipynb` → `/Volumes/.../models/DMRetriever-33M`
- `scripts/dbx_rerank.py`、`setup_nbk.ipynb` → `/Volumes/.../models/ms-marco-MiniLM-L-6-v2`

### 3.4 LLM 端点 / 凭据

- `scripts/LLMs_call.py` → `base_url="https://adb-3300405005568038.18.azuredatabricks.net/serving-endpoints"`，`DATABRICKS_TOKEN`（读自 `config.yaml`）
- `scripts/LLMs_call.py`、`scripts/rag_pipeline.py`、`query_nbk.ipynb` → `gpt_api_key`（读自 `config.yaml`）

---

## 4. 遗留 / 待清理资产

| 资产 | 状态 | 建议 |
|------|------|------|
| `tdis_dev_data_catalog.tdir.rag_chunks_vs`（旧向量索引，无 `optimized_` 前缀） | 活跃代码已不再引用，仅遗留在 `archives/` 的旧 notebook 中 | 迁移前确认无人依赖后删除该索引 |
| `archives/`（旧 notebook + `dbx_retrieve.py`） | 已被 `query_nbk` / `setup_nbk` / `scripts/` 取代 | 仅作历史参考，可在确认后清理 |

---

## 5. 迁移注意事项（Migration Checklist）

1. **集中配置**：catalog/schema/路径/端点仍为硬编码（§3，已收敛到 `scripts/` + 两个 notebook）。迁移前建议进一步抽到统一配置（如 `config.yaml` 或 `scripts/settings.py` 常量模块），把引用收敛到一处，避免漏改。
2. **跨环境边界**：唯一跨 prod/dev 依赖是「读 PROD bronze 湖 → 写 DEV catalog」。迁移目标 catalog 需保证对 `tdis-data-bronze` 容器有读权限。
3. **派生资产无需搬运**：#3–#6 不必跨环境拷贝，迁移后在目标环境**重跑 `setup_nbk`**（预处理 → 建索引 → BM25 建表）即可重建。真正要保留/迁移的只有源数据访问权与代码。
4. **模型一致性**：query 端与文档端必须用**同一** DMRetriever-33M（384 维）。Volume 路径含 catalog 名（`/Volumes/tdis_dev_data_catalog/...`），迁移 catalog 时此路径同步变更，且需重新 `snapshot_download`（嵌入模型 + reranker 各一个）。
5. **重建顺序依赖**：
   `0_preprocess`（建 #3）→ Vector Search 建 #4 → `2_keyword_search_BM25`（建 #5）→ `3_hybrid_search` 可用 →（可选）`6_RAG_test_eval`（建 #6）。
6. **凭据**：`config.yaml`（含 `gpt_api_key`、`DATABRICKS_TOKEN`）已被 `.gitignore` 排除，不入库；迁移目标环境需单独配置。
