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
├── ingest_nbk.ipynb    # 增量注入：新 chunks+embeddings 上线 → 同步向量索引
├── scripts/            # 可复用 .py 模块
│   ├── settings.py                 # ★ 所有 path / 表名 / 端点的单一来源（可用 TDIS_* 环境变量覆盖）
│   ├── lake_io.py                  # ★ 共享读取器：chunks.jsonl + embeddings.npy 解码
│   ├── ingest.py                   # ★ 增量注入（按 chunk_id MERGE）+ 同步 VS 索引 + 重建 BM25
│   ├── dbx_vector_search.py        # 向量检索
│   ├── dbx_keyword_search_bm25.py  # BM25 关键词检索
│   ├── dbx_hybrid_search.py        # RRF 融合
│   ├── dbx_rerank.py               # cross-encoder rerank
│   ├── LLMs_call.py                # LLM 调用（OpenAI / Databricks）
│   └── rag_pipeline.py             # 端到端编排 rag_pipe_main
├── archives/           # 旧 notebook + 冗余脚本
└── DATA.md
```

> 全程只使用 `optimized_*` 家族数据；所有 FQN/路径集中在 `scripts/settings.py`，换环境改一处即可。
> 旧索引 `rag_chunks_vs` 等不再被引用（仅存于 archives/）。

### 配置集中化（settings.py）

`scripts/settings.py` 是所有非机密标识符的单一来源，支持环境变量覆盖以便跨 dev/prod：

| 环境变量 | 默认值 | 作用 |
|----------|--------|------|
| `TDIS_CATALOG` | `tdis_dev_data_catalog` | Catalog |
| `TDIS_SCHEMA` | `tdir` | Schema |
| `TDIS_VOLUME` | `tdir` | UC Volume 名（存模型） |
| `TDIS_LAKE_BASE` | `abfss://tdis-data-bronze@.../RAG_files/` | PROD 源数据根 |
| `TDIS_VS_ENDPOINT` | （空，需设置） | Vector Search endpoint |
| `TDIS_DBX_BASE_URL` | `https://adb-3300405005568038.18...` | Databricks serving endpoint |

机密（`gpt_api_key`、`DATABRICKS_TOKEN`）仍只放 `config.yaml`。

### 增量注入（ingest.py / ingest_nbk）

HPC 处理好的新 `chunks + embeddings` → `ingest.load_new()` 读取 → `ingest.ingest()` 按 `chunk_id`
幂等 MERGE 进 `optimized_chunks_text / optimized_embeddings_dmretriever33m / optimized_rag_chunks`。
混合检索两路都要看到新数据，所以 `ingest()` 默认**同时**刷新两侧：
- **关键词侧**：`rebuild_keyword_index=True` 重建 BM25 倒排表（必做，否则新 chunks 在关键词召回中检索不到）；
- **向量侧**：`sync_index=True` 触发 Vector Search 索引同步。

索引 `optimized_rag_chunks_vs` 是 **Triggered** Delta Sync（endpoint `tdis-ai-rag-light`），不会自动更新，
因此同步是必需的。多次小批量注入时可两个开关都设 `False`，最后统一 `rebuild_bm25()` + `sync_vector_index()` 一次。

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

**重构后已全部集中到 `scripts/settings.py`** —— 所有 catalog/schema、表名、源路径、模型路径、
索引 FQN、serving 端点都在这一个文件定义，其余模块/notebook 一律 `import settings` 取用。
迁移时**只需改 `settings.py`**（或设置 `TDIS_*` 环境变量），不再需要逐文件替换。

| 类别 | 在 settings.py 中的项 |
|------|----------------------|
| Catalog/Schema/Volume | `CATALOG` / `SCHEMA` / `VOLUME`（均可被 `TDIS_*` 覆盖） |
| optimized_* 表名 | `CHUNKS_TEXT_TABLE` / `EMBEDDINGS_TABLE` / `RAG_CHUNKS_TABLE` / `KW_*_TABLE` / `QA_EVAL_TABLE` |
| PROD 源路径 | `LAKE_BASE` / `CHUNKS_GLOB` / `EMBED_NPY_GLOB` / `EMBED_IDS_GLOB` |
| 模型 Volume 路径 | `EMBED_MODEL_DIR` / `RERANKER_MODEL_DIR`（+ 对应 HF id） |
| 向量索引 | `VS_INDEX_FQN` / `VS_ENDPOINT` / `EMBED_DIM` |
| LLM 端点 | `DBX_BASE_URL` |

机密仍只在 `config.yaml`：`gpt_api_key`、`DATABRICKS_TOKEN`（已 gitignore）。

---

## 4. 遗留 / 待清理资产

| 资产 | 状态 | 建议 |
|------|------|------|
| `tdis_dev_data_catalog.tdir.rag_chunks_vs`（旧向量索引，无 `optimized_` 前缀） | 活跃代码已不再引用，仅遗留在 `archives/` 的旧 notebook 中 | 迁移前确认无人依赖后删除该索引 |
| `archives/`（旧 notebook + `dbx_retrieve.py`） | 已被 `query_nbk` / `setup_nbk` / `scripts/` 取代 | 仅作历史参考，可在确认后清理 |

---

## 5. 迁移注意事项（Migration Checklist）

1. **集中配置（已完成）**：catalog/schema/路径/端点已全部收敛到 `scripts/settings.py`，并支持 `TDIS_*` 环境变量覆盖。迁移到新环境时改这一个文件即可，无需逐文件替换。
2. **跨环境边界**：唯一跨 prod/dev 依赖是「读 PROD bronze 湖 → 写 DEV catalog」。迁移目标 catalog 需保证对 `tdis-data-bronze` 容器有读权限。
3. **派生资产无需搬运**：#3–#6 不必跨环境拷贝，迁移后在目标环境**重跑 `setup_nbk`**（预处理 → 建索引 → BM25 建表）即可重建；后续新增数据走 `ingest_nbk`。真正要保留/迁移的只有源数据访问权与代码。
4. **模型一致性**：query 端与文档端必须用**同一** DMRetriever-33M（384 维）。Volume 路径含 catalog 名（`/Volumes/tdis_dev_data_catalog/...`），迁移 catalog 时此路径同步变更，且需重新 `snapshot_download`（嵌入模型 + reranker 各一个）。
5. **重建顺序依赖**：
   `0_preprocess`（建 #3）→ Vector Search 建 #4 → `2_keyword_search_BM25`（建 #5）→ `3_hybrid_search` 可用 →（可选）`6_RAG_test_eval`（建 #6）。
6. **凭据**：`config.yaml`（含 `gpt_api_key`、`DATABRICKS_TOKEN`）已被 `.gitignore` 排除，不入库；迁移目标环境需单独配置。
