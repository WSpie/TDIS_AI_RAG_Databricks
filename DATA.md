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
              ▼  dbx_hybrid_search.py  (RRF 融合, rrf_k=20/60)
        top-10 chunks（拼接为 CONTEXT）
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

## 2. 数据资产清单与归属

| # | 资产 | 位置 / FQN | 归属层 | 产生方 / 拥有方 | 可重建 |
|---|------|-----------|--------|----------------|:-----:|
| 1 | 原始 chunks + embeddings | `abfss://tdis-data-bronze@tdisproddatalakehouse.dfs.core.windows.net/RAG_files/` | **PROD 数据湖 (bronze)** | 上游 TDIS 生产管线，本项目**只读** | — (源) |
| 2 | 嵌入模型 DMRetriever-33M | UC Volume `/Volumes/tdis_dev_data_catalog/tdir/tdir/models/DMRetriever-33M` | DEV（外部下载） | HuggingFace `DMIR01/DMRetriever-33M`，第三方 | ✅ 重新下载 |
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

迁移涉及的标识符当前**散落在 11 个文件、约 38 处**，没有集中配置。下表给出每个标识符出现的文件，供迁移时全量替换。

### 3.1 Catalog / Schema 前缀 `tdis_dev_data_catalog.tdir`

| 文件 | 说明 |
|------|------|
| `0_preprocess.ipynb` | 写入 3a/3b/3c，校验 count |
| `1_vector_search.ipynb` | 查询核心宽表、INDEX_FQN（含旧索引） |
| `2_keyword_search_BM25.ipynb` | 构建 + 读取 5a–5d、核心宽表 |
| `2_keyword_search_DB.ipynb` | 旧索引 `rag_chunks_vs` |
| `3_hybrid_search.ipynb` | INDEX_FQN、CHUNKS_TABLE |
| `6_RAG_test_eval.ipynb` | 写入 `qa_dict_eval` |
| `dbx_keyword_search_bm25.py` | 模块顶部 `CHUNKS/POST/DFT/DST/META` 常量 |
| `dbx_retrieve.py` | `MODEL_DIR`（Volume 路径含 catalog） |
| `dbx_vector_search.py` | `MODEL_DIR` |
| `dbx_hybrid_search.py` | `DEFAULT_INDEX_FQN`、`DEFAULT_CHUNKS_TABLE` |

### 3.2 PROD 源路径

- `0_preprocess.ipynb` → `base = "abfss://tdis-data-bronze@tdisproddatalakehouse.dfs.core.windows.net/RAG_files/"`

### 3.3 模型 Volume 路径

- `dbx_retrieve.py`、`dbx_vector_search.py`、`1_vector_search.ipynb`
  → `/Volumes/tdis_dev_data_catalog/tdir/tdir/models/DMRetriever-33M`

### 3.4 LLM 端点 / 凭据

- `LLMs_call.py` → `base_url="https://adb-3300405005568038.18.azuredatabricks.net/serving-endpoints"`，`DATABRICKS_TOKEN`（读自 `config.yaml`）
- `LLMs_call.py`、`4_query_my_LLMs (backups).ipynb`、`5_RAG_pipeline_demo`、`6_RAG_test_eval` → `gpt_api_key`（读自 `config.yaml`）

---

## 4. 遗留 / 待清理资产

| 资产 | 状态 | 建议 |
|------|------|------|
| `tdis_dev_data_catalog.tdir.rag_chunks_vs`（旧向量索引，无 `optimized_` 前缀） | 出现在 `1_vector_search.ipynb` 早期 cell、`2_keyword_search_DB.ipynb`；现行代码默认用 `optimized_rag_chunks_vs` | 迁移前确认无人依赖后删除 |
| 根目录归档脚本（`dbx_*.py` 之外的旧 notebook） | 早期版本，已被 `optimized_*` 流程取代 | 迁移时归并到 `archives/` 或删除 |

---

## 5. 迁移注意事项（Migration Checklist）

1. **集中配置**：当前 catalog/schema/路径/端点全为硬编码（§3）。迁移前建议抽到统一配置（如 `config.yaml` 或一个 `settings.py`/常量模块），把 38 处引用收敛到一处，避免漏改。
2. **跨环境边界**：唯一跨 prod/dev 依赖是「读 PROD bronze 湖 → 写 DEV catalog」。迁移目标 catalog 需保证对 `tdis-data-bronze` 容器有读权限。
3. **派生资产无需搬运**：#3–#6 不必跨环境拷贝，迁移后在目标环境**重跑** `0_preprocess` → `2_keyword_search_BM25` →（建索引）即可重建。真正要保留/迁移的只有源数据访问权与代码。
4. **模型一致性**：query 端与文档端必须用**同一** DMRetriever-33M（384 维）。Volume 路径含 catalog 名（`/Volumes/tdis_dev_data_catalog/...`），迁移 catalog 时此路径同步变更，且需重新 `snapshot_download`。
5. **重建顺序依赖**：
   `0_preprocess`（建 #3）→ Vector Search 建 #4 → `2_keyword_search_BM25`（建 #5）→ `3_hybrid_search` 可用 →（可选）`6_RAG_test_eval`（建 #6）。
6. **凭据**：`config.yaml`（含 `gpt_api_key`、`DATABRICKS_TOKEN`）已被 `.gitignore` 排除，不入库；迁移目标环境需单独配置。
