# TDIS AI RAG (Databricks)

A Retrieval-Augmented Generation system for **disaster-risk question answering** over TDIS
(Texas Disaster Information System) reports. It runs on Databricks and combines dense vector
search, BM25 keyword search, reciprocal-rank fusion, cross-encoder reranking, and an LLM that
answers strictly from the retrieved context.

The source documents are **public disaster reports**. The default answer model is OpenAI
**gpt-4o-mini**.

---

## Quickstart

### Prerequisites

1. A Databricks workspace with this repo cloned as a **Databricks Repo**, access to the
   `tdis_dev_data_catalog` Unity Catalog, and a Vector Search endpoint.
2. A `config.yaml` at the repo root holding the secrets (git-ignored — never committed):

   ```yaml
   gpt_api_key: "sk-..."          # OpenAI key, used by the default gpt-4o-mini path
   DATABRICKS_TOKEN: "dapi..."    # only needed if you opt into Databricks-served models
   ```
3. Python deps available on the cluster: `transformers`, `torch`, `openai`, `markdown`,
   `tqdm`, `mlflow`, `databricks-vectorsearch`.

### 1. One-time setup — `setup_nbk.ipynb`

Run top to bottom **once per environment** (or when the full source data changes). It:
- downloads the embedding + reranker models into the UC Volume,
- builds the core table `optimized_rag_chunks` from the data lake,
- (guidance to) create the Vector Search index,
- builds the BM25 tables.

### 2. Ask questions — `query_nbk.ipynb`

Set `query` in the Config cell and run. It executes hybrid retrieval → rerank → gpt-4o-mini
and renders the answer. The bottom section runs a fixed QA set for batch evaluation and writes
results to `optimized_rag_chunks` → `qa_dict_eval`.

### 3. Add new documents — `ingest_nbk.ipynb`

Point `NEW_BASE` at a batch of upstream-processed chunks + embeddings (e.g. produced on HPC),
then `ingest()` upserts them and refreshes both retrieval paths (BM25 incrementally + Vector
Search sync).

---

## Where things are stored

Everything resolves through **`scripts/settings.py`** (override with `TDIS_*` env vars). The
authoritative source of truth is the data lake; everything else is derived and reproducible.

| Asset | Location | Notes |
|-------|----------|-------|
| Source chunks + embeddings | `abfss://tdis-data-bronze@tdisproddatalakehouse.dfs.core.windows.net/RAG_files/` | PROD data lake, **read-only** |
| Embedding model | UC Volume `…/models/DMRetriever-33M` | HF `DMIR01/DMRetriever-33M`, 384-dim |
| Reranker model | UC Volume `…/models/ms-marco-MiniLM-L-6-v2` | HF `cross-encoder/ms-marco-MiniLM-L-6-v2` |
| Core wide table | `tdis_dev_data_catalog.tdir.optimized_rag_chunks` | `chunk_id, text, embedding, …` — VS source |
| Vector Search index | `…tdir.optimized_rag_chunks_vs` | Delta Sync (**Triggered**), endpoint `tdis-ai-rag-light` |
| BM25 tables | `…tdir.optimized_kw_postings / _df / _doc_stats / _meta` | inverted index for keyword search |
| Eval results | `…tdir.qa_dict_eval` | written by `query_nbk` batch eval |

Secrets (`gpt_api_key`, `DATABRICKS_TOKEN`) live only in `config.yaml`.

### Repository layout

```
.
├── query_nbk.ipynb     # ask questions: retrieve → RRF → rerank → LLM → (optional) batch eval
├── setup_nbk.ipynb     # one-time build: download models → preprocess → VS index → BM25
├── ingest_nbk.ipynb    # incremental ingest of new chunks+embeddings → refresh indexes
├── scripts/
│   ├── settings.py                 # single source of truth for paths/tables/endpoints
│   ├── lake_io.py                  # shared readers (chunks.jsonl + embeddings.npy)
│   ├── ingest.py                   # MERGE upsert, incremental BM25 update, VS index sync
│   ├── dbx_vector_search.py        # dense vector retrieval
│   ├── dbx_keyword_search_bm25.py  # BM25 keyword retrieval (domain term weights)
│   ├── dbx_hybrid_search.py        # RRF fusion of vector + BM25
│   ├── dbx_rerank.py               # cross-encoder reranking
│   ├── LLMs_call.py                # LLM clients (OpenAI + Databricks)
│   └── rag_pipeline.py             # rag_pipe_main: end-to-end orchestration
├── archives/           # legacy notebooks/scripts (historical reference)
├── DATA.md             # detailed data lineage, ownership, migration notes
└── README.md
```

---

## Pipeline / flow design

```
query
  │
  ▼  embed (DMRetriever-33M, 384-dim)
  ├─────────────────────────────┐
  ▼                             ▼
vector search                 BM25 keyword search
(optimized_rag_chunks_vs)     (optimized_kw_* tables, domain term weights)
  └──────────────┬──────────────┘
                 ▼  RRF fusion              (dbx_hybrid_search.hybrid_search_rrf)
            top_n candidates
                 ▼  cross-encoder rerank    (dbx_rerank.rerank_dataframe)
            top_k context chunks
                 ▼  prompt + generate       (rag_pipeline.rag_pipe_main → gpt-4o-mini)
              answer
```

1. **Retrieve (two paths).**
   - *Vector*: the query is embedded with DMRetriever-33M and matched against the Vector Search
     index via the Spark SQL `vector_search()` function.
   - *BM25*: a Spark BM25 implementation over the inverted-index tables, with static
     disaster-domain term weights (e.g. `fema`, `flood`, `harris`) plus query-adaptive boosts.
2. **Fuse (RRF).** The two ranked lists are merged by Reciprocal Rank Fusion, deduplicated, and
   tagged by source (`VEC` / `KW` / `VEC+KW`).
3. **Rerank.** A cross-encoder re-scores each `(query, chunk)` pair and keeps the top `k` — this
   is the precision step that decides what the LLM actually sees.
4. **Generate.** `build_messages` assembles the reranked chunks into a strict, context-only
   prompt: the model must answer **only** from the provided context and returns
   `OUT_OF_KNOWLEDGE` when the context is insufficient (no outside knowledge, no fabrication).

`rag_pipe_main()` wires all four stages together; default parameters are
`top_each=100, top_n=20, rerank_top_k=10, rrf_k=20`.

### Configuration

`scripts/settings.py` centralizes every path, table name, and endpoint, each overridable via an
environment variable so the same code runs across environments:

| Env var | Default | Purpose |
|---------|---------|---------|
| `TDIS_CATALOG` / `TDIS_SCHEMA` / `TDIS_VOLUME` | `tdis_dev_data_catalog` / `tdir` / `tdir` | Unity Catalog location |
| `TDIS_LAKE_BASE` | `abfss://…/RAG_files/` | source data lake root |
| `TDIS_VS_ENDPOINT` | `tdis-ai-rag-light` | Vector Search endpoint |
| `TDIS_DBX_BASE_URL` | `https://adb-…/serving-endpoints` | Databricks serving base URL |

Migrating to a new environment is a matter of editing `settings.py` (or setting these vars) —
no per-file edits required. See **DATA.md** for full data lineage, ownership, and migration notes.

---

## Notes

- **LLM choice.** The pipeline defaults to OpenAI **gpt-4o-mini**. The Databricks-served models
  were dropped from the default workflow; the dual-backend code path still exists in
  `LLMs_call.py` if needed (`use_dbx_model=True`).
- **Index sync.** `optimized_rag_chunks_vs` is a *Triggered* Delta Sync index, so new rows are
  only searchable after a sync — `ingest()` triggers it automatically.
- **BM25 freshness.** Ingestion updates BM25 incrementally (only the affected chunks are
  re-tokenized). Use `rebuild_bm25()` for a full rebuild/repair.
