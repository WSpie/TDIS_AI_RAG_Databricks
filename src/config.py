# src/config.py
# Central place for all paths, table names, index FQNs, and model ids.
# Notebooks should import constants from here instead of hard-coding strings.

# ---- ADLS (data lake) ----
ADLS_BASE = "abfss://tdis-data-bronze@tdisproddatalakehouse.dfs.core.windows.net/RAG_files/"

# New-pipeline input: raw whole-document txts (already includes HH + Qs)
SRC_TXT_DIR = ADLS_BASE + "converted_raw_txts/"

# Legacy assets (kept only for before/after comparison)
HH_OPT_CHUNKS = ADLS_BASE + "optimized_HH_chunks/"
HH_OPT_EMB    = ADLS_BASE + "optimized_HH_embeddings/"

# ---- Unity Catalog ----
CATALOG = "tdis_dev_data_catalog"
SCHEMA  = "tdir"


def t(name: str) -> str:
    # Build a fully-qualified table/index name
    return f"{CATALOG}.{SCHEMA}.{name}"


# New (rebuild) tables / index -- unified `rag_` prefix (replaces optimized_* / stage*)
TBL_CHUNKS = t("rag_chunks")        # chunk text (+ embedding column after build_embeddings)
VS_INDEX   = t("rag_chunks_vs")     # vector search index
KW_POST    = t("rag_kw_postings")
KW_DF      = t("rag_kw_df")
KW_DST     = t("rag_kw_doc_stats")
KW_META    = t("rag_kw_meta")

# ---- Models ----
EMBED_MODEL     = "DMIR01/DMRetriever-33M"
EMBED_MODEL_DIR = "/Volumes/tdis_dev_data_catalog/tdir/tdir/models/DMRetriever-33M"
# Small open-source English cross-encoder reranker
RERANK_MODEL    = "cross-encoder/ms-marco-MiniLM-L-6-v2"

# ---- Chunking ----
CHUNK_MAX_LEN = 1000
