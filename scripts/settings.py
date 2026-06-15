# settings.py
# Single source of truth for all non-secret paths / identifiers.
# Change environment by editing the defaults here, or by setting the TDIS_* env vars.
# Secrets (gpt_api_key, DATABRICKS_TOKEN) stay in config.yaml.

import os

# -----------------------------------------------------------------------------
# Base identifiers (env-overridable so the same code runs across dev / prod)
# -----------------------------------------------------------------------------
CATALOG = os.environ.get("TDIS_CATALOG", "tdis_dev_data_catalog")
SCHEMA  = os.environ.get("TDIS_SCHEMA", "tdir")
VOLUME  = os.environ.get("TDIS_VOLUME", "tdir")   # UC Volume name (holds models)


def table(name: str) -> str:
    """Fully-qualified name for a table in the active catalog.schema."""
    return f"{CATALOG}.{SCHEMA}.{name}"


# -----------------------------------------------------------------------------
# Data lake source (PROD bronze) — read-only
# -----------------------------------------------------------------------------
LAKE_BASE = os.environ.get(
    "TDIS_LAKE_BASE",
    "abfss://tdis-data-bronze@tdisproddatalakehouse.dfs.core.windows.net/RAG_files/",
)
EMBED_MODEL_NAME = os.environ.get("TDIS_EMBED_MODEL_NAME", "DMRetriever-33M")

# Glob patterns for the chunk text + precomputed embeddings produced upstream
CHUNKS_GLOB    = LAKE_BASE + "optimized_chunks/*/chunks.jsonl"
EMBED_NPY_GLOB = LAKE_BASE + f"optimized_embeddings/{EMBED_MODEL_NAME}/*/embeddings.npy"
EMBED_IDS_GLOB = LAKE_BASE + f"optimized_embeddings/{EMBED_MODEL_NAME}/*/chunk_ids.json"

# -----------------------------------------------------------------------------
# optimized_* family tables
# -----------------------------------------------------------------------------
CHUNKS_TEXT_TABLE = table("optimized_chunks_text")
EMBEDDINGS_TABLE  = table("optimized_embeddings_dmretriever33m")
RAG_CHUNKS_TABLE  = table("optimized_rag_chunks")          # core wide table (text + embedding)

KW_POSTINGS_TABLE  = table("optimized_kw_postings")
KW_DF_TABLE        = table("optimized_kw_df")
KW_DOC_STATS_TABLE = table("optimized_kw_doc_stats")
KW_META_TABLE      = table("optimized_kw_meta")

QA_EVAL_TABLE = table("qa_dict_eval")

# -----------------------------------------------------------------------------
# Vector Search
# -----------------------------------------------------------------------------
VS_INDEX_FQN   = table("optimized_rag_chunks_vs")
VS_ENDPOINT    = os.environ.get("TDIS_VS_ENDPOINT", "")   # set per environment
VS_PRIMARY_KEY = "chunk_id"
VS_EMBED_COL   = "embedding"
EMBED_DIM      = int(os.environ.get("TDIS_EMBED_DIM", "384"))

# -----------------------------------------------------------------------------
# Models (UC Volume)
# -----------------------------------------------------------------------------
MODELS_DIR = os.environ.get("TDIS_MODELS_DIR", f"/Volumes/{CATALOG}/{SCHEMA}/{VOLUME}/models")

EMBED_MODEL_DIR = f"{MODELS_DIR}/{EMBED_MODEL_NAME}"
EMBED_MODEL_HF  = os.environ.get("TDIS_EMBED_MODEL_HF", "DMIR01/DMRetriever-33M")

RERANKER_MODEL_NAME = os.environ.get("TDIS_RERANKER_MODEL_NAME", "ms-marco-MiniLM-L-6-v2")
RERANKER_MODEL_DIR  = f"{MODELS_DIR}/{RERANKER_MODEL_NAME}"
RERANKER_MODEL_HF   = os.environ.get("TDIS_RERANKER_MODEL_HF", "cross-encoder/ms-marco-MiniLM-L-6-v2")

# -----------------------------------------------------------------------------
# LLM serving
# -----------------------------------------------------------------------------
DBX_BASE_URL = os.environ.get(
    "TDIS_DBX_BASE_URL",
    "https://adb-3300405005568038.18.azuredatabricks.net/serving-endpoints",
)
CONFIG_PATH = os.environ.get("TDIS_CONFIG_PATH", "config.yaml")
