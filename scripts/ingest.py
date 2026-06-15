# ingest.py
# Incremental ingestion of new chunks + precomputed embeddings (processed upstream, e.g. on HPC)
# into the optimized_* family, followed by an optional Vector Search index sync.
#
# Idempotent: rows are MERGEd on chunk_id, so re-running with overlapping ids updates in place
# instead of creating duplicates.
#
# Typical flow:
#   chunks_df, emb_df = load_new(spark, base_dir="abfss://.../incoming/")
#   report = ingest(spark, chunks_df, emb_df, sync_index=True)

from typing import Dict, Optional, Tuple

from pyspark.sql import DataFrame, SparkSession

import settings
from lake_io import read_chunks_jsonl, read_embeddings_npy


# -----------------------------------------------------------------------------
# Load new data (upstream layout: chunks.jsonl + embeddings.npy + chunk_ids.json)
# -----------------------------------------------------------------------------
def load_new(
    spark: SparkSession,
    base_dir: str,
    embed_model_name: Optional[str] = None,
) -> Tuple[DataFrame, DataFrame]:
    """Read new chunks + embeddings from a base dir laid out like the lake.

    Expects:
      {base_dir}/optimized_chunks/*/chunks.jsonl
      {base_dir}/optimized_embeddings/{model}/*/embeddings.npy
      {base_dir}/optimized_embeddings/{model}/*/chunk_ids.json
    """
    model = embed_model_name or settings.EMBED_MODEL_NAME
    base = base_dir if base_dir.endswith("/") else base_dir + "/"

    chunks_df = read_chunks_jsonl(spark, base + "optimized_chunks/*/chunks.jsonl")
    emb_df = read_embeddings_npy(
        spark,
        npy_glob=base + f"optimized_embeddings/{model}/*/embeddings.npy",
        ids_glob=base + f"optimized_embeddings/{model}/*/chunk_ids.json",
        model_name=model,
    )
    return chunks_df, emb_df


# -----------------------------------------------------------------------------
# Idempotent MERGE helper
# -----------------------------------------------------------------------------
def _merge_on_chunk_id(spark: SparkSession, target_fqn: str, source_df: DataFrame) -> int:
    """Upsert source_df into target_fqn keyed on chunk_id. Creates the table if missing.

    Returns the number of distinct chunk_ids in source_df.
    """
    n = source_df.select("chunk_id").distinct().count()

    if not spark.catalog.tableExists(target_fqn):
        source_df.write.mode("overwrite").saveAsTable(target_fqn)
        return n

    from delta.tables import DeltaTable

    tgt = DeltaTable.forName(spark, target_fqn)
    (tgt.alias("t")
        .merge(source_df.alias("s"), "t.chunk_id = s.chunk_id")
        .whenMatchedUpdateAll()
        .whenNotMatchedInsertAll()
        .execute())
    return n


# -----------------------------------------------------------------------------
# Main ingestion
# -----------------------------------------------------------------------------
def ingest(
    spark: SparkSession,
    chunks_df: DataFrame,
    emb_df: DataFrame,
    sync_index: bool = True,
    wait_for_sync: bool = False,
) -> Dict[str, int]:
    """Upsert new chunks + embeddings into the optimized_* family and optionally sync the index.

    Updates three tables, keyed on chunk_id:
      - optimized_chunks_text                (chunk_id, source_file, chunk_index_in_file, text)
      - optimized_embeddings_dmretriever33m  (chunk_id, embedding)
      - optimized_rag_chunks                 (inner join of the two — the VS source table)

    NOTE: the BM25 tables (optimized_kw_*) are NOT updated here; rebuild them with
    rebuild_bm25() when keyword recall needs to reflect the new chunks.

    Returns a small report dict with row counts.
    """
    chunks_df = chunks_df.select("chunk_id", "source_file", "chunk_index_in_file", "text")
    emb_df = emb_df.select("chunk_id", "embedding")

    n_chunks = _merge_on_chunk_id(spark, settings.CHUNKS_TEXT_TABLE, chunks_df)
    n_emb = _merge_on_chunk_id(spark, settings.EMBEDDINGS_TABLE, emb_df)

    rag_df = chunks_df.join(emb_df, on="chunk_id", how="inner")
    n_rag = _merge_on_chunk_id(spark, settings.RAG_CHUNKS_TABLE, rag_df)

    report = {"chunks_text": n_chunks, "embeddings": n_emb, "rag_chunks": n_rag}

    if sync_index:
        if settings.VS_ENDPOINT:
            sync_vector_index(wait=wait_for_sync)
            report["index_sync"] = "triggered"
        else:
            # No endpoint configured. A Continuous Delta Sync index picks up the new rows
            # automatically; only a Triggered index needs an explicit sync (set VS_ENDPOINT).
            report["index_sync"] = "skipped (VS_ENDPOINT not set)"

    return report


# -----------------------------------------------------------------------------
# Vector Search index sync
# -----------------------------------------------------------------------------
def sync_vector_index(
    index_fqn: Optional[str] = None,
    endpoint: Optional[str] = None,
    wait: bool = False,
) -> bool:
    """Trigger a sync of the Delta Sync Vector Search index so it picks up new rows.

    Requires the `databricks-vectorsearch` SDK and settings.VS_ENDPOINT (or env override).
    """
    index_fqn = index_fqn or settings.VS_INDEX_FQN
    endpoint = endpoint or settings.VS_ENDPOINT
    if not endpoint:
        raise ValueError(
            "Vector Search endpoint not set. Set settings.VS_ENDPOINT or the "
            "TDIS_VS_ENDPOINT env var before syncing."
        )

    from databricks.vector_search.client import VectorSearchClient

    vsc = VectorSearchClient()
    index = vsc.get_index(endpoint_name=endpoint, index_name=index_fqn)
    index.sync()
    if wait:
        index.wait_until_ready(verbose=True)
    return True


# -----------------------------------------------------------------------------
# BM25 rebuild (full recompute over the current optimized_rag_chunks)
# -----------------------------------------------------------------------------
def rebuild_bm25(spark: SparkSession) -> Dict[str, int]:
    """Recompute the BM25 inverted-index tables from the full optimized_rag_chunks.

    Call this after ingest() when keyword search must reflect newly added chunks.
    """
    from pyspark.sql import functions as F

    chunks = spark.table(settings.RAG_CHUNKS_TABLE).select("chunk_id", "text")

    terms = (chunks
        .select("chunk_id",
                F.explode(F.split(F.lower(F.regexp_replace(F.col("text"), r"[^a-z0-9]+", " ")), r"\s+")).alias("term"))
        .filter(F.length("term") >= 3)
        .filter(~F.col("term").rlike(r"^\d+$")))

    postings = terms.groupBy("term", "chunk_id").agg(F.count("*").cast("int").alias("tf"))
    postings.write.mode("overwrite").saveAsTable(settings.KW_POSTINGS_TABLE)

    doc_stats = postings.groupBy("chunk_id").agg(F.sum("tf").cast("int").alias("doc_len"))
    doc_stats.write.mode("overwrite").saveAsTable(settings.KW_DOC_STATS_TABLE)

    df_table = postings.groupBy("term").agg(F.count("*").cast("long").alias("df"))
    df_table.write.mode("overwrite").saveAsTable(settings.KW_DF_TABLE)

    meta = doc_stats.agg(F.count("*").cast("long").alias("N"), F.avg("doc_len").cast("double").alias("avgdl"))
    meta.write.mode("overwrite").saveAsTable(settings.KW_META_TABLE)

    return {
        "postings": spark.table(settings.KW_POSTINGS_TABLE).count(),
        "doc_stats": spark.table(settings.KW_DOC_STATS_TABLE).count(),
        "df": spark.table(settings.KW_DF_TABLE).count(),
    }
