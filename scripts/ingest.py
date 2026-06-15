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
from pyspark.sql import functions as F

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
    update_keyword_index: bool = True,
) -> Dict[str, int]:
    """Upsert new chunks + embeddings into the optimized_* family, refresh BM25, and sync the index.

    Updates three tables, keyed on chunk_id:
      - optimized_chunks_text                (chunk_id, source_file, chunk_index_in_file, text)
      - optimized_embeddings_dmretriever33m  (chunk_id, embedding)
      - optimized_rag_chunks                 (inner join of the two — the VS source table)

    Both retrieval paths must see the new chunks for hybrid search to be complete:
      - vector side: the Vector Search index (synced when sync_index=True),
      - keyword side: the BM25 tables (rebuilt when update_keyword_index=True).
    Skipping the BM25 update leaves new chunks invisible to keyword recall, so it defaults to True.
    The update is incremental (only the affected chunks are re-tokenized). For many small batches
    you can set sync_index=False and sync once at the end to avoid repeated index syncs.

    Returns a small report dict.
    """
    chunks_df = chunks_df.select("chunk_id", "source_file", "chunk_index_in_file", "text")
    emb_df = emb_df.select("chunk_id", "embedding")

    n_chunks = _merge_on_chunk_id(spark, settings.CHUNKS_TEXT_TABLE, chunks_df)
    n_emb = _merge_on_chunk_id(spark, settings.EMBEDDINGS_TABLE, emb_df)

    rag_df = chunks_df.join(emb_df, on="chunk_id", how="inner")
    n_rag = _merge_on_chunk_id(spark, settings.RAG_CHUNKS_TABLE, rag_df)

    report = {"chunks_text": n_chunks, "embeddings": n_emb, "rag_chunks": n_rag}

    # Keyword side: update BM25 so new chunks are reachable via keyword recall too.
    # Incremental by default (only re-tokenizes the affected chunks); full rebuild on request.
    if update_keyword_index:
        report["bm25"] = update_bm25(spark, chunks_df.select("chunk_id", "text"))

    # Vector side: refresh the Vector Search index
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
# BM25 index maintenance
# -----------------------------------------------------------------------------
# BM25 tables (must match scripts/dbx_keyword_search_bm25.py):
#   optimized_kw_postings   (term, chunk_id, tf)   one row per (term, chunk_id)
#   optimized_kw_doc_stats  (chunk_id, doc_len)     one row per chunk
#   optimized_kw_df         (term, df)              df = #chunks containing term
#   optimized_kw_meta       (N, avgdl)              N = #chunks, avgdl = mean doc_len

def _tokenize_postings(chunks_df: DataFrame) -> DataFrame:
    """Tokenize chunk text into postings: (term, chunk_id, tf).

    Same analyzer as the original build: lowercase, split on non-alphanumeric,
    drop tokens shorter than 3 chars and purely numeric tokens.
    """
    terms = (chunks_df
        .select("chunk_id",
                F.explode(F.split(F.lower(F.regexp_replace(F.col("text"), r"[^a-z0-9]+", " ")), r"\s+")).alias("term"))
        .filter(F.length("term") >= 3)
        .filter(~F.col("term").rlike(r"^\d+$")))
    return terms.groupBy("term", "chunk_id").agg(F.count("*").cast("int").alias("tf"))


def _recompute_meta(spark: SparkSession) -> None:
    """Recompute N + avgdl from the doc_stats table (cheap: one row per chunk)."""
    meta = (spark.table(settings.KW_DOC_STATS_TABLE)
            .agg(F.count("*").cast("long").alias("N"),
                 F.avg("doc_len").cast("double").alias("avgdl")))
    meta.write.mode("overwrite").saveAsTable(settings.KW_META_TABLE)


def rebuild_bm25(spark: SparkSession) -> Dict[str, int]:
    """Full recompute of the BM25 tables from the entire optimized_rag_chunks.

    Use for the initial build, or to repair the index. For incremental ingestion use update_bm25().
    """
    chunks = spark.table(settings.RAG_CHUNKS_TABLE).select("chunk_id", "text")
    postings = _tokenize_postings(chunks)
    postings.write.mode("overwrite").saveAsTable(settings.KW_POSTINGS_TABLE)

    doc_stats = postings.groupBy("chunk_id").agg(F.sum("tf").cast("int").alias("doc_len"))
    doc_stats.write.mode("overwrite").saveAsTable(settings.KW_DOC_STATS_TABLE)

    df_table = postings.groupBy("term").agg(F.count("*").cast("long").alias("df"))
    df_table.write.mode("overwrite").saveAsTable(settings.KW_DF_TABLE)

    _recompute_meta(spark)

    return {
        "mode": "full",
        "postings": spark.table(settings.KW_POSTINGS_TABLE).count(),
        "doc_stats": spark.table(settings.KW_DOC_STATS_TABLE).count(),
        "df": spark.table(settings.KW_DF_TABLE).count(),
    }


def update_bm25(spark: SparkSession, chunks_df: DataFrame) -> Dict[str, int]:
    """Incrementally update the BM25 tables for just the chunks in chunks_df.

    Only the affected chunks are re-tokenized; the four tables are then patched in place.
    Correctly handles re-ingested chunk_ids (text changes): the affected chunks' OLD
    contributions are subtracted before the NEW ones are added, so the result is identical
    to a full rebuild and re-running the same batch is a no-op (idempotent).

    Falls back to a full rebuild_bm25() if the BM25 tables do not exist yet.
    """
    if not spark.catalog.tableExists(settings.KW_POSTINGS_TABLE):
        return rebuild_bm25(spark)

    from delta.tables import DeltaTable

    chunks_df = chunks_df.select("chunk_id", "text")
    affected = chunks_df.select("chunk_id").distinct()

    # New postings for the affected chunks (cache: reused for df/postings/doc_stats)
    new_post = _tokenize_postings(chunks_df).persist()
    n_new_post = new_post.count()  # materialize before any table mutation

    # --- 1) df delta: change in per-term document count contributed by affected chunks ---
    # One postings row == one (term, chunk) pair, so counting rows per term == #docs per term.
    new_cnt = new_post.groupBy("term").agg(F.count("*").cast("long").alias("new_cnt"))
    old_cnt = (spark.table(settings.KW_POSTINGS_TABLE)
               .join(F.broadcast(affected), "chunk_id", "inner")
               .groupBy("term").agg(F.count("*").cast("long").alias("old_cnt")))

    df_delta = (new_cnt.join(old_cnt, "term", "full_outer")
                .select("term",
                        (F.coalesce("new_cnt", F.lit(0)) - F.coalesce("old_cnt", F.lit(0))).cast("long").alias("delta"))
                .filter(F.col("delta") != 0)
                .persist())
    n_df_changed = df_delta.count()  # materialize df_delta BEFORE mutating postings

    # Apply df changes: update / insert / delete-when-zero
    df_tbl = DeltaTable.forName(spark, settings.KW_DF_TABLE)
    (df_tbl.alias("t")
        .merge(df_delta.alias("s"), "t.term = s.term")
        .whenMatchedDelete(condition="(t.df + s.delta) <= 0")
        .whenMatchedUpdate(set={"df": "t.df + s.delta"})
        .whenNotMatchedInsert(condition="s.delta > 0", values={"term": "s.term", "df": "s.delta"})
        .execute())

    # --- 2) postings: delete affected chunks' rows, then append the freshly tokenized ones ---
    post_tbl = DeltaTable.forName(spark, settings.KW_POSTINGS_TABLE)
    (post_tbl.alias("t")
        .merge(F.broadcast(affected).alias("a"), "t.chunk_id = a.chunk_id")
        .whenMatchedDelete()
        .execute())
    new_post.write.format("delta").mode("append").saveAsTable(settings.KW_POSTINGS_TABLE)

    # --- 3) doc_stats: upsert doc_len for the affected chunks ---
    new_doc_stats = new_post.groupBy("chunk_id").agg(F.sum("tf").cast("int").alias("doc_len"))
    ds_tbl = DeltaTable.forName(spark, settings.KW_DOC_STATS_TABLE)
    (ds_tbl.alias("t")
        .merge(new_doc_stats.alias("s"), "t.chunk_id = s.chunk_id")
        .whenMatchedUpdate(set={"doc_len": "s.doc_len"})
        .whenNotMatchedInsertAll()
        .execute())

    # --- 4) meta: recompute N + avgdl from doc_stats (cheap) ---
    _recompute_meta(spark)

    df_delta.unpersist()
    new_post.unpersist()

    return {
        "mode": "incremental",
        "affected_chunks": affected.count(),
        "new_postings": n_new_post,
        "df_terms_changed": n_df_changed,
    }
