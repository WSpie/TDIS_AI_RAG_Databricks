# dbx_hybrid_search.py
# English comments only, Databricks-friendly

from typing import Dict, Set, Optional

from pyspark.sql import DataFrame, SparkSession
from pyspark.sql import functions as F

from dbx_vector_search import vector_topk
from dbx_keyword_search_bm25 import keyword_search


DEFAULT_INDEX_FQN = "tdis_dev_data_catalog.tdir.optimized_rag_chunks_vs"
DEFAULT_CHUNKS_TABLE = "tdis_dev_data_catalog.tdir.optimized_rag_chunks"


def _clean_text_col(col):
    # Light cleanup: collapse whitespace, remove bullet-like symbols
    return F.trim(
        F.regexp_replace(
            F.regexp_replace(col, r"\s+", " "),
            r"[•·]+",
            " ",
        )
    )


def hybrid_search_rrf(
    spark: SparkSession,
    query: str,
    top_each: int = 10,
    top_n: int = 10,
    rrf_k: int = 60,
    return_with_text: bool = False,
    index_fqn: str = DEFAULT_INDEX_FQN,
    chunks_table: str = DEFAULT_CHUNKS_TABLE,
    clean_text: bool = False,
) -> DataFrame:
    """
    Hybrid retrieval = Vector top-k + BM25 top-k, fused by Reciprocal Rank Fusion (RRF).

    Args:
        spark: SparkSession
        query: user query string
        top_each: retrieve top_k from each retriever before fusion
        top_n: final fused top_n results
        rrf_k: RRF constant (larger => smaller score differences)
        return_with_text: if True, join chunk text + metadata
        index_fqn: Databricks Vector Search index FQN
        chunks_table: Delta table containing chunk_id/text/source_file/chunk_index_in_file
        clean_text: if True, add text_clean column

    Returns:
        Spark DataFrame:
          - if return_with_text=False: [chunk_id, rrf_score, from]
          - else: [chunk_id, rrf_score, from, text, source_file, chunk_index_in_file] (+ optional text_clean)
    """

    # Vector retrieval
    vec_df = (
        vector_topk(spark, index_fqn=index_fqn, query=query, top_k=top_each)
        .select("chunk_id", "search_score")
        .orderBy(F.col("search_score").desc())
        .limit(int(top_each))
    )
    vec_rank = [r["chunk_id"] for r in vec_df.select("chunk_id").collect()]
    print('Vector retrieval done.')

    # BM25 retrieval
    kw_df = (
        keyword_search(spark, query=query, k=top_each, return_df=True)
        .select("chunk_id", "score")
        .orderBy(F.col("score").desc())
        .limit(int(top_each))
    )
    kw_rank = [r["chunk_id"] for r in kw_df.select("chunk_id").collect()]
    print('Keywords retrieval done.')

    # RRF fuse + dedup
    scores: Dict[str, float] = {}
    src: Dict[str, Set[str]] = {}

    for i, cid in enumerate(vec_rank):
        scores[cid] = scores.get(cid, 0.0) + 1.0 / (rrf_k + i + 1)
        src.setdefault(cid, set()).add("VEC")

    for i, cid in enumerate(kw_rank):
        scores[cid] = scores.get(cid, 0.0) + 1.0 / (rrf_k + i + 1)
        src.setdefault(cid, set()).add("KW")

    fused = sorted(scores.items(), key=lambda x: -x[1])[: int(top_n)]
    out_rows = [(cid, float(s), "+".join(sorted(src.get(cid, set())))) for cid, s in fused]

    out_df = spark.createDataFrame(out_rows, ["chunk_id", "rrf_score", "from"]).orderBy(
        F.col("rrf_score").desc()
    )

    if not return_with_text:
        return out_df

    chunks = spark.table(chunks_table).select("chunk_id", "text", "source_file", "chunk_index_in_file")
    joined = out_df.join(chunks, "chunk_id", "left").orderBy(F.col("rrf_score").desc())

    if clean_text:
        joined = joined.withColumn("text_clean", _clean_text_col(F.col("text")))

    print('RRF done.')
    return joined
