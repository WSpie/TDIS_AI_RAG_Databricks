# dbx_keyword_search_bm25.py
# English comments only, Databricks-friendly
# BM25 keyword search over optimized_* tables with static + query-adaptive term weights.

import re
from typing import Dict, List, Optional

from pyspark.sql import DataFrame, SparkSession
from pyspark.sql import functions as F
from pyspark.sql.window import Window

import settings

# -----------------------------
# Tables (optimized_ only)
# -----------------------------
CHUNKS = settings.RAG_CHUNKS_TABLE
POST   = settings.KW_POSTINGS_TABLE
DFT    = settings.KW_DF_TABLE
DST    = settings.KW_DOC_STATS_TABLE
META   = settings.KW_META_TABLE

# -----------------------------
# BM25 parameters
# -----------------------------
k1, b = 1.2, 0.75

# -----------------------------
# Selected terms whitelist
# -----------------------------
selected_terms = set([
    "flood","flooding","floodplain","damage","damaged","loss","losses","estimate","estimates","estimated",
    "hazard","hazards","hazardous","risk","risks","hurricane","hurricanes","storm","storms","surge","winds",
    "rainfall","rain","precipitation","drainage","watershed","basin","river","creek","bay","coast","coastal",
    "water","waters","debris","shelter","evacuation","insurance","housing","homes","inundation",
    "fema","noaa","nws","usda","corps","epa",
    "harris","houston","galveston","louisiana","austin","dallas","antonio","mexico","florida","california","colorado","port","tx",
    "county","counties","city","cities","district","region","regional","area","areas","state","states",
    "central","west","east","north","south","southeast","southwest","western","valley","island","lake"
])

# -----------------------------
# Term weights
# -----------------------------
BASE_TERM_WEIGHT = 0.5

# Start with all selected terms at a low default weight
TERM_WEIGHTS: Dict[str, float] = {t: BASE_TERM_WEIGHT for t in selected_terms}

# Keep your fixed weights
TERM_WEIGHTS.update({
    "harris": 2.0, "houston": 2.0, "galveston": 2.0, "texas": 2.0, "louisiana": 2.0,
    "fema": 3.0, "noaa": 3.0, "nws": 3.0, "usda": 3.0, "corps": 3.0, "epa": 3.0,
    "county": 1.5, "city": 1.5, "district": 1.5, "watershed": 2.0, "river": 2.0,
    "basin": 2.0, "coastal": 2.0,
})

# Hazard / impact core
TERM_WEIGHTS.update({
    "flood": 2.5, "flooding": 2.5, "floodplain": 2.0,
    "damage": 2.5, "damaged": 2.0,
    "loss": 2.0, "losses": 2.0,
    "estimate": 2.0, "estimates": 2.0, "estimated": 2.0,
    "risk": 1.8, "risks": 1.8,
    "hazard": 1.5, "hazards": 1.5, "hazardous": 1.2,
    "inundation": 2.2,
})

# Storm / met drivers
TERM_WEIGHTS.update({
    "hurricane": 1.8, "hurricanes": 1.8,
    "storm": 1.6, "storms": 1.6,
    "surge": 1.7, "winds": 1.2,
    "rainfall": 1.8, "rain": 1.3, "precipitation": 1.7,
})

# Hydro / drainage / waterbody signals
TERM_WEIGHTS.update({
    "drainage": 1.9,
    "creek": 1.8,
    "bay": 1.6,
    "coast": 1.5,
    "water": 1.2, "waters": 1.2,
})

# Response / exposure / soc-impact
TERM_WEIGHTS.update({
    "insurance": 1.6,
    "housing": 1.4, "homes": 1.4,
    "evacuation": 1.4,
    "shelter": 1.2,
    "debris": 1.1,
})

# Other geo tokens (keep modest)
TERM_WEIGHTS.update({
    "port": 1.2,
    "tx": 0.8,
    "austin": 1.2,
    "dallas": 1.2,
    "antonio": 1.0,
    "mexico": 1.1,
    "florida": 1.1,
    "california": 1.1,
    "colorado": 1.1,
})

# Admin/general location words (downweight)
TERM_WEIGHTS.update({
    "counties": 1.3,
    "cities": 1.2,
    "region": 1.0, "regional": 1.0,
    "area": 0.9, "areas": 0.9,
    "state": 0.9, "states": 0.9,
    "central": 0.8, "west": 0.8, "east": 0.8, "north": 0.8, "south": 0.8,
    "southeast": 0.8, "southwest": 0.8, "western": 0.8,
    "valley": 0.9, "island": 0.9, "lake": 1.0,
})


def _tokenize_query(query: str) -> List[str]:
    q_terms = [t for t in re.sub(r"[^a-z0-9]+", " ", (query or "").lower()).split() if t]
    return list(dict.fromkeys(q_terms))


def _build_query_boost(q_terms: List[str]) -> Dict[str, float]:
    """
    Query-adaptive term boost.
    Returns per-term multiplicative factors (>= 1.0).
    """
    qset = set(q_terms)

    has_method = any(t in qset for t in ["methodology", "method", "framework", "model", "hazus"])
    has_damage = any(t in qset for t in ["damage", "damaged", "loss", "losses", "estimate", "estimates", "estimated"])
    has_agency = any(t in qset for t in ["fema", "noaa", "nws", "usda", "corps", "epa"])
    has_geo_tx = any(t in qset for t in ["harris", "houston", "galveston", "texas", "tx"])

    boost: Dict[str, float] = {}

    if has_method or has_damage:
        for t in [
            "methodology", "method", "framework", "model", "hazus",
            "estimate", "estimated", "estimates",
            "damage", "loss", "losses",
            "flood", "flooding", "floodplain"
        ]:
            boost[t] = max(boost.get(t, 1.0), 1.4)

    if has_agency:
        for t in ["fema", "noaa", "nws", "usda", "corps", "epa", "hazard", "risk", "insurance"]:
            boost[t] = max(boost.get(t, 1.0), 1.5)

    if has_geo_tx:
        for t in ["harris", "houston", "galveston", "texas", "tx", "county", "district", "watershed", "basin", "river", "coastal"]:
            boost[t] = max(boost.get(t, 1.0), 1.3)

    for t in q_terms:
        boost[t] = max(boost.get(t, 1.0), 1.1)

    return boost


# -----------------------------
# Lazy-loaded globals (cache)
# -----------------------------
_LOADED = False
_META_N: float = 0.0
_META_AVGDL: float = 0.0
_POST_DF: Optional[DataFrame] = None
_DF_DF: Optional[DataFrame] = None
_DST_DF: Optional[DataFrame] = None
_CH_DF: Optional[DataFrame] = None


def load_tables(spark: SparkSession) -> None:
    """
    Load meta + tables once (cached globals).
    Call this once per cluster session (or it will auto-load on first search).
    """
    global _LOADED, _META_N, _META_AVGDL, _POST_DF, _DF_DF, _DST_DF, _CH_DF

    if _LOADED:
        return

    meta_row = spark.table(META).first()
    _META_N = float(meta_row["N"])
    _META_AVGDL = float(meta_row["avgdl"])

    _POST_DF = spark.table(POST).select("term", "chunk_id", "tf")
    _DF_DF = spark.table(DFT).select("term", "df").withColumn("df_ratio", F.col("df") / F.lit(_META_N))
    _DST_DF = spark.table(DST).select("chunk_id", "doc_len")
    _CH_DF = spark.table(CHUNKS).select("chunk_id", "text", "source_file", "chunk_index_in_file")

    _LOADED = True


def keyword_search(
    spark: SparkSession,
    query: str,
    k: int = 50,
    df_ratio_max: float = 0.03,
    top_terms: int = 5,
    min_terms: int = 2,
    english_ratio_min: float = 0.35,
    return_df: bool = True,
) -> DataFrame | Dict:
    """
    BM25 keyword retrieval with static term weights and query-adaptive boosts.

    Returns:
      - Spark DataFrame if return_df=True
      - dict {"result": {"data_array": ...}} if return_df=False
    """
    load_tables(spark)

    assert _POST_DF is not None and _DF_DF is not None and _DST_DF is not None and _CH_DF is not None

    q_terms = _tokenize_query(query)
    if not q_terms:
        empty = spark.createDataFrame(
            [],
            "chunk_id string, matched_terms array<string>, text string, source_file string, chunk_index_in_file long, score double"
        )
        return empty if return_df else {"result": {"data_array": []}}

    qdf = spark.createDataFrame([(t,) for t in q_terms], ["term"])

    keep_rows = [(t, 1 if t in selected_terms else 0) for t in q_terms]
    keep_df = spark.createDataFrame(keep_rows, ["term", "is_whitelist"])

    q_keep = (
        qdf.join(_DF_DF.select("term", "df", "df_ratio"), "term", "left")
           .join(F.broadcast(keep_df), "term", "left")
           .filter(F.col("df").isNotNull())
           .filter((F.col("is_whitelist") == 1) | (F.col("df_ratio") <= F.lit(df_ratio_max)))
           .select("term")
           .dropDuplicates()
    )

    if q_keep.count() < min_terms:
        q_keep = (
            qdf.join(_DF_DF.select("term", "df"), "term", "left")
               .filter(F.col("df").isNotNull())
               .select("term")
               .dropDuplicates()
        )

    q_boost = _build_query_boost(q_terms)

    w_rows = []
    for t in q_terms:
        static_w = float(TERM_WEIGHTS.get(t, BASE_TERM_WEIGHT))
        boost_w = float(q_boost.get(t, 1.0))
        w_rows.append((t, static_w * boost_w))

    wdf = spark.createDataFrame(w_rows, ["term", "term_weight"])

    q_keep_w = (
        q_keep.join(F.broadcast(wdf), "term", "left")
              .fillna({"term_weight": BASE_TERM_WEIGHT})
    )

    post = _POST_DF.join(F.broadcast(q_keep_w.select("term", "term_weight")), "term", "inner")
    dft = _DF_DF.select("term", "df").join(F.broadcast(q_keep_w.select("term")), "term", "inner")

    idf = F.log((F.lit(_META_N) - F.col("df") + F.lit(0.5)) / (F.col("df") + F.lit(0.5)) + F.lit(1.0))
    denom = F.col("tf") + F.lit(k1) * (F.lit(1.0) - F.lit(b) + F.lit(b) * F.col("doc_len") / F.lit(_META_AVGDL))
    bm25_raw = idf * (F.col("tf") * F.lit(k1 + 1.0)) / denom

    scored = (
        post.join(dft, "term")
            .join(_DST_DF, "chunk_id")
            .withColumn("bm25_term", bm25_raw * F.col("term_weight"))
            .select("chunk_id", "term", "tf", "bm25_term")
    )

    hit_cnt = scored.groupBy("chunk_id").agg(F.countDistinct("term").alias("hit_terms"))
    scored2 = scored.join(hit_cnt, "chunk_id").filter(F.col("hit_terms") >= F.lit(int(min_terms)))

    top_chunks = (
        scored2.groupBy("chunk_id")
               .agg(F.sum("bm25_term").alias("score"))
               .orderBy(F.desc("score"))
               .limit(int(k) * 5)
    )

    out0 = top_chunks.join(_CH_DF, "chunk_id", "left")

    if english_ratio_min is not None:
        out0 = (
            out0.withColumn("len_all", F.length("text"))
                .withColumn("len_en", F.length(F.regexp_replace(F.lower(F.col("text")), r"[^a-z]+", "")))
                .withColumn(
                    "en_ratio",
                    F.when(F.col("len_all") > 0, F.col("len_en") / F.col("len_all")).otherwise(F.lit(0.0))
                )
                .filter(F.col("en_ratio") >= F.lit(float(english_ratio_min)))
                .drop("len_all", "len_en", "en_ratio")
        )

    w = Window.partitionBy("chunk_id").orderBy(F.desc("bm25_term"))
    matched = (
        scored2.join(out0.select("chunk_id"), "chunk_id", "inner")
              .withColumn("rk", F.row_number().over(w))
              .filter(F.col("rk") <= F.lit(int(top_terms)))
              .withColumn("term_tf", F.concat(F.col("term"), F.lit("("), F.col("tf").cast("string"), F.lit(")")))
              .groupBy("chunk_id")
              .agg(F.collect_list("term_tf").alias("matched_terms"))
    )

    out = (
        out0.join(matched, "chunk_id", "left")
            .select("chunk_id", "matched_terms", "text", "source_file", "chunk_index_in_file", "score")
            .orderBy(F.desc("score"))
            .limit(int(k))
    )

    if return_df:
        return out

    rows = out.collect()
    data_array = [
        [
            r["chunk_id"],
            r["matched_terms"],
            r["text"],
            r["source_file"],
            r["chunk_index_in_file"],
            float(r["score"]),
        ]
        for r in rows
    ]
    return {"result": {"data_array": data_array}}
