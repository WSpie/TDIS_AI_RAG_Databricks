# dbx_rerank.py
# English comments only, Databricks-friendly
# Cross-encoder reranking stage: re-scores (query, chunk_text) pairs after RRF fusion.
#
# Pipeline position:
#   vector + BM25  ->  RRF fusion (dbx_hybrid_search)  ->  [THIS] rerank  ->  top_k  ->  LLM
#
# Default model is a generic English passage cross-encoder. Download it to a UC Volume
# the same way as the embedding model (see setup_nbk), then point RERANKER_DIR at it.

from typing import List, Optional, Sequence

import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

from pyspark.sql import DataFrame, SparkSession
from pyspark.sql import functions as F

import settings

# Local model path in UC Volume (download via huggingface_hub.snapshot_download)
# HF id: cross-encoder/ms-marco-MiniLM-L-6-v2
RERANKER_DIR = settings.RERANKER_MODEL_DIR

# Lazy-loaded globals (cache across calls in one session)
_tokenizer = None
_model = None


def _load_model() -> None:
    global _tokenizer, _model
    if _tokenizer is None or _model is None:
        _tokenizer = AutoTokenizer.from_pretrained(RERANKER_DIR)
        _model = AutoModelForSequenceClassification.from_pretrained(RERANKER_DIR)
        _model.eval()


def rerank_scores(
    query: str,
    texts: Sequence[str],
    max_length: int = 512,
    batch_size: int = 16,
) -> List[float]:
    """Return one relevance score per text for the given query (higher = more relevant)."""
    _load_model()
    if not texts:
        return []

    scores: List[float] = []
    with torch.no_grad():
        for start in range(0, len(texts), batch_size):
            batch = list(texts[start:start + batch_size])
            inputs = _tokenizer(
                [query] * len(batch),
                batch,
                padding=True,
                truncation=True,
                max_length=max_length,
                return_tensors="pt",
            )
            logits = _model(**inputs).logits  # [batch, 1] for ms-marco cross-encoders
            scores.extend(logits.squeeze(-1).cpu().tolist())
    return scores


def rerank_dataframe(
    spark: SparkSession,
    df: DataFrame,
    query: str,
    top_k: int = 10,
    text_col: str = "text",
    score_col: str = "rerank_score",
    candidate_limit: Optional[int] = 50,
) -> DataFrame:
    """Rerank a retrieval DataFrame with a cross-encoder.

    Steps:
      1) collect up to `candidate_limit` candidate rows (the fused top_n is usually small),
      2) score each (query, text) pair,
      3) return a new DataFrame ordered by rerank score, limited to `top_k`.

    The returned DataFrame keeps all original columns plus `score_col`.
    """
    rows = df.limit(candidate_limit).collect() if candidate_limit else df.collect()
    if not rows:
        return df.withColumn(score_col, F.lit(None).cast("double")).limit(0)

    texts = [(r[text_col] or "") for r in rows]
    scores = rerank_scores(query, texts)

    enriched = [{**r.asDict(), score_col: float(s)} for r, s in zip(rows, scores)]
    enriched.sort(key=lambda d: d[score_col], reverse=True)
    enriched = enriched[: int(top_k)]

    out_cols = list(df.columns) + [score_col]
    return spark.createDataFrame(enriched).select(*out_cols).orderBy(F.col(score_col).desc())
