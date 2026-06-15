# dbx_vector_search.py
# English comments only, Databricks-friendly
# Implements the SAME Vector Search method as your 1_vector_search notebook:
# Spark SQL `vector_search()` with a temp view holding `query_vector`.

from typing import List, Optional
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel

import settings

# Local model path in UC Volume (DMIR01/DMRetriever-33M snapshot)
MODEL_DIR = settings.EMBED_MODEL_DIR

_tokenizer = None
_model = None

def _load_model():
    global _tokenizer, _model
    if _tokenizer is None or _model is None:
        _tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
        _model = AutoModel.from_pretrained(MODEL_DIR)
        _model.eval()

def embed_query(query: str, max_length: int = 512) -> List[float]:
    """Compute DMRetriever-33M query embedding (mean pooling + L2 norm)."""
    _load_model()
    with torch.no_grad():
        inputs = _tokenizer(
            query,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )
        outputs = _model(**inputs)
        last_hidden = outputs.last_hidden_state  # [1, seq_len, hidden]
        mask = inputs["attention_mask"].unsqueeze(-1)  # [1, seq_len, 1]
        pooled = (last_hidden * mask).sum(dim=1) / mask.sum(dim=1)  # [1, hidden]
        pooled = F.normalize(pooled, p=2, dim=1)
    return pooled[0].cpu().tolist()

def register_query_vector(spark, query_vector: List[float], view_name: str = "tmp_query"):
    """Register the query vector as a Spark temp view with column `query_vector`."""
    spark.createDataFrame([(query_vector,)], ["query_vector"]).createOrReplaceTempView(view_name)

def vector_search(
    spark,
    index_fqn: str,
    query_vector: Optional[List[float]] = None,
    query_text: Optional[str] = None,
    top_k: int = 10,
    view_name: str = "tmp_query",
):
    """Run Databricks SQL vector_search() and return a Spark DataFrame.

    Note: This mirrors the notebook behavior: query_vector is taken from the temp view.
    If query_vector is provided, it will be registered to the view automatically.
    """
    if query_vector is None:
        if query_text is None:
            raise ValueError("Provide query_vector or query_text")
        query_vector = embed_query(query_text)
    register_query_vector(spark, query_vector, view_name=view_name)

    sql = f"""
    SELECT
      r.search_score,
      r.text,
      r.chunk_id,
      r.source_file,
      r.chunk_index_in_file
    FROM {view_name} q,
    LATERAL (
      SELECT search_score, text, chunk_id, source_file, chunk_index_in_file
      FROM vector_search(
        index => '{index_fqn}',
        query_vector => q.query_vector,
        num_results => {top_k}
      )
    ) r
    """
    return spark.sql(sql)

def vector_topk(spark, index_fqn: str, query: str, top_k: int = 10):
    """One-liner helper: embed -> register -> vector_search."""
    return vector_search(spark, index_fqn=index_fqn, query_text=query, top_k=top_k)
