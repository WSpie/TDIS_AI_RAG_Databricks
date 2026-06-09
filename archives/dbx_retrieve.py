# dbx_retrieve.py
# English comments only, Databricks-friendly

from typing import List
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel

# Local model path in UC Volume
MODEL_DIR = "/Volumes/tdis_dev_data_catalog/tdir/tdir/models/DMRetriever-33M"

# Lazy-loaded globals
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
    """Register the query vector as a Spark temp view."""
    spark.createDataFrame([(query_vector,)], ["query_vector"]).createOrReplaceTempView(view_name)

def vector_search(spark, index_fqn: str, top_k: int = 5, view_name: str = "tmp_query"):
    """Run Databricks SQL vector_search() and return only lightweight columns."""
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

