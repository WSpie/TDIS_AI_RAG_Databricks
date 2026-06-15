# rag_pipeline.py
# English comments only, Databricks-friendly
# End-to-end RAG orchestration: hybrid retrieval -> RRF fusion -> cross-encoder rerank -> LLM.

from typing import List, Optional, Tuple

import openai
from pyspark.sql import DataFrame, SparkSession
from pyspark.sql import functions as F

from dbx_hybrid_search import hybrid_search_rrf
from dbx_rerank import rerank_dataframe
from LLMs_call import load_api_key, ask_gpt, build_messages, dbx_llm_chat


def rag_pipe_main(
    spark: SparkSession,
    query: str,
    llm_model: str,
    use_dbx_model: bool = True,
    temperature: Optional[float] = None,
    top_each: int = 100,
    top_n: int = 20,
    rerank_top_k: int = 10,
    rrf_k: int = 20,
    config_path: str = "config.yaml",
) -> Tuple[DataFrame, str]:
    """Run the full RAG pipeline and return (reranked_df, answer).

    Stages:
      1) hybrid_search_rrf: vector + BM25 retrieval fused by RRF (returns top_n with text),
      2) rerank_dataframe : cross-encoder re-scores the fused candidates, keeps rerank_top_k,
      3) LLM generation   : build CONTEXT from reranked text and prompt the chosen model.
    """
    # 1) Hybrid retrieval + RRF fusion
    fused_df = hybrid_search_rrf(
        spark, query, top_each=top_each, top_n=top_n, rrf_k=rrf_k, return_with_text=True
    )

    # 2) Cross-encoder rerank
    reranked_df = rerank_dataframe(spark, fused_df, query, top_k=rerank_top_k)

    # 3) Build CONTEXT from reranked chunk text
    text_collection = [r["text"] for r in reranked_df.select("text").collect()]
    messages = build_messages(text_collection=text_collection, user_query=query)

    # 4) Generate
    if use_dbx_model:
        answer = dbx_llm_chat(messages=messages, model_name=llm_model, temperature=temperature)
    else:
        openai.api_key = load_api_key("gpt_api_key", config_path)
        answer = ask_gpt(messages=messages, model=llm_model, temperature=temperature)

    return reranked_df, answer
