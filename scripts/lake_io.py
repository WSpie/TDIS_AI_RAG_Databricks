# lake_io.py
# Shared readers for the upstream RAG file layout (chunks.jsonl + embeddings.npy + chunk_ids.json).
# Used by both the one-time setup build and the incremental ingestion module, so the decode
# logic lives in exactly one place.

import io
import json

import numpy as np
import pandas as pd
from pyspark.sql import DataFrame, SparkSession
from pyspark.sql.functions import col, regexp_extract
from pyspark.sql.types import StructType, StructField, StringType, ArrayType, FloatType

import settings

_EMB_SCHEMA = StructType([
    StructField("chunk_id", StringType(), False),
    StructField("embedding", ArrayType(FloatType()), False),
])


def read_chunks_jsonl(spark: SparkSession, glob: str = None) -> DataFrame:
    """Read chunk text records: chunk_id, source_file, chunk_index_in_file, text."""
    glob = glob or settings.CHUNKS_GLOB
    return spark.read.json(glob).select(
        "chunk_id", "source_file", "chunk_index_in_file", "text"
    )


def read_embeddings_npy(
    spark: SparkSession,
    npy_glob: str = None,
    ids_glob: str = None,
    model_name: str = None,
) -> DataFrame:
    """Read and decode precomputed embeddings into a DataFrame: chunk_id, embedding.

    Each source dir holds an embeddings.npy ([n, dim]) plus a chunk_ids.json (n ids),
    paired by the dir name. Decoding runs distributed via mapInPandas.
    """
    npy_glob = npy_glob or settings.EMBED_NPY_GLOB
    ids_glob = ids_glob or settings.EMBED_IDS_GLOB
    model_name = model_name or settings.EMBED_MODEL_NAME

    npy_re = rf"/{model_name}/([^/]+)/embeddings\.npy$"
    ids_re = rf"/{model_name}/([^/]+)/chunk_ids\.json$"

    df_npy = (spark.read.format("binaryFile").load(npy_glob)
              .withColumn("source_dir", regexp_extract(col("path"), npy_re, 1))
              .select("source_dir", col("content").alias("npy_bytes")))

    df_ids = (spark.read.format("binaryFile").load(ids_glob)
              .withColumn("source_dir", regexp_extract(col("path"), ids_re, 1))
              .select("source_dir", col("content").alias("ids_bytes")))

    df_pair = df_npy.join(df_ids, on="source_dir", how="inner").select(
        "source_dir", "npy_bytes", "ids_bytes"
    )

    def _decode(pdf_iter):
        # Decode .npy + ids inside executors (parallel)
        for pdf in pdf_iter:
            out = []
            for _, r in pdf.iterrows():
                ids = json.loads(bytes(r["ids_bytes"]).decode("utf-8"))
                vecs = np.load(io.BytesIO(bytes(r["npy_bytes"])))
                out.extend((cid, v.astype("float32").tolist()) for cid, v in zip(ids, vecs))
            yield pd.DataFrame(out, columns=["chunk_id", "embedding"])

    return df_pair.mapInPandas(_decode, schema=_EMB_SCHEMA)
