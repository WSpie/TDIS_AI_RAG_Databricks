# src/stats.py
# Data-volume statistics helpers for the RAG rebuild pipeline.
# All functions take an active SparkSession; no dbutils dependency (Spark-native).

from __future__ import annotations

import re
from typing import Optional

from pyspark.sql import SparkSession
from pyspark.sql import functions as F


def bytes_human(n: Optional[int]) -> str:
    # Human-readable byte size
    if not n:
        return "0 B"
    x = float(n)
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if abs(x) < 1024.0:
            return f"{x:.1f} {unit}"
        x /= 1024.0
    return f"{x:.1f} PB"


def fs_stats(spark: SparkSession, path: str, glob: str = "*", recursive: bool = True) -> dict:
    # Count files + total bytes under a path using the binaryFile reader.
    df = (
        spark.read.format("binaryFile")
        .option("pathGlobFilter", glob)
        .option("recursiveFileLookup", "true" if recursive else "false")
        .load(path)
        .agg(
            F.count("*").alias("n_files"),
            F.sum("length").alias("total_bytes"),
        )
    )
    row = df.first()
    return {
        "path": path,
        "n_files": int(row["n_files"] or 0),
        "total_bytes": int(row["total_bytes"] or 0),
    }


def count_doc_folders(spark: SparkSession, base_path: str, filename: str = "chunks.jsonl") -> int:
    # Count distinct document folders that contain `filename` under base_path.
    fn = re.escape(filename)
    df = (
        spark.read.format("binaryFile")
        .option("pathGlobFilter", filename)
        .option("recursiveFileLookup", "true")
        .load(base_path)
        .withColumn("doc", F.regexp_extract("path", r".*/([^/]+)/" + fn + r"$", 1))
    )
    return df.select("doc").distinct().count()


def jsonl_chunk_stats(spark: SparkSession, glob: str) -> dict:
    # Count chunk records + distinct source docs across chunks.jsonl files.
    df = spark.read.json(glob)
    n_chunks = df.count()
    n_docs = df.select("source_file").distinct().count() if "source_file" in df.columns else None
    return {"n_chunks": n_chunks, "n_docs": n_docs}


def table_stats(spark: SparkSession, table: str) -> dict:
    # Row count + on-disk size of a Delta table (via DESCRIBE DETAIL).
    n_rows = spark.table(table).count()
    detail = spark.sql(f"DESCRIBE DETAIL {table}").select("sizeInBytes", "numFiles").first()
    return {
        "n_rows": n_rows,
        "size_bytes": int(detail["sizeInBytes"] or 0),
        "num_files": int(detail["numFiles"] or 0),
    }
