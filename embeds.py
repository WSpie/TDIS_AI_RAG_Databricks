# dbx_embed.py
# English comments only, Databricks-friendly, ABFSS supported

from __future__ import annotations

import os
import json
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer
from tqdm.auto import tqdm


# -------------------------
# Core embedding
# -------------------------

@torch.no_grad()
def _mean_pool(last_hidden_state: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    mask = attention_mask.unsqueeze(-1).type_as(last_hidden_state)
    summed = (last_hidden_state * mask).sum(dim=1)
    counts = mask.sum(dim=1).clamp(min=1e-9)
    return summed / counts


@dataclass
class EmbedConfig:
    model_id: str
    batch_size: int
    max_length: int
    device: str
    fp16: bool = False


class Embedder:
    def __init__(self, cfg: EmbedConfig):
        self.cfg = cfg
        self.tok = AutoTokenizer.from_pretrained(cfg.model_id, use_fast=True)
        self.model = AutoModel.from_pretrained(cfg.model_id).to(cfg.device).eval()

    @torch.no_grad()
    def embed_batch(self, texts: List[str]) -> np.ndarray:
        enc = self.tok(
            texts,
            padding=True,
            truncation=True,
            max_length=self.cfg.max_length,
            return_tensors="pt",
        ).to(self.cfg.device)

        if self.cfg.fp16 and self.cfg.device.startswith("cuda"):
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                out = self.model(**enc)
        else:
            out = self.model(**enc)

        emb = _mean_pool(out.last_hidden_state, enc["attention_mask"])
        emb = F.normalize(emb, p=2, dim=1)
        return emb.detach().cpu().numpy().astype(np.float32)


# -------------------------
# Helpers
# -------------------------

def _auto_workers() -> int:
    try:
        return max(1, len(os.sched_getaffinity(0)))
    except Exception:
        return max(1, os.cpu_count() or 1)


def _is_abfs(p: str) -> bool:
    return p.startswith("abfss://") or p.startswith("abfs://")


def _get_dbutils(spark=None):
    if spark is None:
        from pyspark.sql import SparkSession
        spark = SparkSession.getActiveSession() or SparkSession.builder.getOrCreate()
    try:
        from pyspark.dbutils import DBUtils
        return DBUtils(spark)
    except Exception:
        return globals()["dbutils"]


def _read_jsonl_local(path: Path) -> Tuple[List[str], List[str], dict]:
    chunk_ids, texts = [], []
    meta0 = {}
    with path.open("r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if not line.strip():
                continue
            obj = json.loads(line)
            if i == 0:
                meta0 = {
                    "source_file": obj.get("source_file"),
                    "source_pdf": obj.get("source_pdf"),
                }
            chunk_ids.append(obj["chunk_id"])
            texts.append(obj.get("text", ""))
    return chunk_ids, texts, meta0


# -------------------------
# Main entry
# -------------------------

def embed_main(
    optimized_chunks: str,                  # abfss://.../optimized_chunks/
    out_base: str,                          # abfss://.../optimized_embeddings/
    model_id: str = "DMIR01/DMRetriever-33M",
    model_dirname: str = "DMRetriever-33M",
    batch_size: int = 64,
    max_length: int = 512,
    device: str = "cpu",                    # default CPU
    fp16: bool = False,
    n_workers: Optional[int] = None,        # auto
    mode: str = "process",                  # default process
    spark=None,
) -> dict:

    if n_workers is None:
        n_workers = _auto_workers()

    if not str(device).startswith("cuda"):
        fp16 = False

    if not (_is_abfs(optimized_chunks) and _is_abfs(out_base)):
        raise ValueError("optimized_chunks and out_base must be abfss:// paths")

    dbutils = _get_dbutils(spark)

    # List doc folders
    doc_dirs = [
        fi.path.rstrip("/")
        for fi in dbutils.fs.ls(optimized_chunks)
        if fi.isDir()
    ]

    if not doc_dirs:
        raise FileNotFoundError("No doc folders found")

    embedder = Embedder(EmbedConfig(model_id, batch_size, max_length, device, fp16))

    tmp_root = Path("/tmp") / f"embed_{uuid.uuid4().hex}"
    tmp_root.mkdir(parents=True, exist_ok=True)

    total_chunks = 0

    for doc_dir in tqdm(doc_dirs, desc="Embedding"):
        doc_id = doc_dir.split("/")[-1]
        chunks_abfs = f"{doc_dir}/chunks.jsonl"

        # Local staging
        local_doc_dir = tmp_root / doc_id
        local_doc_dir.mkdir(parents=True, exist_ok=True)
        local_jsonl = local_doc_dir / "chunks.jsonl"

        dbutils.fs.cp(chunks_abfs, f"file:{local_jsonl}", True)

        chunk_ids, texts, meta0 = _read_jsonl_local(local_jsonl)
        if not texts:
            continue

        # Embed
        embs = []
        for i in range(0, len(texts), batch_size):
            embs.append(embedder.embed_batch(texts[i:i + batch_size]))
        embs = np.vstack(embs).astype(np.float32)

        # Save local
        local_out = local_doc_dir / "out"
        local_out.mkdir(parents=True, exist_ok=True)

        np.save(local_out / "embeddings.npy", embs)
        (local_out / "chunk_ids.json").write_text(
            json.dumps(chunk_ids, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

        meta = {
            "name": model_dirname,
            "hf_id": model_id,
            "type": "embedding",
            "doc_id": doc_id,
            "n_chunks": len(texts),
            "max_length": max_length,
            "batch_size": batch_size,
            **meta0,
        }

        (local_out / "model_meta.json").write_text(
            json.dumps(meta, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

        # Upload back
        out_doc_abfs = f"{out_base.rstrip('/')}/{model_dirname}/{doc_id}"
        dbutils.fs.mkdirs(out_doc_abfs)

        dbutils.fs.cp(f"file:{local_out/'embeddings.npy'}", f"{out_doc_abfs}/embeddings.npy", True)
        dbutils.fs.cp(f"file:{local_out/'chunk_ids.json'}", f"{out_doc_abfs}/chunk_ids.json", True)
        dbutils.fs.cp(f"file:{local_out/'model_meta.json'}", f"{out_doc_abfs}/model_meta.json", True)

        total_chunks += len(texts)

    return {
        "model": model_id,
        "files": len(doc_dirs),
        "chunks": total_chunks,
        "out": f"{out_base.rstrip('/')}/{model_dirname}",
    }