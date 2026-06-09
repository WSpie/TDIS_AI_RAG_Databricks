# functions/clean.py
# -*- coding: utf-8 -*-

from __future__ import annotations
from typing import Dict, List, Any
import re

# ---------- text cleanup (KEEP only) ----------
_re_hyphen_wrap = re.compile(r"(?<=\w)-\s*\n(?=\w)")
_re_ws = re.compile(r"\s+")
_re_bad = re.compile(r"[^A-Za-z0-9\s]+")

def clean_text_keep(text: str) -> str:
    # Fix PDF hyphenation across line breaks
    t = (text or "").replace("\r", "\n")
    t = _re_hyphen_wrap.sub("", t)

    # Single paragraph
    t = t.replace("\n", " ")

    # Replace all non-alnum symbols with space
    t = _re_bad.sub(" ", t)

    # Collapse spaces
    t = _re_ws.sub(" ", t).strip()
    return t


# ---------- batch decision prompt ----------
def build_batch_decision_system_prompt() -> str:
    return (
        "For each chunk, output exactly one line: <i>\\tKEEP or <i>\\tDROP (i is the chunk number).\n"
        "KEEP only if the chunk has substantive flood/storm/disaster risk, hazard, damage/impact, inundation, hydrology/drainage, mitigation, "
        "or agency report content (NOAA/NWS/FEMA/USACE). A few URLs in a real report are OK.\n"
        "DROP if mainly ads/marketing/CTA, contact info, navigation/boilerplate, or mostly links/listing with little content, or off-topic.\n"
        "If unsure -> KEEP.\n"
        "No extra text. Do not output reasoning."
    )

def build_batch_decision_user_prompt(items: List[Dict[str, Any]]) -> str:
    # items: [{"i":1,"text":"..."}, ...]
    parts = []
    for it in items:
        i = int(it["i"])
        txt = str(it["text"])
        parts.append(f"[{i}]\n{txt}")
    return "\n\n".join(parts)

def build_batch_decision_messages(items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    return [
        {"role": "system", "content": build_batch_decision_system_prompt()},
        {"role": "user", "content": build_batch_decision_user_prompt(items)},
    ]