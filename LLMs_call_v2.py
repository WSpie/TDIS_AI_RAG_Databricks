# llm_dual_stage.py
# English comments only

import yaml
from typing import List, Dict, Any, Optional
from openai import OpenAI


# -------------------------
# Config
# -------------------------

def load_config(config_path: str = "config.yaml") -> dict:
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def get_dbx_client(
    config_path: str = "config.yaml",
    base_url: str = "https://adb-3300405005568038.18.azuredatabricks.net/serving-endpoints",
) -> OpenAI:
    cfg = load_config(config_path)
    token = cfg["DATABRICKS_TOKEN"]
    return OpenAI(api_key=token, base_url=base_url)


# -------------------------
# Message builders
# -------------------------

def build_messages_stage1(user_query: str) -> List[Dict[str, str]]:
    system_prompt = (
        "You are a disaster risk domain expert.\n"
        "Answer the QUESTION using your internal knowledge only.\n"
        "Do NOT mention any context or knowledge base.\n"
        "Output must be plain text only (no markdown, no bullets, no numbering).\n"
        "Output exactly two lines:\n"
        "Answers: <ONE paragraph only, no line breaks>\n"
        "Data sources: <ONE paragraph only, no line breaks>\n"
    )

    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": f"QUESTION:\n{user_query}\n"},
    ]


def build_messages_stage2(text_collection: str, user_query: str) -> List[Dict[str, str]]:
    system_prompt = (
        "You are a disaster risk domain expert.\n"
        "You MUST answer using ONLY the provided CONTEXT.\n"
        "Do NOT use outside knowledge. Do NOT guess. Do NOT fabricate.\n"
        "If the CONTEXT does not contain enough information, reply exactly:\n"
        "OUT_OF_KNOWLEDGE\n"
        "Output must be plain text only (no markdown, no bullets, no numbering, no extra symbols).\n"
        "If you can answer, output exactly two lines:\n"
        "Answers: <ONE paragraph only, no line breaks>\n"
        "Data sources: <ONE paragraph only, no line breaks>\n"
        "If OUT_OF_KNOWLEDGE, output ONLY that single line and nothing else.\n"
    )

    return [
        {"role": "system", "content": system_prompt},
        {
            "role": "user",
            "content": f"CONTEXT:\n{text_collection}\n\nQUESTION:\n{user_query}\n",
        },
    ]


# -------------------------
# LLM call
# -------------------------

def _normalize_content(content) -> str:
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        out = []
        for p in content:
            if isinstance(p, dict) and p.get("type") == "text":
                t = p.get("text")
                if isinstance(t, str):
                    out.append(t)
        return "".join(out)
    return str(content)


def dbx_chat(
    client: OpenAI,
    messages: List[Dict[str, str]],
    model_name: str,
    temperature: Optional[float] = None,
    max_tokens: int = 5000,
) -> str:

    params: Dict[str, Any] = {
        "model": model_name,
        "messages": messages,
        "max_tokens": max_tokens,
    }

    if temperature is not None:
        params["temperature"] = float(temperature)

    resp = client.chat.completions.create(**params)
    return _normalize_content(resp.choices[0].message.content).strip()


# -------------------------
# Dual-stage pipeline
# -------------------------

def dual_stage_answer(
    user_query: str,
    text_collection: str,
    model_name: str,
    temperature: Optional[float] = None,
    config_path: str = "config.yaml",
) -> Dict[str, str]:

    client = get_dbx_client(config_path=config_path)

    # Stage 1: LLM only
    msg1 = build_messages_stage1(user_query)
    ans1 = dbx_chat(client, msg1, model_name, temperature)

    # Stage 2: KB grounded
    msg2 = build_messages_stage2(text_collection, user_query)
    ans2 = dbx_chat(client, msg2, model_name, temperature)

    final = ans2 if ans2.strip() != "OUT_OF_KNOWLEDGE" else ans1

    return {
        "final_answer": final,
        "stage2_answer": ans2,
        "stage1_answer": ans1,
    }