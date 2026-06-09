# functions/LLM_call.py
# -*- coding: utf-8 -*-

from __future__ import annotations
from typing import Any, Dict, List, Optional, Union

import yaml
from openai import OpenAI


def load_config(config_path: str = "../config.yaml") -> dict:
    # Load YAML config
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _disable_mlflow_autolog() -> None:
    # Disable MLflow OpenAI autologging to avoid pydantic issues
    try:
        import mlflow
        try:
            import mlflow.openai
            mlflow.openai.autolog(disable=True)
        except Exception:
            pass
        try:
            mlflow.autolog(disable=True)
        except Exception:
            pass
    except Exception:
        pass



def _normalize_messages(messages: Union[List[Dict[str, Any]], Dict[str, Any]]) -> List[Dict[str, str]]:
    # Accept list[{role,content},...] OR dict{system,user}
    chat_msgs: List[Dict[str, Any]] = messages if isinstance(messages, list) else []
    if not chat_msgs and isinstance(messages, dict):
        if messages.get("system") is not None:
            chat_msgs.append({"role": "system", "content": messages["system"]})
        if messages.get("user") is not None:
            chat_msgs.append({"role": "user", "content": messages["user"]})

    # Force content to pure string
    out: List[Dict[str, str]] = []
    for m in chat_msgs:
        out.append({"role": str(m.get("role", "")), "content": _content_to_text(m.get("content"))})
    return out


def get_dbx_client(
    config_path: str = "../config.yaml",
    base_url: str = "https://adb-3300405005568038.18.azuredatabricks.net/serving-endpoints",
    token_key: str = "DATABRICKS_TOKEN",
) -> OpenAI:
    # Create Databricks OpenAI-compatible client
    cfg = load_config(config_path)
    token = cfg[token_key]
    return OpenAI(api_key=token, base_url=base_url)


def llm_chat(
    messages: Union[List[Dict[str, Any]], Dict[str, Any]],
    model_name: str = "databricks-gpt-oss-20b",
    temperature: Optional[float] = None,
    max_tokens: int = 5000,
    config_path: str = "../config.yaml",
    base_url: str = "https://adb-3300405005568038.18.azuredatabricks.net/serving-endpoints",
    token_key: str = "DATABRICKS_TOKEN",
) -> str:
    # Generic chat call (Databricks serving endpoints)
    _disable_mlflow_autolog()

    client = get_dbx_client(config_path=config_path, base_url=base_url, token_key=token_key)
    chat_msgs = _normalize_messages(messages)

    params: Dict[str, Any] = {"model": model_name, "messages": chat_msgs, "max_tokens": max_tokens}
    if temperature is not None:
        params["temperature"] = float(temperature)

    resp = client.chat.completions.create(**params)
    return resp.choices[0].message.content[1]['text']