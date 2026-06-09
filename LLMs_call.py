import yaml
import openai
from openai import OpenAI
from typing import Dict, Tuple, List, Any

def load_api_key(key: str = "gpt_api_key", config_path: str = "config.yaml") -> str:
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    if key not in cfg:
        raise KeyError(f"Expected '{key}' in {config_path}")
    return cfg[key]

def ask_gpt(
    messages: List[Dict[str, str]],
    model: str = "gpt-4o-mini",
    temperature: float | None = None,
    **kwargs: Any,
) -> str:
    """
    Unified wrapper for OpenAI chat completions.
    - If temperature is provided (not None), pass it explicitly.
    - If temperature is None, do NOT include it (so models like gpt-5-mini won't reject it).
    """
    params: Dict[str, Any] = {
        "model": model,
        "messages": messages,
    }

    if temperature is not None:
        params["temperature"] = float(temperature)

    params.update(kwargs)

    response = openai.chat.completions.create(**params)
    print('LLM prompting done.')
    return (response.choices[0].message.content or "").strip()

def build_messages(text_collection: str, user_query: str) -> List[Dict[str, str]]:
    # Enforce plain-text, single-paragraph sections, strict OOK behavior
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

    user_prompt = (
        "CONTEXT:\n"
        f"{text_collection}\n\n"
        "QUESTION:\n"
        f"{user_query}\n"
    )

    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]


def load_config(config_path: str = "config.yaml") -> dict:
    # Load YAML config file
    with open(config_path, "r") as f:
        return yaml.safe_load(f)

def dbx_llm_chat(
    messages,
    model_name: str,
    temperature=None,
    max_tokens: int = 5000,
    config_path: str = "config.yaml",
    base_url: str = "https://adb-3300405005568038.18.azuredatabricks.net/serving-endpoints",
) -> str:
    # Disable MLflow OpenAI autologging to avoid pydantic errors on non-standard content parts
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

    # Normalize OpenAI-style content (str | list[parts]) to plain text
    # Keep only parts with type == "text"; drop unknown types like "reasoning"
    def _content_to_text(content):
        if content is None:
            return ""
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            out = []
            for p in content:
                if not isinstance(p, dict):
                    continue
                if p.get("type") == "text":
                    t = p.get("text")
                    if isinstance(t, str):
                        out.append(t)
            return "".join(out)
        return str(content)

    # Read token from config.yaml
    cfg = load_config(config_path)
    token = cfg["DATABRICKS_TOKEN"]

    # Create Databricks OpenAI-compatible client
    client = OpenAI(api_key=token, base_url=base_url)

    # Accept both dict {system,user} and list[{role,content},...]
    chat_msgs = messages if isinstance(messages, list) else []
    if not chat_msgs:
        if messages.get("system") is not None:
            chat_msgs.append({"role": "system", "content": messages["system"]})
        if messages.get("user") is not None:
            chat_msgs.append({"role": "user", "content": messages["user"]})

    # Normalize all input message contents to plain strings
    for m in chat_msgs:
        m["content"] = _content_to_text(m.get("content"))

    # Build params; only pass temperature if provided
    params = dict(model=model_name, messages=chat_msgs, max_tokens=max_tokens)
    if temperature is not None:
        params["temperature"] = temperature

    resp = client.chat.completions.create(**params)

    # Normalize output content to plain string
    return _content_to_text(resp.choices[0].message.content)




