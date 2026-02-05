import yaml
import openai
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
    system_prompt = """You are a disaster risk domain expert.
    You MUST answer using ONLY the provided CONTEXT (retrieved snippets).
    Do NOT use outside knowledge. Do NOT guess. Do NOT fabricate citations.
    If the CONTEXT does not contain enough information, reply exactly: OUT_OF_KNOWLEDGE
    When you answer, be concise and factual.
    """

    user_prompt = f"""CONTEXT:
    {text_collection}

    QUESTION:
    {user_query}

    RESPONSE FORMAT:
    - User Query: {user_query}
    - Answers:
    - Data sources:
    """

    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]
