"""LM Studio (OpenAI-compatible) client used as the categorization fallback."""
from __future__ import annotations

import json
import re
from concurrent.futures import ThreadPoolExecutor

import requests

from config import AGENT_MAX_TOKENS, AGENT_TIMEOUT, LLM_BASE_URL, LLM_MAX_TOKENS, LLM_MODEL, LLM_PARALLEL, LLM_TIMEOUT


def is_available() -> bool:
    try:
        return requests.get(f"{LLM_BASE_URL}/models", timeout=3).ok
    except requests.RequestException:
        return False


def list_models() -> list[dict]:
    """Chat-capable models the server offers: [{id, loaded}], loaded first. [] when unreachable.

    Uses LM Studio's /api/v0/models (which reports load state) and falls back to the standard
    /v1/models list (loaded=None) for other OpenAI-compatible servers. Embedding models are skipped.
    """
    root = LLM_BASE_URL.rstrip("/").removesuffix("/v1")
    try:
        r = requests.get(f"{root}/api/v0/models", timeout=3)
        r.raise_for_status()
        models = [{"id": m["id"], "loaded": m.get("state") == "loaded"}
                  for m in r.json()["data"] if m.get("type") != "embeddings"]
    except (requests.RequestException, KeyError, ValueError):
        try:
            r = requests.get(f"{LLM_BASE_URL}/models", timeout=3)
            r.raise_for_status()
            models = [{"id": m["id"], "loaded": None} for m in r.json()["data"]]
        except (requests.RequestException, KeyError, ValueError):
            return []
    models = [m for m in models if "embed" not in m["id"].lower()]
    return sorted(models, key=lambda m: not m["loaded"])


def resolve_model(preferred: str | None = None, models: list[dict] | None = None) -> str | None:
    """`preferred` (default LLM_MODEL) if the server offers it, else the loaded model, else the first listed.

    Returns None when the server is unreachable or offers no chat models.
    """
    preferred = preferred or LLM_MODEL
    models = list_models() if models is None else models
    if not models:
        return None
    ids = [m["id"] for m in models]
    if preferred in ids:
        return preferred
    loaded = [m["id"] for m in models if m["loaded"]]
    return loaded[0] if loaded else ids[0]


def _extract_json(text: str) -> dict | None:
    # Strip reasoning blocks some local models emit, then grab the first JSON object.
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.S)
    m = re.search(r"\{.*?\}", text, flags=re.S)
    if not m:
        return None
    try:
        return json.loads(m.group(0))
    except json.JSONDecodeError:
        return None


def build_prompt(description: str, amount: float, categories: list[str], hint: str | None = None,
                 examples: list[tuple[str, str, int]] | None = None,
                 glossary: dict[str, list[str]] | None = None) -> tuple[str, str]:
    """(system, user) messages. `examples` are similar past transactions (description, category, times used)
    and `glossary` maps categories to typical merchants; both come from retrieval.ExampleIndex."""
    system = (
        "You categorize one person's expense transactions. Choose exactly one category from the "
        "provided list. Respond with ONLY a JSON object: "
        '{"category": "<one of the list>", "confidence": <number 0-1>}. Never invent categories.'
    )
    if examples or glossary:
        system += (
            " The categories are this person's own; follow how they used them in the past (the category "
            "examples and similar past transactions) over general knowledge. Give confidence above 0.8 only "
            "when similar past transactions clearly support your choice; give 0.5 or less when they don't."
        )
    else:
        system += " Set confidence low if the merchant is ambiguous or unfamiliar."
    lines = []
    if glossary:
        lines.append("Categories, with merchants this person filed under each:")
        lines += [f"- {c}: " + "; ".join(d[:40] for d in glossary.get(c, [])) for c in categories]
    else:
        lines.append(f"Categories: {json.dumps(categories)}")
    if examples:
        lines.append("Most similar past transactions (description -> category, times):")
        lines += [f'- "{d[:60]}" -> {c} ({n}x)' for d, c, n in examples]
    lines.append(f"Transaction to categorize: {description}\nAmount: {amount:.2f}")
    if hint:
        lines.append(f"A text classifier's unsure guess was '{hint}'.")
    return system, "\n".join(lines)


def categorize(description: str, amount: float, categories: list[str], hint: str | None = None,
               model: str | None = None, examples: list[tuple[str, str, int]] | None = None,
               glossary: dict[str, list[str]] | None = None) -> tuple[str, float]:
    """Ask the local LLM to pick a category. Returns (category, confidence 0-1).

    Returns ('', 0.0) on any failure so the pipeline falls through to manual review.
    """
    system, user = build_prompt(description, amount, categories, hint, examples, glossary)
    try:
        r = requests.post(
            f"{LLM_BASE_URL}/chat/completions",
            json={
                "model": model or LLM_MODEL,
                "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
                "temperature": 0.0,
                "max_tokens": LLM_MAX_TOKENS,
            },
            timeout=LLM_TIMEOUT,
        )
        r.raise_for_status()
        content = r.json()["choices"][0]["message"]["content"]
    except (requests.RequestException, KeyError, ValueError):
        return "", 0.0

    data = _extract_json(content)
    if not data:
        return "", 0.0
    cat = str(data.get("category", "")).strip()
    # Case-insensitive match back to a real category; reject anything off-list.
    lookup = {c.lower(): c for c in categories}
    if cat.lower() not in lookup:
        return "", 0.0
    try:
        conf = max(0.0, min(1.0, float(data.get("confidence", 0.0))))
    except (TypeError, ValueError):
        conf = 0.0
    return lookup[cat.lower()], conf


def categorize_many(items: list[tuple], categories: list[str], model: str | None = None,
                    workers: int = LLM_PARALLEL, glossary: dict[str, list[str]] | None = None) -> list[tuple[str, float]]:
    """categorize() for each (description, amount, hint[, examples]), up to `workers` requests at once.

    Results keep input order.
    """
    if not items:
        return []

    def one(it):
        return categorize(it[0], it[1], categories, hint=it[2], model=model,
                          examples=it[3] if len(it) > 3 else None, glossary=glossary)

    with ThreadPoolExecutor(max_workers=max(1, int(workers))) as pool:
        return list(pool.map(one, items))


class LLMError(RuntimeError):
    """The LLM server was unreachable or answered with something unusable."""


def chat(messages: list[dict], tools: list[dict] | None = None, model: str | None = None) -> dict:
    """One chat-completions call at temperature 0 (the agent's call). Returns the assistant message dict
    ({"role", "content", optional "tool_calls"}). Raises LLMError on any failure."""
    body = {"model": model or LLM_MODEL, "messages": messages, "temperature": 0.0, "max_tokens": AGENT_MAX_TOKENS}
    if tools:
        body["tools"] = tools
    try:
        r = requests.post(f"{LLM_BASE_URL}/chat/completions", json=body, timeout=AGENT_TIMEOUT)
        r.raise_for_status()
        return r.json()["choices"][0]["message"]
    except requests.HTTPError as e:
        raise LLMError(f"LLM server returned {e.response.status_code}: {e.response.text[:300]}") from e
    except (requests.RequestException, KeyError, IndexError, ValueError) as e:
        raise LLMError(f"Could not get an answer from the LLM at {LLM_BASE_URL}: {e}") from e
