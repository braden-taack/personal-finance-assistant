"""LM Studio (OpenAI-compatible) client used as the categorization fallback."""
from __future__ import annotations

import json
import re

import requests

from config import LLM_BASE_URL, LLM_MODEL, LLM_TIMEOUT


def is_available() -> bool:
    try:
        return requests.get(f"{LLM_BASE_URL}/models", timeout=3).ok
    except requests.RequestException:
        return False


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


def categorize(description: str, amount: float, categories: list[str], hint: str | None = None) -> tuple[str, float]:
    """Ask the local LLM to pick a category. Returns (category, confidence 0-1).

    Returns ('', 0.0) on any failure so the pipeline falls through to manual review.
    """
    system = (
        "You categorize personal expense transactions. Choose exactly one category from the "
        "provided list. Respond with ONLY a JSON object: "
        '{"category": "<one of the list>", "confidence": <number 0-1>}. '
        "Set confidence low if the merchant is ambiguous or unfamiliar. Never invent categories."
    )
    user = (
        f"Categories: {json.dumps(categories)}\n"
        f"Transaction description: {description}\nAmount: {amount:.2f}\n"
        + (f"The classical model's best guess was '{hint}' but was unsure.\n" if hint else "")
    )
    try:
        r = requests.post(
            f"{LLM_BASE_URL}/chat/completions",
            json={
                "model": LLM_MODEL,
                "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
                "temperature": 0.0,
                "max_tokens": 200,
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
