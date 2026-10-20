"""Nebius Token Factory chat client (OpenAI-compatible) with retry on transient errors."""
from __future__ import annotations

import asyncio
import os

import httpx

BASE_URL = "https://api.tokenfactory.nebius.com/v1"
DEFAULT_MODEL = "nvidia/nemotron-3-super-120b-a12b"
RETRY_STATUS = {429, 500, 502, 503}


class LLMError(RuntimeError):
    pass


async def chat(messages: list[dict], model: str = DEFAULT_MODEL, max_tokens: int = 8000, attempts: int = 3) -> dict:
    """Return the raw chat-completions JSON. Raises LLMError on a non-retryable failure.

    Reasoning models spend part of max_tokens on hidden reasoning, so keep it generous:
    a too-small budget shows up as an empty reply, not as an error.
    """
    key = os.environ.get("NEBIUS_API_KEY")
    if not key:
        raise LLMError("NEBIUS_API_KEY is not set")
    body = {"model": model, "messages": messages, "max_tokens": max_tokens, "temperature": 0.2}
    headers = {"Authorization": f"Bearer {key}"}
    async with httpx.AsyncClient(timeout=180) as http:
        for attempt in range(attempts):
            resp = await http.post(f"{BASE_URL}/chat/completions", json=body, headers=headers)
            if resp.status_code == 200:
                data = resp.json()
                # Guard against silent model routing: the response must name the model we asked for.
                if data.get("model") != model:
                    raise LLMError(f"model mismatch: asked {model}, got {data.get('model')}")
                return data
            if resp.status_code in RETRY_STATUS and attempt < attempts - 1:
                await asyncio.sleep(2 ** attempt)
                continue
            raise LLMError(f"HTTP {resp.status_code}: {resp.text[:200]}")
    raise LLMError("unreachable")
