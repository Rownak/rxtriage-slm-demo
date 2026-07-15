"""Model callers for baseline evaluation — one `generate()` interface.

Both the local base model and the frontier model are reached through the same
OpenAI SDK: Ollama exposes an OpenAI-compatible endpoint, so the only
difference is base_url + api_key. Keeping one code path means the harness treats
"base" and "frontier" identically and the only variable is the model itself.

temperature=0 for determinism; JSON mode requested so both models emit a single
JSON object. On any API error we return the exception text as the "output" so
the harness scores it as an (invalid) response rather than crashing the run —
an honest failure is data, not an exception to swallow silently elsewhere.
"""

from __future__ import annotations

import os

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()  # pull OPENAI_API_KEY from .env

OLLAMA_BASE_URL = "http://localhost:11434/v1"
MAX_TOKENS = 2048  # outputs run ~1k tokens; 2k leaves headroom without runaway


class _ChatModel:
    """Thin wrapper over an OpenAI-compatible chat endpoint."""

    def __init__(self, client: OpenAI, model: str, use_json_mode: bool = True):
        self.client = client
        self.model = model
        self.use_json_mode = use_json_mode

    def generate(self, messages: list[dict]) -> str:
        kwargs = dict(
            model=self.model,
            messages=messages,
            temperature=0,
            max_tokens=MAX_TOKENS,
        )
        if self.use_json_mode:
            kwargs["response_format"] = {"type": "json_object"}
        try:
            resp = self.client.chat.completions.create(**kwargs)
            return resp.choices[0].message.content or ""
        except Exception as e:  # network/API error → scored as an invalid output
            return f"__ERROR__ {type(e).__name__}: {e}"


def ollama_model(model: str = "qwen2.5:3b-instruct") -> _ChatModel:
    """Local base model via Ollama's OpenAI-compatible endpoint (api_key is a dummy)."""
    client = OpenAI(base_url=OLLAMA_BASE_URL, api_key="ollama")
    return _ChatModel(client, model)


def openai_model(model: str = "gpt-4o-mini") -> _ChatModel:
    """Frontier baseline. Requires OPENAI_API_KEY in the environment / .env."""
    if not os.getenv("OPENAI_API_KEY"):
        raise RuntimeError("OPENAI_API_KEY not set — add it to .env")
    return _ChatModel(OpenAI(), model)
