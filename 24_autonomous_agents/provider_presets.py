"""Provider presets - the one place base URLs and default models are fixed.

Students never type a base URL. They pick a provider by name, paste one key, and
(optionally) name a model:

    LLM_PROVIDER=deepseek          # which provider - base URL is predefined below
    LLM_API_KEY=sk-...             # your key for that provider
    LLM_MODEL=deepseek-v4-pro      # optional; omit to use the default below

Everything else (the OpenAI-compatible base URL, the default model) is predefined
here, per provider, and is not a user setting. This table is the only place in the
chapter that names an endpoint: a notebook that hardcoded one would have to be
edited in step with every other, which is how a chapter ends up pointing half its
notebooks at a host that has moved.

Fields per provider:
  - ``client``        which client class builds it. OpenAI-compatible hosts
                      (openai, deepseek, openrouter, and any vLLM/LM-Studio/GLM
                      endpoint) all share the one OpenAI client, differing only
                      by ``base_url``.
  - ``base_url``      the fixed endpoint. ``None`` means the client's native
                      default (OpenAI's own URL, Anthropic, Google).
  - ``default_model`` used when ``LLM_MODEL`` is unset.
"""

from __future__ import annotations

PROVIDER_PRESETS: dict[str, dict[str, str | None]] = {
    "deepseek": {
        "client": "openai",
        "base_url": "https://api.deepseek.com",
        "default_model": "deepseek-v4-pro",
    },
    "openrouter": {
        "client": "openai",
        "base_url": "https://openrouter.ai/api/v1",
        "default_model": "deepseek/deepseek-chat-v3.1",
    },
    "openai": {
        "client": "openai",
        "base_url": None,
        "default_model": "gpt-4.1-mini",
    },
    "anthropic": {
        "client": "anthropic",
        "base_url": None,
        "default_model": "claude-sonnet-4-20250514",
    },
    "google": {
        "client": "google",
        "base_url": None,
        "default_model": "gemini-2.5-flash",
    },
    "ollama": {
        "client": "ollama",
        "base_url": "http://localhost:11434",
        "default_model": "qwen3:8b",
    },
    "mock": {
        "client": "mock",
        "base_url": None,
        "default_model": "mock-model",
    },
}

# Providers a user may name in LLM_PROVIDER, in preference order for messages.
SUPPORTED_PROVIDERS: tuple[str, ...] = tuple(PROVIDER_PRESETS)
