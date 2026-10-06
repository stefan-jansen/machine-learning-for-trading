"""LLM provider abstraction with auto-detection cascade.

Providers: Anthropic, OpenAI, Google Gemini, OpenRouter (any model), Ollama
(local), Mock (explicit tests). Any OpenAI-compatible endpoint (OpenLLM, vLLM,
LM Studio, …) plugs in through the OpenAI client by passing a base_url.

Auto-detect priority: ANTHROPIC_API_KEY -> OPENAI_API_KEY -> GOOGLE_API_KEY ->
OPENROUTER_API_KEY -> Ollama (if running). Missing live integrations raise errors.

No disk cache, no structured logging — teaching code stays transparent.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable

from agent_schemas import TokenUsage
from provider_presets import PROVIDER_PRESETS, SUPPORTED_PROVIDERS

# ---------------------------------------------------------------------------
# Protocol
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class ChatMessage:
    """A single message in the conversation."""

    role: str  # system, user, assistant, tool
    content: str


@runtime_checkable
class LLMClient(Protocol):
    """Protocol for LLM providers -- all agents program against this interface."""

    @property
    def model_name(self) -> str: ...

    def complete(self, messages: list[ChatMessage], json_mode: bool = False) -> str: ...

    def complete_with_usage(
        self, messages: list[ChatMessage], json_mode: bool = False
    ) -> tuple[str, TokenUsage]: ...


# ---------------------------------------------------------------------------
# Mock provider (deterministic, for CI / testing)
# ---------------------------------------------------------------------------


class MockLLMClient:
    """Deterministic mock for CI and offline runs.

    Flow: (1) search once, (2) forecast. Supervisor returns "medium"
    confidence so the system falls back to the ensemble mean.
    """

    @property
    def model_name(self) -> str:
        return "mock-model"

    def complete(self, messages: list[ChatMessage], json_mode: bool = False) -> str:
        text, _ = self.complete_with_usage(messages, json_mode)
        return text

    def complete_with_usage(
        self, messages: list[ChatMessage], json_mode: bool = False
    ) -> tuple[str, TokenUsage]:
        last = messages[-1].content.lower()

        # Supervisor: identify disagreements
        if "supervisor" in last and ("disagree" in last or "identify" in last):
            response = json.dumps(
                {
                    "disagreements": ["Agents differ on base rate anchoring"],
                    "queries": ["base rate for event"],
                }
            )
        # Supervisor: finalize
        elif "supervisor" in last and (
            "final" in last or "updated forecast" in last or "output" in last
        ):
            response = json.dumps(
                {
                    "p_yes": 0.58,
                    "confidence": "medium",
                    "rationale": "Mock supervisor: ensemble looks reasonable",
                }
            )
        # Debate: bull
        elif "bull" in last and "higher probability" in last:
            response = json.dumps(
                {
                    "argument": "Mock bull case: positive evidence supports YES",
                    "p_yes": 0.68,
                    "key_evidence": ["Mock evidence point"],
                }
            )
        # Debate: bear
        elif "bear" in last and "lower probability" in last:
            response = json.dumps(
                {
                    "argument": "Mock bear case: uncertainty warrants caution",
                    "p_yes": 0.52,
                    "key_evidence": ["Mock risk factor"],
                }
            )
        # Agent: ReAct loop
        elif "action" in last and ("search" in last or "forecast" in last):
            has_tool_result = any(m.role == "tool" for m in messages)
            if has_tool_result:
                response = json.dumps(
                    {
                        "action": "forecast",
                        "p_yes": 0.62,
                        "rationale": "Synthetic test support: https://example.org/default-0",
                        "confidence": 0.65,
                        "key_findings": ["Mock finding from search results"],
                    }
                )
            else:
                response = json.dumps(
                    {"action": "search", "query": "latest evidence for this question"}
                )
        # Fallback
        else:
            response = json.dumps(
                {
                    "action": "forecast",
                    "p_yes": 0.60,
                    "rationale": "Mock default forecast",
                    "confidence": 0.6,
                }
            )

        input_tokens = sum(len(m.content.split()) * 2 for m in messages)
        output_tokens = len(response.split()) * 2
        usage = TokenUsage(
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            total_tokens=input_tokens + output_tokens,
        )
        return response, usage


# ---------------------------------------------------------------------------
# Anthropic Claude
# ---------------------------------------------------------------------------


class AnthropicChatClient:
    """Anthropic Claude client (claude-sonnet-4-20250514, etc.)."""

    def __init__(
        self,
        api_key: str,
        model: str = "claude-sonnet-4-20250514",
        temperature: float = 0.7,
        max_tokens: int = 1200,
    ) -> None:
        try:
            from anthropic import Anthropic
        except ImportError as e:
            raise ImportError("pip install anthropic  (or: uv add anthropic)") from e
        self._client = Anthropic(api_key=api_key)
        self._model = model
        self._temperature = temperature
        self._max_tokens = max_tokens

    @property
    def model_name(self) -> str:
        return self._model

    def complete(self, messages: list[ChatMessage], json_mode: bool = False) -> str:
        text, _ = self.complete_with_usage(messages, json_mode)
        return text

    def complete_with_usage(
        self, messages: list[ChatMessage], json_mode: bool = False
    ) -> tuple[str, TokenUsage]:
        system_content = None
        conversation: list[dict[str, Any]] = []

        for msg in messages:
            if msg.role == "system":
                system_content = msg.content
            else:
                role = "user" if msg.role == "tool" else msg.role
                conversation.append({"role": role, "content": msg.content})

        kwargs: dict[str, Any] = {
            "model": self._model,
            "messages": conversation,
            "max_tokens": self._max_tokens,
            "temperature": self._temperature,
        }
        if system_content:
            suffix = (
                "\n\nYou must respond with valid JSON only. No other text." if json_mode else ""
            )
            kwargs["system"] = system_content + suffix
        elif json_mode:
            kwargs["system"] = "You must respond with valid JSON only. No other text."

        response = self._client.messages.create(**kwargs)
        text = "".join(b.text for b in response.content if hasattr(b, "text"))

        usage = TokenUsage(
            input_tokens=response.usage.input_tokens,
            output_tokens=response.usage.output_tokens,
            total_tokens=response.usage.input_tokens + response.usage.output_tokens,
        )
        return text, usage


# ---------------------------------------------------------------------------
# OpenAI / OpenAI-compatible
# ---------------------------------------------------------------------------


class OpenAIChatClient:
    """OpenAI client (gpt-4.1-mini, gpt-4o, etc.)."""

    def __init__(
        self,
        api_key: str,
        model: str = "gpt-4.1-mini",
        temperature: float = 0.7,
        max_tokens: int = 4000,
        base_url: str | None = None,
    ) -> None:
        # 4000, not 1200: reasoning models routed through this OpenAI-compatible
        # client (e.g. deepseek-v4-pro via OpenRouter) spend completion tokens on
        # an internal reasoning trace BEFORE the visible answer. With a 1200 cap,
        # a long reasoning turn exhausts the budget and the API returns empty
        # `content` (finish_reason="length") - the agent then degenerates to a
        # 0.5 "JSON parse error" forecast. The production deepseek profile sets
        # 8000 for the same reason; 4000 is ample for the single-agent notebooks.
        # max_tokens is only a ceiling, so non-reasoning models pay nothing extra.
        try:
            from openai import OpenAI
        except ImportError as e:
            raise ImportError("pip install openai  (or: uv add openai)") from e
        self._client = OpenAI(api_key=api_key, base_url=base_url)
        self._model = model
        self._temperature = temperature
        self._max_tokens = max_tokens

    @property
    def model_name(self) -> str:
        return self._model

    def complete(self, messages: list[ChatMessage], json_mode: bool = False) -> str:
        text, _ = self.complete_with_usage(messages, json_mode)
        return text

    def complete_with_usage(
        self, messages: list[ChatMessage], json_mode: bool = False
    ) -> tuple[str, TokenUsage]:
        request_messages = []
        for m in messages:
            role = m.role
            content = m.content
            if role == "tool":
                role = "user"
                content = f"[Tool Result]\n{content}"
            request_messages.append({"role": role, "content": content})

        kwargs: dict[str, Any] = {}
        if json_mode:
            kwargs["response_format"] = {"type": "json_object"}

        resp = self._client.chat.completions.create(
            model=self._model,
            messages=request_messages,
            temperature=self._temperature,
            max_tokens=self._max_tokens,
            **kwargs,
        )
        text = resp.choices[0].message.content or ""

        usage = TokenUsage(
            input_tokens=resp.usage.prompt_tokens if resp.usage else 0,
            output_tokens=resp.usage.completion_tokens if resp.usage else 0,
            total_tokens=resp.usage.total_tokens if resp.usage else 0,
        )
        return text, usage


# ---------------------------------------------------------------------------
# Google Gemini
# ---------------------------------------------------------------------------


class GoogleGeminiClient:
    """Google Gemini client (gemini-2.5-pro, gemini-2.5-flash, etc.)."""

    def __init__(
        self,
        api_key: str,
        model: str = "gemini-2.5-flash",
        temperature: float = 0.7,
        max_tokens: int = 1200,
    ) -> None:
        try:
            from google import genai
        except ImportError as e:
            raise ImportError("pip install google-genai  (or: uv add google-genai)") from e
        self._genai = genai
        self._client = genai.Client(api_key=api_key)
        self._model = model
        self._temperature = temperature
        self._max_tokens = max_tokens

    @property
    def model_name(self) -> str:
        return self._model

    def complete(self, messages: list[ChatMessage], json_mode: bool = False) -> str:
        text, _ = self.complete_with_usage(messages, json_mode)
        return text

    def complete_with_usage(
        self, messages: list[ChatMessage], json_mode: bool = False
    ) -> tuple[str, TokenUsage]:
        from google.genai import types

        system_content: str | None = None
        contents: list[types.Content] = []

        for msg in messages:
            if msg.role == "system":
                system_content = msg.content
                continue
            # Gemini uses "user" and "model" roles; map "assistant" -> "model"
            # and "tool" -> "user" (with a labeled prefix for clarity).
            role = "model" if msg.role == "assistant" else "user"
            content = msg.content
            if msg.role == "tool":
                content = f"[Tool Result]\n{content}"
            contents.append(types.Content(role=role, parts=[types.Part.from_text(content)]))

        config_kwargs: dict[str, Any] = {
            "temperature": self._temperature,
            "max_output_tokens": self._max_tokens,
        }
        if system_content:
            config_kwargs["system_instruction"] = system_content
        if json_mode:
            config_kwargs["response_mime_type"] = "application/json"

        response = self._client.models.generate_content(
            model=self._model,
            contents=contents,
            config=types.GenerateContentConfig(**config_kwargs),
        )

        text = response.text or ""

        usage_obj = getattr(response, "usage_metadata", None)
        input_tokens = getattr(usage_obj, "prompt_token_count", 0) or 0
        output_tokens = getattr(usage_obj, "candidates_token_count", 0) or 0
        usage = TokenUsage(
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            total_tokens=input_tokens + output_tokens,
        )
        return text, usage


# ---------------------------------------------------------------------------
# Ollama (local)
# ---------------------------------------------------------------------------


class OllamaChatClient:
    """Ollama client for local LLM inference via HTTP API."""

    def __init__(
        self,
        model: str = "qwen2.5:32b",
        base_url: str = "http://localhost:11434",
        temperature: float = 0.7,
        max_tokens: int = 1200,
    ) -> None:
        try:
            import httpx  # noqa: F401
        except ImportError as e:
            raise ImportError("pip install httpx  (or: uv add httpx)") from e
        self._model = model
        self._base_url = base_url.rstrip("/")
        self._temperature = temperature
        self._max_tokens = max_tokens

    @property
    def model_name(self) -> str:
        return self._model

    def complete(self, messages: list[ChatMessage], json_mode: bool = False) -> str:
        text, _ = self.complete_with_usage(messages, json_mode)
        return text

    def complete_with_usage(
        self, messages: list[ChatMessage], json_mode: bool = False
    ) -> tuple[str, TokenUsage]:
        import httpx

        ollama_messages = []
        for msg in messages:
            role = msg.role
            content = msg.content
            if role == "tool":
                role = "user"
                content = f"[Tool Result]\n{content}"
            if json_mode and role == "system":
                content += "\n\nYou must respond with valid JSON only. No other text."
            ollama_messages.append({"role": role, "content": content})

        payload: dict[str, Any] = {
            "model": self._model,
            "messages": ollama_messages,
            "stream": False,
            "options": {"temperature": self._temperature, "num_predict": self._max_tokens},
        }
        if json_mode:
            payload["format"] = "json"

        url = f"{self._base_url}/api/chat"
        try:
            with httpx.Client(timeout=600) as client:
                response = client.post(url, json=payload)
                response.raise_for_status()
                data = response.json()
        except httpx.ConnectError as e:
            raise ConnectionError(
                f"Cannot connect to Ollama at {self._base_url}. Start with: ollama serve"
            ) from e

        text = data.get("message", {}).get("content", "")
        input_tokens = data.get("prompt_eval_count", 0)
        output_tokens = data.get("eval_count", 0)
        usage = TokenUsage(
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            total_tokens=input_tokens + output_tokens,
        )
        return text, usage


# ---------------------------------------------------------------------------
# Factory with auto-detection cascade
# ---------------------------------------------------------------------------


def create_llm_client(provider: str = "") -> LLMClient:
    """Create an LLM client from the unified ``.env`` config.

    The clean path is three intent-named variables - the same ones
    system reads, so one ``.env`` drives the notebooks and the system:

        LLM_PROVIDER   one of: deepseek, openrouter, openai, anthropic, google,
                       ollama, mock. The base URL for each is PREDEFINED in
                       ``provider_presets.PROVIDER_PRESETS`` - never a user setting.
        LLM_API_KEY    your key for that provider.
        LLM_MODEL      optional model override; omit to use the provider default.

    Precedence: the ``provider`` argument wins, else ``LLM_PROVIDER`` from the
    environment. With neither set, fall back to the legacy auto-detect cascade
    (``ANTHROPIC_API_KEY`` → ``OPENAI_API_KEY`` → … → Ollama) so older
    ``.env`` files keep working.
    """
    provider = (provider or os.environ.get("LLM_PROVIDER", "")).strip().lower()

    if provider in PROVIDER_PRESETS:
        return _client_from_preset(provider)

    if provider:
        raise ValueError(
            f"Unknown LLM provider {provider!r}. Supported: {', '.join(SUPPORTED_PROVIDERS)}"
        )

    return _legacy_autodetect()


def _client_from_preset(provider: str) -> LLMClient:
    """Build a client for a named provider using its predefined preset."""
    preset = PROVIDER_PRESETS[provider]
    client = preset["client"]
    model = os.environ.get("LLM_MODEL") or preset["default_model"]

    if client == "mock":
        return MockLLMClient()

    if client == "ollama":
        return OllamaChatClient(model=model, base_url=preset["base_url"])

    key = os.environ.get("LLM_API_KEY", "")
    if not key:
        raise RuntimeError(
            f"LLM_PROVIDER={provider} requires LLM_API_KEY. Select mock explicitly for a synthetic teaching fixture."
        )

    if client == "anthropic":
        return AnthropicChatClient(api_key=key, model=model)
    if client == "google":
        return GoogleGeminiClient(api_key=key, model=model)
    # OpenAI-compatible: openai, deepseek, openrouter (base_url predefined).
    return OpenAIChatClient(api_key=key, model=model, base_url=preset["base_url"])


def _legacy_autodetect() -> LLMClient:
    """Back-compat cascade for ``.env`` files that predate the LLM_* scheme.

    Honors the older per-provider vars (``ANTHROPIC_API_KEY``, ``OPENAI_API_KEY``
    with ``OPENAI_BASE_URL``/``OPENAI_MODEL``, ``GOOGLE_API_KEY``,
    ``OPENROUTER_API_KEY``) and a locally-running Ollama. No available live provider raises an error.
    """
    if os.environ.get("ANTHROPIC_API_KEY"):
        return AnthropicChatClient(
            api_key=os.environ["ANTHROPIC_API_KEY"],
            model=os.environ.get("ANTHROPIC_MODEL", "claude-sonnet-4-20250514"),
        )

    if os.environ.get("OPENAI_API_KEY"):
        return OpenAIChatClient(
            api_key=os.environ["OPENAI_API_KEY"],
            model=os.environ.get("OPENAI_MODEL", "gpt-4.1-mini"),
            base_url=os.environ.get("OPENAI_BASE_URL") or None,
        )

    if os.environ.get("GOOGLE_API_KEY"):
        return GoogleGeminiClient(
            api_key=os.environ["GOOGLE_API_KEY"],
            model=os.environ.get("GOOGLE_MODEL", "gemini-2.5-flash"),
        )

    if os.environ.get("OPENROUTER_API_KEY"):
        return OpenAIChatClient(
            api_key=os.environ["OPENROUTER_API_KEY"],
            model=os.environ.get("OPENROUTER_MODEL", "anthropic/claude-sonnet-4"),
            base_url="https://openrouter.ai/api/v1",
        )

    try:
        import httpx

        r = httpx.get("http://localhost:11434/api/tags", timeout=2)
        if r.status_code == 200:
            return OllamaChatClient(model=os.environ.get("OLLAMA_MODEL", "qwen2.5:32b"))
    except Exception:
        pass

    raise RuntimeError(
        "No LLM provider is available. Configure a provider or select mock explicitly for a synthetic teaching fixture."
    )
