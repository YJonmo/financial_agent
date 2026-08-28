"""Swap between a paid LLM provider (OpenAI) and a free one (Groq).

Groq's API is OpenAI-compatible (same request/response shape, including tool
calling), so we reuse llama-index's `OpenAI` LLM class pointed at Groq's base
URL instead of depending on the separate `llama-index-llms-groq` package.
That class's `metadata` property looks model names up in an OpenAI-only
registry though, so `GroqLLM` overrides it with Groq's own context windows
and forces `is_function_calling_model=True` (Groq's tool-calling models do
support it, they're just not in llama-index's OpenAI model list).

On networks that TLS-inspect outbound HTTPS (e.g. a corporate proxy like
Cloudflare Gateway), the OS trust store trusts the proxy's root CA but the
`certifi` bundle that `httpx`/`openai` use by default does not - which
surfaces as a "self-signed certificate in certificate chain" error for
non-allow-listed hosts like api.groq.com. `truststore.inject_into_ssl()`
below makes Python's ssl module (and therefore httpx/openai) verify against
the OS trust store instead, matching what a browser would accept.
"""

import os

import truststore

truststore.inject_into_ssl()

from llama_index.core.llms import LLMMetadata
from llama_index.core.llms.function_calling import FunctionCallingLLM
from llama_index.llms.openai import OpenAI

OPENAI_CONTEXT_WINDOWS = {"gpt-4.1-mini": 1_047_576}

GROQ_API_BASE = "https://api.groq.com/openai/v1"
# Meta's Llama-3.3-70b (also tried here) intermittently emits its native
# `<function=name>{...}</function>` text format instead of a proper
# structured tool call, which Groq rejects outright (tool_use_failed) -
# breaking the agent mid-conversation. OpenAI's own open-weight gpt-oss
# models are trained specifically for structured tool calling and were
# reliable across testing, so they're the default free model here.
DEFAULT_GROQ_MODEL = "openai/gpt-oss-120b"
DEFAULT_MODEL_BY_PROVIDER = {"openai": "gpt-4o-mini", "groq": DEFAULT_GROQ_MODEL}

# Context windows for Groq's tool-calling-capable models (see
# https://console.groq.com/docs/models for the current list/limits).
GROQ_CONTEXT_WINDOWS = {
    "openai/gpt-oss-120b": 128_000,
    "openai/gpt-oss-20b": 128_000,
    "llama-3.3-70b-versatile": 128_000,
    "llama-3.1-8b-instant": 128_000,
}


def resolve_model_name(provider: str, model: str | None) -> str:
    """Return the model name that would be used, without constructing an LLM."""
    if model:
        return model
    if provider not in DEFAULT_MODEL_BY_PROVIDER:
        raise ValueError(f"Unknown provider {provider!r}, expected 'openai' or 'groq'")
    return DEFAULT_MODEL_BY_PROVIDER[provider]


class GroqLLM(OpenAI):
    """llama-index's OpenAI() LLM, retargeted at Groq's OpenAI-compatible API."""

    @property
    def metadata(self) -> LLMMetadata:
        return LLMMetadata(
            context_window=GROQ_CONTEXT_WINDOWS.get(self.model, 32_000),
            num_output=self.max_tokens or -1,
            is_chat_model=True,
            is_function_calling_model=True,
            model_name=self.model,
        )


class ResilientOpenAI(OpenAI):
    """OpenAI() LLM that doesn't hard-fail on model names newer than the
    installed llama-index-llms-openai package's hardcoded registry (e.g.
    gpt-4.1-mini isn't in llama-index-llms-openai==0.3.2's model list)."""

    @property
    def metadata(self) -> LLMMetadata:
        try:
            return super().metadata
        except ValueError:
            return LLMMetadata(
                context_window=OPENAI_CONTEXT_WINDOWS.get(self.model, 128_000),
                num_output=self.max_tokens or -1,
                is_chat_model=True,
                is_function_calling_model=True,
                model_name=self.model,
            )


def build_llm(provider: str = "openai", model: str | None = None) -> FunctionCallingLLM:
    """Build the chat LLM for the given provider ('openai' or 'groq')."""
    model = resolve_model_name(provider, model)

    if provider == "openai":
        return ResilientOpenAI(
            model=model, temperature=0, api_base=None, api_version=None, max_tokens=None
        )

    if provider == "groq":
        api_key = os.environ.get("GROQ_API_KEY")
        if not api_key:
            raise RuntimeError(
                "GROQ_API_KEY is not set. Get a free key at "
                "https://console.groq.com/keys and add GROQ_API_KEY=... to your .env file."
            )
        return GroqLLM(
            model=model,
            api_base=GROQ_API_BASE,
            api_key=api_key,
            api_version=None,
            max_tokens=None,
            temperature=0,
        )

    raise ValueError(f"Unknown provider {provider!r}, expected 'openai' or 'groq'")
