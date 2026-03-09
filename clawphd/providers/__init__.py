"""LLM provider abstraction module."""

from clawphd.providers.base import LLMProvider, LLMResponse
from clawphd.providers.litellm_provider import LiteLLMProvider

try:
    from clawphd.providers.openai_codex_provider import OpenAICodexProvider
except ImportError:
    OpenAICodexProvider = None  # type: ignore[assignment,misc]

try:
    from clawphd.providers.azure_openai_provider import AzureOpenAIProvider
except ImportError:
    AzureOpenAIProvider = None  # type: ignore[assignment,misc]

__all__ = ["LLMProvider", "LLMResponse", "LiteLLMProvider", "OpenAICodexProvider", "AzureOpenAIProvider"]
