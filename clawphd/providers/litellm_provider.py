"""LiteLLM provider implementation for multi-provider support."""

import hashlib
import json
import os
import secrets
import string
from typing import Any

import json_repair
import litellm
from litellm import acompletion
from loguru import logger

from clawphd.providers.base import LLMProvider, LLMResponse, ToolCallRequest

_ALLOWED_MSG_KEYS = frozenset({"role", "content", "tool_calls", "tool_call_id", "name", "reasoning_content"})
_ANTHROPIC_EXTRA_KEYS = frozenset({"thinking_blocks"})
_ALNUM = string.ascii_letters + string.digits

def _short_tool_id() -> str:
    """Generate a 9-char alphanumeric ID compatible with all providers."""
    return "".join(secrets.choice(_ALNUM) for _ in range(9))


class LiteLLMProvider(LLMProvider):
    """LLM provider using LiteLLM for multi-provider support."""

    def __init__(
        self,
        api_key: str | None = None,
        api_base: str | None = None,
        default_model: str = "anthropic/claude-opus-4-5",
        extra_headers: dict[str, str] | None = None,
        provider_name: str | None = None,
    ):
        super().__init__(api_key, api_base)
        self.default_model = default_model
        self.extra_headers = extra_headers or {}
        self._provider_name = provider_name

        # Detect OpenRouter by api_key prefix or explicit api_base
        self.is_openrouter = (
            provider_name == "openrouter" or
            (api_key and api_key.startswith("sk-or-")) or
            (api_base and "openrouter" in (api_base or ""))
        )

        self.is_vllm = provider_name == "vllm" or (bool(api_base) and not self.is_openrouter and provider_name not in (
            "anthropic", "openai", "deepseek", "gemini", "zhipu", "dashscope",
            "groq", "moonshot", "minimax", "aihubmix", "siliconflow", "volcengine",
        ))

        if api_key:
            self._setup_env(api_key, api_base, default_model)

        if api_base:
            litellm.api_base = api_base

        litellm.suppress_debug_info = True
        litellm.drop_params = True

    def _setup_env(self, api_key: str, api_base: str | None, model: str) -> None:
        """Set environment variables based on detected provider."""
        if self.is_openrouter:
            os.environ["OPENROUTER_API_KEY"] = api_key
        elif self.is_vllm:
            os.environ["HOSTED_VLLM_API_KEY"] = api_key
        elif "deepseek" in model:
            os.environ.setdefault("DEEPSEEK_API_KEY", api_key)
        elif "anthropic" in model or "claude" in model:
            os.environ.setdefault("ANTHROPIC_API_KEY", api_key)
        elif "openai" in model or "gpt" in model:
            os.environ.setdefault("OPENAI_API_KEY", api_key)
        elif "gemini" in model.lower():
            os.environ.setdefault("GEMINI_API_KEY", api_key)
        elif any(k in model for k in ("zhipu", "glm", "zai")):
            os.environ.setdefault("ZAI_API_KEY", api_key)
        elif "dashscope" in model or "qwen" in model.lower():
            os.environ.setdefault("DASHSCOPE_API_KEY", api_key)
        elif "groq" in model:
            os.environ.setdefault("GROQ_API_KEY", api_key)
        elif "moonshot" in model or "kimi" in model:
            os.environ.setdefault("MOONSHOT_API_KEY", api_key)
            os.environ.setdefault("MOONSHOT_API_BASE", api_base or "https://api.moonshot.cn/v1")
        elif "minimax" in model:
            os.environ.setdefault("MINIMAX_API_KEY", api_key)

    def _resolve_model(self, model: str) -> str:
        """Resolve model name by applying provider/gateway prefixes."""
        if self.is_openrouter and not model.startswith("openrouter/"):
            return f"openrouter/{model}"

        if self.is_vllm:
            return f"hosted_vllm/{model}"

        lower = model.lower()

        if ("glm" in lower or "zhipu" in lower) and not any(
            model.startswith(p) for p in ("zhipu/", "zai/", "openrouter/")
        ):
            return f"zai/{model}"

        if ("qwen" in lower or "dashscope" in lower) and not any(
            model.startswith(p) for p in ("dashscope/", "openrouter/")
        ):
            return f"dashscope/{model}"

        if ("moonshot" in lower or "kimi" in lower) and not any(
            model.startswith(p) for p in ("moonshot/", "openrouter/")
        ):
            return f"moonshot/{model}"

        if "gemini" in lower and not model.startswith("gemini/"):
            return f"gemini/{model}"

        return model

    def _supports_cache_control(self, model: str) -> bool:
        """Return True when the provider supports cache_control on content blocks."""
        lower = model.lower()
        return "claude" in lower or "anthropic" in lower

    def _apply_cache_control(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None,
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]] | None]:
        """Return copies of messages and tools with cache_control injected."""
        new_messages = []
        for msg in messages:
            if msg.get("role") == "system":
                content = msg["content"]
                if isinstance(content, str):
                    new_content = [{"type": "text", "text": content, "cache_control": {"type": "ephemeral"}}]
                else:
                    new_content = list(content)
                    new_content[-1] = {**new_content[-1], "cache_control": {"type": "ephemeral"}}
                new_messages.append({**msg, "content": new_content})
            else:
                new_messages.append(msg)

        new_tools = tools
        if tools:
            new_tools = list(tools)
            new_tools[-1] = {**new_tools[-1], "cache_control": {"type": "ephemeral"}}

        return new_messages, new_tools

    @staticmethod
    def _extra_msg_keys(original_model: str, resolved_model: str) -> frozenset[str]:
        """Return provider-specific extra keys to preserve in request messages."""
        if "claude" in original_model.lower() or resolved_model.startswith("anthropic/"):
            return _ANTHROPIC_EXTRA_KEYS
        return frozenset()

    @staticmethod
    def _normalize_tool_call_id(tool_call_id: Any) -> Any:
        """Normalize tool_call_id to a provider-safe 9-char alphanumeric form."""
        if not isinstance(tool_call_id, str):
            return tool_call_id
        if len(tool_call_id) == 9 and tool_call_id.isalnum():
            return tool_call_id
        return hashlib.sha1(tool_call_id.encode()).hexdigest()[:9]

    @staticmethod
    def _ensure_dict_arguments(fn: dict[str, Any]) -> None:
        """Ensure function.arguments is a JSON string representing a dict.

        Anthropic requires tool_use.input to be a dict. When OpenRouter
        converts OpenAI-style messages, it parses function.arguments; if
        the parsed value is not a dict (e.g. a bare string, list, or null),
        Anthropic rejects the request with 'Input should be a valid dictionary'.
        """
        raw = fn.get("arguments")
        if isinstance(raw, dict):
            fn["arguments"] = json.dumps(raw, ensure_ascii=False)
            return
        if not isinstance(raw, str):
            fn["arguments"] = json.dumps({} if raw is None else {"value": raw}, ensure_ascii=False)
            return
        try:
            parsed = json_repair.loads(raw)
        except Exception:
            fn["arguments"] = json.dumps({"raw": raw}, ensure_ascii=False)
            return
        if not isinstance(parsed, dict):
            fn["arguments"] = json.dumps({} if parsed is None else {"value": parsed}, ensure_ascii=False)

    @staticmethod
    def _sanitize_messages(messages: list[dict[str, Any]], extra_keys: frozenset[str] = frozenset()) -> list[dict[str, Any]]:
        """Strip non-standard keys and ensure assistant messages have a content key."""
        allowed = _ALLOWED_MSG_KEYS | extra_keys
        sanitized = LLMProvider._sanitize_request_messages(messages, allowed)
        id_map: dict[str, str] = {}

        def map_id(value: Any) -> Any:
            if not isinstance(value, str):
                return value
            return id_map.setdefault(value, LiteLLMProvider._normalize_tool_call_id(value))

        for clean in sanitized:
            if isinstance(clean.get("tool_calls"), list):
                normalized_tool_calls = []
                for tc in clean["tool_calls"]:
                    if not isinstance(tc, dict):
                        normalized_tool_calls.append(tc)
                        continue
                    tc_clean = dict(tc)
                    tc_clean["id"] = map_id(tc_clean.get("id"))
                    fn = tc_clean.get("function")
                    if isinstance(fn, dict):
                        LiteLLMProvider._ensure_dict_arguments(fn)
                    normalized_tool_calls.append(tc_clean)
                clean["tool_calls"] = normalized_tool_calls

            if "tool_call_id" in clean and clean["tool_call_id"]:
                clean["tool_call_id"] = map_id(clean["tool_call_id"])
        return sanitized

    async def chat(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        model: str | None = None,
        max_tokens: int = 4096,
        temperature: float = 0.7,
        reasoning_effort: str | None = None,
    ) -> LLMResponse:
        """Send a chat completion request via LiteLLM."""
        original_model = model or self.default_model
        model = self._resolve_model(original_model)
        extra_msg_keys = self._extra_msg_keys(original_model, model)

        if self._supports_cache_control(original_model):
            messages, tools = self._apply_cache_control(messages, tools)

        max_tokens = max(1, max_tokens)

        # kimi-k2.5 only supports temperature=1.0
        if "kimi-k2.5" in model.lower():
            temperature = 1.0

        kwargs: dict[str, Any] = {
            "model": model,
            "messages": self._sanitize_messages(self._sanitize_empty_content(messages), extra_keys=extra_msg_keys),
            "max_tokens": max_tokens,
            "temperature": temperature,
        }

        if self.api_key:
            kwargs["api_key"] = self.api_key

        if self.api_base:
            kwargs["api_base"] = self.api_base

        if self.extra_headers:
            kwargs["extra_headers"] = self.extra_headers

        if reasoning_effort:
            kwargs["reasoning_effort"] = reasoning_effort
            kwargs["drop_params"] = True

        if tools:
            kwargs["tools"] = tools
            kwargs["tool_choice"] = "auto"

        try:
            response = await acompletion(**kwargs)
            return self._parse_response(response)
        except Exception as e:
            return LLMResponse(
                content=f"Error calling LLM: {str(e)}",
                finish_reason="error",
            )

    def _parse_response(self, response: Any) -> LLMResponse:
        """Parse LiteLLM response into our standard format."""
        choice = response.choices[0]
        message = choice.message
        content = message.content
        finish_reason = choice.finish_reason

        raw_tool_calls = []
        for ch in response.choices:
            msg = ch.message
            if hasattr(msg, "tool_calls") and msg.tool_calls:
                raw_tool_calls.extend(msg.tool_calls)
                if ch.finish_reason in ("tool_calls", "stop"):
                    finish_reason = ch.finish_reason
            if not content and msg.content:
                content = msg.content

        if len(response.choices) > 1:
            logger.debug("LiteLLM response has {} choices, merged {} tool_calls",
                         len(response.choices), len(raw_tool_calls))

        tool_calls = []
        for tc in raw_tool_calls:
            args = tc.function.arguments
            if isinstance(args, str):
                args = json_repair.loads(args)
            if not isinstance(args, dict):
                args = {} if args is None else {"value": args}

            tool_calls.append(ToolCallRequest(
                id=_short_tool_id(),
                name=tc.function.name,
                arguments=args,
            ))

        usage = {}
        if hasattr(response, "usage") and response.usage:
            usage = {
                "prompt_tokens": response.usage.prompt_tokens,
                "completion_tokens": response.usage.completion_tokens,
                "total_tokens": response.usage.total_tokens,
            }

        reasoning_content = getattr(message, "reasoning_content", None) or None
        thinking_blocks = getattr(message, "thinking_blocks", None) or None

        return LLMResponse(
            content=content,
            tool_calls=tool_calls,
            finish_reason=finish_reason or "stop",
            usage=usage,
            reasoning_content=reasoning_content,
            thinking_blocks=thinking_blocks,
        )

    def get_default_model(self) -> str:
        """Get the default model."""
        return self.default_model
