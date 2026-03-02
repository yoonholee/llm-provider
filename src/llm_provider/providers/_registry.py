"""Provider registry -- config + client factory for OpenAI-compatible providers."""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Callable

_OPENAI_PREFIXES = ("gpt-", "o1", "o3", "o4", "chatgpt-")

_OPENROUTER_DEFAULT_PROVIDER = {"sort": "price"}


def _inject_openrouter_kwargs(kwargs: dict) -> dict:
    """Merge default provider config into extra_body if not already set."""
    kwargs = dict(kwargs)
    extra = kwargs.get("extra_body") or {}
    if "provider" not in extra:
        extra = {**extra, "provider": _OPENROUTER_DEFAULT_PROVIDER}
        kwargs["extra_body"] = extra
    return kwargs


@dataclass(frozen=True)
class OpenAICompatProvider:
    """Configuration for an OpenAI-compatible API provider."""

    name: str
    prefix: str
    base_url: str | None = None
    keys_env: str | None = None  # env var for comma-separated key list
    single_key_envs: tuple[str, ...] = ()
    http2: bool = False
    default_api_key: str | None = None  # e.g. "unused" for local
    error_msg: str = ""
    inject_kwargs: Callable[[dict], dict] | None = None
    base_url_env: str | None = None  # env var to override base_url

    def model_id(self, model: str) -> str:
        """Strip provider prefix from model name."""
        return model.removeprefix(self.prefix)

    def _get_api_key(self) -> str | list[str] | None:
        """Resolve API key(s) from environment. None = SDK default."""
        if self.keys_env:
            keys_str = os.environ.get(self.keys_env)
            if keys_str:
                keys = [k.strip() for k in keys_str.split(",") if k.strip()]
                if keys:
                    return keys
        for env in self.single_key_envs:
            val = os.environ.get(env)
            if val:
                return val
        if self.default_api_key is not None:
            return self.default_api_key
        if self.error_msg:
            raise ValueError(self.error_msg)
        return None  # SDK reads from env (e.g. OPENAI_API_KEY)

    def _get_base_url(self) -> str | None:
        if self.base_url_env:
            return os.environ.get(self.base_url_env, self.base_url)
        return self.base_url

    def _http_client_kwargs(self, on_headers: Any, is_async: bool) -> dict:
        """Build httpx client kwargs for HTTP/2 and optional header hooks."""
        if not self.http2:
            return {}
        import httpx

        event_hooks: dict = {}
        if on_headers:
            from llm_provider.providers._headers import parse_openai_headers

            if is_async:

                async def _hook(response: httpx.Response) -> None:
                    info = parse_openai_headers(response.headers)
                    if info:
                        on_headers(info["remaining"], info["limit"])
            else:

                def _hook(response: httpx.Response) -> None:  # type: ignore[misc]
                    info = parse_openai_headers(response.headers)
                    if info:
                        on_headers(info["remaining"], info["limit"])

            event_hooks["response"] = [_hook]

        if is_async:
            return {
                "http_client": httpx.AsyncClient(http2=True, event_hooks=event_hooks)
            }
        return {"http_client": httpx.Client(http2=True, event_hooks=event_hooks)}

    def create_async_client(self, max_retries: int = 2, on_headers: Any = None):
        from openai import AsyncOpenAI

        from llm_provider.providers._pool import ClientPool

        base_url = self._get_base_url()
        key = self._get_api_key()

        def _make(api_key: str | None = None) -> AsyncOpenAI:
            kw: dict[str, Any] = {
                "max_retries": max_retries,
                **self._http_client_kwargs(on_headers, is_async=True),
            }
            if base_url is not None:
                kw["base_url"] = base_url
            if api_key is not None:
                kw["api_key"] = api_key
            return AsyncOpenAI(**kw)

        if isinstance(key, list):
            return ClientPool([_make(k) for k in key])
        return _make(key)

    def create_sync_client(self, max_retries: int = 2, on_headers: Any = None):
        from openai import OpenAI

        from llm_provider.providers._pool import ClientPool

        base_url = self._get_base_url()
        key = self._get_api_key()

        def _make(api_key: str | None = None) -> OpenAI:
            kw: dict[str, Any] = {
                "max_retries": max_retries,
                **self._http_client_kwargs(on_headers, is_async=False),
            }
            if base_url is not None:
                kw["base_url"] = base_url
            if api_key is not None:
                kw["api_key"] = api_key
            return OpenAI(**kw)

        if isinstance(key, list):
            return ClientPool([_make(k) for k in key])
        return _make(key)


# --- Provider configs ---

PROVIDERS: dict[str, OpenAICompatProvider] = {
    "openai": OpenAICompatProvider(
        name="openai",
        prefix="openai/",
        keys_env="OPENAI_KEYS",
        http2=True,
    ),
    "together": OpenAICompatProvider(
        name="together",
        prefix="together_ai/",
        base_url="https://api.together.xyz/v1",
        keys_env="TOGETHER_KEYS",
        single_key_envs=("TOGETHER_API_KEY", "TOGETHERAI_API_KEY"),
        error_msg="TOGETHER_API_KEY or TOGETHER_KEYS required for Together models",
    ),
    "sambanova": OpenAICompatProvider(
        name="sambanova",
        prefix="sambanova/",
        base_url="https://api.sambanova.ai/v1",
        keys_env="SAMBANOVA_KEYS",
        single_key_envs=("SAMBANOVA_API_KEY",),
        error_msg="SAMBANOVA_API_KEY or SAMBANOVA_KEYS required for SambaNova models",
    ),
    "openrouter": OpenAICompatProvider(
        name="openrouter",
        prefix="openrouter/",
        base_url="https://openrouter.ai/api/v1",
        keys_env="OPENROUTER_KEYS",
        single_key_envs=("OPENROUTER_API_KEY",),
        error_msg="OPENROUTER_API_KEY or OPENROUTER_KEYS required for OpenRouter models",
        inject_kwargs=_inject_openrouter_kwargs,
    ),
    "local": OpenAICompatProvider(
        name="local",
        prefix="local/",
        base_url="http://localhost:30000/v1",
        default_api_key="unused",
        base_url_env="LOCAL_BASE_URL",
    ),
}


def detect(model: str) -> OpenAICompatProvider | None:
    """Detect which OpenAI-compatible provider handles this model.

    Returns None for Gemini and litellm models.
    """
    # OpenAI: special prefix-based detection (gpt-*, o1*, o3*, o4*, chatgpt-*)
    m = model.removeprefix("openai/")
    if any(m.startswith(p) for p in _OPENAI_PREFIXES):
        return PROVIDERS["openai"]
    # Others: simple prefix match
    for name, prov in PROVIDERS.items():
        if name != "openai" and model.startswith(prov.prefix):
            return prov
    return None
