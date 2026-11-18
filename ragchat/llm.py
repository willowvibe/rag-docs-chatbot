"""LLM clients: Ollama Cloud by default, OpenRouter fallbacks, Anthropic optional."""

from __future__ import annotations

import httpx

OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"


def _flatten_content(content) -> str:
    """Turn Anthropic-style text blocks into the plain string Ollama expects."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for block in content:
            if isinstance(block, dict):
                if block.get("type", "text") == "text":
                    parts.append(str(block.get("text", "")))
            elif getattr(block, "type", "text") == "text":
                parts.append(str(getattr(block, "text", "")))
        return "".join(parts)
    return str(content)


class _TextBlock:
    type = "text"

    def __init__(self, text: str):
        self.text = text


class _Message:
    def __init__(self, text: str):
        self.content = [_TextBlock(text)]


class _Messages:
    def __init__(self, client):
        self._client = client

    def create(self, model: str, max_tokens: int, temperature: float,
               system: str, messages: list[dict]) -> _Message:
        return self._client._create(
            model=model,
            max_tokens=max_tokens,
            temperature=temperature,
            system=system,
            messages=messages,
        )


class OllamaClient:
    """Duck-types the subset of the Anthropic SDK this project uses."""

    def __init__(self, api_key: str, host: str, timeout: float = 120, think: bool = False):
        self.api_key = api_key
        self.host = host.rstrip("/")
        self.timeout = timeout
        self.think = think
        self._http = httpx.Client(timeout=timeout)
        self.messages = _Messages(self)

    def _create(self, model: str, max_tokens: int, temperature: float,
                system: str, messages: list[dict]) -> _Message:
        payload = {
            "model": model,
            "messages": [
                {"role": "system", "content": system},
                *[
                    {
                        "role": message["role"],
                        "content": _flatten_content(message.get("content", "")),
                    }
                    for message in messages
                ],
            ],
            "stream": False,
            "think": self.think,
            "options": {"temperature": temperature, "num_predict": max_tokens},
        }
        try:
            response = self._http.post(
                f"{self.host}/api/chat",
                headers={"Authorization": f"Bearer {self.api_key}"},
                json=payload,
            )
        except httpx.HTTPError as exc:
            raise RuntimeError(f"Ollama request failed: {exc}") from None
        if not response.is_success:
            excerpt = response.text[:200].strip()
            raise RuntimeError(
                f"Ollama request failed with status {response.status_code}: {excerpt}"
            )
        data = response.json()
        content = _flatten_content(data["message"]["content"])
        if not content.strip():
            done_reason = data.get("done_reason")
            raise RuntimeError(
                f"empty completion from {model} (done_reason={done_reason})"
            )
        return _Message(content)


class OpenRouterClient:
    """Duck-types the same Anthropic-style subset as OllamaClient."""

    def __init__(self, api_key: str, timeout: float = 120):
        self.api_key = api_key
        self.timeout = timeout
        self._http = httpx.Client(timeout=timeout)
        self.messages = _Messages(self)

    def _create(self, model: str, max_tokens: int, temperature: float,
                system: str, messages: list[dict]) -> _Message:
        payload = {
            "model": model,
            "messages": [
                {"role": "system", "content": system},
                *[
                    {
                        "role": message["role"],
                        "content": _flatten_content(message.get("content", "")),
                    }
                    for message in messages
                ],
            ],
            "temperature": temperature,
            "max_tokens": max_tokens,
            # Free reasoning models spend the whole token budget thinking and return
            # empty content, so reasoning is always disabled here.
            "reasoning": {"enabled": False},
        }
        try:
            response = self._http.post(
                OPENROUTER_URL,
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "X-Title": "rag-docs-chatbot",
                },
                json=payload,
            )
        except httpx.HTTPError as exc:
            raise RuntimeError(f"OpenRouter request failed: {exc}") from None
        if not response.is_success:
            excerpt = response.text[:200].strip()
            raise RuntimeError(
                f"OpenRouter request failed with status {response.status_code}: {excerpt}"
            )
        data = response.json()
        choices = data.get("choices") or []
        choice = choices[0] if choices else {}
        content = _flatten_content((choice.get("message") or {}).get("content") or "")
        if not content.strip():
            finish_reason = choice.get("finish_reason")
            raise RuntimeError(
                f"empty completion from {model} (finish_reason={finish_reason})"
            )
        return _Message(content)


def make_provider_client(settings, provider: str):
    if provider == "ollama":
        return OllamaClient(
            settings.ollama_api_key,
            settings.ollama_host,
            think=getattr(settings, "llm_think", False),
        )
    if provider == "openrouter":
        return OpenRouterClient(settings.openrouter_api_key)
    if provider == "anthropic":
        import anthropic

        return anthropic.Anthropic(api_key=settings.anthropic_api_key)
    raise RuntimeError(f"unsupported LLM provider: {provider!r}")


def make_client(settings):
    return make_provider_client(settings, settings.llm_provider)
