"""LLM clients: Ollama Cloud by default, Anthropic Claude optional."""

from __future__ import annotations

import httpx


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
    def __init__(self, client: OllamaClient):
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

    def __init__(self, api_key: str, host: str, timeout: float = 120):
        self.api_key = api_key
        self.host = host.rstrip("/")
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
            "stream": False,
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
        return _Message(data["message"]["content"])


def make_client(settings) -> OllamaClient:
    if settings.llm_provider == "ollama":
        return OllamaClient(settings.ollama_api_key, settings.ollama_host)
    if settings.llm_provider == "anthropic":
        import anthropic

        return anthropic.Anthropic(api_key=settings.anthropic_api_key)
    raise RuntimeError(f"unsupported LLM_PROVIDER: {settings.llm_provider!r}")
