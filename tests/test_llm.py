import json
from types import SimpleNamespace

import httpx
import pytest

from ragchat.llm import OllamaClient, make_client


def _client(handler, api_key="secret-key", host="https://ollama.example/"):
    client = OllamaClient(api_key, host)
    client._http = httpx.Client(transport=httpx.MockTransport(handler))
    return client


def test_create_posts_the_expected_request():
    captured = {}

    def handler(request):
        captured["url"] = str(request.url)
        captured["authorization"] = request.headers["authorization"]
        captured["body"] = json.loads(request.content)
        return httpx.Response(200, json={"message": {"content": "hello"}})

    client = _client(handler)
    result = client.messages.create(
        model="gpt-oss:120b",
        max_tokens=700,
        temperature=0,
        system="system rules",
        messages=[{"role": "user", "content": "the question"}],
    )

    assert captured["url"] == "https://ollama.example/api/chat"
    assert captured["authorization"] == "Bearer secret-key"
    assert captured["body"] == {
        "model": "gpt-oss:120b",
        "messages": [
            {"role": "system", "content": "system rules"},
            {"role": "user", "content": "the question"},
        ],
        "stream": False,
        "options": {"temperature": 0, "num_predict": 700},
    }
    assert result.content[0].type == "text"
    assert result.content[0].text == "hello"


def test_create_flattens_anthropic_style_content_blocks():
    captured = {}

    def handler(request):
        captured["body"] = json.loads(request.content)
        return httpx.Response(200, json={"message": {"content": "ok"}})

    client = _client(handler)
    client.messages.create(
        model="m",
        max_tokens=10,
        temperature=0,
        system="s",
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "first "},
                    {"type": "text", "text": "second"},
                ],
            }
        ],
    )

    assert captured["body"]["messages"][1] == {
        "role": "user",
        "content": "first second",
    }


def test_non_2xx_raises_without_leaking_the_key():
    def handler(request):
        return httpx.Response(401, json={"error": "unauthorized"})

    client = _client(handler, api_key="super-secret-key")

    with pytest.raises(RuntimeError) as exc_info:
        client.messages.create(
            model="m",
            max_tokens=10,
            temperature=0,
            system="s",
            messages=[{"role": "user", "content": "q"}],
        )

    message = str(exc_info.value)
    assert "401" in message
    assert "unauthorized" in message
    assert "super-secret-key" not in message


def test_make_client_returns_ollama_for_the_default_provider():
    settings = SimpleNamespace(
        llm_provider="ollama",
        ollama_api_key="k",
        ollama_host="https://ollama.com",
        anthropic_api_key="a",
    )

    client = make_client(settings)

    assert isinstance(client, OllamaClient)
    assert client.host == "https://ollama.com"


def test_make_client_returns_anthropic_when_selected():
    anthropic = pytest.importorskip("anthropic")
    settings = SimpleNamespace(
        llm_provider="anthropic",
        ollama_api_key="k",
        ollama_host="https://ollama.com",
        anthropic_api_key="a",
    )

    client = make_client(settings)

    assert isinstance(client, anthropic.Anthropic)
