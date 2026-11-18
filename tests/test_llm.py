import json
from types import SimpleNamespace

import httpx
import pytest

from ragchat.llm import (
    OpenRouterClient,
    OllamaClient,
    make_client,
    make_provider_client,
)


def _client(handler, api_key="secret-key", host="https://ollama.example/", think=False):
    client = OllamaClient(api_key, host, think=think)
    client._http = httpx.Client(transport=httpx.MockTransport(handler))
    return client


def _openrouter_client(handler, api_key="secret-key"):
    client = OpenRouterClient(api_key)
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
        "think": False,
        "options": {"temperature": 0, "num_predict": 700},
    }
    assert result.content[0].type == "text"
    assert result.content[0].text == "hello"


def test_create_includes_think_true_when_configured():
    captured = {}

    def handler(request):
        captured["body"] = json.loads(request.content)
        return httpx.Response(200, json={"message": {"content": "hello"}})

    client = _client(handler, think=True)
    client.messages.create(
        model="m",
        max_tokens=10,
        temperature=0,
        system="s",
        messages=[{"role": "user", "content": "q"}],
    )

    assert captured["body"]["think"] is True


def test_empty_content_raises_with_the_model_and_done_reason():
    def handler(request):
        return httpx.Response(
            200,
            json={
                "message": {"content": "", "thinking": "reasoning used the budget"},
                "done_reason": "length",
            },
        )

    client = _client(handler, api_key="super-secret-key")

    with pytest.raises(RuntimeError) as exc_info:
        client.messages.create(
            model="glm-5.3-flash",
            max_tokens=700,
            temperature=0,
            system="s",
            messages=[{"role": "user", "content": "q"}],
        )

    message = str(exc_info.value)
    assert "empty completion" in message
    assert "glm-5.3-flash" in message
    assert "done_reason=length" in message
    assert "super-secret-key" not in message


def test_whitespace_only_content_raises():
    def handler(request):
        return httpx.Response(
            200,
            json={"message": {"content": "   \n"}, "done_reason": "stop"},
        )

    client = _client(handler)

    with pytest.raises(RuntimeError, match="done_reason=stop"):
        client.messages.create(
            model="m",
            max_tokens=10,
            temperature=0,
            system="s",
            messages=[{"role": "user", "content": "q"}],
        )


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


def test_make_client_threads_llm_think_into_the_ollama_client():
    settings = SimpleNamespace(
        llm_provider="ollama",
        ollama_api_key="k",
        ollama_host="https://ollama.com",
        anthropic_api_key="a",
        llm_think=True,
    )

    client = make_client(settings)

    assert client.think is True


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


def test_openrouter_create_posts_the_openai_style_request():
    captured = {}

    def handler(request):
        captured["url"] = str(request.url)
        captured["authorization"] = request.headers["authorization"]
        captured["title"] = request.headers["x-title"]
        captured["body"] = json.loads(request.content)
        return httpx.Response(
            200,
            json={
                "choices": [
                    {"message": {"content": "hello"}, "finish_reason": "stop"}
                ]
            },
        )

    client = _openrouter_client(handler)
    result = client.messages.create(
        model="inclusionai/ling-3.2-flash:free",
        max_tokens=700,
        temperature=0,
        system="system rules",
        messages=[{"role": "user", "content": "the question"}],
    )

    assert captured["url"] == "https://openrouter.ai/api/v1/chat/completions"
    assert captured["authorization"] == "Bearer secret-key"
    assert captured["title"] == "rag-docs-chatbot"
    assert captured["body"] == {
        "model": "inclusionai/ling-3.2-flash:free",
        "messages": [
            {"role": "system", "content": "system rules"},
            {"role": "user", "content": "the question"},
        ],
        "temperature": 0,
        "max_tokens": 700,
        "reasoning": {"enabled": False},
    }
    assert result.content[0].type == "text"
    assert result.content[0].text == "hello"


def test_openrouter_flattens_anthropic_style_content_blocks():
    captured = {}

    def handler(request):
        captured["body"] = json.loads(request.content)
        return httpx.Response(
            200, json={"choices": [{"message": {"content": "ok"}}]}
        )

    client = _openrouter_client(handler)
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


def test_openrouter_non_200_raises_without_leaking_the_key():
    def handler(request):
        return httpx.Response(429, json={"error": "rate limited"})

    client = _openrouter_client(handler, api_key="super-secret-key")

    with pytest.raises(RuntimeError) as exc_info:
        client.messages.create(
            model="m",
            max_tokens=10,
            temperature=0,
            system="s",
            messages=[{"role": "user", "content": "q"}],
        )

    message = str(exc_info.value)
    assert "429" in message
    assert "rate limited" in message
    assert "super-secret-key" not in message


def test_openrouter_empty_content_raises_with_the_model_and_finish_reason():
    def handler(request):
        return httpx.Response(
            200,
            json={
                "choices": [
                    {"message": {"content": "   "}, "finish_reason": "length"}
                ]
            },
        )

    client = _openrouter_client(handler, api_key="super-secret-key")

    with pytest.raises(RuntimeError) as exc_info:
        client.messages.create(
            model="some-free-model:free",
            max_tokens=700,
            temperature=0,
            system="s",
            messages=[{"role": "user", "content": "q"}],
        )

    message = str(exc_info.value)
    assert "empty completion" in message
    assert "some-free-model:free" in message
    assert "finish_reason=length" in message
    assert "super-secret-key" not in message


def test_make_provider_client_returns_openrouter():
    settings = SimpleNamespace(openrouter_api_key="or-key")

    client = make_provider_client(settings, "openrouter")

    assert isinstance(client, OpenRouterClient)
    assert client.api_key == "or-key"


def test_make_provider_client_rejects_an_unknown_provider():
    settings = SimpleNamespace()

    with pytest.raises(RuntimeError, match="unsupported LLM provider"):
        make_provider_client(settings, "bogus")
