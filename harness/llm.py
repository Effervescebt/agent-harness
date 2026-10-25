"""Model clients behind one interface.

Every client takes a list of chat messages plus tool schemas and returns a `Completion` with the text,
any tool calls, token usage and wall-clock latency. Three implementations:

- `OpenAICompatibleClient`: OpenAI, DeepSeek, OpenRouter, Together, and self-hosted vLLM / Ollama servers
  (all speak the /v1/chat/completions protocol).
- `AnthropicClient`: the Anthropic Messages API.
- `MockClient`: scripted responses for tests and dry runs; needs no network or API key.

Only the standard library is used (urllib), so the harness runs anywhere.
"""
from __future__ import annotations

import json
import os
import random
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from typing import Any, Callable


@dataclass
class ToolCall:
    id: str
    name: str
    arguments: str  # raw JSON text exactly as the model produced it; validated later


@dataclass
class Usage:
    """input_tokens is the TOTAL prompt size; cached_input_tokens of it were cache reads and
    cache_write_tokens were cache writes (Anthropic bills writes at a premium)."""
    input_tokens: int = 0
    output_tokens: int = 0
    cached_input_tokens: int = 0
    cache_write_tokens: int = 0

    def __add__(self, other: "Usage") -> "Usage":
        return Usage(self.input_tokens + other.input_tokens, self.output_tokens + other.output_tokens,
                     self.cached_input_tokens + other.cached_input_tokens,
                     self.cache_write_tokens + other.cache_write_tokens)


@dataclass
class Completion:
    text: str
    tool_calls: list[ToolCall]
    usage: Usage
    latency_s: float
    model: str
    raw: dict = field(default_factory=dict, repr=False)


class LLMError(RuntimeError):
    pass


def _post_json(url: str, payload: dict, headers: dict, timeout: float) -> dict:
    req = urllib.request.Request(url, data=json.dumps(payload).encode(), method="POST",
                                 headers={"Content-Type": "application/json",
                                          "User-Agent": "agent-harness/0.1",  # proxies (e.g. Cloudflare) block Python-urllib
                                          **headers})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())


def _with_retries(fn: Callable[[], dict], max_retries: int) -> dict:
    """Retry rate limits, server errors and network failures with exponential backoff and jitter."""
    for attempt in range(max_retries + 1):
        try:
            return fn()
        except urllib.error.HTTPError as e:
            body = e.read().decode(errors="replace")[:500]
            if e.code not in (408, 409, 429, 500, 502, 503, 504, 529) or attempt == max_retries:
                raise LLMError(f"HTTP {e.code}: {body}") from e
        except (urllib.error.URLError, TimeoutError) as e:
            if attempt == max_retries:
                raise LLMError(f"network error: {e}") from e
        time.sleep(min(30.0, 2 ** attempt) * (0.5 + random.random()))
    raise AssertionError("unreachable")


class LLMClient:
    model: str

    def complete(self, messages: list[dict], tools: list[dict] | None = None,
                 max_tokens: int = 1024, temperature: float = 0.0) -> Completion:
        raise NotImplementedError


class OpenAICompatibleClient(LLMClient):
    """Chat Completions protocol. `base_url` examples:
    https://api.openai.com/v1, https://api.deepseek.com/v1, https://openrouter.ai/api/v1,
    http://localhost:8000/v1 (vLLM), http://localhost:11434/v1 (Ollama)."""

    def __init__(self, model: str, base_url: str = "https://api.openai.com/v1",
                 api_key_env: str = "OPENAI_API_KEY", timeout: float = 120.0, max_retries: int = 4,
                 native_tools: bool = True, send_temperature: bool = True, extra_body: dict | None = None):
        self.model, self.base_url = model, base_url.rstrip("/")
        self.api_key = os.environ.get(api_key_env, "")
        self.timeout, self.max_retries, self.native_tools = timeout, max_retries, native_tools
        self.send_temperature = send_temperature
        self.extra_body = extra_body or {}  # e.g. vLLM: {"chat_template_kwargs": {"enable_thinking": false}}
        self.salt_requests = False

    def complete(self, messages, tools=None, max_tokens=1024, temperature=0.0):
        # caching on these servers is automatic prefix caching; nothing to mark
        payload: dict[str, Any] = {"model": self.model, "messages": messages, "max_tokens": max_tokens}
        if self.send_temperature:
            payload["temperature"] = temperature
        payload.update(self.extra_body)
        if self.salt_requests:  # vLLM: a unique cache_salt per request disables prefix-cache reuse
            payload["cache_salt"] = os.urandom(8).hex()
        if tools and self.native_tools:
            payload["tools"] = [{"type": "function", "function": t} for t in tools]
        headers = {"Authorization": f"Bearer {self.api_key}"} if self.api_key else {}
        t0 = time.perf_counter()
        data = _with_retries(lambda: _post_json(f"{self.base_url}/chat/completions", payload, headers,
                                                self.timeout), self.max_retries)
        latency = time.perf_counter() - t0
        msg = data["choices"][0]["message"]
        calls = [ToolCall(c.get("id") or f"call_{i}", c["function"]["name"], c["function"].get("arguments") or "{}")
                 for i, c in enumerate(msg.get("tool_calls") or [])]
        u = data.get("usage") or {}
        # OpenAI: prompt_tokens_details.cached_tokens; DeepSeek: prompt_cache_hit_tokens
        cached = ((u.get("prompt_tokens_details") or {}).get("cached_tokens") or u.get("prompt_cache_hit_tokens") or 0)
        return Completion(msg.get("content") or "", calls,
                          Usage(u.get("prompt_tokens", 0), u.get("completion_tokens", 0), cached),
                          latency, self.model, data)


class AnthropicClient(LLMClient):
    """Anthropic Messages API. Converts the harness's OpenAI-style messages on the way in and out."""

    def __init__(self, model: str, api_key_env: str = "ANTHROPIC_API_KEY", timeout: float = 120.0,
                 max_retries: int = 4, base_url: str = "https://api.anthropic.com/v1", cache: bool = False,
                 thinking: dict | None = None, send_temperature: bool = False):
        """cache: mark the static prefix (tools + system) with an explicit breakpoint and let top-level
        automatic caching follow the growing conversation tail. thinking: passed through as-is (e.g.
        {"type": "between_tools"} on Claude Sonnet 5.5 or {"type": "disabled"} on Claude Haiku 5.5 to turn
        it off). Current models reject non-default temperature, so it is not sent unless asked."""
        self.model, self.base_url = model, base_url.rstrip("/")
        self.api_key = os.environ.get(api_key_env, "")
        self.timeout, self.max_retries = timeout, max_retries
        self.cache, self.thinking, self.send_temperature = cache, thinking, send_temperature

    @staticmethod
    def _convert(messages: list[dict]) -> tuple[str, list[dict]]:
        system, out = "", []
        for m in messages:
            role = m["role"]
            if role == "system":
                system += m["content"] + "\n"
            elif role == "tool":
                block = {"type": "tool_result", "tool_use_id": m["tool_call_id"], "content": m["content"]}
                if out and out[-1]["role"] == "user" and isinstance(out[-1]["content"], list):
                    out[-1]["content"].append(block)
                else:
                    out.append({"role": "user", "content": [block]})
            elif role == "assistant" and m.get("tool_calls"):
                blocks = [{"type": "text", "text": m["content"]}] if m.get("content") else []
                for c in m["tool_calls"]:
                    try:
                        args = json.loads(c["function"]["arguments"] or "{}")
                    except json.JSONDecodeError:
                        args = {}
                    blocks.append({"type": "tool_use", "id": c["id"], "name": c["function"]["name"], "input": args})
                out.append({"role": "assistant", "content": blocks})
            else:
                out.append({"role": role, "content": m["content"]})
        return system.strip(), out

    def complete(self, messages, tools=None, max_tokens=1024, temperature=0.0):
        system, msgs = self._convert(messages)
        payload: dict[str, Any] = {"model": self.model, "messages": msgs, "max_tokens": max_tokens}
        if self.send_temperature:
            payload["temperature"] = temperature
        if self.thinking:
            payload["thinking"] = self.thinking
        if system:
            block = {"type": "text", "text": system}
            if self.cache:
                block["cache_control"] = {"type": "ephemeral"}
            payload["system"] = [block]
        if self.cache:
            payload["cache_control"] = {"type": "ephemeral"}  # automatic breakpoint on the conversation tail
        if tools:
            payload["tools"] = [{"name": t["name"], "description": t.get("description", ""),
                                 "input_schema": t["parameters"]} for t in tools]
        headers = {"x-api-key": self.api_key, "anthropic-version": "2023-06-01"}
        t0 = time.perf_counter()
        data = _with_retries(lambda: _post_json(f"{self.base_url}/messages", payload, headers, self.timeout),
                             self.max_retries)
        latency = time.perf_counter() - t0
        text = "".join(b.get("text", "") for b in data.get("content", []) if b.get("type") == "text")
        calls = [ToolCall(b["id"], b["name"], json.dumps(b.get("input", {})))
                 for b in data.get("content", []) if b.get("type") == "tool_use"]
        u = data.get("usage") or {}
        read, write = u.get("cache_read_input_tokens", 0) or 0, u.get("cache_creation_input_tokens", 0) or 0
        # Anthropic's input_tokens is only the uncached remainder; total prompt = input + writes + reads
        total_in = (u.get("input_tokens", 0) or 0) + read + write
        return Completion(text, calls, Usage(total_in, u.get("output_tokens", 0), read, write),
                          latency, self.model, data)


class MockClient(LLMClient):
    """Replays scripted completions. Each script item is either a Completion, a string (plain text reply),
    or a callable(messages, tools) -> Completion. Used by the unit tests and `--dry-run`."""

    def __init__(self, script: list | Callable, model: str = "mock", latency_s: float = 0.0):
        self.script, self.model, self.latency_s, self.calls = script, model, latency_s, 0

    def complete(self, messages, tools=None, max_tokens=1024, temperature=0.0):
        if callable(self.script):
            item = self.script(messages, tools)
        else:
            if self.calls >= len(self.script):
                raise LLMError("mock script exhausted")
            item = self.script[self.calls]
        self.calls += 1
        if callable(item) and not isinstance(item, Completion):
            item = item(messages, tools)
        if isinstance(item, str):
            item = Completion(item, [], Usage(), 0.0, self.model)
        if self.latency_s:
            time.sleep(self.latency_s)
        item.usage = item.usage if (item.usage.input_tokens or item.usage.output_tokens) else Usage(
            sum(len(str(m.get("content", ""))) for m in messages) // 4, len(item.text) // 4 + 8)
        item.latency_s = item.latency_s or self.latency_s
        return item


def make_client(spec: dict) -> LLMClient:
    """Build a client from a config table, e.g. {provider = "openai", model = "gpt-4o-mini"}."""
    provider = spec.get("provider", "openai")
    if provider == "anthropic":
        return AnthropicClient(spec["model"], api_key_env=spec.get("api_key_env", "ANTHROPIC_API_KEY"),
                               cache=spec.get("cache", False), thinking=spec.get("thinking"),
                               send_temperature=spec.get("send_temperature", False))
    if provider == "mock":
        from .mock_policy import scripted_policy
        return MockClient(scripted_policy, model=spec.get("model", "mock"))
    if "base_url_env" in spec:  # e.g. a rented GPU's address, kept out of the repo
        spec = {**spec, "base_url": os.environ[spec["base_url_env"]]}
    client = OpenAICompatibleClient(spec["model"], base_url=spec.get("base_url", "https://api.openai.com/v1"),
                                  api_key_env=spec.get("api_key_env", "OPENAI_API_KEY"),
                                  native_tools=spec.get("native_tools", True),
                                  send_temperature=spec.get("send_temperature", True),
                                  extra_body=spec.get("extra_body"))
    client.salt_requests = spec.get("cache") is False and spec.get("salt_when_uncached", False)
    return client
