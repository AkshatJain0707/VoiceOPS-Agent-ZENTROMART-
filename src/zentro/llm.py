"""Thin provider abstraction with forced tool use. Neutral message format in, provider format out."""
from __future__ import annotations
import json
import os
import time
from dataclasses import dataclass, field


@dataclass
class LLMResponse:
    text: str = ""
    tool_calls: list[dict] = field(default_factory=list)  # {id, name, input}


class LLM:
    def __init__(self, provider: str, model: str, fast_model: str | None = None, base_url: str = ""):
        self.provider, self.model, self.fast_model = provider, model, fast_model or model
        self.base_url = base_url
        self._client = None

    def _ensure(self):
        if self._client:
            return
        if self.provider == "anthropic":
            import anthropic
            self._client = anthropic.Anthropic()
        elif self.provider == "openrouter":
            import openai  # OpenRouter speaks the OpenAI API
            self._client = openai.OpenAI(
                base_url=self.base_url or "https://openrouter.ai/api/v1",
                api_key=os.environ.get("OPENROUTER_API_KEY", ""),
                default_headers={"X-Title": "ZentroMart VoiceOps agent"})
        else:
            import openai
            kw = {"base_url": self.base_url} if self.base_url else {}
            self._client = openai.OpenAI(**kw)

    def chat(self, system: str, messages: list[dict], tools: list[dict] | None = None, force_tool: bool = False,
             max_tokens: int = 1200, fast: bool = False) -> LLMResponse:
        self._ensure()
        model = self.fast_model if fast else self.model
        last: Exception | None = None
        for attempt in range(4):
            try:
                if self.provider == "anthropic":
                    return self._anthropic(model, system, messages, tools, force_tool, max_tokens)
                return self._openai(model, system, messages, tools, force_tool, max_tokens)
            except Exception as e:  # rate limits, transient 5xx
                last = e
                time.sleep(1.5 * (2 ** attempt))
        raise RuntimeError(f"LLM call failed: {last}")

    # ---- Anthropic ----
    def _anthropic(self, model, system, messages, tools, force_tool, max_tokens) -> LLMResponse:
        msgs: list[dict] = []
        for m in messages:
            if m["role"] == "user":
                msgs.append({"role": "user", "content": m["content"]})
            elif m["role"] == "assistant":
                blocks = []
                if m.get("content"):
                    blocks.append({"type": "text", "text": m["content"]})
                for tc in m.get("tool_calls", []):
                    blocks.append({"type": "tool_use", "id": tc["id"], "name": tc["name"], "input": tc["input"]})
                msgs.append({"role": "assistant", "content": blocks})
            elif m["role"] == "tool":
                blk = {"type": "tool_result", "tool_use_id": m["tool_call_id"], "content": m["content"]}
                if msgs and msgs[-1]["role"] == "user" and isinstance(msgs[-1]["content"], list):
                    msgs[-1]["content"].append(blk)
                else:
                    msgs.append({"role": "user", "content": [blk]})
        kw = dict(model=model, system=system, messages=msgs, max_tokens=max_tokens, temperature=0)
        if tools:
            kw["tools"] = [{"name": t["name"], "description": t["description"], "input_schema": t["parameters"]} for t in tools]
            kw["tool_choice"] = {"type": "any"} if force_tool else {"type": "auto"}
        r = self._client.messages.create(**kw)
        out = LLMResponse()
        for b in r.content:
            if b.type == "text":
                out.text += b.text
            elif b.type == "tool_use":
                out.tool_calls.append({"id": b.id, "name": b.name, "input": dict(b.input)})
        return out

    # ---- OpenAI-compatible (OpenAI, Gemini, Groq, Ollama, OpenRouter...) ----
    def _openai(self, model, system, messages, tools, force_tool, max_tokens) -> LLMResponse:
        msgs: list[dict] = [{"role": "system", "content": system}]
        for m in messages:
            if m["role"] == "assistant" and m.get("tool_calls"):
                msgs.append({"role": "assistant", "content": m.get("content") or None, "tool_calls": [
                    {"id": tc["id"], "type": "function",
                     "function": {"name": tc["name"], "arguments": json.dumps(tc["input"])}} for tc in m["tool_calls"]]})
            elif m["role"] == "tool":
                msgs.append({"role": "tool", "tool_call_id": m["tool_call_id"], "content": m["content"]})
            else:
                msgs.append({"role": m["role"], "content": m["content"]})
        kw = dict(model=model, messages=msgs, max_tokens=max_tokens, temperature=0)
        if tools:
            kw["tools"] = [{"type": "function", "function": {"name": t["name"], "description": t["description"],
                                                             "parameters": t["parameters"]}} for t in tools]
            kw["tool_choice"] = "required" if force_tool else "auto"
        r = self._client.chat.completions.create(**kw)
        msg = r.choices[0].message
        out = LLMResponse(text=msg.content or "")
        for tc in msg.tool_calls or []:
            try:
                args = json.loads(tc.function.arguments or "{}")
            except json.JSONDecodeError:
                args = {}
            out.tool_calls.append({"id": tc.id, "name": tc.function.name, "input": args})
        return out


def make_llm(cfg) -> LLM:
    return LLM(cfg.llm_provider, cfg.llm_model, cfg.llm_fast_model, cfg.openai_base_url)
