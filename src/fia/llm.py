"""LLM layer. Claude via the Anthropic API for grounded answer writing; a deterministic
extractive fallback keeps tests/offline demos runnable (and never invents facts)."""
from __future__ import annotations

import logging
import os
import re

from . import config
from .embedders import STOP, tokenize

log = logging.getLogger("fia.llm")

SYSTEM = """You are an insurance decision-support assistant for an Indian family.
Answer ONLY from the CONTEXT blocks. Rules:
1. Every sentence that states a fact (amount, period, limit, exclusion, condition) must end with a citation tag
   copied exactly from the context, e.g. [P1§2.2]. Use only tags that appear in the context.
2. If the context does not contain the answer, say you could not find it in the policy documents. Never guess.
3. Mention waiting periods, sub-limits, co-payments and exclusions when they affect the answer.
4. Never tell the family to file or not file a claim. State what the policy says and what is uncertain;
   a human reviewer makes the decision.
5. Be concise: at most 6 short sentences or bullet points."""


def split_lines(text: str) -> list[str]:
    parts = []
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        parts.extend(s.strip() for s in re.split(r"(?<=[.!?])\s+(?=[A-Z₹•\-])", line) if s.strip())
    return parts


class ExtractiveLLM:
    """No-model fallback: quote the best-matching sentences of the top chunks, with citations."""
    name = "extractive-fallback"

    def answer(self, question: str, context_blocks: list[tuple[str, str, str]]) -> str:
        q = set(tokenize(question)) - STOP
        out, seen = [], set()
        for cid, title, text in context_blocks[:2]:
            sents = split_lines(text)
            ranked = sorted(sents, key=lambda s: -len(q & set(tokenize(s))))
            best = [s for s in ranked[:2] if q & set(tokenize(s))] or sents[:1]
            for s in sorted(best, key=sents.index):
                if s not in seen:
                    seen.add(s)
                    out.append(f"- {s.rstrip('.')}. [{cid}]")
        return "\n".join(out)


class AnthropicLLM:
    def __init__(self, model: str):
        import anthropic
        self.client = anthropic.Anthropic()
        self.model = model
        self.name = f"anthropic:{model}"

    def answer(self, question: str, context_blocks: list[tuple[str, str, str]]) -> str:
        ctx = "\n\n".join(f"[{cid}] {title}\n{text}" for cid, title, text in context_blocks)
        msg = self.client.messages.create(
            model=self.model, max_tokens=700, temperature=0, system=SYSTEM,
            messages=[{"role": "user", "content": f"CONTEXT:\n{ctx}\n\nQUESTION: {question}"}])
        return "".join(b.text for b in msg.content if getattr(b, "type", "") == "text").strip()


class OpenAICompatLLM:
    """Any OpenAI-compatible /chat/completions endpoint: Ollama (local, free), Groq, Gemini, OpenRouter, ..."""

    def __init__(self, base_url: str, api_key: str, model: str):
        self.base_url, self.api_key, self.model = base_url.rstrip("/"), api_key or "none", model
        self.name = f"openai-compat:{model}@{self.base_url}"

    def answer(self, question: str, context_blocks: list[tuple[str, str, str]]) -> str:
        import requests
        ctx = "\n\n".join(f"[{cid}] {title}\n{text}" for cid, title, text in context_blocks)
        r = requests.post(f"{self.base_url}/chat/completions", timeout=120,
                          headers={"Authorization": f"Bearer {self.api_key}"},
                          json={"model": self.model, "temperature": 0, "max_tokens": 700,
                                "messages": [{"role": "system", "content": SYSTEM},
                                             {"role": "user", "content": f"CONTEXT:\n{ctx}\n\nQUESTION: {question}"}]})
        r.raise_for_status()
        return r.json()["choices"][0]["message"]["content"].strip()


def get_llm():
    if config.LLM_BACKEND == "openai" or (config.LLM_BACKEND == "auto" and config.LLM_BASE_URL):
        if not config.LLM_BASE_URL:
            raise RuntimeError("FIA_LLM=openai needs FIA_LLM_BASE_URL")
        return OpenAICompatLLM(config.LLM_BASE_URL, config.LLM_API_KEY, config.LLM_MODEL)
    if config.LLM_BACKEND in ("auto", "anthropic") and os.getenv("ANTHROPIC_API_KEY"):
        try:
            return AnthropicLLM(config.LLM_MODEL)
        except Exception as e:  # noqa: BLE001
            if config.LLM_BACKEND == "anthropic":
                raise
            log.warning("Anthropic unavailable (%s) - using extractive fallback", e)
    elif config.LLM_BACKEND == "anthropic":
        raise RuntimeError("FIA_LLM=anthropic but ANTHROPIC_API_KEY is not set")
    return ExtractiveLLM()
