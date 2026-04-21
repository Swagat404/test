"""Async judge clients for OpenAI (GPT-5.4) and Anthropic (Claude Sonnet 4.6).

Both judges support the same multi-turn protocol:
  baseline_score(...) -> (score, raw_response, message_history)
  followup_score(...) -> (score, raw_response, message_history)

The message_history returned by baseline_score is reused as the prefix for
followup_score so that the challenge appears in the same conversation.

Day-1 task: confirm exact snapshot strings from each provider's docs and
update OPENAI_MODEL / ANTHROPIC_MODEL constants below.
"""
from __future__ import annotations

import asyncio
import os
from dataclasses import dataclass, field
from typing import Optional

from anthropic import AsyncAnthropic
from openai import AsyncOpenAI

from .parse import parse_score

# ---------------------------------------------------------------------------
# Model snapshot pins — TODO(Day 1): replace with exact dated snapshot IDs
# from provider dashboards before locking the experiment.
# ---------------------------------------------------------------------------
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-5.4")
ANTHROPIC_MODEL = os.getenv("ANTHROPIC_MODEL", "claude-sonnet-4-6")

DEFAULT_TEMPERATURE = 0.0
DEFAULT_MAX_TOKENS = 256
DEFAULT_SEED = 42


@dataclass
class JudgeResult:
    score: Optional[int]
    raw: str
    history: list[dict] = field(default_factory=list)
    error: Optional[str] = None


class BaseJudge:
    name: str = "base"

    async def baseline_score(self, system: str, user: str) -> JudgeResult:
        raise NotImplementedError

    async def followup_score(self, history: list[dict], followup_user: str) -> JudgeResult:
        raise NotImplementedError


# ---------------------------------------------------------------------------
# OpenAI (GPT-5.4)
# ---------------------------------------------------------------------------

class OpenAIJudge(BaseJudge):
    name = "openai"

    def __init__(self, model: str = OPENAI_MODEL, max_concurrent: int = 8):
        self.model = model
        self.client = AsyncOpenAI()
        self._sem = asyncio.Semaphore(max_concurrent)

    async def _chat(self, messages: list[dict]) -> tuple[str, Optional[str]]:
        async with self._sem:
            for attempt in range(4):
                try:
                    resp = await self.client.chat.completions.create(
                        model=self.model,
                        messages=messages,
                        temperature=DEFAULT_TEMPERATURE,
                        max_tokens=DEFAULT_MAX_TOKENS,
                        seed=DEFAULT_SEED,
                    )
                    return resp.choices[0].message.content or "", None
                except Exception as e:
                    if attempt == 3:
                        return "", f"{type(e).__name__}: {e}"
                    await asyncio.sleep(2 ** attempt)
        return "", "exhausted retries"

    async def baseline_score(self, system: str, user: str) -> JudgeResult:
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ]
        raw, err = await self._chat(messages)
        history = messages + [{"role": "assistant", "content": raw}]
        return JudgeResult(score=parse_score(raw), raw=raw, history=history, error=err)

    async def followup_score(self, history: list[dict], followup_user: str) -> JudgeResult:
        messages = list(history) + [{"role": "user", "content": followup_user}]
        raw, err = await self._chat(messages)
        new_history = messages + [{"role": "assistant", "content": raw}]
        return JudgeResult(score=parse_score(raw), raw=raw, history=new_history, error=err)


# ---------------------------------------------------------------------------
# Anthropic (Claude Sonnet 4.6)
# ---------------------------------------------------------------------------

class AnthropicJudge(BaseJudge):
    name = "anthropic"

    def __init__(self, model: str = ANTHROPIC_MODEL, max_concurrent: int = 4):
        self.model = model
        self.client = AsyncAnthropic()
        self._sem = asyncio.Semaphore(max_concurrent)

    async def _chat(self, system: str, messages: list[dict]) -> tuple[str, Optional[str]]:
        async with self._sem:
            for attempt in range(4):
                try:
                    resp = await self.client.messages.create(
                        model=self.model,
                        system=system,
                        messages=messages,
                        temperature=DEFAULT_TEMPERATURE,
                        max_tokens=DEFAULT_MAX_TOKENS,
                    )
                    parts = [b.text for b in resp.content if getattr(b, "type", None) == "text"]
                    return "".join(parts), None
                except Exception as e:
                    if attempt == 3:
                        return "", f"{type(e).__name__}: {e}"
                    await asyncio.sleep(2 ** attempt)
        return "", "exhausted retries"

    async def baseline_score(self, system: str, user: str) -> JudgeResult:
        msgs = [{"role": "user", "content": user}]
        raw, err = await self._chat(system, msgs)
        history = [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
            {"role": "assistant", "content": raw},
        ]
        return JudgeResult(score=parse_score(raw), raw=raw, history=history, error=err)

    async def followup_score(self, history: list[dict], followup_user: str) -> JudgeResult:
        system = next((m["content"] for m in history if m["role"] == "system"), "")
        anthropic_msgs = [m for m in history if m["role"] != "system"]
        anthropic_msgs.append({"role": "user", "content": followup_user})
        raw, err = await self._chat(system, anthropic_msgs)
        new_history = (
            [{"role": "system", "content": system}]
            + anthropic_msgs
            + [{"role": "assistant", "content": raw}]
        )
        return JudgeResult(score=parse_score(raw), raw=raw, history=new_history, error=err)


def get_judge(name: str) -> BaseJudge:
    if name == "openai":
        return OpenAIJudge()
    if name == "anthropic":
        return AnthropicJudge()
    raise ValueError(f"unknown judge: {name}")
