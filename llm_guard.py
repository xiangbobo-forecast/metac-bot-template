"""
Rate-limit guard for the free Gemini tier.

- A sliding-window input-token limiter for Gemma (free tier: 16k input tokens/min).
- Retries on 429 / RESOURCE_EXHAUSTED / 503 / timeouts with the server's retry delay.
- Falls back to cheaper Gemini models when Gemma keeps failing, so every question
  still gets a forecast.
"""

from __future__ import annotations

import asyncio
import logging
import re
import time

from forecasting_tools import GeneralLlm

logger = logging.getLogger(__name__)


class TokenWindowLimiter:
    def __init__(self, tokens_per_minute: int):
        self.tpm = tokens_per_minute
        self.events: list[tuple[float, int]] = []
        self._lock: asyncio.Lock | None = None
        self._loop = None

    def _get_lock(self) -> asyncio.Lock:
        loop = asyncio.get_running_loop()
        if self._lock is None or self._loop is not loop:
            self._lock = asyncio.Lock()
            self._loop = loop
        return self._lock

    async def acquire(self, tokens: int) -> None:
        tokens = min(tokens, self.tpm)
        async with self._get_lock():
            while True:
                now = time.time()
                self.events = [(t, n) for t, n in self.events if now - t < 61]
                used = sum(n for _, n in self.events)
                if used + tokens <= self.tpm:
                    self.events.append((now, tokens))
                    return
                wait = 61 - (now - self.events[0][0]) + 0.5
                logger.info(f"Gemma token limiter: {used} tokens used in last minute, waiting {wait:.0f}s")
                await asyncio.sleep(max(wait, 1))


def estimate_tokens(text: str) -> int:
    # Gemma tokenizer averages ~3.5-4 chars/token for English; be conservative.
    return int(len(text) / 3.3) + 50


def _is_rate_limit(e: Exception) -> bool:
    s = f"{type(e).__name__} {e}".lower()
    return any(k in s for k in ["429", "resource_exhausted", "ratelimit", "rate limit", "quota", "tokens per minute", "perminute"])


def _is_transient(e: Exception) -> bool:
    s = f"{type(e).__name__} {e}".lower()
    return any(k in s for k in ["503", "500", "unavailable", "overloaded", "timeout", "timed out", "internalservererror", "connection", "empty string"])


def _retry_delay(e: Exception, default: float) -> float:
    m = re.search(r"retry(?:delay)?[\"' :]*\s*(?:in\s*)?([0-9]+(?:\.[0-9]+)?)\s*s", str(e), re.I)
    if m:
        return min(float(m.group(1)) + 2, 70)
    return default


class GuardedForecaster:
    """Invoke Gemma with throttling and retries; fall back to other models."""

    def __init__(
        self,
        primary: str = "gemini/gemma-4-31b-it",
        fallbacks: tuple[str, ...] = ("gemini/gemini-3.1-flash-lite", "gemini/gemini-3.5-flash"),
        primary_tpm: int = 14000,
        primary_attempts: int = 4,
        temperature: float = 0.3,
        timeout: int = 150,
    ):
        self.primary_name = primary
        self.primary = GeneralLlm(model=primary, temperature=temperature, timeout=timeout, allowed_tries=1)
        self.fallbacks = [
            GeneralLlm(model=f, temperature=temperature, timeout=timeout, allowed_tries=1) for f in fallbacks
        ]
        self.limiter = TokenWindowLimiter(primary_tpm)
        self.primary_attempts = primary_attempts
        self.max_primary_prompt_tokens = primary_tpm
        self.last_model_used = primary

    async def _try(self, llm: GeneralLlm, prompt: str, attempts: int, limiter: TokenWindowLimiter | None) -> str:
        last: Exception | None = None
        for i in range(attempts):
            try:
                if limiter:
                    await limiter.acquire(estimate_tokens(prompt))
                return await llm.invoke(prompt)
            except Exception as e:
                last = e
                if _is_rate_limit(e):
                    d = _retry_delay(e, 30)
                elif _is_transient(e):
                    d = 10 * (i + 1)
                else:
                    d = 5
                logger.warning(f"{llm.model} attempt {i+1}/{attempts} failed ({str(e)[:160]}); retry in {d:.0f}s")
                if i < attempts - 1:
                    await asyncio.sleep(d)
        assert last is not None
        raise last

    async def invoke(self, prompt: str) -> str:
        if estimate_tokens(prompt) < self.max_primary_prompt_tokens:
            try:
                out = await self._try(self.primary, prompt, self.primary_attempts, self.limiter)
                self.last_model_used = self.primary_name
                return out
            except Exception as e:
                logger.warning(f"Primary forecaster {self.primary_name} failed, falling back: {str(e)[:200]}")
        else:
            logger.warning("Prompt too large for Gemma free-tier TPM; using fallback model")
        last: Exception | None = None
        for fb in self.fallbacks:
            try:
                out = await self._try(fb, prompt, 3, None)
                self.last_model_used = fb.model
                logger.info(f"Forecast produced by fallback model {fb.model}")
                return out
            except Exception as e:
                last = e
                logger.warning(f"Fallback {fb.model} failed: {str(e)[:200]}")
        assert last is not None
        raise last


async def guarded_invoke(llm: GeneralLlm | str, prompt: str, attempts: int = 4) -> str:
    """Retry helper for helper models (query writer / summarizer)."""
    if isinstance(llm, str):
        llm = GeneralLlm(model=llm, allowed_tries=1, timeout=90)
    last: Exception | None = None
    for i in range(attempts):
        try:
            return await llm.invoke(prompt)
        except Exception as e:
            last = e
            d = _retry_delay(e, 20) if _is_rate_limit(e) else 6 * (i + 1)
            logger.warning(f"{llm.model} helper attempt {i+1} failed ({str(e)[:120]}); retry in {d:.0f}s")
            if i < attempts - 1:
                await asyncio.sleep(d)
    assert last is not None
    raise last
