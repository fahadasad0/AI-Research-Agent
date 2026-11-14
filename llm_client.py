"""LLM access layer.

Responsibilities
----------------
1. ``clean_api_key``  - removes hidden / non-ASCII characters from a pasted key.
   This is the fix for::

       GroqException - 'ascii' codec can't encode characters in position 63-94

   HTTP headers must be ASCII. ``Authorization: Bearer <key>`` is 7 characters
   plus a 56-character Groq key = 63, so "position 63" means the junk sits
   *after* the key (smart quotes, zero-width spaces, pasted text, ...).
2. ``TokenBudget``    - a sliding 60-second token limiter so the Groq free tier
   (about 8,000 tokens/minute for gpt-oss-120b) does not throw 429 errors.
3. ``run_agent``      - runs one CrewAI agent/task and retries on rate limits.
"""
from __future__ import annotations

import os
import re
import time
import unicodedata
from collections import deque
from typing import Callable, Optional

os.environ.setdefault("OTEL_SDK_DISABLED", "true")  # silence CrewAI telemetry noise

_INVISIBLE = {0x200B, 0x200C, 0x200D, 0x200E, 0x200F, 0x2060, 0xFEFF, 0x00A0}
_QUOTES = "\"'`\u201c\u201d\u2018\u2019\u201e\u00ab\u00bb"


class ApiKeyError(ValueError):
    """The API key is missing or contains characters that cannot be sent in an HTTP header."""


class DailyQuotaError(RuntimeError):
    """The provider's daily token/request quota is exhausted; waiting a minute will not help."""


def clean_api_key(raw: Optional[str], label: str = "GROQ_API_KEY") -> str:
    """Return a header-safe API key or raise ``ApiKeyError`` with a precise diagnosis."""
    key = "".join(ch for ch in (raw or "") if ord(ch) not in _INVISIBLE).strip()
    key = key.strip(_QUOTES).strip()
    if key.lower().startswith("bearer "):
        key = key[7:].strip()
    if key.upper().startswith(label + "="):  # user pasted the whole KEY=value line
        key = key.split("=", 1)[1].strip().strip(_QUOTES).strip()

    if not key:
        raise ApiKeyError(f"{label} is missing. Add it in Streamlit Cloud > App settings > Secrets, "
                          "or paste it into the sidebar.")

    bad = [(i, ch) for i, ch in enumerate(key) if ord(ch) > 127 or ch.isspace() or ord(ch) < 33]
    if bad:
        shown = ", ".join(
            f"position {i} = U+{ord(ch):04X} ({unicodedata.name(ch, 'unnamed character')})"
            for i, ch in bad[:5]
        )
        raise ApiKeyError(
            f"{label} contains {len(bad)} character(s) that are not valid in an API key ({shown}). "
            "Delete the key and paste it again as plain text, e.g. "
            f'GROQ_API_KEY = "gsk_..." using straight quotes (not curly quotes) and nothing after it.'
        )
    return key


def est_tokens(text: str) -> int:
    """Deliberately conservative token estimate (about 3.2 characters per token)."""
    return int(len(text) / 3.2) + 1


class TokenBudget:
    """Sliding-window tokens-per-minute limiter shared by every LLM call in a run."""

    def __init__(self, tokens_per_minute: int, on_wait: Optional[Callable[[float], None]] = None):
        self.tpm = max(int(tokens_per_minute), 1000)
        self.on_wait = on_wait
        self.events: deque[list] = deque()  # [timestamp, tokens]

    def _expire(self, now: float) -> None:
        while self.events and now - self.events[0][0] >= 60:
            self.events.popleft()

    def acquire(self, tokens: int) -> list:
        tokens = min(int(tokens), self.tpm)  # a single call can never be "bigger" than the budget
        while True:
            now = time.monotonic()
            self._expire(now)
            used = sum(t for _, t in self.events)
            if used + tokens <= self.tpm:
                event = [now, tokens]
                self.events.append(event)
                return event
            wait = 60 - (now - self.events[0][0]) + 0.5
            if self.on_wait:
                self.on_wait(wait)
            time.sleep(max(wait, 0.5))

    @staticmethod
    def record_actual(event: list, actual_tokens: Optional[int]) -> None:
        if actual_tokens and actual_tokens > 0:
            event[1] = int(actual_tokens)


def make_llm(model: str, api_key: str, max_tokens: int = 2500, temperature: float = 0.3):
    """Create a CrewAI ``LLM`` for Groq."""
    from crewai import LLM

    os.environ["GROQ_API_KEY"] = api_key
    model_id = model if model.startswith("groq/") else f"groq/{model}"
    kwargs = dict(model=model_id, api_key=api_key, temperature=temperature, max_tokens=max_tokens)
    if "gpt-oss" in model_id:
        try:  # keep hidden reasoning short: it counts against the tokens-per-minute limit
            return LLM(**kwargs, reasoning_effort="low")
        except TypeError:
            pass
    return LLM(**kwargs)


def _retry_after_seconds(message: str) -> float:
    m = re.search(r"try again in\s*(?:(\d+)h)?\s*(?:(\d+)m(?!s))?\s*(?:(\d+(?:\.\d+)?)s)?", message, re.I)
    if m and any(m.groups()):
        h, mi, s = (float(g) if g else 0.0 for g in m.groups())
        return h * 3600 + mi * 60 + s
    return 20.0


def run_agent(
    *,
    model: str,
    api_key: str,
    role: str,
    goal: str,
    backstory: str,
    prompt: str,
    expected_output: str,
    budget: TokenBudget,
    expected_out_tokens: int = 1200,
    max_tokens: int = 2500,
    max_retries: int = 4,
    on_status: Optional[Callable[[str], None]] = None,
) -> str:
    """Run one CrewAI agent on one task and return the raw text answer."""
    from crewai import Agent, Crew, Process, Task

    # CrewAI may treat {braces} as template variables; abstracts can contain LaTeX braces.
    prompt = prompt.replace("{", "(").replace("}", ")")
    estimate = est_tokens(prompt) + est_tokens(role + goal + backstory + expected_output) + 600 + expected_out_tokens

    last_error: Optional[Exception] = None
    for attempt in range(max_retries + 1):
        event = budget.acquire(estimate)
        try:
            llm = make_llm(model, api_key, max_tokens=max_tokens)
            agent = Agent(role=role, goal=goal, backstory=backstory, llm=llm, tools=[],
                          verbose=False, allow_delegation=False, max_iter=3)
            task = Task(description=prompt, expected_output=expected_output, agent=agent)
            crew = Crew(agents=[agent], tasks=[task], process=Process.sequential, verbose=False)
            result = crew.kickoff()
            usage = getattr(result, "token_usage", None)
            TokenBudget.record_actual(event, getattr(usage, "total_tokens", None))
            text = str(getattr(result, "raw", result) or "").strip()
            if text:
                return text
            last_error = RuntimeError("The model returned an empty answer.")
        except Exception as exc:  # noqa: BLE001 - we classify and re-raise below
            last_error = exc
            msg = str(exc)
            low = msg.lower()
            if "ascii" in low and "codec" in low:
                raise ApiKeyError(
                    "Python could not encode the request as ASCII. This almost always means the API key "
                    "contains hidden non-ASCII characters. Re-paste the key as plain text."
                ) from exc
            if re.search(r"tokens per day|requests per day|\(TPD\)|\(RPD\)", msg, re.I):
                raise DailyQuotaError(
                    "Groq's daily quota for this model is used up (free tier: about 200K tokens/day). "
                    f"{msg[:300]}"
                ) from exc
            if "request too large" in low or "reduce your message size" in low:
                raise RuntimeError(
                    "A single request exceeded Groq's per-minute token limit. "
                    "Lower the number of papers in the sidebar and try again."
                ) from exc
            if "rate limit" in low or "429" in low or "ratelimit" in low:
                wait = min(_retry_after_seconds(msg) + 1.5, 90)
                if on_status:
                    on_status(f"Groq rate limit reached; waiting {wait:.0f}s before retrying...")
                time.sleep(wait)
                continue
            if any(t in low for t in ("timeout", "timed out", "502", "503", "500", "overloaded", "connection")):
                wait = 5 * (attempt + 1)
                if on_status:
                    on_status(f"Temporary API problem; retrying in {wait}s...")
                time.sleep(wait)
                continue
            raise
    raise RuntimeError(f"The LLM call failed after {max_retries + 1} attempts: {last_error}")
