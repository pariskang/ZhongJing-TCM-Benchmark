"""Unified LLM client with caching, retries and an offline mock provider.

Every module that talks to an LLM goes through :func:`call_text` /
:func:`call_json` so that caching, retry/back-off and provider selection live in
one place (manual §10.3).

Providers
---------
* ``openai`` — the official OpenAI SDK; also works against any OpenAI-compatible
  endpoint (vLLM, Together, DeepSeek, …) via ``base_url``.
* ``mock``  — fully offline, deterministic.  Returns schema-valid JSON for the
  question-generation / quality-judge / STAGER prompts so the whole pipeline can
  be exercised (and unit-tested) without API keys or network.

Select the provider with the ``ZHONGJING_LLM_PROVIDER`` env var, the ``llm``
section of ``configs/pipeline.yaml``, or by passing ``model="mock"``.
"""
from __future__ import annotations

import ast
import hashlib
import json
import os
import re
import time
from typing import Any, Callable, Optional

from utils import PROJECT_ROOT, get_logger

_log = get_logger("llm_client")

# --------------------------------------------------------------------------- #
# Robust JSON extraction                                                        #
# --------------------------------------------------------------------------- #

_FENCE_RE = re.compile(r"```(?:json)?\s*(.*?)```", re.DOTALL)


def extract_json(text: str) -> dict:
    """Best-effort parse of a JSON object out of an LLM response.

    Handles ```json fences, leading/trailing prose and single-quoted dicts.
    Raises ``ValueError`` if nothing parseable is found.
    """
    if text is None:
        raise ValueError("empty LLM response")
    candidates: list[str] = []
    m = _FENCE_RE.search(text)
    if m:
        candidates.append(m.group(1).strip())
    # Greedy outermost braces.
    start, end = text.find("{"), text.rfind("}")
    if start != -1 and end != -1 and end > start:
        candidates.append(text[start : end + 1])
    candidates.append(text.strip())

    for cand in candidates:
        try:
            return json.loads(cand)
        except Exception:
            try:
                obj = ast.literal_eval(cand)
                if isinstance(obj, dict):
                    return obj
            except Exception:
                continue
    raise ValueError(f"Could not extract JSON from response: {text[:200]!r}")


# --------------------------------------------------------------------------- #
# Optional dependencies (cache / retry) loaded lazily                           #
# --------------------------------------------------------------------------- #


def _make_cache(cache_dir: str):
    try:
        import diskcache  # type: ignore

        return diskcache.Cache(str(cache_dir))
    except Exception as exc:  # pragma: no cover - optional dep
        _log.debug("diskcache unavailable (%s); caching disabled", exc)
        return None


def _retry(fn: Callable, *, attempts: int = 5, base_delay: float = 2.0):
    """Run *fn* with exponential back-off.  Uses tenacity if installed."""
    try:
        from tenacity import (  # type: ignore
            retry,
            stop_after_attempt,
            wait_exponential,
            retry_if_exception_type,
        )

        wrapped = retry(
            stop=stop_after_attempt(attempts),
            wait=wait_exponential(multiplier=base_delay, min=base_delay, max=60),
            retry=retry_if_exception_type(Exception),
            reraise=True,
        )(fn)
        return wrapped()
    except ImportError:  # pragma: no cover - fallback path
        last: Optional[Exception] = None
        for i in range(attempts):
            try:
                return fn()
            except Exception as exc:  # noqa: BLE001
                last = exc
                if i == attempts - 1:
                    break
                time.sleep(base_delay * (2 ** i))
        raise last  # type: ignore[misc]


# --------------------------------------------------------------------------- #
# Client                                                                        #
# --------------------------------------------------------------------------- #


class LLMClient:
    """Caching, retrying wrapper over one or more chat-completion backends."""

    def __init__(
        self,
        provider: Optional[str] = None,
        default_model: str = "gpt-4o",
        base_url: Optional[str] = None,
        api_key_env: str = "OPENAI_API_KEY",
        cache_dir: Optional[str] = None,
        temperature: float = 0.2,
        max_tokens: int = 1024,
        timeout: float = 60.0,
        use_cache: bool = True,
    ):
        self.provider = (
            provider or os.environ.get("ZHONGJING_LLM_PROVIDER") or "openai"
        ).lower()
        self.default_model = default_model
        self.base_url = base_url or os.environ.get("OPENAI_BASE_URL")
        self.api_key_env = api_key_env
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.timeout = timeout
        self.use_cache = use_cache
        cache_dir = cache_dir or os.environ.get(
            "ZHONGJING_CACHE_DIR", str(PROJECT_ROOT / ".cache" / "llm")
        )
        self._cache = _make_cache(cache_dir) if use_cache else None
        self._openai_client = None  # lazily created

    # -- provider resolution ------------------------------------------------- #
    def _resolve_provider(self, model: str) -> str:
        if model and model.lower().startswith("mock"):
            return "mock"
        return self.provider

    # -- caching ------------------------------------------------------------- #
    @staticmethod
    def _cache_key(provider: str, model: str, system: str, prompt: str,
                   temperature: float, max_tokens: int) -> str:
        h = hashlib.sha256()
        for part in (provider, model, system or "", prompt, f"{temperature}", f"{max_tokens}"):
            h.update(part.encode("utf-8"))
            h.update(b"\x00")
        return h.hexdigest()

    # -- public API ---------------------------------------------------------- #
    def call_text(
        self,
        prompt: str,
        model: Optional[str] = None,
        system: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        use_cache: Optional[bool] = None,
    ) -> str:
        """Return the raw text completion for *prompt*."""
        model = model or self.default_model
        system = system or ""
        temperature = self.temperature if temperature is None else temperature
        max_tokens = self.max_tokens if max_tokens is None else max_tokens
        provider = self._resolve_provider(model)
        do_cache = self.use_cache if use_cache is None else use_cache

        key = self._cache_key(provider, model, system, prompt, temperature, max_tokens)
        if do_cache and self._cache is not None and key in self._cache:
            return self._cache[key]

        if provider == "mock":
            text = mock_completion(prompt, model)
        elif provider == "openai":
            text = self._call_openai(prompt, model, system, temperature, max_tokens)
        else:
            raise ValueError(f"Unknown LLM provider: {provider!r}")

        if do_cache and self._cache is not None:
            self._cache[key] = text
        return text

    def call_json(
        self,
        prompt: str,
        model: Optional[str] = None,
        system: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        use_cache: Optional[bool] = None,
    ) -> dict:
        """Call the model and parse a JSON object from the response."""
        text = self.call_text(
            prompt, model=model, system=system, temperature=temperature,
            max_tokens=max_tokens, use_cache=use_cache,
        )
        return extract_json(text)

    # -- backends ------------------------------------------------------------ #
    def _ensure_openai(self):
        if self._openai_client is None:
            try:
                from openai import OpenAI  # type: ignore
            except ImportError as exc:  # pragma: no cover
                raise RuntimeError(
                    "openai package not installed; `pip install openai` or use "
                    "the mock provider (model='mock')."
                ) from exc
            api_key = os.environ.get(self.api_key_env)
            self._openai_client = OpenAI(
                api_key=api_key, base_url=self.base_url, timeout=self.timeout
            )
        return self._openai_client

    def _call_openai(self, prompt, model, system, temperature, max_tokens) -> str:
        client = self._ensure_openai()
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        def _do():
            resp = client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
            )
            return resp.choices[0].message.content or ""

        return _retry(_do)


# --------------------------------------------------------------------------- #
# Module-level default client + convenience wrappers                            #
# --------------------------------------------------------------------------- #

_DEFAULT_CLIENT: Optional[LLMClient] = None


def get_client() -> LLMClient:
    """Return the process-wide default client (configured from pipeline.yaml)."""
    global _DEFAULT_CLIENT
    if _DEFAULT_CLIENT is None:
        try:
            from config import load_config

            cfg = load_config()
            llm = cfg.section("llm")
        except Exception:  # pragma: no cover - config optional
            llm = {}
        # Precedence: ZHONGJING_LLM_PROVIDER env var > pipeline.yaml > default.
        provider = os.environ.get("ZHONGJING_LLM_PROVIDER") or llm.get("provider")
        _DEFAULT_CLIENT = LLMClient(
            provider=provider,
            temperature=llm.get("temperature", 0.2),
            max_tokens=llm.get("max_tokens", 1024),
            timeout=llm.get("timeout", 60.0),
            use_cache=llm.get("use_cache", True),
        )
    return _DEFAULT_CLIENT


def set_default_client(client: LLMClient) -> None:
    global _DEFAULT_CLIENT
    _DEFAULT_CLIENT = client


def call_text(prompt: str, model: Optional[str] = None, **kwargs: Any) -> str:
    return get_client().call_text(prompt, model=model, **kwargs)


def call_json(prompt: str, model: Optional[str] = None, **kwargs: Any) -> dict:
    return get_client().call_json(prompt, model=model, **kwargs)


# --------------------------------------------------------------------------- #
# Async fan-out (manual §5.2 / §10.3: asyncio + Semaphore rate-limiting)         #
# --------------------------------------------------------------------------- #


async def map_async(
    prompts: list[str],
    fn: Callable[[str], Any],
    max_concurrency: int = 4,
) -> list[Any]:
    """Apply blocking *fn* over *prompts* with bounded concurrency."""
    import asyncio

    sem = asyncio.Semaphore(max_concurrency)

    async def _one(p: str):
        async with sem:
            return await asyncio.to_thread(fn, p)

    return await asyncio.gather(*[_one(p) for p in prompts])


# --------------------------------------------------------------------------- #
# Mock provider                                                                 #
# --------------------------------------------------------------------------- #

_MOCK_OPTIONS = {
    "A": "肝郁气滞证",
    "B": "脾胃虚弱证",
    "C": "肝胆湿热证",
    "D": "气血两虚证",
}


def mock_completion(prompt: str, model: str = "mock") -> str:
    """Deterministic offline responses keyed off the prompt family."""
    p = prompt

    # 1) Quality judge prompt -> three-dimensional scores.
    if ("professionalism" in p) or ("专业性" in p and "科普性" in p):
        seed = int(hashlib.sha256(p.encode("utf-8")).hexdigest(), 16)
        prof = 6.0 + (seed % 40) / 10.0          # 6.0 – 9.9
        pop = 6.0 + ((seed // 7) % 40) / 10.0
        prac = 6.0 + ((seed // 13) % 40) / 10.0
        return json.dumps(
            {
                "professionalism": round(prof, 1),
                "popularization": round(pop, 1),
                "practicality": round(prac, 1),
                "reason": "（mock）三维评分用于离线流水线演示。",
            },
            ensure_ascii=False,
        )

    # 2) STAGER evaluation prompt -> structured answer block.
    if ("[Answer]" in p) or ("答案选择" in p):
        return (
            "1. 答案选择\n   - [Answer] A\n"
            "2. 详细分析\n   - [Analysis]\n"
            "     · 理论依据: （mock）依据脏腑辨证。\n"
            "     · 关键要点: 抓主症、辨病位病性。\n"
            "     · 常见误区: 忽略舌脉。"
        )

    # 3) Question-generation prompt -> schema-valid question JSON.
    if ("源文本" in p) or ('"stem"' in p):
        # Parse the *requested* type from the "题型: X" line (the instructions
        # list all three names, so a plain substring check would be ambiguous).
        m = re.search(r"题型[:：]\s*(single_choice|multiple_response|short_answer)", p)
        qtype = m.group(1) if m else "single_choice"
        stem = (
            "患者女性，35岁，胸胁胀痛、善太息、情志抑郁，舌淡红苔薄白，脉弦。"
            "根据源文本，其证型最宜辨为下列哪一项？"
        )
        explanation = (
            "第一步：抓主症——胸胁胀痛、善太息、脉弦，均为肝气郁结之象；"
            "第二步：辨病位在肝，病性属气滞；第三步：故辨为肝郁气滞证，治宜疏肝理气。"
        )
        if qtype == "short_answer":
            return json.dumps(
                {
                    "stem": stem,
                    "options": {},
                    "answer": [],
                    "reference_answer": "肝郁气滞证；治以疏肝理气解郁，方选柴胡疏肝散加减。",
                    "explanation": explanation,
                    "theoretical_basis": "肝主疏泄、调畅情志；脏腑辨证。",
                },
                ensure_ascii=False,
            )
        answer = ["A", "C"] if qtype == "multiple_response" else ["A"]
        return json.dumps(
            {
                "stem": stem,
                "options": dict(_MOCK_OPTIONS),
                "answer": answer,
                "reference_answer": None,
                "explanation": explanation,
                "theoretical_basis": "肝主疏泄、调畅情志；脏腑辨证。",
            },
            ensure_ascii=False,
        )

    # 4) Generic yes/no validity judge -> default to valid.
    if "仅输出" in p and ("true" in p.lower() or "valid" in p.lower()):
        return json.dumps({"valid": True, "reason": "（mock）"}, ensure_ascii=False)

    return "MOCK_RESPONSE"
