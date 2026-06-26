"""Unified LLM client with caching, retries and an offline mock provider.

Every module that talks to an LLM goes through :func:`call_text` /
:func:`call_json` so that caching, retry/back-off and provider selection live in
one place (manual §10.3).

Providers
---------
* ``openai``  — the official OpenAI SDK; also works against any OpenAI-compatible
  endpoint (vLLM, Together, DeepSeek, …) via ``base_url``.
* ``minimax`` — MiniMax's OpenAI-compatible endpoint
  (``https://api.minimaxi.com/v1``).  Reuses the OpenAI SDK; reads the key from
  ``MINIMAX_API_KEY`` and defaults the model to ``MiniMax-M3``.
* ``azure``   — Azure OpenAI's v1 (OpenAI-compatible) endpoint, e.g.
  ``https://<resource>.openai.azure.com/openai/v1``.  Reuses the OpenAI SDK;
  reads the key from ``AZURE_API_KEY`` / the endpoint from ``AZURE_BASE_URL`` and
  defaults the model (deployment name) to ``Kimi-K2.6``.  Reasoning deployments
  (o4-mini, Phi-4-reasoning, grok-*-reasoning, …) that reject ``temperature`` or
  require ``max_completion_tokens`` are handled transparently (see
  ``_call_openai``).
* ``mock``    — fully offline, deterministic.  Returns schema-valid JSON for the
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


def _parse_one(cand: str):
    """Try strict JSON then a Python-literal parse; return a dict or ``None``."""
    if not cand:
        return None
    try:
        return json.loads(cand)
    except Exception:
        try:
            obj = ast.literal_eval(cand)
            return obj if isinstance(obj, dict) else None
        except Exception:
            return None


def _repair_json(cand: str):
    """Repair malformed JSON with the optional ``json-repair`` lib (else ``None``).

    Handles the usual LLM JSON breakage — trailing commas, unquoted keys,
    truncated/missing braces, smart quotes, stray prose — that the strict parsers
    above reject.
    """
    if not cand:
        return None
    try:
        from json_repair import repair_json  # type: ignore

        obj = repair_json(cand, return_objects=True)
        return obj if isinstance(obj, dict) else None
    except Exception:  # pragma: no cover - optional dep / unrepairable
        return None


def extract_json(text: str) -> dict:
    """Best-effort parse of a JSON object out of an LLM response.

    Tries, in order: ```json fences, the greedy outermost ``{...}`` span, and the
    whole string — first with strict JSON / Python-literal parsing, then with the
    ``json-repair`` library as a last resort.  Raises ``ValueError`` if nothing
    parseable is found.
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
        obj = _parse_one(cand)
        if obj is not None:
            return obj

    # Last resort: repair the most JSON-like candidate(s).
    for cand in candidates:
        obj = _repair_json(cand)
        if obj is not None:
            return obj
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
# Local / HuggingFace model registry                                            #
# --------------------------------------------------------------------------- #

# Maps a friendly model name (the string passed as ``model=`` to ``call_text``)
# to per-model loading options consumed by ``LLMClient._load_hf``. Populated from
# ``configs/hf_models.yaml`` via :func:`register_local_models`; the Colab/notebook
# can also extend it at runtime. Recognised keys per entry:
#   repo, quant, dtype, trust_remote_code, loader ("vl"), auto_class,
#   chat_template, template_kwargs, no_system, think_prefix.
LOCAL_MODEL_OPTS: dict[str, dict] = {}


def register_local_models(mapping: dict) -> None:
    """Merge *mapping* (friendly name -> options dict) into ``LOCAL_MODEL_OPTS``."""
    for name, opts in (mapping or {}).items():
        if isinstance(opts, dict):
            LOCAL_MODEL_OPTS[name] = dict(opts)


def load_local_registry(path: Optional[str] = None) -> dict:
    """Load the HF model registry YAML into ``LOCAL_MODEL_OPTS`` and return it.

    The file is keyed by friendly name; each value is the options dict above.
    Missing file / PyYAML simply yields an empty registry (no hard dependency).
    """
    try:
        import yaml  # type: ignore

        p = path or os.environ.get(
            "ZHONGJING_HF_REGISTRY", str(PROJECT_ROOT / "configs" / "hf_models.yaml")
        )
        data = yaml.safe_load(open(p, encoding="utf-8")) or {}
        models = data.get("models", data) if isinstance(data, dict) else {}
        register_local_models(models)
    except Exception as exc:  # pragma: no cover - optional
        _log.debug("local registry not loaded (%s)", exc)
    return LOCAL_MODEL_OPTS


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
        max_tokens: int = 8192,
        timeout: float = 60.0,
        use_cache: bool = True,
    ):
        self.provider = (
            provider or os.environ.get("ZHONGJING_LLM_PROVIDER") or "openai"
        ).lower()
        # MiniMax speaks the OpenAI wire protocol: default its endpoint, key env
        # and model unless the caller overrode them explicitly.
        if self.provider == "minimax":
            self.base_url = (
                base_url
                or os.environ.get("MINIMAX_BASE_URL")
                or "https://api.minimaxi.com/v1"
            )
            self.api_key_env = (
                "MINIMAX_API_KEY" if api_key_env == "OPENAI_API_KEY" else api_key_env
            )
            if default_model == "gpt-4o":
                default_model = os.environ.get("MINIMAX_MODEL", "MiniMax-M3")
        elif self.provider == "azure":
            # Azure OpenAI v1 endpoint is OpenAI-compatible: a plain OpenAI client
            # with base_url=".../openai/v1" and api_key=<key> (no api-version dance).
            self.base_url = (
                base_url
                or os.environ.get("AZURE_BASE_URL")
                or os.environ.get("AZURE_OPENAI_BASE_URL")
                or "https://fosterpearson-ft-5186-resource.openai.azure.com/openai/v1"
            )
            self.api_key_env = (
                "AZURE_API_KEY" if api_key_env == "OPENAI_API_KEY" else api_key_env
            )
            if default_model == "gpt-4o":
                default_model = os.environ.get("AZURE_MODEL", "Kimi-K2.6")
        else:
            self.base_url = base_url or os.environ.get("OPENAI_BASE_URL")
            self.api_key_env = api_key_env
        self.default_model = default_model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.timeout = timeout
        self.use_cache = use_cache
        cache_dir = cache_dir or os.environ.get(
            "ZHONGJING_CACHE_DIR", str(PROJECT_ROOT / ".cache" / "llm")
        )
        self._cache = _make_cache(cache_dir) if use_cache else None
        self._openai_client = None  # lazily created
        # Per-model parameter quirks learned at runtime (Azure reasoning models):
        #   {"no_temperature": bool, "use_max_completion_tokens": bool}
        self._param_quirks: dict[str, dict] = {}
        # Single-GPU-slot cache for the local/HF backend: {"id", "model", "tok"}.
        # Only one model is held at a time; switching models frees the previous.
        self._hf_slot: Optional[dict] = None

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
        elif provider in ("openai", "minimax", "azure"):
            # MiniMax and Azure (v1) are OpenAI-compatible; base_url / api_key_env
            # already point at the right endpoint (set in __init__).
            text = self._call_openai(prompt, model, system, temperature, max_tokens)
        elif provider in ("local", "hf"):
            text = self._call_local(prompt, model, system, temperature, max_tokens)
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

        quirks = self._param_quirks.setdefault(model, {})

        def _build_kwargs() -> dict:
            kw: dict[str, Any] = {"model": model, "messages": messages}
            # Reasoning deployments (o4-mini, Phi-4-reasoning, grok-*-reasoning, …)
            # only accept the default temperature; omit it for them.
            if not quirks.get("no_temperature"):
                kw["temperature"] = temperature
            # Same models renamed the output cap to `max_completion_tokens`.
            if quirks.get("use_max_completion_tokens"):
                kw["max_completion_tokens"] = max_tokens
            else:
                kw["max_tokens"] = max_tokens
            return kw

        def _do():
            # Try the full call; on a parameter-unsupported error, learn the quirk
            # (cached on the client so later calls skip the failed attempt) and retry.
            for _ in range(3):
                try:
                    resp = client.chat.completions.create(**_build_kwargs())
                    return resp.choices[0].message.content or ""
                except Exception as exc:  # noqa: BLE001
                    msg = str(exc).lower()
                    learned = False
                    if ("max_completion_tokens" in msg or "max_tokens" in msg) and not quirks.get(
                        "use_max_completion_tokens"
                    ):
                        quirks["use_max_completion_tokens"] = True
                        learned = True
                    if "temperature" in msg and not quirks.get("no_temperature"):
                        quirks["no_temperature"] = True
                        learned = True
                    if not learned:
                        raise
            return client.chat.completions.create(**_build_kwargs()).choices[0].message.content or ""

        return _retry(_do)

    # -- local / HuggingFace backend ----------------------------------------- #
    def _load_hf(self, model_id: str):
        """Load (or reuse) a local HF model in the single GPU slot.

        Only one model is resident at a time: requesting a different ``model_id``
        unloads the previous one and frees the GPU before loading the new weights.
        Per-model loading options (quantization, ``trust_remote_code``, dtype, a
        fallback chat template for models whose tokenizer ships none) are looked
        up from :data:`LOCAL_MODEL_OPTS` (populated from ``configs/hf_models.yaml``
        by the caller) and overridable via ``HF_*`` environment variables.
        """
        if self._hf_slot and self._hf_slot["id"] == model_id:
            return self._hf_slot

        if not LOCAL_MODEL_OPTS:  # lazily populate from configs/hf_models.yaml
            load_local_registry()

        # Evict the previous model and reclaim VRAM before loading another.
        if self._hf_slot is not None:
            self._hf_slot.clear()
            self._hf_slot = None
            try:
                import gc

                import torch  # type: ignore

                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except Exception:  # pragma: no cover - best-effort cleanup
                pass

        import torch  # type: ignore
        import transformers  # type: ignore
        from transformers import AutoTokenizer  # type: ignore

        opts = dict(LOCAL_MODEL_OPTS.get(model_id, {}))
        repo = opts.get("repo", model_id)

        def _envflag(name: str, default: bool) -> bool:
            v = os.environ.get(name)
            return default if v is None else v.strip().lower() in ("1", "true", "yes", "on")

        trust = _envflag("HF_TRUST_REMOTE_CODE", bool(opts.get("trust_remote_code", True)))
        # Quantization: HF_QUANT in {none,4bit,8bit}; default per-model (else none).
        quant = (os.environ.get("HF_QUANT") or opts.get("quant") or "none").lower()
        dtype_name = os.environ.get("HF_DTYPE") or opts.get("dtype") or "bfloat16"
        dtype = getattr(torch, dtype_name, torch.bfloat16)
        token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
        # Multimodal models (Qwen2.5-VL / Gemma3 / …) need an image-text-to-text
        # class + a processor; we still drive them text-only for this benchmark.
        is_vl = str(opts.get("loader", "")).lower() in ("vl", "image-text-to-text")
        auto_cls_name = opts.get("auto_class") or (
            "AutoModelForImageTextToText" if is_vl else "AutoModelForCausalLM"
        )
        AutoModelCls = getattr(transformers, auto_cls_name)

        _log.info("loading local model %s (quant=%s, dtype=%s, cls=%s)",
                  repo, quant, dtype_name, auto_cls_name)
        if is_vl:
            from transformers import AutoProcessor  # type: ignore

            processor = AutoProcessor.from_pretrained(repo, trust_remote_code=trust, token=token)
            tok = getattr(processor, "tokenizer", None) or processor
        else:
            processor = None
            tok = AutoTokenizer.from_pretrained(repo, trust_remote_code=trust, token=token)

        model_kwargs: dict[str, Any] = {
            "trust_remote_code": trust,
            "device_map": os.environ.get("HF_DEVICE_MAP", "auto"),
            "token": token,
        }
        if quant in ("4bit", "8bit"):
            try:
                from transformers import BitsAndBytesConfig  # type: ignore

                if quant == "4bit":
                    model_kwargs["quantization_config"] = BitsAndBytesConfig(
                        load_in_4bit=True,
                        bnb_4bit_compute_dtype=dtype,
                        bnb_4bit_quant_type="nf4",
                        bnb_4bit_use_double_quant=True,
                    )
                else:
                    model_kwargs["quantization_config"] = BitsAndBytesConfig(load_in_8bit=True)
            except Exception as exc:  # pragma: no cover - bitsandbytes missing
                _log.warning("bitsandbytes unavailable (%s); loading unquantized", exc)
                model_kwargs["torch_dtype"] = dtype
        else:
            model_kwargs["torch_dtype"] = dtype

        model = AutoModelCls.from_pretrained(repo, **model_kwargs)
        model.eval()
        if getattr(tok, "pad_token_id", None) is None and getattr(tok, "eos_token_id", None) is not None:
            tok.pad_token = tok.eos_token

        self._hf_slot = {
            "id": model_id,
            "model": model,
            "tok": tok,
            "processor": processor,                       # set for VL models, else None
            "chat_template": opts.get("chat_template"),    # fallback if tokenizer ships none
            "no_system": bool(opts.get("no_system", False)),
            "think_prefix": opts.get("think_prefix"),      # e.g. "<think>\n" for R1-style
            "extra_template_kwargs": opts.get("template_kwargs") or {},  # e.g. thinking_mode
        }
        return self._hf_slot

    def _call_local(self, prompt, model, system, temperature, max_tokens) -> str:
        import torch  # type: ignore

        slot = self._load_hf(model)
        tok, mdl = slot["tok"], slot["model"]
        processor = slot["processor"]

        is_vl = processor is not None

        def _content(s: str):
            # VL processors (Qwen2.5-VL / Gemma3) want typed content parts even for
            # text-only turns; plain LMs want a bare string.
            return [{"type": "text", "text": s}] if is_vl else s

        messages = []
        if system and not slot["no_system"]:
            messages.append({"role": "system", "content": _content(system)})
        elif system and slot["no_system"]:
            # R1-style: no system role — fold instructions into the user turn.
            prompt = f"{system}\n\n{prompt}"
        messages.append({"role": "user", "content": _content(prompt)})

        # Render the prompt: prefer the tokenizer/processor chat template, fall back
        # to a registry-supplied template, else a plain rendering.
        if slot["chat_template"]:
            tok.chat_template = slot["chat_template"]
        tmpl_kwargs = dict(add_generation_prompt=True, **slot["extra_template_kwargs"])
        try:
            text = tok.apply_chat_template(messages, tokenize=False, **tmpl_kwargs)
        except Exception:
            sys_part = f"{system}\n\n" if (system and not slot["no_system"]) else ""
            text = f"{sys_part}{prompt}\n"
        if slot["think_prefix"]:
            text = text + slot["think_prefix"]

        def _do():
            # VL models use their processor to tokenize (text-only here); plain LMs
            # use the tokenizer directly.
            enc = (processor if processor is not None else tok)(text, return_tensors="pt")
            enc = {k: v.to(mdl.device) for k, v in enc.items() if hasattr(v, "to")}
            in_len = enc["input_ids"].shape[1]
            gen = {
                "max_new_tokens": int(max_tokens),
                "do_sample": temperature is not None and temperature > 0,
                "pad_token_id": getattr(tok, "pad_token_id", None),
            }
            if gen["do_sample"]:
                gen["temperature"] = float(temperature)
                gen["top_p"] = float(os.environ.get("HF_TOP_P", "0.95"))
            with torch.no_grad():
                out = mdl.generate(**enc, **gen)
            new_tokens = out[0][in_len:]
            decoder = processor if processor is not None else tok
            return decoder.decode(new_tokens, skip_special_tokens=True).strip()

        return _retry(_do, attempts=2, base_delay=1.0)


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
            base_url=llm.get("base_url"),
            temperature=llm.get("temperature", 0.2),
            max_tokens=llm.get("max_tokens", 8192),
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

    # 1b) T2 expert-inquiry prompt -> a JSON action (ask 主症 → 舌 → 脉 → diagnose).
    if "接诊医生" in p:
        convo = p.rsplit("【对话】", 1)[-1]
        if "部位" not in convo:
            q = "请问主要不适的部位和性质如何？还有哪些伴随症状？"
        elif "舌" not in convo:
            q = "请问舌象如何？"
        elif "脉" not in convo:
            q = "请问脉象如何？"
        else:
            return json.dumps({"action": "diagnose", "answer": "（mock 辨证，离线占位）"}, ensure_ascii=False)
        return json.dumps({"action": "ask", "query": q}, ensure_ascii=False)

    # 0b) Confidence-elicitation prompt -> first option label + an over-confident 0.9.
    if ("置信度" in p) and ('"answer"' in p):
        tail = p.rsplit("题目", 1)[-1] if "题目" in p else p
        lm = re.search(r"(?m)^\s*([A-D甲乙丙丁戊己1-6])[).、．。:：]\s*\S", tail)
        label = lm.group(1) if lm else "A"
        return json.dumps({"answer": [label], "confidence": 0.9}, ensure_ascii=False)

    # 0c) T5 MDT specialty agent -> vote the first option label (homogeneous panel).
    if "多学科会诊" in p:
        tail = p.rsplit("选项", 1)[-1] if "选项" in p else p
        lm = re.search(r"(?m)^\s*([A-D甲乙丙丁戊己1-6])[).、．。:：]\s*\S", tail)
        label = lm.group(1) if lm else "A"
        return json.dumps(
            {"vote": [label], "confidence": 0.7, "rationale": "（mock）本专科意见"},
            ensure_ascii=False,
        )

    # 1a) T3 tool-use agent -> call the contraindication checker, then ground the answer.
    if ("中医临床智能体" in p) and ("可用工具" in p):
        convo = p.rsplit("【历史】", 1)[-1]
        if "工具结果" not in convo:                       # not yet consulted a tool
            task = p.split("任务:", 1)[-1].split("可用工具", 1)[0]
            m_rx = re.search(r"[:：]\s*([^。\n]*(?:、[^。\n、]+)+)", task)
            herbs = [h.strip() for h in m_rx.group(1).split("、")] if m_rx else []
            return json.dumps(
                {"action": "call_tool", "tool": "contraindication_check", "args": {"herbs": herbs}},
                ensure_ascii=False,
            )
        conflict = ('"conflict": true' in convo.lower()) or ('"conflict":true' in convo.lower())
        return json.dumps({"action": "final", "answer": "有禁忌" if conflict else "安全"}, ensure_ascii=False)

    # 1c) L2 step-PRM preference prompt -> pick the sounder next action.
    if ("更优" in p) and ("候选A" in p):
        a = re.search(r"候选A[:：]\s*(.+)", p)
        b = re.search(r"候选B[:：]\s*(.+)", p)
        ta = a.group(1).strip() if a else ""
        tb = b.group(1).strip() if b else ""
        good = ("追问", "采集", "判别", "四诊", "辨证", "据此")
        bad = ("立即", "尚未", "未问", "直接判定", "已问", "已采集", "无关", "寒暄", "重复")
        sa = sum(g in ta for g in good) - sum(x in ta for x in bad)
        sb = sum(g in tb for g in good) - sum(x in tb for x in bad)
        return json.dumps({"better": "A" if sa >= sb else "B"}, ensure_ascii=False)

    # 2) STAGER evaluation prompt -> structured answer block.
    if ("[Answer]" in p) or ("答案选择" in p):
        # Label-aware: answer the first option label (A–D / 甲乙丙丁 / 1–4) that
        # appears in the question block (after "题目:"), so option-order / symbol
        # perturbation demos run coherently offline.
        tail = p.rsplit("题目", 1)[-1] if "题目" in p else p
        lm = re.search(r"(?m)^\s*([A-D甲乙丙丁戊己1-6])[).、．。:：]\s*\S", tail)
        label = lm.group(1) if lm else "A"
        return (
            f"1. 答案选择\n   - [Answer] {label}\n"
            "2. 详细分析\n   - [Analysis]\n"
            "     · 理论依据: （mock）依据脏腑辨证。\n"
            "     · 关键要点: 抓主症、辨病位病性。\n"
            "     · 常见误区: 忽略舌脉。"
        )

    # 2b) T1 counterfactual-pair prompt (also contains 源文本) -> a flipped pair.
    if "反事实最小对" in p:
        return json.dumps(
            {
                "cf_feature": "舌脉",
                "base_value": "舌淡胖、苔白滑，脉沉迟",
                "cf_value": "舌红、苔黄腻，脉滑数",
                "options": {
                    "A": "脾胃虚寒证",
                    "B": "脾胃湿热证",
                    "C": "肝郁气滞证",
                    "D": "气血两虚证",
                },
                "base_stem": "患者男，48岁，脘腹冷痛、喜温喜按、纳呆便溏，舌淡胖、苔白滑，脉沉迟。其证型最宜辨为？",
                "base_answer": ["A"],
                "variant_stem": "患者男，48岁，脘腹冷痛、喜温喜按、纳呆便溏，舌红、苔黄腻，脉滑数。其证型最宜辨为？",
                "cf_answer": ["B"],
                "explanation": "舌脉由虚寒之象转为湿热之象时，证型相应由脾胃虚寒翻转为脾胃湿热。",
            },
            ensure_ascii=False,
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

    # 4) Short-answer semantic judge: reference_answer + student_answer present.
    if "学生答案" in p and "参考答案" in p:
        return json.dumps(
            {"correct": True, "score": 0.9, "reason": "（mock）语义一致，核心要点覆盖。"},
            ensure_ascii=False,
        )

    # 5) Generic yes/no validity judge -> default to valid.
    if "仅输出" in p and ("true" in p.lower() or "valid" in p.lower()):
        return json.dumps({"valid": True, "reason": "（mock）"}, ensure_ascii=False)

    return "MOCK_RESPONSE"
