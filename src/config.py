"""Configuration loading for the pipeline.

A thin wrapper around ``configs/pipeline.yaml`` that supports dotted-key access
(``cfg.get("dtqf.qualify_threshold")``) and resolves any ``*_dir`` / ``*_path``
values against the project root.
"""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any, Optional

import yaml

from utils import PROJECT_ROOT, get_logger, resolve_path

DEFAULT_CONFIG_PATH = PROJECT_ROOT / "configs" / "pipeline.yaml"
_log = get_logger("config")


class Config:
    """Dict-backed config with dotted lookups and path resolution."""

    def __init__(self, data: dict, source: Optional[Path] = None):
        self._data = data or {}
        self.source = source

    # -- access -------------------------------------------------------------- #
    def get(self, dotted_key: str, default: Any = None) -> Any:
        node: Any = self._data
        for part in dotted_key.split("."):
            if isinstance(node, dict) and part in node:
                node = node[part]
            else:
                return default
        return node

    def __getitem__(self, key: str) -> Any:
        value = self.get(key, _MISSING)
        if value is _MISSING:
            raise KeyError(key)
        return value

    def section(self, key: str) -> dict:
        return dict(self.get(key, {}) or {})

    def path(self, dotted_key: str, default: Optional[str] = None) -> Path:
        """Return a config value resolved to an absolute :class:`Path`."""
        value = self.get(dotted_key, default)
        if value is None:
            raise KeyError(f"No path configured for {dotted_key!r}")
        return resolve_path(value)

    @property
    def data(self) -> dict:
        return self._data


_MISSING = object()


@lru_cache(maxsize=8)
def load_config(path: "str | Path | None" = None) -> Config:
    """Load (and cache) the pipeline configuration."""
    cfg_path = Path(path) if path else DEFAULT_CONFIG_PATH
    if not cfg_path.exists():
        _log.warning("config file %s not found — using defaults", cfg_path)
        return Config(DEFAULT_CONFIG, source=None)
    with cfg_path.open("r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh) or {}
    # Shallow-merge over defaults so missing keys still resolve.
    merged = _deep_merge(DEFAULT_CONFIG, data)
    return Config(merged, source=cfg_path)


def _deep_merge(base: dict, override: dict) -> dict:
    out = dict(base)
    for k, v in override.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = v
    return out


#: Fallback defaults — mirror ``configs/pipeline.yaml`` so code works even if the
#: file is absent.  The YAML on disk is the source of truth in practice.
DEFAULT_CONFIG: dict = {
    "paths": {
        "raw_dir": "data/raw",
        "interim_dir": "data/interim",
        "final_dir": "data/final",
        "results_dir": "results",
        "topic_model_dir": "data/interim/topic_model",
        "cache_dir": ".cache/llm",
    },
    "lexicons": {
        "tcm_terms": "lexicons/tcm_terms.txt",
        "stopwords": "lexicons/stopwords_zh.txt",
        "category_anchors": "lexicons/category_anchors.yaml",
    },
    "prompts": {
        "gen_question": "prompts/gen_question.v3.txt",
        "stager_eval": "prompts/stager_eval.txt",
        "judge_quality": "prompts/judge_quality.txt",
    },
    "ingest": {
        "min_chars": 100,
        "dedup_threshold": 0.85,
        "minhash_perm": 128,
        "extensions": [".txt", ".md", ".html", ".htm", ".docx"],
    },
    "quality": {
        "min_chars": 300,
        "min_tcm_density": 0.04,
        "max_ad_hits": 8,
        "overall_threshold": 6.0,
        "judge_model": "gpt-4o",
    },
    "topic": {
        "embedding_model": "BAAI/bge-large-zh-v1.5",
        "max_len": 400,
        "overlap": 80,
        "min_passage_len": 50,
        "min_cluster_size": 15,
        "umap_components": 5,
        "nr_topics": 175,
        "random_state": 42,
    },
    "generate": {"model": "gpt-4o", "max_passage_chars": 1500},
    "dtqf": {
        "sample_size": 100,
        "max_iter": 20,
        "qualify_threshold": 0.95,
        "kappa_threshold": 0.8,
        "use_llm_judge": False,
        "w2v": {"vector_size": 200, "window": 5, "min_count": 2},
    },
    "assemble": {"diagnostic_frac": 0.10, "token_encoding": "cl100k_base"},
    "evaluate": {"n_runs": 3, "models": ["gpt-4o"]},
    "stats": {"num_segments": 2},
    "llm": {
        "provider": "openai",  # openai | minimax | mock
        "base_url": None,
        "temperature": 0.2,
        "max_tokens": 8192,
        "timeout": 60,
        "max_concurrency": 4,
        "use_cache": True,
    },
}
