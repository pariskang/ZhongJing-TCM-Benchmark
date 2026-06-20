"""Shared helpers: project paths, JSONL I/O, logging.

Kept dependency-free (stdlib + pydantic only) so every module can import it
without pulling in the heavy ML stack.
"""
from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Any, Iterable, Iterator, Optional, Type, TypeVar

from pydantic import BaseModel

#: Repository root (``src/`` lives directly under it).
PROJECT_ROOT = Path(__file__).resolve().parent.parent

T = TypeVar("T", bound=BaseModel)

_LOG_LEVEL = os.environ.get("ZHONGJING_LOG_LEVEL", "INFO").upper()


def get_logger(name: str) -> logging.Logger:
    """Return a configured logger; idempotent across calls."""
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(
            logging.Formatter("%(asctime)s | %(levelname)-7s | %(name)s | %(message)s")
        )
        logger.addHandler(handler)
        logger.setLevel(_LOG_LEVEL)
        logger.propagate = False
    return logger


def resolve_path(path: "str | Path") -> Path:
    """Resolve *path* against the project root unless it is already absolute."""
    p = Path(path)
    return p if p.is_absolute() else (PROJECT_ROOT / p)


def ensure_parent(path: "str | Path") -> Path:
    """Make sure the parent directory of *path* exists; return the resolved path."""
    p = resolve_path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    return p


def _to_dict(record: Any) -> dict:
    """Serialise a record to a JSON-safe dict (enums -> values, etc.)."""
    if isinstance(record, BaseModel):
        return record.model_dump(mode="json")
    if isinstance(record, dict):
        return record
    raise TypeError(f"Cannot serialise object of type {type(record)!r} to JSONL")


def save_jsonl(records: Iterable[Any], path: "str | Path") -> Path:
    """Write *records* (pydantic models or dicts) as one JSON object per line."""
    out = ensure_parent(path)
    n = 0
    with out.open("w", encoding="utf-8") as fh:
        for rec in records:
            fh.write(json.dumps(_to_dict(rec), ensure_ascii=False))
            fh.write("\n")
            n += 1
    get_logger("utils").info("wrote %d records -> %s", n, out)
    return out


def iter_jsonl(path: "str | Path") -> Iterator[dict]:
    """Lazily yield dict records from a JSONL file (skips blank lines)."""
    p = resolve_path(path)
    with p.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                yield json.loads(line)


def load_jsonl(path: "str | Path") -> list[dict]:
    """Read an entire JSONL file into a list of dicts."""
    return list(iter_jsonl(path))


def load_jsonl_as(path: "str | Path", model: Type[T]) -> list[T]:
    """Read a JSONL file and validate each line into ``model`` instances."""
    return [model.model_validate(rec) for rec in iter_jsonl(path)]


def read_text(path: "str | Path", default: str = "") -> str:
    """Read a UTF-8 text file, returning *default* if it does not exist."""
    p = resolve_path(path)
    if not p.exists():
        return default
    return p.read_text(encoding="utf-8")


def read_lines(path: "str | Path") -> list[str]:
    """Read non-empty, non-comment (``#``) stripped lines from a text file."""
    out: list[str] = []
    for line in read_text(path).splitlines():
        s = line.strip()
        if s and not s.startswith("#"):
            out.append(s)
    return out
