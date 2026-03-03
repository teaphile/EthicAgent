"""General utility helpers used across the project.

A grab-bag of small functions that don't belong anywhere else.
If you're adding something here, think twice about whether it should
be in a more specific module instead.
"""

from __future__ import annotations

import hashlib
import re
import time
import logging
from datetime import datetime, timezone
from typing import Any

logger = logging.getLogger(__name__)


def slugify(text: str) -> str:
    """Turn 'Some Text Here' into 'some-text-here'."""
    text = text.lower().strip()
    text = re.sub(r"[^\w\s-]", "", text)
    text = re.sub(r"[\s_]+", "-", text)
    return text.strip("-")


def truncate(text: str, max_len: int = 200, suffix: str = "...") -> str:
    """Truncate text to max_len characters, adding suffix if truncated."""
    if len(text) <= max_len:
        return text
    return text[: max_len - len(suffix)] + suffix


def hash_text(text: str) -> str:
    """SHA256 hash of text — useful for cache keys and dedup."""
    return hashlib.sha256(text.encode()).hexdigest()[:16]


def clamp(value: float, low: float = 0.0, high: float = 1.0) -> float:
    """Clamp a value to [low, high]."""
    return max(low, min(high, value))


def safe_divide(numerator: float, denominator: float, default: float = 0.0) -> float:
    """Division that doesn't blow up on zero."""
    if denominator == 0:
        return default
    return numerator / denominator


def timer(func=None):
    """Dual-purpose timer: works as both a decorator and context manager.

    As decorator::

        @timer
        def slow_func():
            ...

    As context manager::

        with timer() as t:
            do_stuff()
        print(t.elapsed_ms)
    """
    if func is not None:
        # Used as @timer (decorator mode)
        import functools

        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            t = Timer()
            t.__enter__()
            try:
                return func(*args, **kwargs)
            finally:
                t.__exit__(None, None, None)

        return wrapper

    # Used as timer() (context manager mode)
    return Timer()


class Timer:
    """Simple timing context manager.

    Usage::

        with timer() as t:
            do_stuff()
        print(f"Took {t.elapsed_ms:.1f}ms")
    """

    def __init__(self) -> None:
        self.start_time = 0.0
        self.end_time = 0.0
        self.elapsed_ms = 0.0

    def __enter__(self) -> Timer:
        self.start_time = time.perf_counter()
        return self

    def __exit__(self, *args: Any) -> None:
        self.end_time = time.perf_counter()
        self.elapsed_ms = (self.end_time - self.start_time) * 1000.0


def flatten_dict(d: dict, parent_key: str = "", sep: str = ".") -> dict[str, Any]:
    """Flatten a nested dict: {'a': {'b': 1}} → {'a.b': 1}."""
    items: list[tuple[str, Any]] = []
    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, dict):
            items.extend(flatten_dict(v, new_key, sep).items())
        else:
            items.append((new_key, v))
    return dict(items)


def batch_iter(items: list, batch_size: int):
    """Yield successive batches from a list."""
    for i in range(0, len(items), batch_size):
        yield items[i : i + batch_size]


def format_score(score: float, width: int = 6) -> str:
    """Format a 0-1 score as a readable string with visual bar."""
    bar_len = int(score * 10)
    bar = "█" * bar_len + "░" * (10 - bar_len)
    return f"{score:{width}.3f} {bar}"


def pluralize(word: str, count: int) -> str:
    """Good enough pluralization for English nouns."""
    if count == 1:
        return word
    if word.endswith("y") and word[-2] not in "aeiou":
        return word[:-1] + "ies"
    if word.endswith(("s", "sh", "ch", "x", "z")):
        return word + "es"
    return word + "s"


def now_iso() -> str:
    """Return the current UTC time as an ISO-8601 string."""
    return datetime.now(timezone.utc).isoformat()


# ── Backward-compat aliases expected by tests ────────────────

def hash_dict(d: dict) -> str:
    """Deterministic hash of a dict (order-independent)."""
    import json
    canonical = json.dumps(d, sort_keys=True, default=str)
    return hashlib.sha256(canonical.encode()).hexdigest()[:16]


def merge_dicts(base: dict, override: dict) -> dict:
    """Deep merge two dicts, with override taking precedence."""
    result = dict(base)
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = merge_dicts(result[key], value)
        else:
            result[key] = value
    return result


def truncate_text(text: str, max_len: int = 200, suffix: str = "...") -> str:
    """Alias for truncate()."""
    return truncate(text, max_len, suffix)


def format_score_breakdown(scores: dict[str, float]) -> str:
    """Format philosophy scores as a human-readable string."""
    lines = []
    for name, score in sorted(scores.items()):
        lines.append(f"  {name}: {score:.3f}")
    return "\n".join(lines)
