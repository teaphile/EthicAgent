"""Input validators for pipeline data.

These are the bouncers at the door — they reject bad data before
it gets into the pipeline and causes confusing errors downstream.
"""

from __future__ import annotations

import logging
import re
from typing import Any

logger = logging.getLogger(__name__)

VALID_DOMAINS = {"healthcare", "finance", "hiring", "disaster", "general"}
VALID_VERDICTS = {"AUTO_APPROVE", "ESCALATE", "REJECT", "HARD_BLOCK"}


class ValidationError(ValueError):
    """Raised when input validation fails."""

    pass


def validate_task(task: str) -> str:
    """Validate and clean a task description."""
    if not task or not isinstance(task, str):
        raise ValidationError("Task must be a non-empty string")
    task = task.strip()
    if len(task) < 5:
        raise ValidationError(f"Task too short ({len(task)} chars), minimum is 5")
    if len(task) > 10000:
        raise ValidationError(f"Task too long ({len(task)} chars), maximum is 10000")
    return task


def validate_domain(domain: str) -> bool:
    """Validate a domain identifier.

    Returns True if *domain* is a recognized domain name, False otherwise.
    """
    if not domain or not isinstance(domain, str):
        return False
    domain_clean = domain.lower().strip()
    return domain_clean in VALID_DOMAINS


def _validate_domain_strict(domain: str) -> str:
    """Strict version that raises on invalid domain."""
    domain = domain.lower().strip()
    if domain not in VALID_DOMAINS:
        raise ValidationError(
            f"Invalid domain '{domain}'. Must be one of: {', '.join(sorted(VALID_DOMAINS))}"
        )
    return domain


def validate_score(score: float, name: str = "score") -> float:
    """Validate a 0-1 score."""
    if not isinstance(score, (int, float)):
        raise ValidationError(f"{name} must be a number, got {type(score).__name__}")
    if score < 0.0 or score > 1.0:
        raise ValidationError(f"{name} must be between 0 and 1, got {score}")
    return float(score)


def validate_weights(weights: dict[str, float]) -> bool:
    """Validate philosophy weights.

    Returns True if all values are non-negative numbers that sum to ~1.0.
    """
    if not isinstance(weights, dict) or not weights:
        return False

    for val in weights.values():
        if not isinstance(val, (int, float)) or val < 0:
            return False

    total = sum(weights.values())
    return not abs(total - 1.0) > 0.01


def validate_context(context: dict[str, Any]) -> dict[str, Any]:
    """Basic validation of pipeline context dict."""
    if not isinstance(context, dict):
        raise ValidationError("Context must be a dict")
    if "domain" in context and not validate_domain(context["domain"]):
        raise ValidationError(f"Invalid domain: {context['domain']}")
    return context


def sanitize_text(text: str) -> str:
    """Remove potentially dangerous content from text input."""
    # Strip control characters (except newline, tab)
    text = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]", "", text)
    # Limit excessive whitespace
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r" {5,}", "    ", text)
    return text.strip()


def validate_task_input(task: str) -> dict:
    """Validate task input. Returns dict with 'valid' key and optional 'errors'.

    Used by the orchestrator which expects ``result["valid"]``.
    """
    errors: list[str] = []
    if not task or not isinstance(task, str):
        errors.append("Task must be a non-empty string")
    else:
        stripped = task.strip()
        if len(stripped) < 5:
            errors.append(f"Task too short ({len(stripped)} chars), minimum is 5")
        if len(stripped) > 10000:
            errors.append(f"Task too long ({len(stripped)} chars), maximum is 10000")
    return {"valid": len(errors) == 0, "errors": errors}


def is_valid_task(task: str) -> bool:
    """Simple boolean check — backward-compat wrapper."""
    return validate_task_input(task)["valid"]


def validate_scores(scores: dict[str, float]) -> bool:
    """Validate that all philosophy scores are in [0, 1]."""
    if not isinstance(scores, dict):
        return False
    for _key, val in scores.items():
        if not isinstance(val, (int, float)):
            return False
        if val < 0.0 or val > 1.0:
            return False
    return True
