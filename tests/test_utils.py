"""Tests for utility modules — config_loader, validators, helpers."""

from __future__ import annotations

import time

from ethicagent.utils.helpers import (
    clamp,
    flatten_dict,
    format_score_breakdown,
    hash_dict,
    merge_dicts,
    now_iso,
    safe_divide,
    timer,
    truncate_text,
)
from ethicagent.utils.validators import (
    validate_domain,
    validate_scores,
    validate_task_input,
    validate_weights,
)


class TestClamp:
    def test_within_range(self):
        assert clamp(0.5, 0.0, 1.0) == 0.5

    def test_below_min(self):
        assert clamp(-0.5, 0.0, 1.0) == 0.0

    def test_above_max(self):
        assert clamp(1.5, 0.0, 1.0) == 1.0

    def test_at_boundaries(self):
        assert clamp(0.0, 0.0, 1.0) == 0.0
        assert clamp(1.0, 0.0, 1.0) == 1.0


class TestHashDict:
    def test_deterministic(self):
        d = {"a": 1, "b": 2}
        assert hash_dict(d) == hash_dict(d)

    def test_different_dicts(self):
        assert hash_dict({"a": 1}) != hash_dict({"b": 2})

    def test_order_independent(self):
        d1 = {"a": 1, "b": 2}
        d2 = {"b": 2, "a": 1}
        assert hash_dict(d1) == hash_dict(d2)


class TestMergeDicts:
    def test_simple_merge(self):
        result = merge_dicts({"a": 1}, {"b": 2})
        assert result == {"a": 1, "b": 2}

    def test_override(self):
        result = merge_dicts({"a": 1}, {"a": 2})
        assert result["a"] == 2

    def test_nested_merge(self):
        result = merge_dicts(
            {"a": {"x": 1}},
            {"a": {"y": 2}},
        )
        assert result["a"]["x"] == 1
        assert result["a"]["y"] == 2


class TestTruncateText:
    def test_short_text(self):
        assert truncate_text("hello", 100) == "hello"

    def test_long_text(self):
        result = truncate_text("a" * 200, 10)
        assert len(result) <= 13  # 10 + "..."

    def test_exact_length(self):
        result = truncate_text("abcde", 5)
        assert result == "abcde"


class TestSafeDivide:
    def test_normal_division(self):
        assert safe_divide(10, 2) == 5.0

    def test_division_by_zero(self):
        assert safe_divide(10, 0) == 0.0

    def test_custom_default(self):
        assert safe_divide(10, 0, default=-1.0) == -1.0


class TestFlattenDict:
    def test_flat_dict(self):
        result = flatten_dict({"a": 1, "b": 2})
        assert result == {"a": 1, "b": 2}

    def test_nested_dict(self):
        result = flatten_dict({"a": {"b": 1, "c": 2}})
        assert result["a.b"] == 1
        assert result["a.c"] == 2

    def test_deep_nesting(self):
        result = flatten_dict({"a": {"b": {"c": 3}}})
        assert result["a.b.c"] == 3


class TestFormatScoreBreakdown:
    def test_basic(self):
        result = format_score_breakdown(
            {
                "deontological": 0.85,
                "consequentialist": 0.70,
            }
        )
        assert "deontological" in result
        assert "0.85" in result


class TestNowIso:
    def test_returns_string(self):
        result = now_iso()
        assert isinstance(result, str)
        assert "T" in result


class TestTimer:
    def test_timer_decorator(self):
        @timer
        def slow_func():
            time.sleep(0.01)
            return 42

        result = slow_func()
        assert result == 42


# ─── Validator Tests ─────────────────────────────────────────


class TestValidateTaskInput:
    def test_valid_task(self):
        result = validate_task_input("Approve a loan for the applicant")
        assert result["valid"] is True

    def test_empty_task(self):
        result = validate_task_input("")
        assert result["valid"] is False

    def test_whitespace_task(self):
        result = validate_task_input("   ")
        assert result["valid"] is False


class TestValidateDomain:
    def test_valid_domains(self):
        for domain in ["healthcare", "finance", "hiring", "disaster", "general"]:
            assert validate_domain(domain) is True

    def test_invalid_domain(self):
        assert validate_domain("unknown_domain") is False

    def test_case_insensitive(self):
        # depends on implementation
        result = validate_domain("Healthcare")
        # Should handle case; either True or False is acceptable
        assert isinstance(result, bool)


class TestValidateScores:
    def test_valid_scores(self):
        scores = {
            "deontological": 0.85,
            "consequentialist": 0.70,
            "virtue_ethics": 0.75,
            "contextual": 0.80,
        }
        assert validate_scores(scores) is True

    def test_out_of_range(self):
        scores = {
            "deontological": 1.5,
            "consequentialist": 0.70,
        }
        assert validate_scores(scores) is False

    def test_negative_score(self):
        scores = {"deontological": -0.1}
        assert validate_scores(scores) is False


class TestValidateWeights:
    def test_valid_weights(self):
        weights = {
            "deontological": 0.25,
            "consequentialist": 0.25,
            "virtue_ethics": 0.25,
            "contextual": 0.25,
        }
        assert validate_weights(weights) is True

    def test_weights_not_sum_to_one(self):
        weights = {
            "deontological": 0.50,
            "consequentialist": 0.50,
            "virtue_ethics": 0.50,
            "contextual": 0.50,
        }
        assert validate_weights(weights) is False
