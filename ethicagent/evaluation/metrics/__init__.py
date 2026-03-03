"""Evaluation Metrics — Core metrics for ethical decision evaluation.

Provides accuracy, fairness, consistency, reliability, and
composite metrics for quantitative evaluation of the EthicAgent
framework.

Every function accepts a flat ``List[Dict[str, Any]]`` of result
records so callers don't need to know about our internal types.
"""

from __future__ import annotations

import logging
import math
from collections import Counter
from typing import Any

logger = logging.getLogger(__name__)


# ── Verdict Accuracy ─────────────────────────────────────────


def verdict_accuracy(
    results: list[dict[str, Any]],
) -> dict[str, Any]:
    """Compute verdict accuracy metrics.

    Args:
        results: List of result dicts with ``actual_verdict`` and
            ``expected_verdict`` keys.

    Returns:
        Dictionary with overall_accuracy, per-verdict precision/recall/f1.
    """
    if not results:
        return {"overall_accuracy": 0.0, "total": 0}

    labeled = [r for r in results if r.get("expected_verdict")]
    if not labeled:
        return {"overall_accuracy": 0.0, "note": "No labeled cases", "total": 0}

    correct = sum(
        1 for r in labeled if r.get("actual_verdict", "").lower() == r["expected_verdict"].lower()
    )
    total = len(labeled)

    verdicts = set(r["expected_verdict"].lower() for r in labeled)
    per_verdict: dict[str, dict[str, float]] = {}

    for v in sorted(verdicts):
        tp = sum(
            1
            for r in labeled
            if r.get("actual_verdict", "").lower() == v and r["expected_verdict"].lower() == v
        )
        fp = sum(
            1
            for r in labeled
            if r.get("actual_verdict", "").lower() == v and r["expected_verdict"].lower() != v
        )
        fn = sum(
            1
            for r in labeled
            if r.get("actual_verdict", "").lower() != v and r["expected_verdict"].lower() == v
        )

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0

        per_verdict[v] = {
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1": round(f1, 4),
            "support": tp + fn,
        }

    return {
        "overall_accuracy": round(correct / total, 4),
        "correct": correct,
        "total": total,
        "per_verdict": per_verdict,
    }


# ── EDS Score Distribution ───────────────────────────────────


def eds_score_metrics(
    results: list[dict[str, Any]],
) -> dict[str, Any]:
    """Compute EDS score distribution metrics.

    Returns mean, std, min, max, quartiles, and histogram bin
    counts for quick plotting.
    """
    scores = [r["eds_score"] for r in results if "eds_score" in r]
    if not scores:
        return {"mean": 0.0, "count": 0}

    n = len(scores)
    mean = sum(scores) / n
    variance = sum((s - mean) ** 2 for s in scores) / max(n - 1, 1)
    std = math.sqrt(variance)

    sorted_scores = sorted(scores)

    def _percentile(data: list[float], p: float) -> float:
        k = (len(data) - 1) * p
        f = int(k)
        c = f + 1 if f + 1 < len(data) else f
        return data[f] + (data[c] - data[f]) * (k - f)

    # Histogram bins for quick dashboard rendering
    bins = [0.0, 0.2, 0.4, 0.5, 0.6, 0.8, 1.01]
    histogram: dict[str, int] = {}
    for lo, hi in zip(bins, bins[1:], strict=False):
        label = f"{lo:.1f}-{hi:.2f}"
        histogram[label] = sum(1 for s in scores if lo <= s < hi)

    return {
        "count": n,
        "mean": round(mean, 4),
        "std": round(std, 4),
        "min": round(sorted_scores[0], 4),
        "max": round(sorted_scores[-1], 4),
        "q25": round(_percentile(sorted_scores, 0.25), 4),
        "median": round(_percentile(sorted_scores, 0.50), 4),
        "q75": round(_percentile(sorted_scores, 0.75), 4),
        "histogram": histogram,
    }


# ── EDS Range Accuracy ───────────────────────────────────────


def eds_range_accuracy(
    results: list[dict[str, Any]],
) -> dict[str, Any]:
    """How often does the EDS score fall within the expected range?

    Also breaks results down by difficulty level.
    """
    labeled = [r for r in results if r.get("expected_eds_range") and "eds_score" in r]
    if not labeled:
        return {"in_range_rate": 0.0, "note": "No range labels", "total": 0}

    in_range = 0
    by_difficulty: dict[str, dict[str, int]] = {}

    for r in labeled:
        eds = r["eds_score"]
        lo, hi = r["expected_eds_range"]
        hit = lo <= eds <= hi

        if hit:
            in_range += 1

        diff = r.get("difficulty", "unknown")
        by_difficulty.setdefault(diff, {"in_range": 0, "total": 0})
        by_difficulty[diff]["total"] += 1
        if hit:
            by_difficulty[diff]["in_range"] += 1

    per_difficulty = {
        diff: round(d["in_range"] / d["total"], 4) if d["total"] else 0.0
        for diff, d in by_difficulty.items()
    }

    return {
        "in_range_rate": round(in_range / len(labeled), 4),
        "in_range": in_range,
        "total": len(labeled),
        "per_difficulty": per_difficulty,
    }


# ── Fairness Metrics ─────────────────────────────────────────


def fairness_metrics(
    results: list[dict[str, Any]],
    group_key: str = "domain",
) -> dict[str, Any]:
    """Compute fairness metrics across groups.

    Calculates statistical parity difference and disparate-impact
    ratio across a grouping key.
    """
    groups: dict[str, list[dict]] = {}
    for r in results:
        g = r.get(group_key, "unknown")
        groups.setdefault(g, []).append(r)

    approve_set = {"approve", "approved"}
    reject_set = {"reject", "rejected", "hard_block", "hard_blocked"}

    group_approval: dict[str, float] = {}
    group_rejection: dict[str, float] = {}

    for g, g_results in groups.items():
        total = len(g_results)
        approved = sum(1 for r in g_results if r.get("actual_verdict", "").lower() in approve_set)
        rejected = sum(1 for r in g_results if r.get("actual_verdict", "").lower() in reject_set)

        group_approval[g] = round(approved / total, 4) if total else 0.0
        group_rejection[g] = round(rejected / total, 4) if total else 0.0

    rates = list(group_approval.values())
    spd = round(max(rates) - min(rates), 4) if rates else 0.0
    dir_val = round(min(rates) / max(rates), 4) if rates and max(rates) > 0 else 0.0

    # Four-fifths rule check (DIR >= 0.8 is fair)
    four_fifths_pass = dir_val >= 0.8 if rates else True

    return {
        "group_approval_rates": group_approval,
        "group_rejection_rates": group_rejection,
        "statistical_parity_difference": spd,
        "disparate_impact_ratio": dir_val,
        "four_fifths_rule_pass": four_fifths_pass,
        "groups_analyzed": sorted(groups.keys()),
    }


# ── Consistency Score ────────────────────────────────────────


def consistency_score(
    results: list[dict[str, Any]],
) -> dict[str, Any]:
    """Measure decision consistency for similar inputs.

    Groups results into EDS score bands and checks if the same
    band always produces the same verdict.  Also computes a
    score-spread consistency (1 − normalised stddev) so that
    widely varying EDS scores pull the value down.
    """
    if len(results) < 2:
        return {"consistency": 1.0, "note": "Insufficient data"}

    bands = [(0.0, 0.5), (0.5, 0.8), (0.8, 1.01)]
    consistent = 0
    total = 0

    for lo, hi in bands:
        band_results = [r for r in results if lo <= r.get("eds_score", 0.0) < hi]
        if len(band_results) < 2:
            continue

        counts = Counter(r.get("actual_verdict", "") for r in band_results)
        most_common_count = counts.most_common(1)[0][1]
        consistent += most_common_count
        total += len(band_results)

    band_consistency = round(consistent / total, 4) if total > 0 else None

    # Score-spread consistency: 1 - 2*stddev of EDS scores (clamped 0-1)
    all_eds = [r.get("eds_score", 0.0) for r in results]
    n = len(all_eds)
    mean_eds = sum(all_eds) / n
    variance = sum((s - mean_eds) ** 2 for s in all_eds) / max(n - 1, 1)
    stddev = variance**0.5
    spread_consistency = max(0.0, min(1.0, 1.0 - 2 * stddev))

    # Combine: prefer band_consistency when available, blend with spread
    if band_consistency is not None:
        final = round((band_consistency + spread_consistency) / 2, 4)
    else:
        final = round(spread_consistency, 4)

    return {
        "consistency": final,
        "consistent_decisions": consistent,
        "total_compared": total,
    }


# ── Philosophy Contribution Analysis ─────────────────────────


def philosophy_contribution_analysis(
    results: list[dict[str, Any]],
) -> dict[str, Any]:
    """Analyse each philosophy's contribution to final decisions.

    For each of the four lenses, reports mean / std / min / max
    and how scores differ between approved and rejected cases.
    """
    philosophies = ["deontological", "consequentialist", "virtue_ethics", "contextual"]
    approve_set = {"approve", "approved"}
    reject_set = {"reject", "rejected", "hard_block", "hard_blocked"}
    analysis: dict[str, dict[str, Any]] = {}

    for phil in philosophies:
        scores = [
            r.get("philosophy_scores", {}).get(phil, 0.0)
            for r in results
            if phil in r.get("philosophy_scores", {})
        ]
        if not scores:
            continue

        n = len(scores)
        mean = sum(scores) / n
        analysis[phil] = {
            "mean": round(mean, 4),
            "min": round(min(scores), 4),
            "max": round(max(scores), 4),
            "std": round(math.sqrt(sum((s - mean) ** 2 for s in scores) / max(n - 1, 1)), 4),
            "n": n,
        }

        # Split by verdict bucket
        for label, vset in [("approved_mean", approve_set), ("rejected_mean", reject_set)]:
            bucket = [
                r.get("philosophy_scores", {}).get(phil, 0.0)
                for r in results
                if r.get("actual_verdict", "").lower() in vset
                and phil in r.get("philosophy_scores", {})
            ]
            analysis[phil][label] = round(sum(bucket) / len(bucket), 4) if bucket else 0.0

    return analysis


# ── Composite Metric ─────────────────────────────────────────


def compute_all_metrics(
    results: list[dict[str, Any]],
) -> dict[str, Any]:
    """Compute every metric in one call.

    This is the main entry point called by the benchmark runner.
    """
    return {
        "verdict_accuracy": verdict_accuracy(results),
        "eds_score_metrics": eds_score_metrics(results),
        "eds_range_accuracy": eds_range_accuracy(results),
        "fairness": fairness_metrics(results),
        "consistency": consistency_score(results),
        "philosophy_contributions": philosophy_contribution_analysis(results),
        "total_cases": len(results),
    }
