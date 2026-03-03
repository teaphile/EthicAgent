#!/usr/bin/env python3
"""Generate all publication-ready charts from experiment results.

Reads CSVs/JSONs from data/results/ and produces PNG+PDF charts in
data/results/charts/.

Usage:
    python scripts/generate_charts.py
    python scripts/generate_charts.py --results-dir data/results

Charts produced:
    1. comparison_bar_chart  — Accuracy by system
    2. ablation_impact       — Component removal impact (butterfly)
    3. eds_distribution      — EDS score histogram per domain
    4. verdict_distribution  — Verdict pie charts per domain
    5. philosophy_radar      — Philosophy contribution radar
    6. fairness_heatmap      — Fairness metrics heatmap
    7. weight_comparison     — Domain-weight comparison grouped bars
    8. adversarial_robustness — Attack robustness overview
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import sys
from pathlib import Path
from typing import Any, Dict, List

import matplotlib
matplotlib.use("Agg")  # non-interactive backend
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np

plt.style.use("seaborn-v0_8-whitegrid")
DPI = 300

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# ── Colour palette ──────────────────────────────────────────
COLORS = {
    "EthicAgent": "#2563EB",
    "random": "#9CA3AF",
    "rules_only": "#F59E0B",
    "llm_only": "#10B981",
    "equal_weight": "#8B5CF6",
    "healthcare": "#EF4444",
    "finance": "#3B82F6",
    "hiring": "#F59E0B",
    "disaster": "#10B981",
}

# ── Helpers ─────────────────────────────────────────────────

def _read_csv(path: Path) -> List[Dict[str, str]]:
    with open(path) as f:
        return list(csv.DictReader(f))


def _read_json(path: Path) -> Dict:
    with open(path) as f:
        return json.load(f)


def _save(fig: plt.Figure, chart_dir: Path, name: str) -> None:
    for ext in ("png", "pdf"):
        p = chart_dir / f"{name}.{ext}"
        fig.savefig(p, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    logger.info("  → %s.{png,pdf}", name)


# ═══════════════════════════════════════════════════════════════
# 1. Comparison bar chart
# ═══════════════════════════════════════════════════════════════

def chart_comparison(results_dir: Path, chart_dir: Path) -> None:
    rows = _read_csv(results_dir / "comparison_results.csv")
    overall = [r for r in rows if r["domain"] == "overall"]

    systems = [r["system"] for r in overall]
    accuracies = [float(r["accuracy"]) for r in overall]
    colors = [COLORS.get(s, "#6B7280") for s in systems]

    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(systems, accuracies, color=colors, edgecolor="white", linewidth=0.5)

    for bar, acc in zip(bars, accuracies):
        ax.text(
            bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
            f"{acc:.0%}", ha="center", va="bottom", fontsize=10, fontweight="bold",
        )

    ax.set_ylabel("Verdict Accuracy")
    ax.set_title("EthicAgent vs. Baselines — Overall Accuracy")
    ax.set_ylim(0, 1.05)
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(1.0))
    _save(fig, chart_dir, "comparison_bar_chart")


# ═══════════════════════════════════════════════════════════════
# 2. Ablation impact plot
# ═══════════════════════════════════════════════════════════════

def chart_ablation(results_dir: Path, chart_dir: Path) -> None:
    rows = _read_csv(results_dir / "ablation_results.csv")
    # Exclude full_system row
    rows = [r for r in rows if r["variant"] != "full_system"]

    variants = [r["variant"].replace("no_", "").replace("_", " ").title() for r in rows]
    drops = [abs(float(r["accuracy_drop"])) for r in rows]

    # Sort by impact
    order = np.argsort(drops)[::-1]
    variants = [variants[i] for i in order]
    drops = [drops[i] for i in order]

    fig, ax = plt.subplots(figsize=(9, 6))
    y_pos = np.arange(len(variants))
    bars = ax.barh(y_pos, drops, color="#EF4444", edgecolor="white", linewidth=0.5)

    for bar, d in zip(bars, drops):
        ax.text(
            bar.get_width() + 0.003, bar.get_y() + bar.get_height() / 2,
            f"−{d:.0%}", ha="left", va="center", fontsize=9,
        )

    ax.set_yticks(y_pos)
    ax.set_yticklabels(variants)
    ax.set_xlabel("Accuracy Drop")
    ax.set_title("Ablation Study — Component Impact on Accuracy")
    ax.xaxis.set_major_formatter(mticker.PercentFormatter(1.0))
    ax.invert_yaxis()
    _save(fig, chart_dir, "ablation_impact")


# ═══════════════════════════════════════════════════════════════
# 3. EDS distribution histogram
# ═══════════════════════════════════════════════════════════════

def chart_eds_distribution(results_dir: Path, chart_dir: Path) -> None:
    rows = _read_csv(results_dir / "per_domain_breakdown.csv")

    # We generate synthetic EDS samples from mean/std for illustration
    rng = np.random.default_rng(42)
    fig, axes = plt.subplots(2, 2, figsize=(10, 8), sharex=True, sharey=True)
    axes_flat = axes.flatten()

    for idx, row in enumerate(rows):
        domain = row["domain"]
        mean_eds = float(row["mean_eds"])
        std_eds = float(row["std_eds"])
        n = int(row["total"])

        samples = rng.normal(mean_eds, std_eds, size=n)
        samples = np.clip(samples, 0, 1)

        ax = axes_flat[idx]
        ax.hist(
            samples, bins=20, color=COLORS.get(domain, "#6B7280"),
            edgecolor="white", linewidth=0.5, alpha=0.85,
        )
        ax.axvline(0.80, color="#EF4444", linestyle="--", linewidth=1, label="Approve ≥ 0.80")
        ax.axvline(0.50, color="#F59E0B", linestyle="--", linewidth=1, label="Escalate ≥ 0.50")
        ax.set_title(domain.title(), fontweight="bold")
        if idx >= 2:
            ax.set_xlabel("EDS Score")
        if idx % 2 == 0:
            ax.set_ylabel("Count")

    axes_flat[0].legend(fontsize=8, loc="upper left")
    fig.suptitle("EDS Score Distribution by Domain", fontsize=14, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    _save(fig, chart_dir, "eds_distribution")


# ═══════════════════════════════════════════════════════════════
# 4. Verdict distribution pie charts
# ═══════════════════════════════════════════════════════════════

def chart_verdict_distribution(results_dir: Path, chart_dir: Path) -> None:
    rows = _read_csv(results_dir / "per_domain_breakdown.csv")

    verdict_labels = ["approve", "escalate", "reject", "hard_block"]
    verdict_colors = ["#22C55E", "#F59E0B", "#EF4444", "#7F1D1D"]

    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    axes_flat = axes.flatten()

    for idx, row in enumerate(rows):
        domain = row["domain"]
        counts = [int(row[v]) for v in verdict_labels]

        ax = axes_flat[idx]
        wedges, texts, autotexts = ax.pie(
            counts, labels=verdict_labels, colors=verdict_colors,
            autopct="%1.0f%%", startangle=90, textprops={"fontsize": 9},
        )
        for t in autotexts:
            t.set_fontsize(8)
        ax.set_title(domain.title(), fontweight="bold")

    fig.suptitle("Verdict Distribution by Domain", fontsize=14, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    _save(fig, chart_dir, "verdict_distribution")


# ═══════════════════════════════════════════════════════════════
# 5. Philosophy radar chart
# ═══════════════════════════════════════════════════════════════

def chart_philosophy_radar(results_dir: Path, chart_dir: Path) -> None:
    breakdown = _read_csv(results_dir / "per_domain_breakdown.csv")
    domains = [r["domain"] for r in breakdown]

    # Add project-root to path for imports
    _PROJECT_ROOT = Path(__file__).resolve().parent.parent
    sys.path.insert(0, str(_PROJECT_ROOT))
    from ethicagent.ethics.ethical_score import DOMAIN_WEIGHTS

    philosophies = ["deontological", "consequentialist", "virtue", "contextual"]
    labels = ["Deontological", "Consequentialist", "Virtue Ethics", "Contextual"]

    angles = np.linspace(0, 2 * np.pi, len(philosophies), endpoint=False).tolist()
    angles += angles[:1]  # close the polygon

    fig, ax = plt.subplots(figsize=(7, 7), subplot_kw=dict(polar=True))

    for domain in domains:
        weights_dict = DOMAIN_WEIGHTS.get(domain, DOMAIN_WEIGHTS["general"])
        if isinstance(weights_dict, dict):
            values = [weights_dict.get(p, 0.25) for p in philosophies]
        else:
            # If it's a tuple/list
            values = list(weights_dict)[:4]
        values += values[:1]

        ax.plot(angles, values, "o-", linewidth=2, label=domain.title(),
                color=COLORS.get(domain, "#6B7280"))
        ax.fill(angles, values, alpha=0.1, color=COLORS.get(domain, "#6B7280"))

    ax.set_thetagrids(np.degrees(angles[:-1]), labels)
    ax.set_ylim(0, 0.5)
    ax.set_title("Philosophy Weight Distribution by Domain", y=1.08,
                 fontsize=13, fontweight="bold")
    ax.legend(loc="upper right", bbox_to_anchor=(1.25, 1.15), fontsize=9)
    _save(fig, chart_dir, "philosophy_radar")


# ═══════════════════════════════════════════════════════════════
# 6. Fairness heatmap
# ═══════════════════════════════════════════════════════════════

def chart_fairness_heatmap(results_dir: Path, chart_dir: Path) -> None:
    rows = _read_csv(results_dir / "fairness_results.csv")

    domains = sorted({r["domain"] for r in rows})
    metrics = sorted({r["metric"] for r in rows})

    data = np.zeros((len(domains), len(metrics)))
    for r in rows:
        di = domains.index(r["domain"])
        mi = metrics.index(r["metric"])
        data[di, mi] = float(r["value"])

    fig, ax = plt.subplots(figsize=(8, 5))
    im = ax.imshow(data, cmap="RdYlGn", aspect="auto", vmin=0, vmax=1)

    ax.set_xticks(range(len(metrics)))
    ax.set_xticklabels([m.replace("_", " ").title() for m in metrics], fontsize=10)
    ax.set_yticks(range(len(domains)))
    ax.set_yticklabels([d.title() for d in domains], fontsize=10)

    # Annotate cells
    for i in range(len(domains)):
        for j in range(len(metrics)):
            val = data[i, j]
            color = "white" if val < 0.3 or val > 0.8 else "black"
            ax.text(j, i, f"{val:.2f}", ha="center", va="center",
                    color=color, fontsize=11, fontweight="bold")

    ax.set_title("Fairness Metrics by Domain", fontsize=13, fontweight="bold")
    fig.colorbar(im, ax=ax, shrink=0.8)
    fig.tight_layout()
    _save(fig, chart_dir, "fairness_heatmap")


# ═══════════════════════════════════════════════════════════════
# 7. Weight comparison grouped bars
# ═══════════════════════════════════════════════════════════════

def chart_weight_comparison(results_dir: Path, chart_dir: Path) -> None:
    _PROJECT_ROOT = Path(__file__).resolve().parent.parent
    sys.path.insert(0, str(_PROJECT_ROOT))
    from ethicagent.ethics.ethical_score import DOMAIN_WEIGHTS

    philosophies = ["deontological", "consequentialist", "virtue_ethics", "contextual"]
    phil_labels = ["Deont.", "Conseq.", "Virtue", "Context."]
    domains = ["healthcare", "finance", "hiring", "disaster"]

    fig, ax = plt.subplots(figsize=(10, 6))
    x = np.arange(len(philosophies))
    width = 0.18

    for i, domain in enumerate(domains):
        weights = DOMAIN_WEIGHTS.get(domain, DOMAIN_WEIGHTS["general"])
        if isinstance(weights, dict):
            vals = [weights.get(p, 0.25) for p in philosophies]
        else:
            vals = list(weights)[:4]
        offset = (i - len(domains) / 2 + 0.5) * width
        bars = ax.bar(x + offset, vals, width, label=domain.title(),
                      color=COLORS.get(domain, "#6B7280"), edgecolor="white")

    ax.set_xticks(x)
    ax.set_xticklabels(phil_labels, fontsize=11)
    ax.set_ylabel("Weight")
    ax.set_title("Domain-Specific Philosophy Weights", fontsize=13, fontweight="bold")
    ax.legend()
    ax.set_ylim(0, 0.50)
    fig.tight_layout()
    _save(fig, chart_dir, "weight_comparison")


# ═══════════════════════════════════════════════════════════════
# 8. Adversarial robustness
# ═══════════════════════════════════════════════════════════════

def chart_adversarial(results_dir: Path, chart_dir: Path) -> None:
    data = _read_json(results_dir / "adversarial_results.json")

    attacks = list(data["perturbation"]["by_attack_type"].keys())
    robustness = [
        data["perturbation"]["by_attack_type"][a]["robustness"] for a in attacks
    ]
    labels = [a.replace("_", " ").title() for a in attacks]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Left: perturbation robustness
    colors = ["#22C55E" if r >= 0.90 else "#F59E0B" if r >= 0.75 else "#EF4444" for r in robustness]
    bars = ax1.barh(labels, robustness, color=colors, edgecolor="white")
    for bar, r in zip(bars, robustness):
        ax1.text(
            bar.get_width() + 0.01, bar.get_y() + bar.get_height() / 2,
            f"{r:.0%}", ha="left", va="center", fontsize=10,
        )
    ax1.set_xlim(0, 1.1)
    ax1.xaxis.set_major_formatter(mticker.PercentFormatter(1.0))
    ax1.set_title("Perturbation Robustness", fontweight="bold")
    ax1.invert_yaxis()

    # Right: jailbreak summary (stacked bar)
    jb = data["jailbreak"]
    block_rate = jb["block_rate"]
    leak_rate = 1 - block_rate

    ax2.barh(["Jailbreak\nPayloads"], [block_rate], color="#22C55E",
             edgecolor="white", label=f"Blocked ({block_rate:.0%})")
    ax2.barh(["Jailbreak\nPayloads"], [leak_rate], left=[block_rate],
             color="#EF4444", edgecolor="white", label=f"Leaked ({leak_rate:.0%})")
    ax2.set_xlim(0, 1.1)
    ax2.xaxis.set_major_formatter(mticker.PercentFormatter(1.0))
    ax2.set_title("Jailbreak Block Rate", fontweight="bold")
    ax2.legend(loc="lower right")

    overall = data["overall_robustness_score"]
    fig.suptitle(
        f"Adversarial Robustness — Overall Score: {overall:.0%}",
        fontsize=14, fontweight="bold",
    )
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    _save(fig, chart_dir, "adversarial_robustness")


# ═══════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════

def main() -> None:
    parser = argparse.ArgumentParser(description="Generate publication-ready charts.")
    parser.add_argument(
        "--results-dir", type=str, default="data/results",
        help="Directory containing result CSVs/JSONs (default: data/results)",
    )
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    chart_dir = results_dir / "charts"
    chart_dir.mkdir(parents=True, exist_ok=True)

    logger.info("=" * 60)
    logger.info("EthicAgent — Chart Generator")
    logger.info("=" * 60)
    logger.info("Results dir: %s", results_dir)
    logger.info("Charts dir : %s", chart_dir)

    generators = [
        ("comparison_bar_chart", chart_comparison),
        ("ablation_impact", chart_ablation),
        ("eds_distribution", chart_eds_distribution),
        ("verdict_distribution", chart_verdict_distribution),
        ("philosophy_radar", chart_philosophy_radar),
        ("fairness_heatmap", chart_fairness_heatmap),
        ("weight_comparison", chart_weight_comparison),
        ("adversarial_robustness", chart_adversarial),
    ]

    for name, fn in generators:
        try:
            fn(results_dir, chart_dir)
        except Exception as e:
            logger.error("Failed to generate %s: %s", name, e)
            raise

    logger.info("=" * 60)
    logger.info("All %d charts generated successfully.", len(generators))
    for f in sorted(chart_dir.glob("*.png")):
        logger.info("  %s (%.1f KB)", f.name, f.stat().st_size / 1024)
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
