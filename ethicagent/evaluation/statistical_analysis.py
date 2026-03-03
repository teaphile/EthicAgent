"""Statistical Analysis — Significance tests and effect-size measures.

Provides statistical rigor for benchmark comparisons:
- Paired t-tests, Wilcoxon signed-rank, McNemar's test
- Cohen's d, Cliff's delta effect sizes
- Confidence intervals and bootstrapping
- Multiple comparison correction (Bonferroni, Holm)

All tests fall back to manual implementations when scipy is not
installed, so the module works in minimal environments.
"""

from __future__ import annotations

import logging
import math
import random
from typing import Any, Dict, List, Optional, Tuple

from ethicagent.utils.helpers import now_iso

logger = logging.getLogger(__name__)


class StatisticalAnalyzer:
    """Statistical analysis for benchmark evaluation.

    Computes significance tests, effect sizes, and confidence
    intervals for comparing EthicAgent against baselines.

    Example::

        analyzer = StatisticalAnalyzer()
        result = analyzer.compare_systems(ea_scores, baseline_scores)
        print(result["significant"])  # True / False
    """

    def __init__(
        self,
        alpha: float = 0.05,
        bootstrap_samples: int = 1000,
    ) -> None:
        self.alpha = alpha
        self.bootstrap_samples = bootstrap_samples

    # ─── Main Comparison ─────────────────────────────────────

    def compare_systems(
        self,
        system_a_scores: List[float],
        system_b_scores: List[float],
        system_a_name: str = "EthicAgent",
        system_b_name: str = "Baseline",
    ) -> Dict[str, Any]:
        """Compare two systems using multiple statistical tests.

        Args:
            system_a_scores: Per-case scores for system A.
            system_b_scores: Per-case scores for system B.
            system_a_name: Display name for system A.
            system_b_name: Display name for system B.

        Returns:
            Comprehensive comparison results.

        Raises:
            ValueError: If score lists differ in length.
        """
        if len(system_a_scores) != len(system_b_scores):
            raise ValueError(
                f"Score lists must have the same length: "
                f"{len(system_a_scores)} != {len(system_b_scores)}"
            )

        n = len(system_a_scores)
        if n < 2:
            return {"error": "Need at least 2 samples for comparison"}

        result: Dict[str, Any] = {
            "system_a": system_a_name,
            "system_b": system_b_name,
            "n": n,
            "descriptive": self._descriptive_stats(
                system_a_scores, system_b_scores,
                system_a_name, system_b_name,
            ),
            "paired_t_test": self._paired_t_test(system_a_scores, system_b_scores),
            "wilcoxon_test": self._wilcoxon_test(system_a_scores, system_b_scores),
            "cohens_d": self._cohens_d(system_a_scores, system_b_scores),
            "cliffs_delta": self._cliffs_delta(system_a_scores, system_b_scores),
            "bootstrap_ci": self._bootstrap_ci(system_a_scores, system_b_scores),
        }

        result["significant"] = (
            result["paired_t_test"].get("p_value", 1.0) < self.alpha
        )
        result["timestamp"] = now_iso()
        return result

    def compare_multiple(
        self,
        system_scores: Dict[str, List[float]],
        reference_system: str = "EthicAgent",
    ) -> Dict[str, Any]:
        """Compare a reference system against several others.

        Applies Bonferroni and Holm corrections for multiple tests.
        """
        if reference_system not in system_scores:
            raise ValueError(f"Reference system '{reference_system}' not found")

        ref = system_scores[reference_system]
        others = [k for k in system_scores if k != reference_system]
        m = len(others)

        result: Dict[str, Any] = {
            "reference": reference_system,
            "n_comparisons": m,
            "alpha": self.alpha,
            "corrected_alpha_bonferroni": self.alpha / max(m, 1),
            "comparisons": {},
        }

        p_values: List[Tuple[str, float]] = []
        for name in others:
            comp = self.compare_systems(ref, system_scores[name], reference_system, name)
            result["comparisons"][name] = comp
            p_values.append((name, comp.get("paired_t_test", {}).get("p_value", 1.0)))

        # Bonferroni
        bonf: Dict[str, Dict[str, Any]] = {}
        for name, p in p_values:
            cp = min(p * m, 1.0)
            bonf[name] = {"original_p": p, "corrected_p": cp, "significant": cp < self.alpha}
        result["bonferroni_correction"] = bonf

        # Holm step-down
        result["holm_correction"] = self._holm_correction(p_values)
        return result

    # ─── Descriptive Statistics ──────────────────────────────

    @staticmethod
    def _descriptive_stats(
        a: List[float], b: List[float],
        name_a: str, name_b: str,
    ) -> Dict[str, Any]:
        def _stats(scores: List[float]) -> Dict[str, float]:
            n = len(scores)
            mean = sum(scores) / n
            var = sum((x - mean) ** 2 for x in scores) / max(n - 1, 1)
            s = sorted(scores)
            median = s[n // 2] if n % 2 else (s[n // 2 - 1] + s[n // 2]) / 2
            return {"mean": mean, "std": math.sqrt(var), "median": median, "min": min(scores), "max": max(scores), "n": n}

        diffs = [x - y for x, y in zip(a, b)]
        return {
            name_a: _stats(a),
            name_b: _stats(b),
            "differences": _stats(diffs),
            "mean_difference": sum(diffs) / len(diffs),
        }

    # ─── Paired t-test ───────────────────────────────────────

    def _paired_t_test(self, a: List[float], b: List[float]) -> Dict[str, Any]:
        try:
            from scipy import stats as sp
            t, p = sp.ttest_rel(a, b)
            return {"t_statistic": float(t), "p_value": float(p), "significant": float(p) < self.alpha, "method": "scipy"}
        except ImportError:
            return self._manual_paired_t(a, b)

    def _manual_paired_t(self, a: List[float], b: List[float]) -> Dict[str, Any]:
        n = len(a)
        d = [x - y for x, y in zip(a, b)]
        md = sum(d) / n
        vd = sum((x - md) ** 2 for x in d) / max(n - 1, 1)
        se = math.sqrt(vd / n) if vd > 0 else 1e-10
        t = md / se
        p = 2 * (1 - self._normal_cdf(abs(t)))
        return {"t_statistic": t, "p_value": p, "significant": p < self.alpha, "method": "manual (normal approx)", "df": n - 1}

    # ─── Wilcoxon ────────────────────────────────────────────

    def _wilcoxon_test(self, a: List[float], b: List[float]) -> Dict[str, Any]:
        try:
            from scipy import stats as sp
            nz = sum(1 for x, y in zip(a, b) if x != y)
            if nz < 10:
                return {"warning": "Too few non-zero differences", "n_non_zero": nz}
            stat, p = sp.wilcoxon(a, b)
            return {"statistic": float(stat), "p_value": float(p), "significant": float(p) < self.alpha, "method": "scipy"}
        except ImportError:
            return {"warning": "scipy not available for Wilcoxon test"}
        except Exception as exc:
            return {"error": str(exc)}

    # ─── Effect Sizes ────────────────────────────────────────

    @staticmethod
    def _cohens_d(a: List[float], b: List[float]) -> Dict[str, Any]:
        n = len(a)
        ma, mb = sum(a) / n, sum(b) / n
        va = sum((x - ma) ** 2 for x in a) / max(n - 1, 1)
        vb = sum((x - mb) ** 2 for x in b) / max(n - 1, 1)
        pooled = math.sqrt((va + vb) / 2)
        d = (ma - mb) / pooled if pooled > 1e-10 else 0.0
        ad = abs(d)
        interp = "negligible" if ad < 0.2 else ("small" if ad < 0.5 else ("medium" if ad < 0.8 else "large"))
        return {"d": round(d, 4), "abs_d": round(ad, 4), "interpretation": interp}

    @staticmethod
    def _cliffs_delta(a: List[float], b: List[float]) -> Dict[str, Any]:
        more = sum(1 for x in a for y in b if x > y)
        less = sum(1 for x in a for y in b if x < y)
        total = len(a) * len(b)
        if total == 0:
            return {"delta": 0.0, "interpretation": "negligible"}

        delta = (more - less) / total
        ad = abs(delta)
        interp = "negligible" if ad < 0.147 else ("small" if ad < 0.33 else ("medium" if ad < 0.474 else "large"))
        return {
            "delta": round(delta, 4), "abs_delta": round(ad, 4),
            "interpretation": interp,
            "more": more, "less": less, "ties": total - more - less,
        }

    # ─── Bootstrap CI ────────────────────────────────────────

    def _bootstrap_ci(
        self, a: List[float], b: List[float], confidence: float = 0.95,
    ) -> Dict[str, Any]:
        n = len(a)
        diffs = [x - y for x, y in zip(a, b)]
        rng = random.Random(42)

        means: List[float] = []
        for _ in range(self.bootstrap_samples):
            sample = [rng.choice(diffs) for _ in range(n)]
            means.append(sum(sample) / n)

        means.sort()
        lo_idx = max(0, int((1 - confidence) / 2 * len(means)))
        hi_idx = min(len(means) - 1, int((1 + confidence) / 2 * len(means)) - 1)

        return {
            "mean_difference": round(sum(diffs) / n, 6),
            "ci_lower": round(means[lo_idx], 6),
            "ci_upper": round(means[hi_idx], 6),
            "confidence_level": confidence,
            "bootstrap_samples": self.bootstrap_samples,
        }

    # ─── Multiple-Comparison Correction ──────────────────────

    def _holm_correction(self, p_values: List[Tuple[str, float]]) -> Dict[str, Dict[str, Any]]:
        m = len(p_values)
        if m == 0:
            return {}

        ranked = sorted(p_values, key=lambda x: x[1])
        out: Dict[str, Dict[str, Any]] = {}
        for i, (name, p) in enumerate(ranked):
            adj_alpha = self.alpha / (m - i)
            cp = min(p * (m - i), 1.0)
            out[name] = {
                "original_p": p, "corrected_p": cp,
                "adjusted_alpha": adj_alpha, "rank": i + 1,
                "significant": p < adj_alpha,
            }
        return out

    # ─── McNemar's Test ──────────────────────────────────────

    def mcnemar_test(
        self,
        correct_a: List[bool],
        correct_b: List[bool],
    ) -> Dict[str, Any]:
        """McNemar's test for comparing classifiers."""
        if len(correct_a) != len(correct_b):
            raise ValueError("Lists must have same length")

        n_both = sum(a and b for a, b in zip(correct_a, correct_b))
        n_a_only = sum(a and not b for a, b in zip(correct_a, correct_b))
        n_b_only = sum(not a and b for a, b in zip(correct_a, correct_b))
        n_neither = sum(not a and not b for a, b in zip(correct_a, correct_b))

        disc = n_a_only + n_b_only
        if disc == 0:
            return {"warning": "No discordant pairs", "chi2": 0.0, "p_value": 1.0, "significant": False}

        chi2 = (abs(n_a_only - n_b_only) - 1) ** 2 / disc
        p = 1 - self._chi2_cdf(chi2, 1)

        return {
            "contingency_table": {
                "both_correct": n_both,
                "a_only_correct": n_a_only,
                "b_only_correct": n_b_only,
                "both_wrong": n_neither,
            },
            "chi2": round(chi2, 4),
            "p_value": round(p, 6),
            "significant": p < self.alpha,
            "method": "McNemar (continuity-corrected)",
        }

    # ─── CDF Approximations ─────────────────────────────────

    @staticmethod
    def _normal_cdf(x: float) -> float:
        """Standard normal CDF — Abramowitz & Stegun 26.2.17."""
        if x < -8:
            return 0.0
        if x > 8:
            return 1.0

        p = 0.2316419
        b1, b2, b3, b4, b5 = 0.319381530, -0.356563782, 1.781477937, -1.821255978, 1.330274429
        t = 1.0 / (1.0 + p * abs(x))
        z = math.exp(-0.5 * x * x) / math.sqrt(2 * math.pi)
        y = 1.0 - z * (b1 * t + b2 * t**2 + b3 * t**3 + b4 * t**4 + b5 * t**5)
        return y if x >= 0 else 1.0 - y

    @staticmethod
    def _chi2_cdf(x: float, df: int) -> float:
        """Chi-squared CDF approximation."""
        if x <= 0:
            return 0.0
        if df == 1:
            return 2 * StatisticalAnalyzer._normal_cdf(math.sqrt(x)) - 1
        z = (x / df) ** (1.0 / 3) - (1 - 2.0 / (9 * df))
        z /= math.sqrt(2.0 / (9 * df))
        return StatisticalAnalyzer._normal_cdf(z)

    # ─── Summary ─────────────────────────────────────────────

    def generate_summary(self, comparison: Dict[str, Any]) -> str:
        """Generate a human-readable summary of statistical analysis."""
        lines: List[str] = ["=" * 60, "STATISTICAL ANALYSIS SUMMARY", "=" * 60]

        if "comparisons" in comparison:
            # Multi-comparison
            ref = comparison.get("reference", "Reference")
            lines.append(f"Reference system: {ref}")
            lines.append(f"Comparisons: {comparison['n_comparisons']}")
            lines.append(
                f"α={comparison['alpha']}, "
                f"Bonferroni-corrected α={comparison['corrected_alpha_bonferroni']:.4f}"
            )
            lines.append("")

            for name, comp in comparison["comparisons"].items():
                lines.append(f"--- vs {name} ---")
                desc = comp.get("descriptive", {})
                lines.append(f"  Mean Δ: {desc.get('mean_difference', 0):.4f}")

                tt = comp.get("paired_t_test", {})
                if "p_value" in tt:
                    sig = "***" if tt["significant"] else "n.s."
                    lines.append(f"  t={tt['t_statistic']:.3f}, p={tt['p_value']:.4f} {sig}")

                cd = comp.get("cohens_d", {})
                if "d" in cd:
                    lines.append(f"  Cohen's d={cd['d']:.3f} ({cd['interpretation']})")
                lines.append("")

            if "bonferroni_correction" in comparison:
                lines.append("Bonferroni-corrected:")
                for name, c in comparison["bonferroni_correction"].items():
                    sig = "sig." if c["significant"] else "n.s."
                    lines.append(f"  {name}: p={c['corrected_p']:.4f} {sig}")

        else:
            # Single comparison
            lines.append(f"Comparing {comparison.get('system_a', 'A')} vs {comparison.get('system_b', 'B')}")
            lines.append(f"N = {comparison.get('n', 0)}")

            desc = comparison.get("descriptive", {})
            lines.append(f"Mean Δ: {desc.get('mean_difference', 0):.4f}")

            tt = comparison.get("paired_t_test", {})
            if tt:
                lines.append(f"t={tt.get('t_statistic', 0):.3f}, p={tt.get('p_value', 1):.4f}")

            cd = comparison.get("cohens_d", {})
            if cd:
                lines.append(f"Cohen's d={cd.get('d', 0):.3f} ({cd.get('interpretation', '—')})")

            ci = comparison.get("bootstrap_ci", {})
            if ci:
                lines.append(f"95% CI: [{ci.get('ci_lower', 0):.4f}, {ci.get('ci_upper', 0):.4f}]")

            sig = comparison.get("significant", False)
            lines.append(f"\nConclusion: {'SIGNIFICANT' if sig else 'NOT SIGNIFICANT'}")

        lines.append("=" * 60)
        return "\n".join(lines)
