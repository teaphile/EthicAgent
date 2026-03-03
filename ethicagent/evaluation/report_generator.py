"""Report Generator — LaTeX, Markdown, and HTML reporting.

Produces publication-ready tables, figures, and report sections
from benchmark results for inclusion in research papers.

The key names used here must match the output of
``compute_all_metrics()`` in the metrics module exactly:
- ``verdict_accuracy`` → ``overall_accuracy``
- ``eds_score_metrics`` → ``mean``, ``std``
- ``eds_range_accuracy`` → ``in_range_rate``
- ``consistency`` → ``consistency``
- ``philosophy_contributions`` → per-philosophy dicts
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

from ethicagent.utils.helpers import now_iso

logger = logging.getLogger(__name__)


class ReportGenerator:
    """Generate formatted reports from benchmark results.

    Supports LaTeX (for paper), Markdown (for README), and
    HTML (for web) output formats.
    """

    def __init__(
        self,
        results: Optional[Dict[str, Any]] = None,
        output_dir: str = "reports",
    ) -> None:
        self.results = results or {}
        self.output_dir = output_dir

    def set_results(self, results: Dict[str, Any]) -> None:
        self.results = results

    # ─── Full Report ─────────────────────────────────────────

    def generate_full_report(
        self,
        formats: Optional[List[str]] = None,
    ) -> Dict[str, str]:
        """Generate reports in all requested formats.

        Args:
            formats: ``['latex', 'markdown', 'html']`` (default: all).

        Returns:
            Dict mapping format name to filepath.
        """
        formats = formats or ["latex", "markdown", "html"]
        os.makedirs(self.output_dir, exist_ok=True)

        generated: Dict[str, str] = {}

        for fmt in formats:
            if fmt == "latex":
                path = os.path.join(self.output_dir, "results.tex")
                content = self.generate_latex()
            elif fmt == "markdown":
                path = os.path.join(self.output_dir, "results.md")
                content = self.generate_markdown()
            elif fmt == "html":
                path = os.path.join(self.output_dir, "results.html")
                content = self.generate_html()
            else:
                logger.warning("Unknown format: %s", fmt)
                continue

            with open(path, "w") as f:
                f.write(content)
            generated[fmt] = path
            logger.info("%s report → %s", fmt.title(), path)

        # Also dump raw JSON
        json_path = os.path.join(self.output_dir, "results.json")
        with open(json_path, "w") as f:
            json.dump(self.results, f, indent=2, default=str)
        generated["json"] = json_path

        return generated

    # ── Metric accessors (keep report code DRY) ──────────────

    def _ea(self) -> Dict[str, Any]:
        """Return the EthicAgent aggregate metrics dict."""
        return self.results.get("aggregate_metrics", {}).get("ethicagent", {})

    def _va(self, metrics: Optional[Dict] = None) -> float:
        m = metrics or self._ea()
        return m.get("verdict_accuracy", {}).get("overall_accuracy", 0.0)

    def _eds_mean(self, metrics: Optional[Dict] = None) -> float:
        m = metrics or self._ea()
        return m.get("eds_score_metrics", {}).get("mean", 0.0)

    def _eds_std(self, metrics: Optional[Dict] = None) -> float:
        m = metrics or self._ea()
        return m.get("eds_score_metrics", {}).get("std", 0.0)

    def _range_acc(self, metrics: Optional[Dict] = None) -> float:
        m = metrics or self._ea()
        return m.get("eds_range_accuracy", {}).get("in_range_rate", 0.0)

    def _consistency(self, metrics: Optional[Dict] = None) -> float:
        m = metrics or self._ea()
        return m.get("consistency", {}).get("consistency", 0.0)

    # ─── LaTeX ───────────────────────────────────────────────

    def generate_latex(self) -> str:
        parts: List[str] = [
            f"% EthicAgent Evaluation Results",
            f"% Generated: {now_iso()}\n",
            self._latex_main_table(),
            "",
            self._latex_scenario_table(),
            "",
            self._latex_baseline_table(),
            "",
            self._latex_ablation_table(),
            "",
            self._latex_philosophy_table(),
        ]
        return "\n".join(parts)

    def _latex_main_table(self) -> str:
        acc = self._va()
        mean = self._eds_mean()
        std = self._eds_std()
        ra = self._range_acc()
        cons = self._consistency()

        return "\n".join([
            r"\begin{table}[htbp]",
            r"\centering",
            r"\caption{Overall EthicAgent Performance}",
            r"\label{tab:main_results}",
            r"\begin{tabular}{lc}",
            r"\toprule",
            r"\textbf{Metric} & \textbf{Value} \\",
            r"\midrule",
            f"Verdict Accuracy & {acc:.1%} \\\\",
            f"Mean EDS & ${mean:.3f} \\pm {std:.3f}$ \\\\",
            f"EDS Range Accuracy & {ra:.1%} \\\\",
            f"Consistency & {cons:.3f} \\\\",
            r"\bottomrule",
            r"\end{tabular}",
            r"\end{table}",
        ])

    def _latex_scenario_table(self) -> str:
        sr = self.results.get("scenario_results", {})
        if not sr:
            return "% No scenario results available"

        rows: List[str] = []
        for name, data in sr.items():
            m = data.get("metrics", {})
            acc = self._va(m)
            mean = self._eds_mean(m)
            n = data.get("total_cases", 0)
            clean = name.replace("_", " ").replace("Scenario", "").strip()
            rows.append(f"{clean} & {acc:.1%} & {mean:.3f} & {n} \\\\")

        return "\n".join([
            r"\begin{table}[htbp]",
            r"\centering",
            r"\caption{Per-Scenario Results}",
            r"\label{tab:scenario_results}",
            r"\begin{tabular}{lccc}",
            r"\toprule",
            r"\textbf{Scenario} & \textbf{Accuracy} & \textbf{Mean EDS} & \textbf{Cases} \\",
            r"\midrule",
            *rows,
            r"\bottomrule",
            r"\end{tabular}",
            r"\end{table}",
        ])

    def _latex_baseline_table(self) -> str:
        agg = self.results.get("aggregate_metrics", {})
        imp = agg.get("improvement_over_baselines", {})
        if not imp:
            return "% No baseline comparison available"

        ea_acc = self._va()
        ea_eds = self._eds_mean()

        rows: List[str] = []
        for name in sorted(imp.keys()):
            bl = agg.get(name, {})
            bl_acc = self._va(bl)
            bl_eds = self._eds_mean(bl)
            iv = imp[name]
            iv_str = f"+{iv:.1f}\\%" if iv != float("inf") else "---"
            clean = name.replace("_", " ").title()
            rows.append(f"{clean} & {bl_acc:.1%} & {bl_eds:.3f} & {iv_str} \\\\")

        return "\n".join([
            r"\begin{table}[htbp]",
            r"\centering",
            r"\caption{Comparison with Baselines}",
            r"\label{tab:baseline_comparison}",
            r"\begin{tabular}{lccc}",
            r"\toprule",
            r"\textbf{Method} & \textbf{Accuracy} & \textbf{Mean EDS} & \textbf{Improvement} \\",
            r"\midrule",
            f"\\textbf{{EthicAgent}} & \\textbf{{{ea_acc:.1%}}} & \\textbf{{{ea_eds:.3f}}} & --- \\\\",
            r"\midrule",
            *rows,
            r"\bottomrule",
            r"\end{tabular}",
            r"\end{table}",
        ])

    def _latex_ablation_table(self) -> str:
        variants = self.results.get("ablation_results", {}).get("variants", {})
        if not variants:
            return "% No ablation results available"

        full_acc = self._va(
            variants.get("full_system", {}).get("metrics", {})
        )

        rows: List[str] = []
        for name, data in sorted(variants.items()):
            if name == "full_system" or "error" in data:
                continue
            acc = self._va(data.get("metrics", {}))
            delta = acc - full_acc
            clean = name.replace("no_", "w/o ").replace("_", " ").title()
            rows.append(f"{clean} & {acc:.1%} & {delta:+.1%} \\\\")

        return "\n".join([
            r"\begin{table}[htbp]",
            r"\centering",
            r"\caption{Ablation Study Results}",
            r"\label{tab:ablation}",
            r"\begin{tabular}{lcc}",
            r"\toprule",
            r"\textbf{Variant} & \textbf{Accuracy} & \textbf{$\Delta$ Accuracy} \\",
            r"\midrule",
            f"Full System & {full_acc:.1%} & --- \\\\",
            r"\midrule",
            *rows,
            r"\bottomrule",
            r"\end{tabular}",
            r"\end{table}",
        ])

    def _latex_philosophy_table(self) -> str:
        phil = self._ea().get("philosophy_contributions", {})
        if not phil:
            return "% No philosophy contribution data available"

        rows: List[str] = []
        for pname, pdata in phil.items():
            m = pdata.get("mean", 0)
            s = pdata.get("std", 0)
            clean = pname.replace("_", " ").title()
            rows.append(f"{clean} & {m:.3f} & {s:.3f} \\\\")

        return "\n".join([
            r"\begin{table}[htbp]",
            r"\centering",
            r"\caption{Philosophy Module Contributions}",
            r"\label{tab:philosophy}",
            r"\begin{tabular}{lcc}",
            r"\toprule",
            r"\textbf{Philosophy} & \textbf{Mean Score} & \textbf{Std Dev} \\",
            r"\midrule",
            *rows,
            r"\bottomrule",
            r"\end{tabular}",
            r"\end{table}",
        ])

    # ─── Markdown ────────────────────────────────────────────

    def generate_markdown(self) -> str:
        parts: List[str] = [
            "# EthicAgent Evaluation Results",
            f"\n*Generated: {now_iso()}*\n",
            "## Overall Performance\n",
            self._md_main_table(),
            "\n## Per-Scenario Results\n",
            self._md_scenario_table(),
            "\n## Baseline Comparison\n",
            self._md_baseline_table(),
            "\n## Ablation Study\n",
            self._md_ablation_table(),
        ]
        return "\n".join(parts)

    def _md_main_table(self) -> str:
        return "\n".join([
            "| Metric | Value |",
            "|--------|-------|",
            f"| Verdict Accuracy | {self._va():.1%} |",
            f"| Mean EDS | {self._eds_mean():.3f} ± {self._eds_std():.3f} |",
            f"| EDS Range Accuracy | {self._range_acc():.1%} |",
            f"| Consistency | {self._consistency():.3f} |",
        ])

    def _md_scenario_table(self) -> str:
        sr = self.results.get("scenario_results", {})
        if not sr:
            return "*No scenario results available*"

        lines = [
            "| Scenario | Accuracy | Mean EDS | Cases |",
            "|----------|----------|----------|-------|",
        ]
        for name, data in sr.items():
            m = data.get("metrics", {})
            clean = name.replace("Scenario", "").replace("_", " ").strip()
            lines.append(
                f"| {clean} | {self._va(m):.1%} | "
                f"{self._eds_mean(m):.3f} | {data.get('total_cases', 0)} |"
            )
        return "\n".join(lines)

    def _md_baseline_table(self) -> str:
        agg = self.results.get("aggregate_metrics", {})
        imp = agg.get("improvement_over_baselines", {})
        if not imp:
            return "*No baseline results available*"

        lines = [
            "| Method | Accuracy | Improvement |",
            "|--------|----------|-------------|",
            f"| **EthicAgent** | **{self._va():.1%}** | — |",
        ]
        for name in sorted(imp.keys()):
            bl = agg.get(name, {})
            acc = self._va(bl)
            iv = imp[name]
            iv_str = f"+{iv:.1f}%" if iv != float("inf") else "—"
            clean = name.replace("_", " ").title()
            lines.append(f"| {clean} | {acc:.1%} | {iv_str} |")
        return "\n".join(lines)

    def _md_ablation_table(self) -> str:
        variants = self.results.get("ablation_results", {}).get("variants", {})
        if not variants:
            return "*No ablation results available*"

        full_acc = self._va(
            variants.get("full_system", {}).get("metrics", {})
        )
        lines = [
            "| Variant | Accuracy | Δ Accuracy |",
            "|---------|----------|------------|",
            f"| Full System | {full_acc:.1%} | — |",
        ]
        for name, data in sorted(variants.items()):
            if name == "full_system" or "error" in data:
                continue
            acc = self._va(data.get("metrics", {}))
            delta = acc - full_acc
            clean = name.replace("no_", "w/o ").replace("_", " ").title()
            lines.append(f"| {clean} | {acc:.1%} | {delta:+.1%} |")
        return "\n".join(lines)

    # ─── HTML ────────────────────────────────────────────────

    def generate_html(self) -> str:
        md = self.generate_markdown()
        body = self._md_to_html(md)

        return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>EthicAgent Evaluation Results</title>
  <style>
    body {{
      font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
      max-width: 900px; margin: 0 auto; padding: 2rem;
      color: #333; line-height: 1.6;
    }}
    h1 {{ color: #1a5276; border-bottom: 2px solid #1a5276; padding-bottom: .5rem; }}
    h2 {{ color: #2c3e50; margin-top: 2rem; }}
    table {{ border-collapse: collapse; width: 100%; margin: 1rem 0; }}
    th, td {{ border: 1px solid #ddd; padding: 8px 12px; text-align: left; }}
    th {{ background: #f2f2f2; font-weight: 600; }}
    tr:nth-child(even) {{ background: #f9f9f9; }}
    tr:hover {{ background: #e8f4fd; }}
    footer {{ margin-top: 3rem; color: #999; font-size: .9rem; }}
  </style>
</head>
<body>
{body}
<footer><p>Generated by EthicAgent Report Generator — {now_iso()}</p></footer>
</body>
</html>"""

    @staticmethod
    def _md_to_html(md: str) -> str:
        """Minimal Markdown-to-HTML converter (headers + tables)."""
        lines = md.split("\n")
        out: List[str] = []
        in_table = False

        for line in lines:
            line = line.strip()
            if not line:
                if in_table:
                    out.append("</tbody></table>")
                    in_table = False
                out.append("")
                continue

            if line.startswith("# "):
                out.append(f"<h1>{line[2:]}</h1>")
            elif line.startswith("## "):
                out.append(f"<h2>{line[3:]}</h2>")
            elif line.startswith("*") and line.endswith("*"):
                out.append(f"<p><em>{line.strip('*')}</em></p>")
            elif line.startswith("|"):
                cells = [c.strip() for c in line.split("|")[1:-1]]
                # Skip separator rows
                if all(set(c) <= {"-", ":", " "} for c in cells if c):
                    continue
                if not in_table:
                    out.append("<table><thead><tr>")
                    for c in cells:
                        out.append(f"  <th>{c.replace('**', '')}</th>")
                    out.append("</tr></thead><tbody>")
                    in_table = True
                else:
                    out.append("<tr>")
                    for c in cells:
                        if c.startswith("**") and c.endswith("**"):
                            c = f"<strong>{c[2:-2]}</strong>"
                        out.append(f"  <td>{c}</td>")
                    out.append("</tr>")
            else:
                out.append(f"<p>{line}</p>")

        if in_table:
            out.append("</tbody></table>")
        return "\n".join(out)
