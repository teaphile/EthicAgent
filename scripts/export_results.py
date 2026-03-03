#!/usr/bin/env python3
"""Export EthicAgent results in various formats.

Reads raw experiment results and generates publication-ready outputs:
- LaTeX tables for journal papers
- Markdown reports for documentation
- HTML reports for sharing
- JSON for programmatic access
- CSV for spreadsheet analysis

Usage:
    python scripts/export_results.py --input results/raw_results.json
    python scripts/export_results.py --input results/ --format latex markdown
    python scripts/export_results.py --input results/ --format all --output exports/
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import sys
from datetime import datetime
from io import StringIO
from pathlib import Path
from typing import Any

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from ethicagent.evaluation import ReportGenerator

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


def find_latest_results(results_dir: Path) -> Path | None:
    """Find the most recent raw_results.json in the results directory."""
    if results_dir.is_file() and results_dir.suffix == ".json":
        return results_dir

    # Search for raw_results.json in subdirectories
    candidates = sorted(results_dir.glob("**/raw_results.json"), reverse=True)
    if candidates:
        return candidates[0]

    # Search for any JSON file
    candidates = sorted(results_dir.glob("**/*.json"), reverse=True)
    if candidates:
        return candidates[0]

    return None


def load_results(input_path: Path) -> dict[str, Any]:
    """Load experiment results from JSON file."""
    if input_path.is_dir():
        json_file = find_latest_results(input_path)
        if json_file is None:
            logger.error(f"No results files found in {input_path}")
            sys.exit(1)
        input_path = json_file

    logger.info(f"Loading results from: {input_path}")
    with open(input_path) as f:
        data = json.load(f)

    logger.info(f"  Timestamp: {data.get('timestamp', 'N/A')}")
    if "scenario_results" in data:
        domains = list(data["scenario_results"].keys())
        logger.info(f"  Domains: {', '.join(domains)}")
    if "baseline_results" in data:
        baselines = list(data["baseline_results"].keys())
        logger.info(f"  Baselines: {', '.join(baselines)}")
    if "ablation_results" in data:
        variants = list(data["ablation_results"].keys())
        logger.info(f"  Ablation variants: {len(variants)}")

    return data


def export_latex(results: dict, output_dir: Path) -> None:
    """Export results as LaTeX tables."""
    generator = ReportGenerator()
    latex = generator.generate_latex(results)

    filepath = output_dir / "tables.tex"
    filepath.write_text(latex)
    logger.info(f"  LaTeX tables: {filepath}")

    # Also generate individual tables
    if "scenario_results" in results:
        scenario_latex = _generate_scenario_latex(results["scenario_results"])
        (output_dir / "table_scenarios.tex").write_text(scenario_latex)

    if "baseline_results" in results:
        baseline_latex = _generate_baseline_latex(results["baseline_results"])
        (output_dir / "table_baselines.tex").write_text(baseline_latex)

    if "ablation_results" in results:
        ablation_latex = _generate_ablation_latex(results["ablation_results"])
        (output_dir / "table_ablation.tex").write_text(ablation_latex)


def _generate_scenario_latex(scenario_results: dict) -> str:
    """Generate LaTeX table for scenario results."""
    lines = [
        r"\begin{table}[htbp]",
        r"\centering",
        r"\caption{Scenario Evaluation Results by Domain}",
        r"\label{tab:scenario_results}",
        r"\begin{tabular}{lrrr}",
        r"\toprule",
        r"\textbf{Domain} & \textbf{Cases} & \textbf{Correct} & \textbf{Accuracy} \\",
        r"\midrule",
    ]

    total_cases = 0
    total_correct = 0
    for domain, data in sorted(scenario_results.items()):
        cases = data.get("total_cases", data.get("cases", 0))
        correct = data.get("correct", 0)
        accuracy = data.get("accuracy", 0)
        total_cases += cases
        total_correct += correct
        lines.append(
            f"  {domain.title()} & {cases} & {correct} & {accuracy:.1%} \\\\"
        )

    overall = total_correct / total_cases if total_cases > 0 else 0
    lines.extend([
        r"\midrule",
        f"  \\textbf{{Overall}} & \\textbf{{{total_cases}}} & \\textbf{{{total_correct}}} & \\textbf{{{overall:.1%}}} \\\\",
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
    ])
    return "\n".join(lines)


def _generate_baseline_latex(baseline_results: dict) -> str:
    """Generate LaTeX table for baseline comparisons."""
    lines = [
        r"\begin{table}[htbp]",
        r"\centering",
        r"\caption{Baseline Comparison Results}",
        r"\label{tab:baseline_results}",
        r"\begin{tabular}{lr}",
        r"\toprule",
        r"\textbf{System} & \textbf{Accuracy} \\",
        r"\midrule",
    ]

    for system, data in sorted(baseline_results.items()):
        accuracy = data.get("accuracy", 0)
        name = system.replace("_", " ").title()
        if system == "EthicAgent":
            lines.append(f"  \\textbf{{{name}}} & \\textbf{{{accuracy:.1%}}} \\\\")
        else:
            lines.append(f"  {name} & {accuracy:.1%} \\\\")

    lines.extend([
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
    ])
    return "\n".join(lines)


def _generate_ablation_latex(ablation_results: dict) -> str:
    """Generate LaTeX table for ablation study."""
    lines = [
        r"\begin{table}[htbp]",
        r"\centering",
        r"\caption{Ablation Study Results}",
        r"\label{tab:ablation_results}",
        r"\begin{tabular}{lrr}",
        r"\toprule",
        r"\textbf{Variant} & \textbf{Accuracy} & \textbf{$\Delta$} \\",
        r"\midrule",
    ]

    full_acc = ablation_results.get("full_system", {}).get("accuracy", 0)
    for variant, data in sorted(ablation_results.items()):
        accuracy = data.get("accuracy", 0)
        delta = data.get("delta", accuracy - full_acc)
        name = variant.replace("_", r"\_")
        delta_str = f"{delta:+.1%}" if variant != "full_system" else "—"
        if variant == "full_system":
            lines.append(f"  \\textbf{{{name}}} & \\textbf{{{accuracy:.1%}}} & {delta_str} \\\\")
        else:
            lines.append(f"  {name} & {accuracy:.1%} & {delta_str} \\\\")

    lines.extend([
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
    ])
    return "\n".join(lines)


def export_markdown(results: dict, output_dir: Path) -> None:
    """Export results as Markdown report."""
    generator = ReportGenerator()
    markdown = generator.generate_markdown(results)

    filepath = output_dir / "report.md"
    filepath.write_text(markdown)
    logger.info(f"  Markdown report: {filepath}")


def export_html(results: dict, output_dir: Path) -> None:
    """Export results as HTML report."""
    generator = ReportGenerator()
    html = generator.generate_html(results)

    filepath = output_dir / "report.html"
    filepath.write_text(html)
    logger.info(f"  HTML report: {filepath}")


def export_json(results: dict, output_dir: Path) -> None:
    """Export results as formatted JSON."""
    filepath = output_dir / "results.json"
    filepath.write_text(json.dumps(results, indent=2, default=str))
    logger.info(f"  JSON results: {filepath}")


def export_csv(results: dict, output_dir: Path) -> None:
    """Export results as CSV files."""
    # Scenario results CSV
    if "scenario_results" in results:
        filepath = output_dir / "scenarios.csv"
        with open(filepath, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["Domain", "Total Cases", "Correct", "Accuracy"])
            for domain, data in sorted(results["scenario_results"].items()):
                writer.writerow([
                    domain,
                    data.get("total_cases", 0),
                    data.get("correct", 0),
                    f'{data.get("accuracy", 0):.4f}',
                ])
        logger.info(f"  CSV scenarios: {filepath}")

    # Baseline results CSV
    if "baseline_results" in results:
        filepath = output_dir / "baselines.csv"
        with open(filepath, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["System", "Accuracy"])
            for system, data in sorted(results["baseline_results"].items()):
                writer.writerow([system, f'{data.get("accuracy", 0):.4f}'])
        logger.info(f"  CSV baselines: {filepath}")

    # Ablation results CSV
    if "ablation_results" in results:
        filepath = output_dir / "ablation.csv"
        with open(filepath, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["Variant", "Accuracy", "Delta"])
            full_acc = results["ablation_results"].get("full_system", {}).get("accuracy", 0)
            for variant, data in sorted(results["ablation_results"].items()):
                acc = data.get("accuracy", 0)
                delta = data.get("delta", acc - full_acc)
                writer.writerow([variant, f"{acc:.4f}", f"{delta:+.4f}"])
        logger.info(f"  CSV ablation: {filepath}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export EthicAgent results in various formats.",
    )
    parser.add_argument(
        "--input", "-i", type=str, default="results",
        help="Input file or directory containing results. Default: results/",
    )
    parser.add_argument(
        "--output", "-o", type=str, default=None,
        help="Output directory. Default: same as input directory",
    )
    parser.add_argument(
        "--format", "-f", nargs="+",
        choices=["json", "latex", "markdown", "html", "csv", "all"],
        default=["all"],
        help="Export formats. Default: all",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    input_path = PROJECT_ROOT / args.input
    results = load_results(input_path)

    # Determine output directory
    if args.output:
        output_dir = PROJECT_ROOT / args.output
    elif input_path.is_file():
        output_dir = input_path.parent / "exports"
    else:
        output_dir = input_path / "exports"

    output_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Output directory: {output_dir}")

    # Determine formats
    formats = args.format
    if "all" in formats:
        formats = ["json", "latex", "markdown", "html", "csv"]

    # Export
    export_funcs = {
        "json": export_json,
        "latex": export_latex,
        "markdown": export_markdown,
        "html": export_html,
        "csv": export_csv,
    }

    for fmt in formats:
        if fmt in export_funcs:
            try:
                export_funcs[fmt](results, output_dir)
            except Exception as e:
                logger.error(f"  Failed to export {fmt}: {e}")

    logger.info(f"\nExport complete: {len(formats)} formats → {output_dir}")


if __name__ == "__main__":
    main()
