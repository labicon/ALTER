#!/usr/bin/env python3
"""Render website/README figures from the paper's Tables I–II.

Run from any directory: python scripts/plot_website_results.py
Requires matplotlib; uses no external data services or image generation.
"""
import argparse
import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

ROOT = Path(__file__).resolve().parents[1]
METHODS = ("ALTER", "FT-mixed", "FS", "FT-multi")
COLORS = {"ALTER": "#2463da", "FT-mixed": "#d77a25", "FS": "#16815e", "FT-multi": "#8490a3"}
LABELS = {"ALTER": "ALTER (ours)", "FT-mixed": "FT-mixed", "FS": "From scratch (FS)", "FT-multi": "FT-multi"}
BUDGETS = (20, 40, 60)
METRICS = (("coordination_pct", "Coordination success"), ("combined_source_pct", "Combined source success"))


def load_data(path):
    with path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    data = {}
    for row in rows:
        budget, method = int(row["multi_agent_demos"]), row["method"]
        key = (budget, method)
        if key in data:
            raise ValueError(f"Duplicate result: {key}")
        values = {field: float(row[field]) for field in (
            "coordination_pct", "place_return_pct", "wipe_pct", "combined_source_pct")}
        if not all(0 <= value <= 100 for value in values.values()):
            raise ValueError(f"Success percentage out of bounds: {key}")
        if abs(values["combined_source_pct"] - (values["place_return_pct"] + values["wipe_pct"]) / 2) > 1e-9:
            raise ValueError(f"Combined source score disagrees with equal task weighting: {key}")
        expected_source = 0 if method == "FT-multi" else budget
        if int(row["distilled_source_demos"]) != expected_source:
            raise ValueError(f"Unexpected source-data budget: {key}")
        data[key] = values
    if set(data) != {(budget, method) for budget in BUDGETS for method in METHODS}:
        raise ValueError("Expected all four methods at all three demonstration budgets")
    return data


def draw(data, output, stacked=False):
    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 12,
        "text.color": "#202c43", "axes.labelcolor": "#59657a",
        "xtick.color": "#59657a", "ytick.color": "#59657a",
        "svg.fonttype": "none", "svg.hashsalt": "alter-results",
        "pdf.fonttype": 42,
    })
    if stacked:
        fig, axes = plt.subplots(2, 1, figsize=(6.4, 10))
        fig.subplots_adjust(left=.115, right=.985, top=.84, bottom=.23, hspace=.72)
    else:
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        fig.subplots_adjust(left=.06, right=.985, top=.81, bottom=.32, wspace=.2)
    width = .19
    for ax, (metric, title) in zip(axes, METRICS):
        for index, method in enumerate(METHODS):
            positions = [i + (index - 1.5) * width for i in range(len(BUDGETS))]
            values = [data[(budget, method)][metric] for budget in BUDGETS]
            ax.bar(positions, values, width=width * .91, color=COLORS[method], zorder=3)
            for x, value in zip(positions, values):
                ax.text(x, value + 1.4, f"{value:g}",
                        ha="center", va="bottom",
                        fontsize=11 if stacked else 11.5,
                        color="#202c43",
                        fontweight="bold" if method == "ALTER" else "normal", zorder=4)
        ax.set_title(title, fontsize=16, fontweight="bold", pad=17)
        ax.set_ylim(0, 100)
        ax.set_yticks([0, 20, 40, 60, 80, 100])
        ax.set_xticks(range(len(BUDGETS)), labels=[f"{b} / {b}" for b in BUDGETS])
        ax.set_xlabel("Adaptation demonstrations\n(multi-agent / distilled single-arm*)", labelpad=10)
        ax.set_ylabel("Success (%)", labelpad=7)
        ax.grid(axis="y", color="#e2e8f1", linewidth=.8, zorder=0)
        ax.tick_params(axis="both", length=0, pad=7)
        for spine in ax.spines.values():
            spine.set_visible(False)
    fig.legend([Patch(facecolor=COLORS[m]) for m in METHODS], [LABELS[m] for m in METHODS],
               loc="upper center", bbox_to_anchor=(.52, .99), ncol=2 if stacked else 4,
               frameon=False, fontsize=12, columnspacing=1.5, handlelength=1.3)
    if stacked:
        notes = ("* Paired counts apply to ALTER, FS, and FT-mixed.\n"
                 "FT-multi uses 20 / 0, 40 / 0, and 60 / 0 demonstrations.\n"
                 "Adaptation budgets exclude base-policy pretraining.\n"
                 "Combined source success: mean of place-return and wipe.\n"
                 "Paper Tables I–II; no uncertainty estimates reported.")
        fig.text(.5, .028, notes, ha="center", fontsize=9, color="#59657a", linespacing=1.55)
    else:
        notes = ("* Paired counts apply to ALTER, FS, and FT-mixed. FT-multi uses 20 / 0, 40 / 0, and 60 / 0 demonstrations.\n"
                 "Adaptation budgets exclude base-policy pretraining. Combined source success averages place-return and wipe success.\n"
                 "Paper Tables I–II; reported values shown without uncertainty estimates.")
        fig.text(.5, .045, notes, ha="center", fontsize=10.5, color="#59657a", linespacing=1.7)
    stem = "results-mobile" if stacked else "results"
    formats = ("svg",) if stacked else ("png", "svg", "pdf")
    for extension in formats:
        fig.savefig(output / f"{stem}.{extension}", dpi=200, facecolor="white")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=ROOT / "docs/static/data/simulation-results.csv")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "docs/pictures")
    args = parser.parse_args()
    data = load_data(args.data)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    draw(data, args.output_dir)
    draw(data, args.output_dir, stacked=True)
    print(f"Wrote results.png/.svg/.pdf and results-mobile.svg to {args.output_dir}")


if __name__ == "__main__":
    main()
