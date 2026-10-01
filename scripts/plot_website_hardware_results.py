#!/usr/bin/env python3
"""Render hardware charts from verified trial counts in hardware-results.csv.

Add a complete three-method budget to the CSV to plot another data regime.
Run: python scripts/plot_website_hardware_results.py
"""
import argparse
import csv
from pathlib import Path

from plot_website_results import ROOT, COLORS, LABELS, Patch, plt

METHODS = ("ALTER", "FT-mixed", "FS")
TASKS = ("lid_removal_successes", "lid_replacement_successes", "bird_pick_place_successes")


def load_data(path):
    data = {}
    with path.open(newline="") as handle:
        for row in csv.DictReader(handle):
            method = row["method"]
            values = {key: int(value) for key, value in row.items() if key != "method" and value != ""}
            budget = (values["multi_agent_demos"], values["distilled_source_demos"])
            key = (budget, method)
            if method not in METHODS or key in data or min(budget) <= 0:
                raise ValueError(f"Invalid method, duplicate row, or budget: {key}")
            trials = values["coordination_trials"]
            source_trials = values["combined_source_trials"]
            if trials <= 0 or source_trials <= 0:
                raise ValueError(f"Trial denominators must be positive: {key}")
            if not 0 <= values["coordination_successes"] <= trials:
                raise ValueError(f"Invalid coordination count: {key}")
            if not 0 <= values["combined_source_successes"] <= source_trials:
                raise ValueError(f"Invalid source count: {key}")
            if any(task in values for task in TASKS):
                per_task = values["source_trials_per_task"]
                if not all(0 <= values[task] <= per_task for task in TASKS):
                    raise ValueError(f"Invalid task count: {key}")
                if sum(values[task] for task in TASKS) != values["combined_source_successes"] or 3 * per_task != source_trials:
                    raise ValueError(f"Source totals disagree with task counts: {key}")
            data[key] = (100 * values["coordination_successes"] / trials,
                         100 * values["combined_source_successes"] / source_trials)
    budgets = sorted({budget for budget, _ in data})
    if not budgets or set(data) != {(budget, method) for budget in budgets for method in METHODS}:
        raise ValueError("Each budget must report ALTER, FT-mixed, and FS")
    return data, budgets


def draw(data, budgets, output, stacked=False):
    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 12,
        "text.color": "#202c43", "axes.labelcolor": "#59657a",
        "xtick.color": "#59657a", "ytick.color": "#59657a",
        "svg.fonttype": "none", "svg.hashsalt": "alter-hardware-results", "pdf.fonttype": 42,
    })
    if stacked:
        fig, axes = plt.subplots(2, 1, figsize=(6.4, 10))
        fig.subplots_adjust(left=.115, right=.985, top=.84, bottom=.23, hspace=.72)
    else:
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        fig.subplots_adjust(left=.06, right=.985, top=.81, bottom=.32, wspace=.2)
    width = .22
    for metric, (ax, title) in enumerate(zip(axes, ("Coordination success", "Combined source success"))):
        for index, method in enumerate(METHODS):
            positions = [i + (index - 1) * width for i in range(len(budgets))]
            values = [data[(budget, method)][metric] for budget in budgets]
            ax.bar(positions, values, width=width * .91, color=COLORS[method], zorder=3)
            for x, value in zip(positions, values):
                ax.text(x, value + 1.4, f"{value:.1f}".removesuffix(".0"),
                        ha="center", va="bottom", fontsize=11.5,
                        fontweight="bold" if method == "ALTER" else "normal")
        ax.set_title(title, fontsize=16, fontweight="bold", pad=17)
        ax.set_ylim(0, 100)
        ax.set_xlim(-.6, len(budgets) - .4)
        ax.set_yticks([0, 20, 40, 60, 80, 100])
        ax.set_xticks(range(len(budgets)), labels=[f"{a} / {b}" for a, b in budgets])
        ax.set_xlabel("Adaptation demonstrations\n(multi-agent / distilled single-arm)", labelpad=10)
        ax.set_ylabel("Success (%)", labelpad=7)
        ax.grid(axis="y", color="#e2e8f1", linewidth=.8, zorder=0)
        ax.tick_params(axis="both", length=0, pad=7)
        for spine in ax.spines.values():
            spine.set_visible(False)
    fig.legend([Patch(facecolor=COLORS[m]) for m in METHODS], [LABELS[m] for m in METHODS],
               loc="upper center", bbox_to_anchor=(.52, .99), ncol=2 if stacked else 3,
               frameon=False, fontsize=12, columnspacing=1.5, handlelength=1.3)
    notes = ("Combined source success pools lid removal, lid replacement,\nand bird pick-place trials.\n"
             "Adaptation budgets exclude base-policy pretraining.\n"
             "Sources: paper Table IV and presentation slide 4.\nNo uncertainty estimates reported.") if stacked else (
             "Combined source success pools lid removal, lid replacement, and bird pick-place trials.\n"
             "Adaptation budgets exclude base-policy pretraining.\n"
             "Sources: paper Table IV and presentation slide 4; no uncertainty estimates reported.")
    fig.text(.5, .028 if stacked else .045, notes, ha="center",
             fontsize=9 if stacked else 10.5, color="#59657a", linespacing=1.7)
    stem = "hardware-results-mobile" if stacked else "hardware-results"
    for extension in (("svg",) if stacked else ("png", "svg", "pdf")):
        fig.savefig(output / f"{stem}.{extension}", dpi=200, facecolor="white")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=ROOT / "docs/static/data/hardware-results.csv")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "docs/pictures")
    args = parser.parse_args()
    data, budgets = load_data(args.data)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    draw(data, budgets, args.output_dir)
    draw(data, budgets, args.output_dir, stacked=True)
    print(f"Wrote hardware charts to {args.output_dir}")


if __name__ == "__main__":
    main()
