import argparse
import csv
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import Patch


PALETTE = {
    "sft": "#6B7280",
    "old": "#C97A5A",
    "gated": "#2B8C7E",
    "gated_light": "#B8DCD6",
    "accent": "#1F4E79",
    "grid": "#D9DEE3",
    "text": "#1F2933",
}


def read_rows(path: Path):
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def as_float(row, key):
    return float(row[key])


def save_publication_figure(fig, output_prefix: Path):
    output_prefix.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_prefix.with_suffix(".svg"), bbox_inches="tight")
    fig.savefig(output_prefix.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(output_prefix.with_suffix(".png"), dpi=300, bbox_inches="tight")
    fig.savefig(output_prefix.with_suffix(".tiff"), dpi=600, bbox_inches="tight")


def main():
    parser = argparse.ArgumentParser(description="Plot VeriSeek SciFact benchmark results.")
    parser.add_argument(
        "--source",
        default="assets/veriseek_scifact_benchmark_source.tsv",
        help="TSV source data with SciFact benchmark metrics.",
    )
    parser.add_argument(
        "--output_prefix",
        default="assets/veriseek_scifact_benchmark",
        help="Output path prefix. SVG, PDF, PNG, and TIFF are written.",
    )
    args = parser.parse_args()

    rows = read_rows(Path(args.source))
    sft = next(row for row in rows if row["training_path"] == "SFT")
    old = next(row for row in rows if row["training_path"] == "SFT+RL old")
    gated = [row for row in rows if row["training_path"] == "SFT+RL gated n=4"]
    gated.sort(key=lambda row: int(row["checkpoint"]))
    best = gated[-1]

    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
            "font.size": 7,
            "axes.spines.right": False,
            "axes.spines.top": False,
            "axes.linewidth": 0.8,
            "legend.frameon": False,
            "xtick.major.size": 2.5,
            "ytick.major.size": 2.5,
        }
    )

    fig = plt.figure(figsize=(7.1, 3.9), constrained_layout=True)
    gs = fig.add_gridspec(2, 3, width_ratios=[1.4, 1.1, 1.0], height_ratios=[1, 1])
    ax_a = fig.add_subplot(gs[:, 0])
    ax_b = fig.add_subplot(gs[0, 1])
    ax_c = fig.add_subplot(gs[1, 1])
    ax_d = fig.add_subplot(gs[:, 2])

    # Panel A: paired endpoint comparison.
    labels = ["SFT", "Old\nSFT+RL", "Gated n=4\nstep 200"]
    acc_values = [as_float(sft, "label_accuracy"), as_float(old, "label_accuracy"), as_float(best, "label_accuracy")]
    ev_values = [as_float(sft, "evidence_f1"), as_float(old, "evidence_f1"), as_float(best, "evidence_f1")]
    x = range(len(labels))
    width = 0.34
    ax_a.bar([i - width / 2 for i in x], acc_values, width, color=PALETTE["accent"], label="Label accuracy")
    ax_a.bar([i + width / 2 for i in x], ev_values, width, color=PALETTE["gated"], label="Evidence F1")
    ax_a.set_ylim(0.25, 0.86)
    ax_a.set_ylabel("SciFact dev score")
    ax_a.set_xticks(list(x), labels)
    ax_a.grid(axis="y", color=PALETTE["grid"], linewidth=0.6, alpha=0.7)
    ax_a.legend(loc="upper left", ncols=1)
    for i, (acc, ev) in enumerate(zip(acc_values, ev_values)):
        ax_a.text(i - width / 2, acc + 0.012, f"{acc:.3f}", ha="center", va="bottom", fontsize=6.5)
        ax_a.text(i + width / 2, ev + 0.012, f"{ev:.3f}", ha="center", va="bottom", fontsize=6.5)

    delta_ev = as_float(best, "evidence_f1") - as_float(sft, "evidence_f1")
    delta_acc = as_float(best, "label_accuracy") - as_float(sft, "label_accuracy")
    ax_a.annotate(
        f"+{delta_acc:.3f} acc\n+{delta_ev:.3f} evidence F1",
        xy=(2, as_float(best, "label_accuracy")),
        xytext=(1.15, 0.84),
        arrowprops={"arrowstyle": "-", "color": PALETTE["text"], "linewidth": 0.7},
        ha="left",
        va="top",
        fontsize=7,
    )

    # Panel B: gated checkpoint trajectory for answer accuracy.
    steps = [int(row["checkpoint"]) for row in gated]
    acc = [as_float(row, "label_accuracy") for row in gated]
    ev = [as_float(row, "evidence_f1") for row in gated]
    ax_b.plot(steps, acc, marker="o", color=PALETTE["accent"], linewidth=1.6, markersize=4)
    ax_b.axhline(as_float(sft, "label_accuracy"), color=PALETTE["sft"], linewidth=1.0, linestyle="--")
    ax_b.set_ylim(0.76, 0.805)
    ax_b.set_ylabel("Label accuracy")
    ax_b.set_xticks(steps)
    ax_b.grid(axis="y", color=PALETTE["grid"], linewidth=0.6, alpha=0.7)
    ax_b.text(51, as_float(sft, "label_accuracy") + 0.001, "SFT baseline", color=PALETTE["sft"], fontsize=6.5)

    # Panel C: gated checkpoint trajectory for evidence F1.
    ax_c.plot(steps, ev, marker="o", color=PALETTE["gated"], linewidth=1.6, markersize=4)
    ax_c.axhline(as_float(sft, "evidence_f1"), color=PALETTE["sft"], linewidth=1.0, linestyle="--")
    ax_c.set_ylim(0.36, 0.415)
    ax_c.set_ylabel("Evidence F1")
    ax_c.set_xlabel("RL step")
    ax_c.set_xticks(steps)
    ax_c.grid(axis="y", color=PALETTE["grid"], linewidth=0.6, alpha=0.7)
    ax_c.text(51, as_float(sft, "evidence_f1") + 0.0015, "SFT baseline", color=PALETTE["sft"], fontsize=6.5)

    # Panel D: unsupported rate moves toward better calibrated evidence use.
    unsupported = [
        as_float(sft, "unsupported_answer_rate"),
        as_float(old, "unsupported_answer_rate"),
        as_float(best, "unsupported_answer_rate"),
    ]
    colors = [PALETTE["sft"], PALETTE["old"], PALETTE["gated"]]
    ax_d.barh(labels, unsupported, color=colors, height=0.55)
    ax_d.invert_yaxis()
    ax_d.set_xlim(0.35, 0.60)
    ax_d.set_xlabel("Unsupported-answer rate")
    ax_d.grid(axis="x", color=PALETTE["grid"], linewidth=0.6, alpha=0.7)
    for y, value in enumerate(unsupported):
        ax_d.text(value + 0.006, y, f"{value:.3f}", va="center", ha="left", fontsize=6.5)
    ax_d.annotate(
        "Lower is better",
        xy=(as_float(best, "unsupported_answer_rate"), 2),
        xytext=(0.505, 2.32),
        arrowprops={"arrowstyle": "-", "color": PALETTE["text"], "linewidth": 0.7},
        ha="left",
        va="center",
        fontsize=6.7,
    )

    for label, ax in zip(["a", "b", "c", "d"], [ax_a, ax_b, ax_c, ax_d]):
        ax.text(-0.16, 1.05, label, transform=ax.transAxes, fontweight="bold", fontsize=9)

    fig.suptitle(
        "Evidence-aware GRPO improves SciFact grounding after SFT",
        x=0.01,
        ha="left",
        fontsize=9,
        fontweight="bold",
        color=PALETTE["text"],
    )

    legend_handles = [
        Patch(facecolor=PALETTE["sft"], label="SFT baseline"),
        Patch(facecolor=PALETTE["old"], label="Old SFT+RL negative control"),
        Patch(facecolor=PALETTE["gated"], label="Gated n=4 SFT+RL"),
    ]
    fig.legend(handles=legend_handles, loc="lower center", bbox_to_anchor=(0.5, -0.03), ncols=3)
    save_publication_figure(fig, Path(args.output_prefix))


if __name__ == "__main__":
    main()
