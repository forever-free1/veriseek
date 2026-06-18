import argparse
import csv
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt


PALETTE = {
    "base": "#A7AFBA",
    "rl": "#7B9BB8",
    "sft": "#1F4E79",
    "sft_rl": "#2B8C7E",
    "evidence": "#57A99A",
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
    parser = argparse.ArgumentParser(description="Plot VeriSeek public SciFact benchmark summary.")
    parser.add_argument(
        "--source",
        default="assets/veriseek_scifact_benchmark_source.tsv",
        help="TSV source data with public VeriSeek benchmark metrics.",
    )
    parser.add_argument(
        "--output_prefix",
        default="assets/veriseek_scifact_benchmark",
        help="Output path prefix. SVG, PDF, PNG, and TIFF are written.",
    )
    args = parser.parse_args()

    rows = read_rows(Path(args.source))
    labels = ["Base", "RL-only", "SFT", "VeriSeek\nSFT+RL"]
    colors = [PALETTE["base"], PALETTE["rl"], PALETTE["sft"], PALETTE["sft_rl"]]
    answer = [as_float(row, "answer_accuracy") for row in rows]
    evidence = [as_float(row, "evidence_f1") for row in rows]

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

    fig, (ax_a, ax_b) = plt.subplots(1, 2, figsize=(5.8, 3.2), constrained_layout=True)

    x = range(len(labels))
    ax_a.bar(x, answer, color=colors, width=0.64)
    ax_a.set_ylim(0, 0.86)
    ax_a.set_ylabel("Answer accuracy")
    ax_a.set_xticks(list(x), labels)
    ax_a.grid(axis="y", color=PALETTE["grid"], linewidth=0.6, alpha=0.7)
    ax_a.set_title("SciFact answer prediction", loc="left", fontsize=8, fontweight="bold")
    for idx, value in enumerate(answer):
        ax_a.text(idx, value + 0.016, f"{value:.3f}", ha="center", va="bottom", fontsize=6.8)
    ax_a.annotate(
        "best final model",
        xy=(3, answer[3]),
        xytext=(2.15, 0.84),
        arrowprops={"arrowstyle": "-", "linewidth": 0.8, "color": PALETTE["text"]},
        ha="left",
        va="center",
        fontsize=7,
    )

    ax_b.bar(x, evidence, color=colors, width=0.64)
    ax_b.set_ylim(0, 0.46)
    ax_b.set_ylabel("Evidence F1")
    ax_b.set_xticks(list(x), labels)
    ax_b.grid(axis="y", color=PALETTE["grid"], linewidth=0.6, alpha=0.7)
    ax_b.set_title("Verifiable evidence grounding", loc="left", fontsize=8, fontweight="bold")
    for idx, value in enumerate(evidence):
        label = f"{value:.3f}" if value > 0 else "n/a"
        ax_b.text(idx, value + (0.015 if value > 0 else 0.018), label, ha="center", va="bottom", fontsize=6.5)

    for label, ax in zip(["a", "b"], [ax_a, ax_b]):
        ax.text(-0.16, 1.04, label, transform=ax.transAxes, fontweight="bold", fontsize=9)

    fig.suptitle(
        "VeriSeek SFT+RL improves answer accuracy and evidence grounding on SciFact dev (n = 300)",
        x=0.01,
        ha="left",
        fontsize=9.5,
        fontweight="bold",
        color=PALETTE["text"],
    )

    save_publication_figure(fig, Path(args.output_prefix))


if __name__ == "__main__":
    main()
