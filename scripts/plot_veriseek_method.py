from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch


PALETTE = {
    "base": "#6B7280",
    "sft": "#1F4E79",
    "rl": "#2B8C7E",
    "old": "#C97A5A",
    "paper": "#F8FAFC",
    "edge": "#1F2933",
    "grid": "#D9DEE3",
}


def box(ax, xy, width, height, text, facecolor, edgecolor="#1F2933", fontsize=8):
    patch = FancyBboxPatch(
        xy,
        width,
        height,
        boxstyle="round,pad=0.015,rounding_size=0.025",
        linewidth=0.9,
        edgecolor=edgecolor,
        facecolor=facecolor,
    )
    ax.add_patch(patch)
    ax.text(
        xy[0] + width / 2,
        xy[1] + height / 2,
        text,
        ha="center",
        va="center",
        fontsize=fontsize,
        color="#111827",
    )
    return patch


def arrow(ax, start, end, color="#1F2933", rad=0.0):
    ax.add_patch(
        FancyArrowPatch(
            start,
            end,
            arrowstyle="-|>",
            mutation_scale=10,
            linewidth=0.9,
            color=color,
            connectionstyle=f"arc3,rad={rad}",
        )
    )


def main():
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
            "font.size": 8,
        }
    )

    fig, ax = plt.subplots(figsize=(7.1, 3.1))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    ax.text(
        0.02,
        0.94,
        "VeriSeek training and evaluation loop",
        fontsize=10,
        fontweight="bold",
        ha="left",
        va="center",
        color="#1F2933",
    )

    box(ax, (0.04, 0.56), 0.18, 0.18, "Qwen3-4B\nThinking base", "#E5E7EB")
    box(ax, (0.33, 0.74), 0.18, 0.16, "SFT\nimitation", "#D6E3F3")
    box(ax, (0.33, 0.47), 0.18, 0.16, "RL-only\nreward from base", "#D9EEE9")
    box(ax, (0.33, 0.20), 0.18, 0.16, "VeriSeek\nSFT+RL", "#CBE7E1")

    box(ax, (0.64, 0.58), 0.20, 0.20, "Deterministic\nreward\nanswer + evidence", "#F8FAFC")
    box(ax, (0.64, 0.25), 0.20, 0.20, "Strict SciFact\nevaluation\nXML + evidence F1", "#F8FAFC")

    box(
        ax,
        (0.86, 0.43),
        0.12,
        0.22,
        "Best:\nVeriSeek\nSFT+RL\nAcc 0.793\nEvF1 0.406",
        "#E8F4F1",
        edgecolor=PALETTE["rl"],
        fontsize=7.2,
    )

    arrow(ax, (0.22, 0.65), (0.33, 0.82), PALETTE["sft"])
    arrow(ax, (0.22, 0.65), (0.33, 0.55), PALETTE["rl"])
    arrow(ax, (0.51, 0.82), (0.33, 0.28), PALETTE["rl"], rad=-0.25)
    arrow(ax, (0.51, 0.28), (0.64, 0.66), PALETTE["rl"], rad=0.15)
    arrow(ax, (0.51, 0.55), (0.64, 0.66), PALETTE["rl"], rad=-0.05)
    arrow(ax, (0.74, 0.58), (0.74, 0.45), PALETTE["edge"])
    arrow(ax, (0.84, 0.35), (0.86, 0.54), PALETTE["rl"])

    ax.text(0.30, 0.08, "No trainer rewrite. No tool-protocol change. No LLM judge.", ha="left", fontsize=7.2, color="#4B5563")
    ax.text(0.64, 0.82, "Hard gate on format;\nweak evidence is capped.", ha="left", fontsize=7.1, color="#4B5563")

    out = Path("assets/veriseek_method_overview")
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out.with_suffix(".svg"), bbox_inches="tight")
    fig.savefig(out.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(out.with_suffix(".png"), dpi=300, bbox_inches="tight")


if __name__ == "__main__":
    main()
