#!/usr/bin/env python3
"""Regenerate the README's RT-DETRv4/YOLOv9 figures from the numbers recorded in
``docs/finetune-rfdetr-ref.md`` (section "Results — RT-DETRv4 / YOLOv9 fine-tune
+ REAL pair-free filter"). The underlying checkpoints were deleted after that run
per this project's checkpoint-hygiene policy — these numbers are the artifact of
record, hand-transcribed from that doc, not re-computed from a live run.

Requires matplotlib, which is not a project dependency (this script is a rarely
run figure regenerator, not part of the main pipeline):
  uv run --with matplotlib python scripts/generate_multimodel_figures.py
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams["font.size"] = 11

GROUPS = ["RT-DETRv4\nlevel 2", "RT-DETRv4\nlevel 3", "YOLOv9\nlevel 2", "YOLOv9\nlevel 3"]
A_AP = [0.5284, 0.5284, 0.2263, 0.2263]
B_AP = [0.5394, 0.4095, 0.0939, 0.1240]
FILTER_AP = [0.5784, 0.4060, 0.2607, 0.1733]


def ap_comparison():
    x = np.arange(len(GROUPS))
    w = 0.25
    fig, ax = plt.subplots(figsize=(11, 5.2), dpi=120)
    bars = [
        ax.bar(x - w, A_AP, w, label="A' (fine-tuned ceiling)", color="#7f8c8d"),
        ax.bar(x, B_AP, w, label="B (shifted, no filter)", color="#e74c3c"),
        ax.bar(x + w, FILTER_AP, w, label="filter(B) — trained pair-free", color="#2ecc71"),
    ]
    for group in bars:
        for rect in group:
            h = rect.get_height()
            ax.annotate(f"{h:.3f}", (rect.get_x() + rect.get_width() / 2, h),
                        xytext=(0, 3), textcoords="offset points",
                        ha="center", va="bottom", fontsize=9)
    ax.set_ylabel("AP@[.5:.95] (class-agnostic, vs SAM3 pseudo-GT)")
    ax.set_title("Trained pair-free filter vs. A'/B — RT-DETRv4 & YOLOv9 (cooktop, held-out test)")
    ax.set_xticks(x)
    ax.set_xticklabels(GROUPS)
    ax.legend(loc="upper right", framealpha=0.9)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_ylim(0, max(A_AP + B_AP + FILTER_AP) * 1.25)
    fig.tight_layout()
    fig.savefig("docs/images/multimodel_ap_comparison.png")
    print("wrote docs/images/multimodel_ap_comparison.png")


def delta_chart():
    deltas = [f - b for f, b in zip(FILTER_AP, B_AP)]
    colors = ["#2ecc71" if d > 0 else "#e74c3c" for d in deltas]
    fig, ax = plt.subplots(figsize=(9, 5.2), dpi=120)
    bars = ax.bar(GROUPS, deltas, color=colors, width=0.55)
    for rect, d in zip(bars, deltas):
        h = rect.get_height()
        va = "bottom" if h >= 0 else "top"
        offset = 3 if h >= 0 else -3
        ax.annotate(f"{d:+.3f} AP", (rect.get_x() + rect.get_width() / 2, h),
                    xytext=(0, offset), textcoords="offset points",
                    ha="center", va=va, fontsize=10, fontweight="bold")
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_ylabel("filter(B) − B  (AP@[.5:.95])")
    ax.set_title("Does the trained pair-free filter help? (positive = filter helps)")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    fig.savefig("docs/images/multimodel_delta.png")
    print("wrote docs/images/multimodel_delta.png")


if __name__ == "__main__":
    ap_comparison()
    delta_chart()
