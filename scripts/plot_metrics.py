"""
Render the control ON vs OFF metrics bar chart for the README from the saved eval JSONs.

    python scripts/plot_metrics.py \
        --on outputs/eval_report.json --off outputs/eval_baseline.json \
        --output assets/metrics_comparison.png

Reads the two evaluate.py reports (control scale 1.0 vs 0.0) and plots the three
control-sensitive metrics side by side. FID/CLIP are intentionally left off — CLIP is
flat by design (control shouldn't change text alignment) and FID is unreliable at n=100.
"""

import argparse
import json

from _plotstyle import apply_style, clean_axes, OFF_C, ON_C, MUTED
import matplotlib.pyplot as plt


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--on", default="outputs/eval_report.json")
    p.add_argument("--off", default="outputs/eval_baseline.json")
    p.add_argument("--output", default="assets/metrics_comparison.png")
    args = p.parse_args()

    on = json.load(open(args.on))
    off = json.load(open(args.off))

    labels = ["Canny\nedge recall", "Canny\nedge F1", "Canny\nSSIM"]
    keys = ["canny_edge_recall", "canny_edge_f1", "canny_ssim"]
    on_vals = [on[k] for k in keys]
    off_vals = [off[k] for k in keys]

    apply_style()
    x = range(len(labels))
    w = 0.38
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    b_off = ax.bar([i - w / 2 for i in x], off_vals, w, label="control OFF (scale=0)",
                   color=OFF_C, edgecolor="white", linewidth=1.2)
    b_on = ax.bar([i + w / 2 for i in x], on_vals, w, label="control ON (scale=1)",
                  color=ON_C, edgecolor="white", linewidth=1.2)

    for bars in (b_off, b_on):
        for bar in bars:
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.015,
                    f"{bar.get_height():.2f}", ha="center", va="bottom", fontsize=9, color=MUTED)

    ax.set_xticks(list(x))
    ax.set_xticklabels(labels)
    ax.set_ylim(0, 1.0)
    ax.set_ylabel("score (higher = follows the condition)")
    ax.set_title(f"ControlNet (Canny) — control ON vs OFF · {on['num_samples']} COCO val samples")
    ax.legend(loc="upper right", ncol=2)
    clean_axes(ax, grid_axis="y")
    fig.tight_layout()
    fig.savefig(args.output)
    print(f"saved {args.output}")


if __name__ == "__main__":
    main()
