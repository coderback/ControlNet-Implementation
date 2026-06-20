"""
Render two portfolio figures for the README from the 12k-step run's recorded measurements:

  1. zero-conv weight magnitude vs training step  -> assets/zero_conv_growth.png
     (ControlNet's "sudden convergence" signature: the zero-init control path climbing off ~0
      toward a normal conv's scale as spatial control is learned.)
  2. per-sample Canny edge-recall, control ON vs OFF -> assets/per_sample_recall.png
     (every diagnostic sample improves with control, not just the mean.)

The numbers below are the recorded outputs of scripts/diagnose.py on the full.yaml run
(checkpoints 2k/4k/12k for the trajectory; the 8-sample ON/OFF diagnostic for the dumbbell).

    python scripts/plot_training.py
"""

from _plotstyle import apply_style, clean_axes, OFF_C, ON_C, MUTED, LINE_COLORS, CONNECT
import matplotlib.pyplot as plt

# --- recorded zero-conv mean|w| per checkpoint (scripts/diagnose.py) ---
STEPS = [2000, 4000, 12000]
ZERO_CONV = {
    "down zero-convs": [3.98e-4, 5.58e-4, 1.07e-3],
    "mid zero-conv": [3.47e-4, 4.93e-4, 8.87e-4],
    "cond-embed out": [1.09e-4, 1.40e-4, 1.77e-3],
}
REF_NORMAL_CONV = 3.50e-2  # a trained conv's scale == "converged" reference

# --- recorded per-sample Canny edge-recall (8-sample diagnostic) ---
OFF = [0.430, 0.266, 0.317, 0.282, 0.243, 0.295, 0.539, 0.143]
ON = [0.856, 0.791, 0.905, 0.898, 0.848, 0.779, 0.861, 0.887]


def plot_zero_conv_growth(path="assets/zero_conv_growth.png"):
    apply_style()
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    for (label, vals), color in zip(ZERO_CONV.items(), LINE_COLORS):
        ax.plot(STEPS, vals, marker="o", markersize=6, linewidth=2.2, color=color,
                markeredgecolor="white", markeredgewidth=1.0, label=label)
    ax.axhline(REF_NORMAL_CONV, ls="--", color="#9aa7b1", linewidth=1.4)
    ax.text(STEPS[-1], REF_NORMAL_CONV * 1.12, f"normal conv ≈ {REF_NORMAL_CONV:.0e}  (converged)",
            ha="right", va="bottom", fontsize=8.5, color=MUTED)
    ax.set_yscale("log")
    ax.set_xlabel("training step")
    ax.set_ylabel("mean |weight|  (log scale)")
    ax.set_title("Zero-conv control path climbing as control emerges")
    ax.legend(loc="lower right")
    clean_axes(ax, grid_axis="y")
    fig.tight_layout()
    fig.savefig(path)
    print(f"saved {path}")


def plot_per_sample_recall(path="assets/per_sample_recall.png"):
    apply_style()
    order = sorted(range(len(ON)), key=lambda i: ON[i])  # lowest ON at bottom -> best at top
    y = list(range(len(order)))
    fig, ax = plt.subplots(figsize=(7.6, 4.6))
    for row, i in enumerate(order):
        ax.plot([OFF[i], ON[i]], [row, row], color=CONNECT, linewidth=3.0,
                solid_capstyle="round", zorder=1)
        ax.annotate(f"+{ON[i] - OFF[i]:.2f}", xy=((OFF[i] + ON[i]) / 2, row), xytext=(0, 8),
                    textcoords="offset points", ha="center", fontsize=8, color=MUTED)
    ax.scatter([OFF[i] for i in order], y, s=95, color=OFF_C, edgecolor="white",
               linewidth=1.3, zorder=2, label="control OFF")
    ax.scatter([ON[i] for i in order], y, s=95, color=ON_C, edgecolor="white",
               linewidth=1.3, zorder=3, label="control ON")
    ax.set_yticks(y)
    ax.set_yticklabels([f"sample {i}" for i in order])
    ax.set_ylim(-0.6, len(order) - 0.2)
    ax.set_xlim(0, 1.0)
    ax.set_xlabel("Canny edge-recall  (fraction of conditioning edges reproduced)")
    ax.set_title("Edge-recall improves on every sample with control ON")
    ax.legend(loc="lower left", ncol=1)
    clean_axes(ax, grid_axis="x")
    fig.tight_layout()
    fig.savefig(path)
    print(f"saved {path}")


if __name__ == "__main__":
    plot_zero_conv_growth()
    plot_per_sample_recall()
