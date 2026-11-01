"""Shared chart styling and a simple loss-curve chart."""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
INK_SECONDARY = "#52514e"
MUTED = "#898781"

plt.rcParams.update(
    {
        "font.family": ["Segoe UI", "DejaVu Sans"],
        "figure.facecolor": "#fcfcfb",
        "axes.facecolor": "#fcfcfb",
        "axes.edgecolor": "#c3c2b7",
        "axes.labelcolor": INK_SECONDARY,
        "axes.titlelocation": "left",
        "axes.titlesize": 12,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "xtick.color": INK_SECONDARY,
        "ytick.color": INK_SECONDARY,
        "grid.color": "#e1e0d9",
        "grid.linewidth": 0.7,
    }
)


def style_axes(ax, grid="y"):
    ax.grid(axis=grid)
    ax.set_axisbelow(True)
    ax.tick_params(length=0)


def plot_loss(history, baselines, title, path):
    """Plot training and validation loss, with horizontal reference lines from `baselines`."""
    fig, ax = plt.subplots(figsize=(9, 5))
    iters = [row["iter"] for row in history]
    ax.plot(iters, [row["train_loss"] for row in history], color=SERIES[0], lw=1.6, label="Training")
    ax.plot(iters, [row["val_loss"] for row in history], color=SERIES[1], lw=1.6, label="Validation")
    for name, value in baselines.items():
        ax.axhline(value, color=MUTED, lw=0.9)
        ax.annotate(name, (iters[-1], value), xytext=(0, 3), textcoords="offset points", ha="right", color=INK_SECONDARY, fontsize=9)
    style_axes(ax)
    ax.set(xlabel="Training iteration", ylabel="Cross-entropy loss")
    ax.set_title(title)
    ax.legend(frameon=False, labelcolor=INK_SECONDARY)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)
