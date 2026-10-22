
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RESULTS_DIR = PROJECT_ROOT / "results"
FIGURES_DIR = RESULTS_DIR / "figures"

EPISODES_FILE = RESULTS_DIR / "sparse7_evaluation_episodes.csv"
MODEL_FILE = RESULTS_DIR / "sparse7_evaluation_by_model.csv"
PAIRED_FILE = RESULTS_DIR / "sparse7_paired_comparison.csv"


def main():
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    for path in (EPISODES_FILE, MODEL_FILE, PAIRED_FILE):
        if not path.exists():
            raise FileNotFoundError(f"Required results file not found: {path}")

    episodes = pd.read_csv(EPISODES_FILE)
    models = pd.read_csv(MODEL_FILE)
    paired = pd.read_csv(PAIRED_FILE)

    print("Evaluation episode columns:", episodes.columns.tolist())
    print("Model summary columns:", models.columns.tolist())
    print("Paired comparison columns:", paired.columns.tolist())


    # Figure 1: mean success rate with standard deviation across training seeds
    per_seed = (
        episodes.groupby(["algorithm", "training_seed"])["success"]
        .mean()
        .reset_index(name="success_rate")
    )

    summary = (
        per_seed.groupby("algorithm")["success_rate"]
        .agg(["mean", "std"])
    )

    order = ["fixed", "adaptive"]
    summary = summary.reindex(order)

    means = summary["mean"]
    stds = summary["std"]

    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(
        means.index,
        means.values * 100,
        yerr=stds.values * 100,
        capsize=7,
        error_kw={"elinewidth": 1.5},
    )

    ax.bar_label(
        bars,
        labels=[f"{value:.1%}" for value in means.values],
        padding=5,
    )

    ax.set_title("SparseGridWorld-v1: Success Rate Across 15 Seeds")
    ax.set_ylabel("Success rate (%)")
    ax.set_ylim(0, 112)
    ax.grid(axis="y", alpha=0.3)
    ax.set_axisbelow(True)

    fig.tight_layout()
    fig.savefig(
        FIGURES_DIR / "success_rate_comparison.png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.close(fig)

    print("\nSuccess-rate summary across training seeds:")
    print(summary)


    # Figure 2: paired success-rate differences across training seeds
    required_paired_cols = {"training_seed", "success_rate_difference"}
    if not required_paired_cols.issubset(paired.columns):
        raise ValueError(
            "The paired CSV does not contain the expected columns. "
            "Check the printed columns and update the script accordingly."
        )

    paired = paired.sort_values("training_seed")
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(
        paired["training_seed"].astype(str),
        paired["success_rate_difference"] * 100,
    )
    ax.axhline(0, linewidth=1)
    ax.set_title("Adaptive PPO − Fixed PPO: Success Rate by Training Seed")
    ax.set_xlabel("Training seed")
    ax.set_ylabel("Difference (percentage points)")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "paired_success_differences.png", dpi=300)
    plt.close(fig)


    # Figure 3: paired episode-step differences across training seeds
    fig, ax = plt.subplots(figsize=(10, 5))

    values = paired["mean_steps_difference"]
    bars = ax.bar(
        paired["training_seed"].astype(str),
        values,
    )

    ax.axhline(0, color="black", linewidth=1.2)

    for bar, value in zip(bars, values):
        ax.annotate(
            f"{value:.2f}",
            xy=(bar.get_x() + bar.get_width() / 2, value),
            xytext=(0, -5),
            textcoords="offset points",
            ha="center",
            va="top",
            fontsize=8,
        )

    ax.set_title("Adaptive PPO vs. Fixed PPO: Episode Steps Across 15 Seeds")
    ax.set_xlabel("Training seed")
    ax.set_ylabel("Difference in mean steps (Adaptive − Fixed)")
    ax.grid(axis="y", alpha=0.3)
    ax.set_axisbelow(True)

    fig.tight_layout()
    fig.savefig(
        FIGURES_DIR / "paired_step_differences.png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.close(fig)


    print(f"\nCharts saved in: {FIGURES_DIR}")
    for path in sorted(FIGURES_DIR.glob("*.png")):
        print(f" - {path.name}")


if __name__ == "__main__":
    main()