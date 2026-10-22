
import csv
from pathlib import Path

import gymnasium as gym
import numpy as np
import torch

import src.environments
from src.models.actor_critic import Actor


ROOT = Path(__file__).resolve().parents[2]
RESULTS_DIR = ROOT / "results"

ENVIRONMENT = "SparseGridWorld-v1"
TRAINING_SEEDS = [
    42, 123, 456, 789, 1024,
    2026, 2027, 2028, 2029, 2030,
    2031, 2032, 2033, 2034, 2035,
]
EVALUATION_EPISODES = 100
BASE_EVAL_SEED = 10000


def evaluate_checkpoint(checkpoint_path, training_seed):
    """Evaluate one checkpoint on a fixed set of test scenarios."""

    checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=True,
    )

    actor = Actor(observation_dim=4, action_dim=4)
    actor.load_state_dict(checkpoint["actor"])
    actor.eval()

    env = gym.make(ENVIRONMENT)
    episode_results = []

    try:
        for episode in range(EVALUATION_EPISODES):
            # The same episode seeds are used for every model.
            observation, _ = env.reset(
                seed=BASE_EVAL_SEED + episode
            )

            total_reward = 0.0
            steps = 0
            terminated = False
            truncated = False
            info = {}

            while not (terminated or truncated):
                state = torch.as_tensor(
                    observation,
                    dtype=torch.float32,
                ).unsqueeze(0)

                with torch.no_grad():
                    logits = actor(state)
                    action = int(torch.argmax(logits, dim=1).item())

                observation, reward, terminated, truncated, info = (
                    env.step(action)
                )

                total_reward += float(reward)
                steps += 1

            success = bool(info.get("reached_goal", False))

            episode_results.append({
                "algorithm": "",
                "training_seed": training_seed,
                "episode": episode + 1,
                "evaluation_seed": BASE_EVAL_SEED + episode,
                "success": int(success),
                "reward": total_reward,
                "steps": steps,
                "successful_steps": steps if success else "",
            })

    finally:
        env.close()

    return episode_results


def summarize(algorithm, training_seed, episodes):
    successes = [row["success"] for row in episodes]
    rewards = [row["reward"] for row in episodes]
    steps = [row["steps"] for row in episodes]
    successful_steps = [
        row["steps"] for row in episodes if row["success"] == 1
    ]

    return {
        "algorithm": algorithm,
        "training_seed": training_seed,
        "episodes": len(episodes),
        "success_rate": float(np.mean(successes)),
        "mean_reward": float(np.mean(rewards)),
        "mean_steps": float(np.mean(steps)),
        "mean_successful_steps": (
            float(np.mean(successful_steps))
            if successful_steps else ""
        ),
    }


def write_csv(path, rows):
    if not rows:
        raise ValueError(f"No results to write: {path}")

    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main():
    all_episode_results = []
    model_summaries = []

    for seed in TRAINING_SEEDS:
        for algorithm in ["fixed", "adaptive"]:
            checkpoint_path = (
                RESULTS_DIR
                / f"ppo_{algorithm}_sparse7_seed_{seed}.pt"
            )

            if not checkpoint_path.exists():
                raise FileNotFoundError(
                    f"Missing checkpoint: {checkpoint_path}"
                )

            print(
                f"Evaluating {algorithm.upper()} PPO "
                f"(training seed {seed})..."
            )

            episodes = evaluate_checkpoint(checkpoint_path, seed)

            for row in episodes:
                row["algorithm"] = algorithm

            all_episode_results.extend(episodes)
            model_summaries.append(
                summarize(algorithm, seed, episodes)
            )

            result = model_summaries[-1]
            print(
                f"  Success: {result['success_rate']:.1%} | "
                f"Mean reward: {result['mean_reward']:.3f} | "
                f"Mean steps: {result['mean_steps']:.2f}"
            )

    # Compare each fixed/adaptive training-seed pair on identical episodes.
    paired_results = []

    for seed in TRAINING_SEEDS:
        fixed = [
            row for row in all_episode_results
            if row["algorithm"] == "fixed"
            and row["training_seed"] == seed
        ]
        adaptive = [
            row for row in all_episode_results
            if row["algorithm"] == "adaptive"
            and row["training_seed"] == seed
        ]

        fixed.sort(key=lambda row: row["episode"])
        adaptive.sort(key=lambda row: row["episode"])

        fixed_success = np.array(
            [row["success"] for row in fixed], dtype=float
        )
        adaptive_success = np.array(
            [row["success"] for row in adaptive], dtype=float
        )
        fixed_steps = np.array(
            [row["steps"] for row in fixed], dtype=float
        )
        adaptive_steps = np.array(
            [row["steps"] for row in adaptive], dtype=float
        )

        paired_results.append({
            "training_seed": seed,
            "fixed_success_rate": float(fixed_success.mean()),
            "adaptive_success_rate": float(adaptive_success.mean()),
            "success_rate_difference": float(
                (adaptive_success - fixed_success).mean()
            ),
            "episodes_with_adaptive_success_advantage": int(
                np.sum(adaptive_success > fixed_success)
            ),
            "episodes_with_fixed_success_advantage": int(
                np.sum(fixed_success > adaptive_success)
            ),
            "episodes_with_equal_success": int(
                np.sum(fixed_success == adaptive_success)
            ),
            "fixed_mean_steps": float(fixed_steps.mean()),
            "adaptive_mean_steps": float(adaptive_steps.mean()),
            "mean_steps_difference": float(
                adaptive_steps.mean() - fixed_steps.mean()
            ),
        })

    write_csv(
        RESULTS_DIR / "sparse7_evaluation_episodes.csv",
        all_episode_results,
    )
    write_csv(
        RESULTS_DIR / "sparse7_evaluation_by_model.csv",
        model_summaries,
    )
    write_csv(
        RESULTS_DIR / "sparse7_paired_comparison.csv",
        paired_results,
    )

    print("\n=== Mean Across Fifteen Training Seeds ===")

    for algorithm in ["fixed", "adaptive"]:
        rows = [
            row for row in model_summaries
            if row["algorithm"] == algorithm
        ]

        print(
            f"{algorithm.upper()} PPO | "
            f"Success: {np.mean([r['success_rate'] for r in rows]):.1%} | "
            f"Mean reward: {np.mean([r['mean_reward'] for r in rows]):.3f} | "
            f"Mean steps: {np.mean([r['mean_steps'] for r in rows]):.2f}"
        )

    print("\n=== Paired Results by Training Seed ===")
    for row in paired_results:
        print(
            f"Seed {row['training_seed']} | "
            f"Success difference (adaptive - fixed): "
            f"{row['success_rate_difference']:+.1%} | "
            f"Step difference (adaptive - fixed): "
            f"{row['mean_steps_difference']:+.2f}"
        )

    print("\nEvaluation complete. CSV files saved:")
    print("  results/sparse7_evaluation_episodes.csv")
    print("  results/sparse7_evaluation_by_model.csv")
    print("  results/sparse7_paired_comparison.csv")


if __name__ == "__main__":
    main()
