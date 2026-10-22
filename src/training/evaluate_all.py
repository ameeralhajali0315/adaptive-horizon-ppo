
import csv
import statistics
from pathlib import Path

import gymnasium as gym
import torch

from src.models.actor_critic import Actor


PROJECT_ROOT = Path(__file__).resolve().parents[2]
RESULTS_DIR = PROJECT_ROOT / "results"
EPISODES = 20
EVALUATION_SEED = 1000
TRAINING_SEEDS = [42, 123, 456, 789, 1024]


def evaluate_model(model_path, episodes=EPISODES, seed=EVALUATION_SEED):
    env = gym.make("CartPole-v1")

    observation_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n

    actor = Actor(observation_dim, action_dim)

    checkpoint = torch.load(
        model_path,
        map_location="cpu",
        weights_only=True,
    )
    actor.load_state_dict(checkpoint["actor"])
    actor.eval()

    rewards = []

    try:
        for episode in range(episodes):
            observation, _ = env.reset(seed=seed + episode)
            total_reward = 0.0
            terminated = False
            truncated = False

            while not (terminated or truncated):
                state = torch.as_tensor(
                    observation,
                    dtype=torch.float32,
                )

                with torch.no_grad():
                    logits = actor(state)
                    action = torch.argmax(logits).item()

                observation, reward, terminated, truncated, _ = (
                    env.step(action)
                )
                total_reward += reward

            rewards.append(total_reward)

    finally:
        env.close()

    return rewards


def main():
    summary_rows = []
    episode_rows = []
    results_by_method_and_seed = {}

    experiments = [
        ("fixed", seed, f"ppo_fixed_seed_{seed}.pt")
        for seed in TRAINING_SEEDS
    ] + [
        ("adaptive", seed, f"ppo_adaptive_seed_{seed}.pt")
        for seed in TRAINING_SEEDS
    ]

    for method, training_seed, filename in experiments:
        model_path = RESULTS_DIR / filename

        if not model_path.exists():
            raise FileNotFoundError(
                f"Missing checkpoint: {model_path}"
            )

        print(f"\nEvaluating {method} PPO | training seed "
              f"{training_seed} ...")

        rewards = evaluate_model(model_path)
        mean_reward = statistics.mean(rewards)
        std_reward = statistics.pstdev(rewards)

        results_by_method_and_seed[(method, training_seed)] = (
            mean_reward
        )

        summary_rows.append({
            "method": method,
            "training_seed": training_seed,
            "episodes": len(rewards),
            "mean_reward": mean_reward,
            "std_reward": std_reward,
            "min_reward": min(rewards),
            "max_reward": max(rewards),
        })

        for episode_index, reward in enumerate(rewards):
            episode_rows.append({
                "method": method,
                "training_seed": training_seed,
                "evaluation_seed": EVALUATION_SEED + episode_index,
                "episode": episode_index + 1,
                "reward": reward,
            })

        print(
            f"Mean: {mean_reward:.2f} | "
            f"Std: {std_reward:.2f} | "
            f"Min: {min(rewards):.0f} | "
            f"Max: {max(rewards):.0f}"
        )

    summary_path = RESULTS_DIR / "multi_seed_summary.csv"
    episodes_path = RESULTS_DIR / "multi_seed_episode_rewards.csv"

    with summary_path.open(
        "w", newline="", encoding="utf-8"
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=[
                "method",
                "training_seed",
                "episodes",
                "mean_reward",
                "std_reward",
                "min_reward",
                "max_reward",
            ],
        )
        writer.writeheader()
        writer.writerows(summary_rows)

    with episodes_path.open(
        "w", newline="", encoding="utf-8"
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=[
                "method",
                "training_seed",
                "evaluation_seed",
                "episode",
                "reward",
            ],
        )
        writer.writeheader()
        writer.writerows(episode_rows)

    fixed_means = [
        results_by_method_and_seed[("fixed", seed)]
        for seed in TRAINING_SEEDS
    ]
    adaptive_means = [
        results_by_method_and_seed[("adaptive", seed)]
        for seed in TRAINING_SEEDS
    ]
    paired_differences = [
        adaptive - fixed
        for adaptive, fixed in zip(adaptive_means, fixed_means)
    ]

    print("\n=== Five-Seed Comparison ===")
    print(f"Fixed PPO mean across runs:    "
          f"{statistics.mean(fixed_means):.2f}")
    print(f"Adaptive PPO mean across runs: "
          f"{statistics.mean(adaptive_means):.2f}")
    print(f"Mean paired difference:        "
          f"{statistics.mean(paired_differences):+.2f}")
    print(f"Paired differences:            "
          f"{[round(value, 2) for value in paired_differences]}")

    print(f"\nSaved summary: {summary_path}")
    print(f"Saved episode results: {episodes_path}")


if __name__ == "__main__":
    main()
