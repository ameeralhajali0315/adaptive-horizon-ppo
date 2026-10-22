
import csv
from pathlib import Path

import gymnasium as gym
import numpy as np
import torch

import src.environments
from src.models.actor_critic import Actor


PROJECT_ROOT = Path(__file__).resolve().parents[2]
RESULTS_DIR = PROJECT_ROOT / "results"

CHECKPOINTS = {
    "Fixed PPO": RESULTS_DIR / "ppo_fixed_sparse.pt",
    "Adaptive PPO": RESULTS_DIR / "ppo_adaptive_sparse.pt",
}

NUM_EPISODES = 100
BASE_SEED = 10_000


def load_actor(checkpoint_path, device):
    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
        weights_only=True,
    )

    actor = Actor(
        observation_dim=4,
        action_dim=4,
    ).to(device)

    actor.load_state_dict(checkpoint["actor"])
    actor.eval()

    return actor


def choose_action(actor, observation, device):
    state = torch.as_tensor(
        observation,
        dtype=torch.float32,
        device=device,
    ).unsqueeze(0)

    with torch.no_grad():
        logits = actor(state)
        action = int(torch.argmax(logits, dim=-1).item())

    return action


def evaluate_model(actor, env, device):
    episode_results = []

    for episode in range(NUM_EPISODES):
        observation, _ = env.reset(seed=BASE_SEED + episode)

        total_reward = 0.0
        steps = 0
        terminated = False
        truncated = False
        info = {}

        while not (terminated or truncated):
            action = choose_action(actor, observation, device)

            observation, reward, terminated, truncated, info = (
                env.step(action)
            )

            total_reward += float(reward)
            steps += 1

        success = bool(info.get("reached_goal", False))

        episode_results.append({
            "success": int(success),
            "reward": total_reward,
            "steps": steps,
        })

    return episode_results


def main():
    device = torch.device("cpu")
    env = gym.make("SparseGridWorld-v0")

    summary_rows = []
    episode_rows = []

    try:
        for model_name, checkpoint_path in CHECKPOINTS.items():
            if not checkpoint_path.exists():
                raise FileNotFoundError(
                    f"Checkpoint not found: {checkpoint_path}"
                )

            print(f"\nEvaluating {model_name}...")
            actor = load_actor(checkpoint_path, device)
            results = evaluate_model(actor, env, device)

            success_rate = float(np.mean(
                [result["success"] for result in results]
            ))
            mean_reward = float(np.mean(
                [result["reward"] for result in results]
            ))
            mean_steps = float(np.mean(
                [result["steps"] for result in results]
            ))

            successful_steps = [
                result["steps"]
                for result in results
                if result["success"] == 1
            ]

            mean_success_steps = (
                float(np.mean(successful_steps))
                if successful_steps
                else None
            )

            summary_rows.append({
                "model": model_name,
                "episodes": NUM_EPISODES,
                "success_rate": success_rate,
                "mean_reward": mean_reward,
                "mean_steps": mean_steps,
                "mean_steps_on_success": mean_success_steps,
            })

            for episode, result in enumerate(results):
                episode_rows.append({
                    "model": model_name,
                    "episode": episode + 1,
                    "seed": BASE_SEED + episode,
                    **result,
                })

            print(f"Success rate: {success_rate:.1%}")
            print(f"Mean reward: {mean_reward:.3f}")
            print(f"Mean episode steps: {mean_steps:.2f}")
            if mean_success_steps is not None:
                print(
                    "Mean steps on successful episodes: "
                    f"{mean_success_steps:.2f}"
                )
            else:
                print("Mean steps on successful episodes: N/A")

    finally:
        env.close()

    summary_path = RESULTS_DIR / "sparse_evaluation_summary.csv"
    episodes_path = RESULTS_DIR / "sparse_evaluation_episodes.csv"

    with summary_path.open(
        "w", newline="", encoding="utf-8"
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=summary_rows[0].keys(),
        )
        writer.writeheader()
        writer.writerows(summary_rows)

    with episodes_path.open(
        "w", newline="", encoding="utf-8"
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=episode_rows[0].keys(),
        )
        writer.writeheader()
        writer.writerows(episode_rows)

    print("\n=== SparseGridWorld Comparison ===")
    for result in summary_rows:
        print(
            f"{result['model']}: "
            f"success={result['success_rate']:.1%}, "
            f"mean reward={result['mean_reward']:.3f}, "
            f"mean steps={result['mean_steps']:.2f}"
        )

    print(f"\nSummary saved to: {summary_path}")
    print(f"Episode results saved to: {episodes_path}")


if __name__ == "__main__":
    main()
