
from pathlib import Path

import gymnasium as gym
import numpy as np
import torch

import src.environments
from src.models.actor_critic import Actor


PROJECT_ROOT = Path(__file__).resolve().parents[2]
RESULTS_DIR = PROJECT_ROOT / "results"


def load_actor(path):
    checkpoint = torch.load(
        path,
        map_location="cpu",
        weights_only=True,
    )

    actor = Actor(observation_dim=4, action_dim=4)
    actor.load_state_dict(checkpoint["actor"])
    actor.eval()
    return actor


def main():
    fixed = load_actor(RESULTS_DIR / "ppo_fixed_sparse.pt")
    adaptive = load_actor(RESULTS_DIR / "ppo_adaptive_sparse.pt")

    rng = np.random.default_rng(2026)

    # Sample valid grid coordinates and normalize them.
    grid_positions = np.arange(5, dtype=np.float32) / 4.0
    observations = []

    for _ in range(10_000):
        agent = rng.integers(0, 5, size=2)
        goal = rng.integers(0, 5, size=2)

        while np.array_equal(agent, goal):
            goal = rng.integers(0, 5, size=2)

        observations.append([
            grid_positions[agent[0]],
            grid_positions[agent[1]],
            grid_positions[goal[0]],
            grid_positions[goal[1]],
        ])

    states = torch.tensor(observations, dtype=torch.float32)

    with torch.no_grad():
        fixed_logits = fixed(states)
        adaptive_logits = adaptive(states)

        fixed_actions = fixed_logits.argmax(dim=1)
        adaptive_actions = adaptive_logits.argmax(dim=1)

    different = fixed_actions != adaptive_actions
    different_count = int(different.sum().item())

    print("Observations compared:", len(observations))
    print("Different selected actions:", different_count)
    print(f"Action disagreement rate: {different_count / len(observations):.2%}")

    if different_count:
        indices = torch.where(different)[0][:10].tolist()
        print("\nFirst differing observations:")
        for index in indices:
            print(
                f"Observation={observations[index]}, "
                f"Fixed={fixed_actions[index].item()}, "
                f"Adaptive={adaptive_actions[index].item()}"
            )


if __name__ == "__main__":
    main()
