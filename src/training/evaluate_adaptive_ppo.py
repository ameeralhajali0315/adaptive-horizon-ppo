
import torch
import gymnasium as gym
from pathlib import Path

from src.models.actor_critic import Actor


def evaluate_policy(episodes=20, seed=1000):
    project_root = Path(__file__).resolve().parents[2]
    model_path = (
        project_root / "results" / "ppo_adaptive_linear.pt"
    )

    if not model_path.exists():
        raise FileNotFoundError(
            f"Adaptive checkpoint not found: {model_path}"
        )

    env = gym.make("CartPole-v1")

    actor = Actor(
        env.observation_space.shape[0],
        env.action_space.n,
    )

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
            total_reward = 0

            while True:
                state = torch.as_tensor(
                    observation, dtype=torch.float32
                )

                with torch.no_grad():
                    action = actor(state).argmax().item()

                observation, reward, terminated, truncated, _ = (
                    env.step(action)
                )
                total_reward += reward

                if terminated or truncated:
                    break

            rewards.append(total_reward)
    finally:
        env.close()

    reward_tensor = torch.tensor(rewards, dtype=torch.float32)

    print("\n=== Adaptive PPO Evaluation ===")
    print(f"Episodes:       {episodes}")
    print(f"Average reward: {reward_tensor.mean().item():.2f}")
    print(
        f"Std deviation:  "
        f"{reward_tensor.std(unbiased=False).item():.2f}"
    )
    print(f"Minimum reward: {reward_tensor.min().item():.0f}")
    print(f"Maximum reward: {reward_tensor.max().item():.0f}")
    print(f"All rewards:    {rewards}")


if __name__ == "__main__":
    evaluate_policy()
