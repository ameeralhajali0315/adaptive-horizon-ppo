import gymnasium as gym
import torch
from torch.distributions import Categorical

from src.models.actor_critic import Actor


def main():
    env = gym.make("CartPole-v1")

    observation_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n

    actor = Actor(observation_dim, action_dim)

    observation, info = env.reset(seed=42)

    total_reward = 0

    for step in range(100):
        # Convert environment observation to PyTorch tensor
        observation_tensor = torch.tensor(
            observation,
            dtype=torch.float32
        ).unsqueeze(0)

        # Actor chooses an action
        logits = actor(observation_tensor)

        distribution = Categorical(logits=logits)

        action = distribution.sample()

        # Send action to CartPole
        observation, reward, terminated, truncated, info = env.step(
            action.item()
        )

        total_reward += reward

        print(
            f"Step: {step + 1:3d} | "
            f"Action: {action.item()} | "
            f"Reward: {reward}"
        )

        if terminated or truncated:
            print("Episode finished.")
            break

    print(f"\nTotal reward: {total_reward}")

    env.close()


if __name__ == "__main__":
    main()