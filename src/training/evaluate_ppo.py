
import torch
import gymnasium as gym
from torch.distributions import Categorical

from src.models.actor_critic import Actor


def evaluate_policy(
    model_path="results/ppo_baseline_retest.pt",
    episodes=20,
    seed=1000,
):
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

    episode_rewards = []

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
                    # Use the most probable action for evaluation.
                    action = torch.argmax(logits).item()

                observation, reward, terminated, truncated, _ = (
                    env.step(action)
                )

                total_reward += reward

            episode_rewards.append(total_reward)

    finally:
        env.close()

    rewards = torch.tensor(episode_rewards, dtype=torch.float32)

    print("\n=== PPO Baseline Evaluation ===")
    print(f"Episodes:       {episodes}")
    print(f"Average reward: {rewards.mean().item():.2f}")
    print(f"Std deviation:  {rewards.std(unbiased=False).item():.2f}")
    print(f"Minimum reward: {rewards.min().item():.0f}")
    print(f"Maximum reward: {rewards.max().item():.0f}")
    print(f"All rewards:    {episode_rewards}")

    return episode_rewards


if __name__ == "__main__":
    evaluate_policy()
