
import torch
import torch.nn as nn
from torch.distributions import Categorical


def ppo_update(
    actor,
    critic,
    actor_optimizer,
    critic_optimizer,
    data,
    clip_epsilon=0.2,
    value_coefficient=0.5,
    entropy_coefficient=0.01,
    epochs=4,
):
    states = data["states"]
    actions = data["actions"]
    old_log_probs = data["log_probs"]
    advantages = data["advantages"]
    returns = data["returns"]

    # Normalize advantages to improve training stability.
    advantages = (
        advantages - advantages.mean()
    ) / (advantages.std(unbiased=False) + 1e-8)

    metrics = {
        "actor_loss": 0.0,
        "critic_loss": 0.0,
        "entropy": 0.0,
    }

    for _ in range(epochs):
        # Evaluate the actions using the updated policy.
        logits = actor(states)
        distribution = Categorical(logits=logits)

        new_log_probs = distribution.log_prob(actions)
        entropy = distribution.entropy().mean()

        # PPO probability ratio.
        ratio = torch.exp(new_log_probs - old_log_probs)

        unclipped = ratio * advantages
        clipped = torch.clamp(
            ratio,
            1 - clip_epsilon,
            1 + clip_epsilon,
        ) * advantages

        actor_loss = -torch.minimum(
            unclipped, clipped
        ).mean()

        # Train the critic to predict discounted returns.
        predicted_values = critic(states).squeeze(-1)
        critic_loss = nn.functional.mse_loss(
            predicted_values, returns
        )

        total_actor_loss = (
            actor_loss - entropy_coefficient * entropy
        )

        actor_optimizer.zero_grad()
        total_actor_loss.backward()
        nn.utils.clip_grad_norm_(actor.parameters(), 0.5)
        actor_optimizer.step()

        critic_optimizer.zero_grad()
        (value_coefficient * critic_loss).backward()
        nn.utils.clip_grad_norm_(critic.parameters(), 0.5)
        critic_optimizer.step()

        metrics["actor_loss"] += actor_loss.item()
        metrics["critic_loss"] += critic_loss.item()
        metrics["entropy"] += entropy.item()

    return {
        name: value / epochs
        for name, value in metrics.items()
    }
