
import torch


def compute_gae(
    rewards,
    values,
    next_values,
    terminated,
    truncated,
    gamma=0.99,
    gae_lambda=0.95,
):
    """
    Calculate GAE and value targets.

    True termination prevents value bootstrapping.
    Truncation allows value bootstrapping, but GAE recursion
    must not carry into the next episode.
    """
    advantages = torch.zeros_like(rewards)
    last_advantage = 0.0

    for t in reversed(range(len(rewards))):
        # A true terminal state has no future value.
        bootstrap_mask = 1.0 - terminated[t].float()

        # Neither termination nor truncation can carry GAE
        # recursion into the next episode.
        continuation_mask = 1.0 - (
            terminated[t] | truncated[t]
        ).float()

        delta = (
            rewards[t]
            + gamma * next_values[t] * bootstrap_mask
            - values[t]
        )

        last_advantage = (
            delta
            + gamma * gae_lambda
            * continuation_mask
            * last_advantage
        )

        advantages[t] = last_advantage

    returns = advantages + values
    return advantages, returns
