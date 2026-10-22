
import torch

from src.algorithms.gae import compute_gae


def test_compute_gae_returns_advantages_and_returns():
    rewards = torch.tensor([1.0, 1.0, 1.0])
    values = torch.tensor([0.5, 0.6, 0.7])
    next_values = torch.tensor([0.6, 0.7, 0.0])
    terminated = torch.tensor([False, False, True])
    truncated = torch.tensor([False, False, False])

    advantages, returns = compute_gae(
        rewards=rewards,
        values=values,
        next_values=next_values,
        terminated=terminated,
        truncated=truncated,
        gamma=0.99,
        gae_lambda=0.95,
    )

    assert advantages.shape == rewards.shape
    assert returns.shape == rewards.shape
    assert torch.isfinite(advantages).all()
    assert torch.isfinite(returns).all()

    # A true terminal state must not bootstrap from next_values.
    assert torch.isclose(returns[-1], rewards[-1])


def test_truncation_bootstraps_but_stops_gae_recursion():
    rewards = torch.tensor([1.0])
    values = torch.tensor([0.5])
    next_values = torch.tensor([2.0])
    terminated = torch.tensor([False])
    truncated = torch.tensor([True])

    advantages, returns = compute_gae(
        rewards=rewards,
        values=values,
        next_values=next_values,
        terminated=terminated,
        truncated=truncated,
        gamma=0.99,
        gae_lambda=0.95,
    )

    expected_return = rewards[0] + 0.99 * next_values[0]

    assert torch.isclose(returns[0], expected_return)
    assert torch.isclose(advantages[0], expected_return - values[0])
