
import pytest
import torch

from src.algorithms.ppo import ppo_update
from src.models.actor_critic import Actor, Critic


@pytest.fixture
def ppo_components():
    torch.manual_seed(42)

    actor = Actor(observation_dim=4, action_dim=4)
    critic = Critic(observation_dim=4)

    actor_optimizer = torch.optim.Adam(actor.parameters(), lr=1e-3)
    critic_optimizer = torch.optim.Adam(critic.parameters(), lr=1e-3)

    batch_size = 16
    states = torch.randn(batch_size, 4)
    actions = torch.randint(0, 4, (batch_size,))

    with torch.no_grad():
        old_log_probs = torch.distributions.Categorical(
            logits=actor(states)
        ).log_prob(actions)

    data = {
        "states": states,
        "actions": actions,
        "log_probs": old_log_probs,
        "advantages": torch.linspace(-1.0, 1.0, batch_size),
        "returns": torch.randn(batch_size),
    }

    return (
        actor,
        critic,
        actor_optimizer,
        critic_optimizer,
        data,
    )


def test_ppo_update_returns_finite_metrics(ppo_components):
    actor, critic, actor_optimizer, critic_optimizer, data = ppo_components

    metrics = ppo_update(
        actor,
        critic,
        actor_optimizer,
        critic_optimizer,
        data,
        epochs=2,
    )

    assert set(metrics) == {"actor_loss", "critic_loss", "entropy"}
    assert all(torch.isfinite(torch.tensor(value)) for value in metrics.values())
    assert metrics["critic_loss"] >= 0.0
    assert metrics["entropy"] >= 0.0


def test_ppo_update_changes_actor_and_critic_parameters(ppo_components):
    actor, critic, actor_optimizer, critic_optimizer, data = ppo_components

    actor_before = [p.detach().clone() for p in actor.parameters()]
    critic_before = [p.detach().clone() for p in critic.parameters()]

    ppo_update(
        actor,
        critic,
        actor_optimizer,
        critic_optimizer,
        data,
        epochs=2,
    )

    assert any(
        not torch.equal(before, after)
        for before, after in zip(actor_before, actor.parameters())
    )
    assert any(
        not torch.equal(before, after)
        for before, after in zip(critic_before, critic.parameters())
    )


def test_ppo_update_keeps_model_parameters_finite(ppo_components):
    actor, critic, actor_optimizer, critic_optimizer, data = ppo_components

    ppo_update(
        actor,
        critic,
        actor_optimizer,
        critic_optimizer,
        data,
        epochs=2,
    )

    assert all(torch.isfinite(p).all() for p in actor.parameters())
    assert all(torch.isfinite(p).all() for p in critic.parameters())


def test_ppo_update_with_one_epoch_returns_metrics(ppo_components):
    actor, critic, actor_optimizer, critic_optimizer, data = ppo_components

    metrics = ppo_update(
        actor,
        critic,
        actor_optimizer,
        critic_optimizer,
        data,
        epochs=1,
    )

    assert all(isinstance(value, float) for value in metrics.values())
