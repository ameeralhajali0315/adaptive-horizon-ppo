
import torch
from torch.distributions import Categorical

from src.models.actor_critic import Actor, Critic


def test_actor_and_critic_forward_pass():
    observation_dim = 4
    action_dim = 2

    actor = Actor(observation_dim, action_dim)
    critic = Critic(observation_dim)

    observation = torch.tensor(
        [[0.0274, -0.0061, 0.0359, 0.0197]],
        dtype=torch.float32,
    )

    logits = actor(observation)
    value = critic(observation)

    assert logits.shape == (1, action_dim)
    assert value.numel() == 1
    assert torch.isfinite(logits).all()
    assert torch.isfinite(value).all()

    distribution = Categorical(logits=logits)
    action = distribution.sample()
    log_probability = distribution.log_prob(action)

    assert action.shape == (1,)
    assert log_probability.shape == (1,)
    assert torch.isfinite(log_probability).all()
