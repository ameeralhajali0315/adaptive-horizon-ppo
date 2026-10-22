
import pytest
import torch

from src.algorithms.rollout_buffer import RolloutBuffer


def test_rollout_buffer_stores_experiences_as_tensors():
    buffer = RolloutBuffer()

    buffer.add(
        state=[0.1, 0.2, 0.3, 0.4],
        action=0,
        reward=1.0,
        log_prob=-0.65,
        value=0.20,
        next_value=0.25,
        terminated=False,
        truncated=False,
    )

    buffer.add(
        state=[0.2, 0.3, 0.4, 0.5],
        action=1,
        reward=1.0,
        log_prob=-0.72,
        value=0.25,
        next_value=0.30,
        terminated=False,
        truncated=False,
    )

    buffer.add(
        state=[0.3, 0.4, 0.5, 0.6],
        action=0,
        reward=1.0,
        log_prob=-0.60,
        value=0.30,
        next_value=0.0,
        terminated=True,
        truncated=False,
    )

    assert len(buffer) == 3

    data = buffer.get_tensors()

    assert data["states"].shape == (3, 4)
    assert data["actions"].shape == (3,)
    assert data["rewards"].shape == (3,)
    assert data["log_probs"].shape == (3,)
    assert data["values"].shape == (3,)
    assert data["next_values"].shape == (3,)
    assert data["terminated"].shape == (3,)
    assert data["truncated"].shape == (3,)

    assert data["states"].dtype == torch.float32
    assert data["actions"].dtype == torch.long
    assert data["terminated"].dtype == torch.bool
    assert data["truncated"].dtype == torch.bool

    assert data["actions"].tolist() == [0, 1, 0]
    assert data["terminated"].tolist() == [False, False, True]
    assert data["truncated"].tolist() == [False, False, False]


def test_rollout_buffer_clear_removes_all_experiences():
    buffer = RolloutBuffer()

    buffer.add(
        state=[0.1, 0.2, 0.3, 0.4],
        action=0,
        reward=1.0,
        log_prob=-0.65,
        value=0.20,
        next_value=0.25,
        terminated=False,
        truncated=False,
    )

    buffer.clear()

    assert len(buffer) == 0
    assert buffer.states == []
    assert buffer.actions == []
    assert buffer.rewards == []
    assert buffer.next_values == []
    assert buffer.terminated == []
    assert buffer.truncated == []

    with pytest.raises(ValueError, match="empty buffer"):
        buffer.get_tensors()
