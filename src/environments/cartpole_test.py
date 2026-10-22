
import gymnasium as gym
import numpy as np


def test_cartpole_reset_returns_valid_observation():
    env = gym.make("CartPole-v1")

    try:
        observation, info = env.reset(seed=42)

        assert observation.shape == env.observation_space.shape
        assert env.observation_space.contains(observation)
        assert isinstance(info, dict)
    finally:
        env.close()


def test_cartpole_step_returns_valid_transition():
    env = gym.make("CartPole-v1")

    try:
        observation, info = env.reset(seed=42)
        action = env.action_space.sample()

        next_observation, reward, terminated, truncated, info = (
            env.step(action)
        )

        assert env.observation_space.contains(next_observation)
        assert isinstance(reward, (int, float, np.number))
        assert isinstance(terminated, bool)
        assert isinstance(truncated, bool)
        assert isinstance(info, dict)
    finally:
        env.close()


def test_cartpole_has_expected_spaces():
    env = gym.make("CartPole-v1")

    try:
        assert env.observation_space.shape == (4,)
        assert env.action_space.n == 2
    finally:
        env.close()
