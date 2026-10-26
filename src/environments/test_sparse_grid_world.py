
import numpy as np
import pytest
from gymnasium.utils.env_checker import check_env

from src.environments.sparse_grid_world import SparseGridWorld


def test_reset_returns_valid_observation_and_info():
    env = SparseGridWorld(grid_size=7, max_steps=40)

    observation, info = env.reset(seed=42)

    assert env.observation_space.contains(observation)
    assert observation.dtype == np.float32
    assert observation.shape == (4,)
    assert info == {}
    assert env.steps == 0
    assert not np.array_equal(env.agent_pos, env.goal_pos)

    env.close()


def test_reset_is_reproducible_with_same_seed():
    env = SparseGridWorld(grid_size=7)

    obs1, _ = env.reset(seed=123)
    agent1 = env.agent_pos.copy()
    goal1 = env.goal_pos.copy()

    obs2, _ = env.reset(seed=123)

    np.testing.assert_array_equal(obs1, obs2)
    np.testing.assert_array_equal(agent1, env.agent_pos)
    np.testing.assert_array_equal(goal1, env.goal_pos)

    env.close()


@pytest.mark.parametrize("grid_size", [2, 5, 7])
def test_observation_and_spaces_for_supported_grid_sizes(grid_size):
    env = SparseGridWorld(grid_size=grid_size)

    observation, _ = env.reset(seed=7)

    assert env.action_space.n == 4
    assert env.observation_space.shape == (4,)
    assert env.observation_space.contains(observation)

    env.close()


def test_movement_is_clipped_at_grid_boundaries():
    env = SparseGridWorld(grid_size=5)
    env.reset(seed=1)

    env.agent_pos = np.array([0, 0], dtype=np.int32)
    env.goal_pos = np.array([4, 4], dtype=np.int32)

    env.step(0)  # Up at the top boundary
    np.testing.assert_array_equal(env.agent_pos, [0, 0])

    env.step(2)  # Left at the left boundary
    np.testing.assert_array_equal(env.agent_pos, [0, 0])

    env.agent_pos = np.array([4, 4], dtype=np.int32)
    env.step(1)  # Down at the bottom boundary
    np.testing.assert_array_equal(env.agent_pos, [4, 4])

    env.step(3)  # Right at the right boundary
    np.testing.assert_array_equal(env.agent_pos, [4, 4])

    env.close()


def test_reaching_goal_gives_reward_and_terminates():
    env = SparseGridWorld(grid_size=3, max_steps=10)
    env.reset(seed=1)

    env.agent_pos = np.array([1, 2], dtype=np.int32)
    env.goal_pos = np.array([2, 2], dtype=np.int32)

    observation, reward, terminated, truncated, info = env.step(3)

    assert reward == 1.0
    assert terminated is True
    assert truncated is False
    assert info["reached_goal"] is True
    assert info["steps"] == 1
    assert env.observation_space.contains(observation)

    env.close()


def test_non_goal_step_has_zero_reward():
    env = SparseGridWorld(grid_size=5, max_steps=10)
    env.reset(seed=1)

    env.agent_pos = np.array([0, 0], dtype=np.int32)
    env.goal_pos = np.array([4, 4], dtype=np.int32)

    _, reward, terminated, truncated, info = env.step(3)

    assert reward == 0.0
    assert terminated is False
    assert truncated is False
    assert info["reached_goal"] is False

    env.close()


def test_episode_truncates_at_max_steps_without_goal():
    env = SparseGridWorld(grid_size=5, max_steps=2)
    env.reset(seed=1)

    env.agent_pos = np.array([0, 0], dtype=np.int32)
    env.goal_pos = np.array([4, 4], dtype=np.int32)

    _, reward1, terminated1, truncated1, _ = env.step(3)
    _, reward2, terminated2, truncated2, info2 = env.step(3)

    assert reward1 == 0.0
    assert terminated1 is False
    assert truncated1 is False

    assert reward2 == 0.0
    assert terminated2 is False
    assert truncated2 is True
    assert info2["steps"] == 2

    env.close()


@pytest.mark.parametrize("grid_size", [0, 1, -1])
def test_invalid_grid_sizes_are_rejected(grid_size):
    with pytest.raises(ValueError, match="grid_size must be at least 2"):
        SparseGridWorld(grid_size=grid_size)


@pytest.mark.parametrize("max_steps", [0, -1])
def test_invalid_max_steps_are_rejected(max_steps):
    with pytest.raises(ValueError, match="max_steps must be positive"):
        SparseGridWorld(max_steps=max_steps)


@pytest.mark.parametrize("action", [-1, 4, 1.5])
def test_invalid_actions_are_rejected(action):
    env = SparseGridWorld(grid_size=5)
    env.reset(seed=42)

    with pytest.raises(ValueError, match="Invalid action"):
        env.step(action)

    env.close()


def test_environment_passes_gymnasium_checker():
    env = SparseGridWorld(grid_size=7, max_steps=40)

    check_env(env, skip_render_check=True)

    env.close()

def test_reaching_goal_on_final_step_is_termination_not_truncation():
    env = SparseGridWorld(grid_size=3, max_steps=1)
    env.reset(seed=42)

    # One move right reaches the goal on the only permitted step.
    env.agent_pos = np.array([1, 2], dtype=np.int32)
    env.goal_pos = np.array([2, 2], dtype=np.int32)

    _, reward, terminated, truncated, info = env.step(3)

    assert reward == 1.0
    assert terminated is True
    assert truncated is False
    assert info["reached_goal"] is True
    assert info["steps"] == 1

    env.close()
