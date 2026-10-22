
import gymnasium as gym
import numpy as np
from gymnasium import spaces


class SparseGridWorld(gym.Env):
    """
    A grid-world environment with sparse rewards.

    The agent receives +1.0 only when it reaches the goal.
    All other steps provide zero reward.
    """

    metadata = {"render_modes": ["human"]}

    def __init__(self, grid_size=5, max_steps=40, render_mode=None):
        super().__init__()

        if grid_size < 2:
            raise ValueError("grid_size must be at least 2.")
        if max_steps < 1:
            raise ValueError("max_steps must be positive.")

        self.grid_size = grid_size
        self.max_steps = max_steps
        self.render_mode = render_mode

        # Actions: 0=up, 1=down, 2=left, 3=right
        self.action_space = spaces.Discrete(4)

        # Observation: agent_x, agent_y, goal_x, goal_y.
        # Every coordinate is normalized to [0, 1].
        self.observation_space = spaces.Box(
            low=0.0,
            high=1.0,
            shape=(4,),
            dtype=np.float32,
        )

        self.agent_pos = np.array([0, 0], dtype=np.int32)
        self.goal_pos = np.array(
            [grid_size - 1, grid_size - 1],
            dtype=np.int32,
        )
        self.steps = 0

    def _get_obs(self):
        scale = float(self.grid_size - 1)

        return np.array(
            [
                self.agent_pos[0] / scale,
                self.agent_pos[1] / scale,
                self.goal_pos[0] / scale,
                self.goal_pos[1] / scale,
            ],
            dtype=np.float32,
        )

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        # Randomize both the start and goal on every episode.
        # The agent can observe the goal coordinates.
        self.agent_pos = self.np_random.integers(
            0, self.grid_size, size=2, dtype=np.int32
        )

        while True:
            self.goal_pos = self.np_random.integers(
                0, self.grid_size, size=2, dtype=np.int32
            )
            if not np.array_equal(self.agent_pos, self.goal_pos):
                break

        self.steps = 0

        return self._get_obs(), {}

    def step(self, action):
        if not self.action_space.contains(action):
            raise ValueError(f"Invalid action: {action}")

        # Movement: x increases to the right, y increases downward.
        if action == 0:       # Up
            self.agent_pos[1] -= 1
        elif action == 1:     # Down
            self.agent_pos[1] += 1
        elif action == 2:     # Left
            self.agent_pos[0] -= 1
        elif action == 3:     # Right
            self.agent_pos[0] += 1

        # Prevent the agent from leaving the grid.
        self.agent_pos = np.clip(
            self.agent_pos, 0, self.grid_size - 1
        )

        self.steps += 1

        reached_goal = np.array_equal(
            self.agent_pos, self.goal_pos
        )

        # Sparse reward: no reward until the goal is reached.
        reward = 1.0 if reached_goal else 0.0

        terminated = bool(reached_goal)
        truncated = bool(
            self.steps >= self.max_steps and not terminated
        )

        info = {
            "reached_goal": bool(reached_goal),
            "steps": self.steps,
        }

        return (
            self._get_obs(),
            reward,
            terminated,
            truncated,
            info,
        )

    def render(self):
        grid = [
            ["."] * self.grid_size
            for _ in range(self.grid_size)
        ]

        gx, gy = self.goal_pos
        ax, ay = self.agent_pos

        grid[gy][gx] = "G"
        grid[ay][ax] = "A" if (ax, ay) != (gx, gy) else "A/G"

        print("\n".join(" ".join(row) for row in grid))
        print()
