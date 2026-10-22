import numpy as np
from dataclasses import dataclass, field

import torch


@dataclass
class RolloutBuffer:
    states: list = field(default_factory=list)
    actions: list = field(default_factory=list)
    rewards: list = field(default_factory=list)
    log_probs: list = field(default_factory=list)
    values: list = field(default_factory=list)
    next_values: list = field(default_factory=list)
    terminated: list = field(default_factory=list)
    truncated: list = field(default_factory=list)

    def add(
        self,
        state,
        action,
        reward,
        log_prob,
        value,
        next_value,
        terminated,
        truncated,
    ):
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)
        self.log_probs.append(log_prob)
        self.values.append(value)
        self.next_values.append(next_value)
        self.terminated.append(terminated)
        self.truncated.append(truncated)

    def __len__(self):
        return len(self.states)

    def clear(self):
        for name in (
            "states", "actions", "rewards", "log_probs",
            "values", "next_values", "terminated", "truncated",
        ):
            getattr(self, name).clear()

    def get_tensors(self):
        if not self.states:
            raise ValueError("Cannot create tensors from an empty buffer.")

        return {
            "states": torch.as_tensor(
                np.asarray(self.states, dtype=np.float32),
                dtype=torch.float32,
            ),
            "actions": torch.as_tensor(self.actions, dtype=torch.long),
            "rewards": torch.as_tensor(self.rewards, dtype=torch.float32),
            "log_probs": torch.as_tensor(self.log_probs, dtype=torch.float32),
            "values": torch.as_tensor(self.values, dtype=torch.float32),
            "next_values": torch.as_tensor(
                self.next_values, dtype=torch.float32
            ),
            "terminated": torch.as_tensor(
                self.terminated, dtype=torch.bool
            ),
            "truncated": torch.as_tensor(
                self.truncated, dtype=torch.bool
            ),
        }
