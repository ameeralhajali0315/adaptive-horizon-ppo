import torch
import torch.nn as nn


class Actor(nn.Module):
    def __init__(self, observation_dim, action_dim):
        super().__init__()

        self.network = nn.Sequential(
            nn.Linear(observation_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, action_dim)
        )

    def forward(self, observation):
        return self.network(observation)


class Critic(nn.Module):
    def __init__(self, observation_dim):
        super().__init__()

        self.network = nn.Sequential(
            nn.Linear(observation_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

    def forward(self, observation):
        return self.network(observation)