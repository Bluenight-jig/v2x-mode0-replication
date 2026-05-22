"""
actor.py — Vehicle Actor.

Path A change: actor architecture is unchanged, but TWO INSTANCES are created
(one per traffic class) by the trainer. Each instance is dedicated to one
class. This eliminates the shared-actor coordination failure where a single
shared network couldn't differentiate M0 and M1 behaviour.
"""
import torch
import torch.nn as nn
from torch.distributions import Categorical


class VehicleActor(nn.Module):

    def __init__(self, obs_dim: int, action_dim: int, hidden: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, action_dim),
        )

    def forward(self, obs: torch.Tensor) -> Categorical:
        return Categorical(logits=self.net(obs))

    def get_action(self, obs: torch.Tensor, deterministic: bool = False):
        dist = self.forward(obs)
        if deterministic:
            action = dist.probs.argmax(dim=-1)
        else:
            action = dist.sample()
        return action, dist.log_prob(action), dist.entropy()
