"""
mappo_gaefix.py -- both trainers with per-agent GAE (aa-achievement.md §7b).

A NEW file. agents/mappo.py and agents/mappo_mode0c.py are not edited, so every
existing run stays reproducible from untouched code.

The problem. The training loop fills the rollout buffer interleaved by vehicle:
index = t * N + i. The original compute_gae walks that flat buffer as ONE time
sequence, so vehicle i's "next" entry is vehicle i+1 at the same step. That
credits vehicles for each other's rewards, shrinks the effective discount to
(gamma*lambda)^N per step, and makes credit depend on list position.

The fix. Reshape to (T, N) and run GAE along each vehicle's own trajectory.
Nothing else changes: same actors, same centralized critic, same PPO update,
same team value V(global state) as each vehicle's baseline.
"""
import numpy as np

from agents.mappo import MAPPOTrainer
from agents.mappo_mode0c import MAPPOTrainerMode0c


def per_agent_gae(rewards, values, next_val, dones, n_agents, gamma, lam):
    """GAE along each vehicle's own trajectory. Inputs are the flat, interleaved
    buffers the training loop builds; outputs keep that same order."""
    rewards = np.asarray(rewards)
    values = np.asarray(values)
    dones = np.asarray(dones)
    if len(rewards) % n_agents:
        raise ValueError(f"buffer length {len(rewards)} is not a multiple of N = {n_agents}")
    T = len(rewards) // n_agents
    R = rewards.reshape(T, n_agents)
    V = values.reshape(T, n_agents)
    D = dones.reshape(T, n_agents)
    A = np.zeros_like(R)
    for i in range(n_agents):
        gae, nv = 0.0, next_val
        for t in reversed(range(T)):
            delta = R[t, i] + gamma * nv * (1 - D[t, i]) - V[t, i]
            gae = delta + gamma * lam * (1 - D[t, i]) * gae
            A[t, i] = gae
            nv = V[t, i]
    adv = A.reshape(-1).astype(rewards.dtype)
    return adv, adv + values


class MAPPOTrainerGAEFix(MAPPOTrainer):
    """Mode 0a (shared per-class actors) with per-agent GAE."""

    def compute_gae(self, rewards, values, next_val, dones):
        return per_agent_gae(rewards, values, next_val, dones, self.N, self.gamma, self.lam)


class MAPPOTrainerMode0cGAEFix(MAPPOTrainerMode0c):
    """Mode 0c (per-vehicle actors) with per-agent GAE."""

    def compute_gae(self, rewards, values, next_val, dones):
        return per_agent_gae(rewards, values, next_val, dones, self.N, self.gamma, self.lam)
