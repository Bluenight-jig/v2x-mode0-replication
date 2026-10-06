"""
mappo_critic_ablation.py -- critic-input ablation for R2-4 (aa-achievement.md §7c).

A NEW file; nothing existing is edited. Every class inherits a per-agent-GAE
trainer from agents/mappo_gaefix.py and changes ONE thing: what the critic sees.

  Paper's critic (the rebuild's standard arm, not in this file):
      V(global state)          one team value per step, shared by all vehicles
  Local critic      (FEATURES = "local"):
      V(o_i)                   vehicle i's own observation only
  Agent-specific    (FEATURES = "agentspecific"):
      V(global state (+) o_i)  everything the paper's critic sees, plus which vehicle

Local vs agent-specific isolates INFORMATION (both give a value per vehicle).
Agent-specific vs the paper's critic isolates STRUCTURE (both see everything).

Held fixed: the critic network (RSUCritic, same hidden size), its optimiser and
learning rate, per-agent GAE, advantage normalisation, and the PPO actor updates,
which run unchanged in the parent class.

How it fits the unchanged runners. The global state is every vehicle's
observation, concatenated in vehicle order, followed by M channel-load counts,
so vehicle i's observation is gs[i*obs_dim:(i+1)*obs_dim]. The runner calls
value(gs) once per rollout step and once more for the bootstrap, then
compute_gae. value() computes all N per-vehicle values, caches them, and returns
their mean (the runner only stores it). compute_gae ignores the runner's values
and uses the cache -- T step entries plus one bootstrap entry, checked, or it
stops with an error. update_from_tensors gives the parent update this critic's
features in place of the global state, so the parent's own critic-loss line
trains the new critic on each vehicle's own return.
"""
import numpy as np
import torch
import torch.optim as optim

from agents.critic import RSUCritic
from agents.mappo_gaefix import MAPPOTrainerGAEFix, MAPPOTrainerMode0cGAEFix


def per_agent_gae_vec(rewards, values_tn, next_vals, dones, n_agents, gamma, lam):
    """Per-agent GAE with a separate value for every vehicle.
    rewards, dones: flat, interleaved (index t*N + i). values_tn: (T, N).
    next_vals: (N,) bootstrap values. Returns flat advantages and returns."""
    rewards = np.asarray(rewards, dtype=np.float32)
    dones = np.asarray(dones, dtype=np.float32)
    if len(rewards) % n_agents:
        raise ValueError(f"buffer length {len(rewards)} is not a multiple of N = {n_agents}")
    T = len(rewards) // n_agents
    V = np.asarray(values_tn, dtype=np.float32)
    if V.shape != (T, n_agents):
        raise ValueError(f"values have shape {V.shape}, expected {(T, n_agents)}")
    nv0 = np.asarray(next_vals, dtype=np.float32).reshape(-1)
    if nv0.shape != (n_agents,):
        raise ValueError(f"bootstrap values have shape {nv0.shape}, expected {(n_agents,)}")
    R = rewards.reshape(T, n_agents)
    D = dones.reshape(T, n_agents)
    A = np.zeros_like(R)
    for i in range(n_agents):
        gae, nv = 0.0, float(nv0[i])
        for t in reversed(range(T)):
            delta = R[t, i] + gamma * nv * (1 - D[t, i]) - V[t, i]
            gae = delta + gamma * lam * (1 - D[t, i]) * gae
            A[t, i] = gae
            nv = V[t, i]
    adv = A.reshape(-1)
    return adv, adv + V.reshape(-1)


class _PerVehicleCriticMixin:
    """Replaces the team-value critic with a per-vehicle one. Must come first in the MRO."""
    FEATURES = None  # "local" or "agentspecific"

    def _setup_critic(self, cfg, obs_dim, global_state_dim):
        if self.FEATURES not in ("local", "agentspecific"):
            raise ValueError(f"unknown FEATURES {self.FEATURES!r}")
        self._od = obs_dim
        self._gsd = global_state_dim
        if global_state_dim < self.N * obs_dim:
            raise ValueError("global state is shorter than N observations")
        in_dim = obs_dim if self.FEATURES == "local" else global_state_dim + obs_dim
        self.critic = RSUCritic(in_dim).to(self.device)
        self.opt_c = optim.Adam(self.critic.parameters(), lr=cfg.get("lr_critic", 3e-4))
        self.critic_in_dim = in_dim
        self._vcache = []

    def _features_np(self, global_state):
        gs = np.asarray(global_state, dtype=np.float32).reshape(-1)
        if gs.shape != (self._gsd,):
            raise ValueError(f"global state has length {gs.shape[0]}, expected {self._gsd}")
        own = gs[: self.N * self._od].reshape(self.N, self._od)
        if self.FEATURES == "local":
            return own
        return np.concatenate([np.repeat(gs[None, :], self.N, axis=0), own], axis=1)

    def _features_t(self, buf):
        if self.FEATURES == "local":
            return buf["obs"]
        return torch.cat([buf["global_state"], buf["obs"]], dim=1)

    def value(self, global_state):
        x = torch.FloatTensor(self._features_np(global_state)).to(self.device)
        with torch.no_grad():
            v = self.critic(x).squeeze(-1).cpu().numpy().astype(np.float32)
        self._vcache.append(v)
        return float(v.mean())

    def compute_gae(self, rewards, values, next_val, dones):
        # the runner's `values` and `next_val` are team-level placeholders; use the cache
        T = len(rewards) // self.N
        n = len(self._vcache)
        if n != T + 1:
            self._vcache = []
            raise RuntimeError(f"critic cache holds {n} value() calls; expected {T + 1} "
                               f"({T} rollout steps + 1 bootstrap)")
        V = np.stack(self._vcache[:T])
        nv = self._vcache[T]
        self._vcache = []
        return per_agent_gae_vec(rewards, V, nv, dones, self.N, self.gamma, self.lam)

    def update_from_tensors(self, buf):
        b = dict(buf)
        b["global_state"] = self._features_t(buf)   # the parent's critic line now trains this critic
        return super().update_from_tensors(b)


class MAPPOTrainerLocalCritic(_PerVehicleCriticMixin, MAPPOTrainerGAEFix):
    """Mode 0a, local critic V(o_i)."""
    FEATURES = "local"

    def __init__(self, cfg, obs_dim, action_dim, global_state_dim):
        super().__init__(cfg, obs_dim, action_dim, global_state_dim)
        self._setup_critic(cfg, obs_dim, global_state_dim)


class MAPPOTrainerASCritic(_PerVehicleCriticMixin, MAPPOTrainerGAEFix):
    """Mode 0a, agent-specific global critic V(gs (+) o_i)."""
    FEATURES = "agentspecific"

    def __init__(self, cfg, obs_dim, action_dim, global_state_dim):
        super().__init__(cfg, obs_dim, action_dim, global_state_dim)
        self._setup_critic(cfg, obs_dim, global_state_dim)


class MAPPOTrainerMode0cLocalCritic(_PerVehicleCriticMixin, MAPPOTrainerMode0cGAEFix):
    """Mode 0c, local critic V(o_i)."""
    FEATURES = "local"

    def __init__(self, cfg, obs_dim, action_dim, global_state_dim):
        super().__init__(cfg, obs_dim, action_dim, global_state_dim)
        self._setup_critic(cfg, obs_dim, global_state_dim)


class MAPPOTrainerMode0cASCritic(_PerVehicleCriticMixin, MAPPOTrainerMode0cGAEFix):
    """Mode 0c, agent-specific global critic V(gs (+) o_i)."""
    FEATURES = "agentspecific"

    def __init__(self, cfg, obs_dim, action_dim, global_state_dim):
        super().__init__(cfg, obs_dim, action_dim, global_state_dim)
        self._setup_critic(cfg, obs_dim, global_state_dim)
