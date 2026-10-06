"""
mappo_mode0b.py -- Mode 0b, the paper's hybrid fleet (Section 5.2; aa-achievement.md §7c, v52 and v57).

A NEW file; nothing existing is edited.

Passive vehicles cannot run an actor of their own, so the RCU serves them Mode 0a's shared per-class
policy. Active vehicles run their own Mode 0c actors. All are coordinated by the RCU and trained jointly.
Critic: agent-specific, V(global state (+) o_i) -- decision D5.

Construction -- a subclass of the Mode 0c trainer with per-agent GAE:
  * Action selection. Every passive vehicle's actor slot points at ONE shared actor per class (the first
    passive vehicle's), so passive vehicles of a class act from the same policy, each with its own
    observation and vehicle index -- exactly as in Mode 0a.
  * Update. Passive vehicles' transitions are routed to that shared actor, which therefore trains once per
    epoch on their POOLED samples (Mode 0a's update); active vehicles keep their own actors and samples.
    Routing happens inside update_from_tensors, AFTER the runner has computed per-agent advantages from the
    flat buffer by position (runner line 175, update line 180), so it cannot merge trajectories.
  * The unused slots' original actors and optimisers stay separate objects and never step -- the parent
    skips any vehicle with no transitions -- so the runner's one learning-rate schedule per optimiser is
    not doubled.
  * The parent averages per-class logging over all slots, counting unused slots as zero; corrected here.
    (The runner's "Actors: ... total params" line counts the shared actors once per slot; cosmetic.)

Passive set: the LAST k vehicles of each class, in vehicle order. Defaults (v52): 2 safety and 3 M1, for
N = 10. CFG keys "mode0b_passive_m0" and "mode0b_passive_m1" override; 0 and 0 give exactly Mode 0c with
the agent-specific critic.
"""
import torch

from agents.mappo_critic_ablation import _PerVehicleCriticMixin
from agents.mappo_gaefix import MAPPOTrainerMode0cGAEFix

PASSIVE_DEFAULT = {0: 2, 1: 3}


class _HybridActorsMixin:
    """Shares one actor per class among passive vehicles. Sits between the critic mixin and the Mode 0c trainer."""

    def _setup_hybrid(self, cfg):
        vc = cfg.get("_vehicle_classes_for_logging")
        if vc is None or len(vc) != self.N:
            raise ValueError("Mode 0b needs CFG['_vehicle_classes_for_logging'] (one class per vehicle) "
                             "set before the trainer is built")
        self._vc = [int(v) for v in vc]
        want = {0: int(cfg.get("mode0b_passive_m0", PASSIVE_DEFAULT[0])),
                1: int(cfg.get("mode0b_passive_m1", PASSIVE_DEFAULT[1]))}
        route = list(range(self.N))
        self.passive, self.shared_slot, aliased = {}, {}, []
        for c, k in want.items():
            idx = [i for i in range(self.N) if self._vc[i] == c]
            if not 0 <= k <= len(idx):
                raise ValueError(f"{k} passive vehicles requested in class M{c}, which has {len(idx)}")
            pas = idx[len(idx) - k:] if k else []
            self.passive[c] = pas
            if pas:
                rep = pas[0]
                self.shared_slot[c] = rep
                for j in pas[1:]:
                    self.actors[j] = self.actors[rep]      # same object: same policy for every passive vehicle
                    route[j] = rep
                    aliased.append(j)
        self.active = [i for i in range(self.N) if not any(i in p for p in self.passive.values())]
        self._aliased = sorted(aliased)
        self._route = torch.as_tensor(route, dtype=torch.long)

    def update_from_tensors(self, buf):
        b = dict(buf)
        vid = buf["vehicle_idx"]
        b["vehicle_idx"] = self._route.to(vid.device)[vid]   # passive -> their class's shared slot
        return self._correct_logging(super().update_from_tensors(b))

    def _correct_logging(self, stats):
        if not self._aliased:
            return stats
        out = dict(stats)
        groups = {"": list(range(self.N)),
                  "_m0": [i for i in range(self.N) if self._vc[i] == 0],
                  "_m1": [i for i in range(self.N) if self._vc[i] == 1]}
        for suffix, slots in groups.items():
            used = [i for i in slots if i not in self._aliased]
            if not used:
                continue
            for base in ("actor_loss", "entropy"):
                key = base + suffix
                if key in out:
                    out[key] = out[key] * len(slots) / len(used)
        return out


class MAPPOTrainerMode0bAS(_PerVehicleCriticMixin, _HybridActorsMixin, MAPPOTrainerMode0cGAEFix):
    """Mode 0b: passive vehicles on shared per-class actors, active on their own; agent-specific critic (D5)."""
    FEATURES = "agentspecific"

    def __init__(self, cfg, obs_dim, action_dim, global_state_dim):
        super().__init__(cfg, obs_dim, action_dim, global_state_dim)
        self._setup_critic(cfg, obs_dim, global_state_dim)
        self._setup_hybrid(cfg)
