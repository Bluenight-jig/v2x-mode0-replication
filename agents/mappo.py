"""
mappo.py — Multi-Agent PPO trainer with Path A: per-class actor specialisation.

Architecture:
  - Two actor networks: actor_m0 and actor_m1.
  - Routing by traffic class: M0 vehicles use actor_m0, M1 vehicles use actor_m1.
  - Single centralized critic (unchanged from prior version).
  - Separate Adam optimizers per actor; one for the critic.

CTDE preserved:
  - Each actor sees only its own class's local observations (decentralized exec).
  - Critic sees global state during training.
"""
import numpy as np
import torch
import torch.optim as optim
from agents.actor  import VehicleActor
from agents.critic import RSUCritic


class MAPPOTrainer:

    def __init__(self, cfg: dict,
                 obs_dim: int, action_dim: int, global_state_dim: int):
        self.cfg = cfg
        self.N   = cfg["n_vehicles"]

        # ── Hyperparameters ───────────────────────────────────────────
        self.gamma  = cfg.get("gamma",      0.99)
        self.lam    = cfg.get("gae_lambda", 0.95)
        self.eps    = cfg.get("clip_eps",   0.2)
        self.epochs = cfg.get("ppo_epochs", 4)

        # ── Entropy schedule (owned by trainer; advanced via step_entropy_schedule) ──
        self._ent_c_start = cfg.get("ent_coef_start", cfg.get("ent_coef", 0.01))
        self._ent_c_end   = cfg.get("ent_coef_end",   cfg.get("ent_coef", 0.01))
        self._ent_c_total = cfg.get("n_episodes", 1)
        self._ent_c_step  = 0
        self.ent_c        = self._ent_c_start

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # ── Path A: TWO actor networks (one per class) ────────────────
        self.actor_m0 = VehicleActor(obs_dim, action_dim).to(self.device)
        self.actor_m1 = VehicleActor(obs_dim, action_dim).to(self.device)

        # Centralized critic (unchanged)
        self.critic = RSUCritic(global_state_dim).to(self.device)

        # ── Optimizers — separate per actor + one for critic ─────────
        self.opt_a_m0 = optim.Adam(self.actor_m0.parameters(),
                                   lr=cfg.get("lr_actor",  3e-4))
        self.opt_a_m1 = optim.Adam(self.actor_m1.parameters(),
                                   lr=cfg.get("lr_actor",  3e-4))
        self.opt_c    = optim.Adam(self.critic.parameters(),
                                   lr=cfg.get("lr_critic", 3e-4))

        # Backwards-compat aliases (so external code expecting trainer.opt_a/.actor
        # doesn't break — these point to the M0 actor as a default).
        self.actor   = self.actor_m0
        self.opt_a   = self.opt_a_m0

    # ------------------------------------------------------------------
    def step_entropy_schedule(self) -> float:
        """Advance entropy schedule by one episode. Returns the new ent_c."""
        progress = self._ent_c_step / max(self._ent_c_total - 1, 1)
        progress = min(progress, 1.0)
        self.ent_c = self._ent_c_start + (
            self._ent_c_end - self._ent_c_start) * progress
        self._ent_c_step += 1
        return self.ent_c

    # ------------------------------------------------------------------
    def select_actions(self, obs: np.ndarray, vehicle_classes: np.ndarray,
                       deterministic: bool = False):
        """
        Path A: route each vehicle's observation to its class's actor.
        obs              : (N, obs_dim) numpy array.
        vehicle_classes  : (N,) int array, 0 = M0, 1 = M1.
        deterministic    : argmax (eval) vs sample (training).
        Returns: actions (N,) int, log_probs (N,) float.
        """
        actions   = np.zeros(self.N, dtype=np.int64)
        log_probs = np.zeros(self.N, dtype=np.float32)

        m0_idx = np.where(vehicle_classes == 0)[0]
        m1_idx = np.where(vehicle_classes == 1)[0]

        with torch.no_grad():
            if len(m0_idx) > 0:
                t = torch.FloatTensor(obs[m0_idx]).to(self.device)
                a, lp, _ = self.actor_m0.get_action(t, deterministic=deterministic)
                actions[m0_idx]   = a.cpu().numpy()
                log_probs[m0_idx] = lp.cpu().numpy()
            if len(m1_idx) > 0:
                t = torch.FloatTensor(obs[m1_idx]).to(self.device)
                a, lp, _ = self.actor_m1.get_action(t, deterministic=deterministic)
                actions[m1_idx]   = a.cpu().numpy()
                log_probs[m1_idx] = lp.cpu().numpy()

        return actions, log_probs

    # ------------------------------------------------------------------
    def value(self, global_state: np.ndarray) -> float:
        t = torch.FloatTensor(global_state).unsqueeze(0).to(self.device)
        with torch.no_grad():
            return self.critic(t).squeeze().cpu().item()

    # ------------------------------------------------------------------
    def compute_gae(self, rewards, values, next_val, dones):
        adv, gae = np.zeros_like(rewards), 0.
        for t in reversed(range(len(rewards))):
            delta = rewards[t] + self.gamma * next_val * (1 - dones[t]) - values[t]
            gae   = delta + self.gamma * self.lam * (1 - dones[t]) * gae
            adv[t] = gae
            next_val = values[t]
        returns = adv + values
        return adv, returns

    # ------------------------------------------------------------------
    def update_from_tensors(self, buf: dict) -> dict:
        """
        PPO update with per-class actor routing.
        buf must include a 'classes' tensor (LongTensor, shape (T,)).
        """
        obs_t = buf["obs"]
        gs_t  = buf["global_state"]
        act_t = buf["actions"]
        olp_t = buf["log_probs"]
        adv_t = buf["advantages"]
        ret_t = buf["returns"]
        cls_t = buf["classes"]   # (T,) LongTensor on device

        # Normalize advantages globally (across both classes)
        adv_t = (adv_t - adv_t.mean()) / (adv_t.std() + 1e-8)

        m0_mask = (cls_t == 0)
        m1_mask = (cls_t == 1)

        a_loss_m0_val = 0.0
        a_loss_m1_val = 0.0
        ent_m0_val    = 0.0
        ent_m1_val    = 0.0
        c_loss_val    = 0.0

        for _ in range(self.epochs):
            # ── Actor M0 update ───────────────────────────────────────
            if m0_mask.any():
                obs_m0  = obs_t[m0_mask]
                act_m0  = act_t[m0_mask]
                olp_m0  = olp_t[m0_mask]
                adv_m0  = adv_t[m0_mask]

                dist_m0 = self.actor_m0.forward(obs_m0)
                ratio   = torch.exp(dist_m0.log_prob(act_m0) - olp_m0)
                s1      = ratio * adv_m0
                s2      = torch.clamp(ratio, 1 - self.eps, 1 + self.eps) * adv_m0
                ent     = dist_m0.entropy().mean()
                a_loss_m0 = -torch.min(s1, s2).mean() - self.ent_c * ent

                self.opt_a_m0.zero_grad()
                a_loss_m0.backward()
                torch.nn.utils.clip_grad_norm_(self.actor_m0.parameters(), 0.5)
                self.opt_a_m0.step()
                a_loss_m0_val = a_loss_m0.item()
                ent_m0_val    = ent.item()

            # ── Actor M1 update ───────────────────────────────────────
            if m1_mask.any():
                obs_m1  = obs_t[m1_mask]
                act_m1  = act_t[m1_mask]
                olp_m1  = olp_t[m1_mask]
                adv_m1  = adv_t[m1_mask]

                dist_m1 = self.actor_m1.forward(obs_m1)
                ratio   = torch.exp(dist_m1.log_prob(act_m1) - olp_m1)
                s1      = ratio * adv_m1
                s2      = torch.clamp(ratio, 1 - self.eps, 1 + self.eps) * adv_m1
                ent     = dist_m1.entropy().mean()
                a_loss_m1 = -torch.min(s1, s2).mean() - self.ent_c * ent

                self.opt_a_m1.zero_grad()
                a_loss_m1.backward()
                torch.nn.utils.clip_grad_norm_(self.actor_m1.parameters(), 0.5)
                self.opt_a_m1.step()
                a_loss_m1_val = a_loss_m1.item()
                ent_m1_val    = ent.item()

            # ── Critic update (joint over both classes) ────────────────
            val    = self.critic(gs_t).squeeze(-1)
            c_loss = 0.5 * ((val - ret_t) ** 2).mean()
            self.opt_c.zero_grad()
            c_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 0.5)
            self.opt_c.step()
            c_loss_val = c_loss.item()

        # Aggregate stats — return weighted by class population for back-compat
        # with TensorBoard "actor_loss" / "entropy" single scalars.
        n_m0 = int(m0_mask.sum().item())
        n_m1 = int(m1_mask.sum().item())
        denom = max(n_m0 + n_m1, 1)
        actor_loss_avg = (a_loss_m0_val * n_m0 + a_loss_m1_val * n_m1) / denom
        entropy_avg    = (ent_m0_val    * n_m0 + ent_m1_val    * n_m1) / denom

        return {
            "actor_loss":     actor_loss_avg,
            "actor_loss_m0":  a_loss_m0_val,
            "actor_loss_m1":  a_loss_m1_val,
            "critic_loss":    c_loss_val,
            "entropy":        entropy_avg,
            "entropy_m0":     ent_m0_val,
            "entropy_m1":     ent_m1_val,
        }
