"""
mappo_mode0c.py — Mode 0c trainer with per-vehicle actors.

Architecture:
  - One actor network per vehicle (N total).
  - Routing by vehicle index, not by class.
  - Centralized critic on global state (unchanged from Mode 0a).
  - All other hyperparameters identical to MAPPOTrainer.
"""
import numpy as np
import torch
import torch.optim as optim
from agents.actor  import VehicleActor
from agents.critic import RSUCritic


class MAPPOTrainerMode0c:

    def __init__(self, cfg: dict,
                 obs_dim: int, action_dim: int, global_state_dim: int):
        self.cfg = cfg
        self.N   = cfg["n_vehicles"]

        self.gamma  = cfg.get("gamma",      0.99)
        self.lam    = cfg.get("gae_lambda", 0.95)
        self.eps    = cfg.get("clip_eps",   0.2)
        self.epochs = cfg.get("ppo_epochs", 4)

        self._ent_c_start = cfg.get("ent_coef_start", cfg.get("ent_coef", 0.01))
        self._ent_c_end   = cfg.get("ent_coef_end",   cfg.get("ent_coef", 0.01))
        self._ent_c_total = cfg.get("n_episodes", 1)
        self._ent_c_step  = 0
        self.ent_c        = self._ent_c_start

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Mode 0c: ONE ACTOR PER VEHICLE (not per class)
        self.actors = [
            VehicleActor(obs_dim, action_dim).to(self.device)
            for _ in range(self.N)
        ]

        self.critic = RSUCritic(global_state_dim).to(self.device)

        self.opts_a = [
            optim.Adam(a.parameters(), lr=cfg.get("lr_actor", 3e-4))
            for a in self.actors
        ]
        self.opt_c = optim.Adam(self.critic.parameters(),
                                lr=cfg.get("lr_critic", 3e-4))

        # Backwards-compat aliases for code expecting trainer.actor / .opt_a
        self.actor = self.actors[0]
        self.opt_a = self.opts_a[0]

    def step_entropy_schedule(self) -> float:
        progress = self._ent_c_step / max(self._ent_c_total - 1, 1)
        progress = min(progress, 1.0)
        self.ent_c = self._ent_c_start + (
            self._ent_c_end - self._ent_c_start) * progress
        self._ent_c_step += 1
        return self.ent_c

    def select_actions(self, obs: np.ndarray, vehicle_classes: np.ndarray,
                       deterministic: bool = False):
        """
        Mode 0c: route each vehicle\'s observation to its OWN actor.
        vehicle_classes is accepted but not used for routing (kept for
        signature compat with the Mode 0a trainer).
        """
        actions   = np.zeros(self.N, dtype=np.int64)
        log_probs = np.zeros(self.N, dtype=np.float32)

        with torch.no_grad():
            for i in range(self.N):
                t = torch.FloatTensor(obs[i]).unsqueeze(0).to(self.device)
                a, lp, _ = self.actors[i].get_action(t, deterministic=deterministic)
                actions[i]   = a.cpu().numpy()[0]
                log_probs[i] = lp.cpu().numpy()[0]

        return actions, log_probs

    def value(self, global_state: np.ndarray) -> float:
        t = torch.FloatTensor(global_state).unsqueeze(0).to(self.device)
        with torch.no_grad():
            return self.critic(t).squeeze().cpu().item()

    def compute_gae(self, rewards, values, next_val, dones):
        adv, gae = np.zeros_like(rewards), 0.
        for t in reversed(range(len(rewards))):
            delta = rewards[t] + self.gamma * next_val * (1 - dones[t]) - values[t]
            gae   = delta + self.gamma * self.lam * (1 - dones[t]) * gae
            adv[t] = gae
            next_val = values[t]
        returns = adv + values
        return adv, returns

    def update_from_tensors(self, buf: dict) -> dict:
        """
        PPO update with per-vehicle actor routing.
        buf['vehicle_idx'] (LongTensor on device) routes each transition
        to the correct actor.
        """
        obs_t = buf["obs"]
        gs_t  = buf["global_state"]
        act_t = buf["actions"]
        olp_t = buf["log_probs"]
        adv_t = buf["advantages"]
        ret_t = buf["returns"]
        vid_t = buf["vehicle_idx"]

        adv_t = (adv_t - adv_t.mean()) / (adv_t.std() + 1e-8)

        # Per-vehicle masks (computed once)
        masks = [(vid_t == i) for i in range(self.N)]

        actor_losses_per = [0.0] * self.N
        entropies_per    = [0.0] * self.N
        c_loss_val       = 0.0

        for _ in range(self.epochs):
            for i in range(self.N):
                mask = masks[i]
                if not mask.any():
                    continue
                obs_i = obs_t[mask]
                act_i = act_t[mask]
                olp_i = olp_t[mask]
                adv_i = adv_t[mask]

                dist = self.actors[i].forward(obs_i)
                ratio = torch.exp(dist.log_prob(act_i) - olp_i)
                s1    = ratio * adv_i
                s2    = torch.clamp(ratio, 1 - self.eps, 1 + self.eps) * adv_i
                ent   = dist.entropy().mean()
                a_loss = -torch.min(s1, s2).mean() - self.ent_c * ent

                self.opts_a[i].zero_grad()
                a_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.actors[i].parameters(), 0.5)
                self.opts_a[i].step()
                actor_losses_per[i] = a_loss.item()
                entropies_per[i]    = ent.item()

            # Critic update (joint)
            val    = self.critic(gs_t).squeeze(-1)
            c_loss = 0.5 * ((val - ret_t) ** 2).mean()
            self.opt_c.zero_grad()
            c_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 0.5)
            self.opt_c.step()
            c_loss_val = c_loss.item()

        # Aggregate stats. Compute mean across vehicles, plus per-class means
        # for backward compat with the existing TensorBoard tags.
        vc = np.array(self.cfg.get("_vehicle_classes_for_logging", [0]*self.N))
        m0_idx = [i for i in range(self.N) if vc[i] == 0]
        m1_idx = [i for i in range(self.N) if vc[i] == 1]

        a_loss_m0 = float(np.mean([actor_losses_per[i] for i in m0_idx])) if m0_idx else 0.0
        a_loss_m1 = float(np.mean([actor_losses_per[i] for i in m1_idx])) if m1_idx else 0.0
        ent_m0    = float(np.mean([entropies_per[i]    for i in m0_idx])) if m0_idx else 0.0
        ent_m1    = float(np.mean([entropies_per[i]    for i in m1_idx])) if m1_idx else 0.0
        actor_loss_avg = float(np.mean(actor_losses_per))
        entropy_avg    = float(np.mean(entropies_per))

        return {
            "actor_loss":     actor_loss_avg,
            "actor_loss_m0":  a_loss_m0,
            "actor_loss_m1":  a_loss_m1,
            "critic_loss":    c_loss_val,
            "entropy":        entropy_avg,
            "entropy_m0":     ent_m0,
            "entropy_m1":     ent_m1,
        }
