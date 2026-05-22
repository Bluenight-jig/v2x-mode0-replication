"""
train.py — MAPPO training, Path A (per-class actors).

Sweep N over: 2, 3, 4, 5, 7, 10  (Config A v3).
For each run, edit the n_vehicles entry in CFG (or use the sed one-liner).
"""
import os, json, pathlib
import numpy as np, torch
from torch.utils.tensorboard import SummaryWriter
from envs.v2x_env  import V2XEnv
from agents.mappo  import MAPPOTrainer

CFG = {
    # ── Environment ──────────────────────────────────────────────────
    "n_vehicles":     4,    # ← CHANGE THIS for each run: 2, 3, 4, 5, 7, 10
    "n_subchannels":  5,
    "sumo_cfg":       "scenarios/highway/highway.sumocfg",
    "max_steps":      200,
    "ema_alpha":      0.3,

    # ── PSM-001: M0/M1 traffic differentiation ────────────────────────
    "m0_ratio":            0.50,
    "m0_subchannel_count": 2,
    "demand_separation":   False,

    # ── Reward weights (Fix A3) ───────────────────────────────────────
    "alpha":      1.0,
    "beta":       0.3,
    "gamma_team": 0.5,
    "delta":      0.3,
    "eta":        0.3,

    # ── Communication bridge ─────────────────────────────────────────
    "bridge_binary": os.path.expanduser("~/v2x_thesis/ns3_bridge/v2x_bridge"),
    "ns3_port":      5556,
    "fc_ghz":         5.9,

    # ── MAPPO hyperparameters ─────────────────────────────────────────
    "gamma":       0.99,
    "gae_lambda":  0.95,
    "clip_eps":    0.2,
    "ppo_epochs":  4,
    "lr_actor":    3e-4,
    "lr_critic":   3e-4,
    "lr_min":      1e-5,

    # ── Entropy schedule (now owned by trainer) ───────────────────────
    "ent_coef_start": 0.05,
    "ent_coef_end":   0.001,

    # ── Training schedule ─────────────────────────────────────────────
    "n_episodes":      3000,
    "rollout_steps":   200,
    "n_eval_episodes": 100,

    # ── Result tracking ───────────────────────────────────────────────
    "config_label": "A_N4_repro_smoke",
    "result_file":  "results/_reproducibility_smoke.json",
}


def move_to_device(buf: dict, device: torch.device) -> dict:
    """Transfer rollout buffer to GPU once before PPO epochs loop."""
    return {
        "obs":          torch.FloatTensor(buf["obs"]).to(device, non_blocking=True),
        "global_state": torch.FloatTensor(buf["global_state"]).to(device, non_blocking=True),
        "actions":      torch.LongTensor(buf["actions"]).to(device, non_blocking=True),
        "log_probs":    torch.FloatTensor(buf["log_probs"]).to(device, non_blocking=True),
        "advantages":   torch.FloatTensor(buf["advantages"]).to(device, non_blocking=True),
        "returns":      torch.FloatTensor(buf["returns"]).to(device, non_blocking=True),
        "classes":      torch.LongTensor(buf["classes"]).to(device, non_blocking=True),
    }


def main():
    if torch.cuda.is_available():
        device = torch.device("cuda:0")
        torch.backends.cudnn.benchmark = True
        print(f"GPU : {torch.cuda.get_device_name(0)}")
        print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory/1e9:.1f} GB")
    else:
        device = torch.device("cpu")
        print("WARNING: CUDA not available — falling back to CPU.")

    obs_dim          = 3 + CFG["n_subchannels"] + 3   # Path A.5 (added vehicle_id_norm)
    action_dim       = CFG["n_subchannels"] * 5
    global_state_dim = CFG["n_vehicles"] * obs_dim + CFG["n_subchannels"]

    env     = V2XEnv(CFG)
    trainer = MAPPOTrainer(CFG, obs_dim, action_dim, global_state_dim)

    N = CFG["n_vehicles"]
    writer = SummaryWriter(f"logs/config_{CFG['config_label']}_N{N}_v3p5")

    print(f"Device          : {device}")
    print(f"Actor M0 params : {sum(p.numel() for p in trainer.actor_m0.parameters()):,}")
    print(f"Actor M1 params : {sum(p.numel() for p in trainer.actor_m1.parameters()):,}")
    print(f"Critic params   : {sum(p.numel() for p in trainer.critic.parameters()):,}")
    print(f"obs_dim={obs_dim}  action_dim={action_dim}  gs_dim={global_state_dim}")
    print(f"M0 count: {env.m0_count}  M1 count: {env.m1_count}")

    # ── LR schedulers — one per optimizer ─────────────────────────────
    sched_a_m0 = torch.optim.lr_scheduler.CosineAnnealingLR(
        trainer.opt_a_m0, T_max=CFG["n_episodes"], eta_min=CFG["lr_min"])
    sched_a_m1 = torch.optim.lr_scheduler.CosineAnnealingLR(
        trainer.opt_a_m1, T_max=CFG["n_episodes"], eta_min=CFG["lr_min"])
    sched_c    = torch.optim.lr_scheduler.CosineAnnealingLR(
        trainer.opt_c,    T_max=CFG["n_episodes"], eta_min=CFG["lr_min"])

    n_ep = CFG["n_episodes"]
    print(f"Training for {n_ep} episodes (Config {CFG['config_label']}, N={N})...\n")

    m0_pdr_history = []
    m1_pdr_history = []

    # ════════════════════════════════════════════════════════════════
    # TRAINING PHASE
    # ════════════════════════════════════════════════════════════════
    vc = env.vehicle_classes   # (N,) — fixed across episodes

    for ep in range(n_ep):
        trainer.step_entropy_schedule()

        obs, info = env.reset()
        buf = {k: [] for k in ["obs", "global_state", "actions",
                                "log_probs", "rewards", "dones", "values",
                                "classes"]}
        ep_rew = 0.

        ep_pdr_m0_steps  = []
        ep_pdr_m1_steps  = []
        ep_sinr_m0_steps = []
        ep_sinr_m1_steps = []
        ep_collision_steps = []   # Path A.5
        ep_m0_coll_wp_steps = []
        ep_m1_coll_wp_steps = []

        for _ in range(CFG["rollout_steps"]):
            gs   = info["global_state"]
            val  = trainer.value(gs)
            acts, lps = trainer.select_actions(obs, vc, deterministic=False)
            nobs, rew, done, trunc, info = env.step(acts)

            if len(info["pdr_m0"]) > 0:
                ep_pdr_m0_steps.append(float(info["pdr_m0"].mean()))
                ep_sinr_m0_steps.append(float(info["sinr_dB_m0"].mean()))
            if len(info["pdr_m1"]) > 0:
                ep_pdr_m1_steps.append(float(info["pdr_m1"].mean()))
                ep_sinr_m1_steps.append(float(info["sinr_dB_m1"].mean()))
            ep_collision_steps.append(float(info.get("m0_collision_rate", 0.0)))
            ep_m0_coll_wp_steps.append(float(info.get("m0_collision_rate_within_pool", 0.0)))
            ep_m1_coll_wp_steps.append(float(info.get("m1_collision_rate_within_pool", 0.0)))

            for i in range(CFG["n_vehicles"]):
                buf["obs"].append(obs[i])
                buf["global_state"].append(gs)
                buf["actions"].append(acts[i])
                buf["log_probs"].append(lps[i])
                buf["rewards"].append(rew[i])
                buf["dones"].append(float(done or trunc))
                buf["values"].append(val)
                buf["classes"].append(int(vc[i]))

            obs    = nobs
            ep_rew += rew.mean()
            if done or trunc:
                break

        # ── PPO update ────────────────────────────────────────────────
        for k in buf:
            buf[k] = np.array(buf[k], dtype=np.float32 if k != "classes" and k != "actions" else np.int64)
        next_val = trainer.value(info["global_state"])
        buf["advantages"], buf["returns"] = trainer.compute_gae(
            buf["rewards"].astype(np.float32), buf["values"].astype(np.float32),
            next_val, buf["dones"].astype(np.float32))

        gpu_buf = move_to_device(buf, trainer.device)
        stats   = trainer.update_from_tensors(gpu_buf)
        del gpu_buf
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        sched_a_m0.step(); sched_a_m1.step(); sched_c.step()

        # ── TensorBoard logging ────────────────────────────────────────
        if ep_pdr_m0_steps:
            ep_m0_pdr  = float(np.mean(ep_pdr_m0_steps))
            ep_m0_p05  = float(np.percentile(ep_pdr_m0_steps, 5))
            ep_m0_sinr = float(np.mean(ep_sinr_m0_steps))
            m0_pdr_history.append(ep_m0_pdr)
            writer.add_scalar("m0_pdr_mean",  ep_m0_pdr,  ep)
            writer.add_scalar("m0_pdr_p05",   ep_m0_p05,  ep)
            writer.add_scalar("m0_sinr_mean", ep_m0_sinr, ep)
        if ep_pdr_m1_steps:
            ep_m1_pdr  = float(np.mean(ep_pdr_m1_steps))
            ep_m1_sinr = float(np.mean(ep_sinr_m1_steps))
            m1_pdr_history.append(ep_m1_pdr)
            writer.add_scalar("m1_pdr_mean",  ep_m1_pdr,  ep)
            writer.add_scalar("m1_sinr_mean", ep_m1_sinr, ep)
        if ep_collision_steps:
            writer.add_scalar("m0_collision_rate", float(np.mean(ep_collision_steps)), ep)
        if ep_m0_coll_wp_steps:
            writer.add_scalar("m0_collision_rate_within_pool", float(np.mean(ep_m0_coll_wp_steps)), ep)
        if ep_m1_coll_wp_steps:
            writer.add_scalar("m1_collision_rate_within_pool", float(np.mean(ep_m1_coll_wp_steps)), ep)

        writer.add_scalar("ep_reward",      ep_rew,                          ep)
        writer.add_scalar("actor_loss",     stats["actor_loss"],             ep)
        writer.add_scalar("actor_loss_m0",  stats["actor_loss_m0"],          ep)
        writer.add_scalar("actor_loss_m1",  stats["actor_loss_m1"],          ep)
        writer.add_scalar("critic_loss",    stats["critic_loss"],            ep)
        writer.add_scalar("entropy",        stats["entropy"],                ep)
        writer.add_scalar("entropy_m0",     stats["entropy_m0"],             ep)
        writer.add_scalar("entropy_m1",     stats["entropy_m1"],             ep)
        writer.add_scalar("ent_coef",       trainer.ent_c,                   ep)
        writer.add_scalar("lr_actor",       sched_a_m0.get_last_lr()[0],     ep)

        if ep % 50 == 0:
            m0p = m0_pdr_history[-1] if m0_pdr_history else 0.
            m1p = m1_pdr_history[-1] if m1_pdr_history else 0.
            vram_s = ""
            if torch.cuda.is_available():
                vram_s = f" VRAM {torch.cuda.max_memory_allocated()/1e6:.0f}MB"
                torch.cuda.reset_peak_memory_stats()
            print(f"Ep{ep:5d} | M0_PDR {m0p:.3f} | M1_PDR {m1p:.3f} | "
                  f"H_m0 {stats['entropy_m0']:.2f} H_m1 {stats['entropy_m1']:.2f} | "
                  f"a{stats['actor_loss']:.4f} c{stats['critic_loss']:.4f} "
                  f"ent{trainer.ent_c:.4f}{vram_s}")

    # ════════════════════════════════════════════════════════════════
    # EVALUATION PHASE (stochastic at terminal entropy)
    # ════════════════════════════════════════════════════════════════
    print(f"\nTraining complete. Running {CFG['n_eval_episodes']} eval episodes...\n")

    eval_m0_pdr  = []
    eval_m1_pdr  = []
    eval_m0_p05  = []
    eval_m0_sinr = []
    eval_m1_sinr = []
    eval_m0_collision = []   # Path A.5
    eval_m0_coll_wp = []
    eval_m1_coll_wp = []

    for ev in range(CFG["n_eval_episodes"]):
        obs, info = env.reset()
        ep_pdr_m0  = []
        ep_pdr_m1  = []
        ep_sinr_m0 = []
        ep_sinr_m1 = []
        ep_collision_steps_eval = []   # Path A.5
        ep_m0_coll_wp_eval = []
        ep_m1_coll_wp_eval = []

        for _ in range(CFG["rollout_steps"]):
            acts, _ = trainer.select_actions(obs, vc, deterministic=False)
            nobs, _, done, trunc, info = env.step(acts)
            if len(info["pdr_m0"]) > 0:
                ep_pdr_m0.append(float(info["pdr_m0"].mean()))
                ep_sinr_m0.append(float(info["sinr_dB_m0"].mean()))
            if len(info["pdr_m1"]) > 0:
                ep_pdr_m1.append(float(info["pdr_m1"].mean()))
                ep_sinr_m1.append(float(info["sinr_dB_m1"].mean()))
            ep_collision_steps_eval.append(float(info.get("m0_collision_rate", 0.0)))
            ep_m0_coll_wp_eval.append(float(info.get("m0_collision_rate_within_pool", 0.0)))
            ep_m1_coll_wp_eval.append(float(info.get("m1_collision_rate_within_pool", 0.0)))
            obs = nobs
            if done or trunc:
                break

        if ep_pdr_m0:
            eval_m0_pdr.append(float(np.mean(ep_pdr_m0)))
            eval_m0_p05.append(float(np.percentile(ep_pdr_m0, 5)))
            eval_m0_sinr.append(float(np.mean(ep_sinr_m0)))
        if ep_pdr_m1:
            eval_m1_pdr.append(float(np.mean(ep_pdr_m1)))
            eval_m1_sinr.append(float(np.mean(ep_sinr_m1)))
        if ep_collision_steps_eval:
            eval_m0_collision.append(float(np.mean(ep_collision_steps_eval)))
        if ep_m0_coll_wp_eval:
            eval_m0_coll_wp.append(float(np.mean(ep_m0_coll_wp_eval)))
        if ep_m1_coll_wp_eval:
            eval_m1_coll_wp.append(float(np.mean(ep_m1_coll_wp_eval)))

        if ev % 20 == 0:
            m0p = eval_m0_pdr[-1] if eval_m0_pdr else 0.
            m1p = eval_m1_pdr[-1] if eval_m1_pdr else 0.
            print(f"  Eval ep {ev:3d}: M0_PDR={m0p:.3f}  M1_PDR={m1p:.3f}")

    m0_mean      = float(np.mean(eval_m0_pdr))         if eval_m0_pdr else 0.0
    m0_p05_inter = float(np.percentile(eval_m0_pdr,5)) if eval_m0_pdr else 0.0
    m0_p05_intra = float(np.mean(eval_m0_p05))         if eval_m0_p05 else 0.0
    m1_mean      = float(np.mean(eval_m1_pdr))         if eval_m1_pdr else 0.0
    m0_sinr      = float(np.mean(eval_m0_sinr))        if eval_m0_sinr else 0.0
    m1_sinr      = float(np.mean(eval_m1_sinr))        if eval_m1_sinr else 0.0
    m0_coll = float(np.mean(eval_m0_collision)) if eval_m0_collision else 0.0
    m0_coll_wp = float(np.mean(eval_m0_coll_wp)) if eval_m0_coll_wp else 0.0
    m1_coll_wp = float(np.mean(eval_m1_coll_wp)) if eval_m1_coll_wp else 0.0

    print(f"\n=== Eval Summary ({CFG['n_eval_episodes']} episodes) ===")
    print(f"  M0_PDR_mean         : {m0_mean:.4f}")
    print(f"  M0_PDR_p05 (across) : {m0_p05_inter:.4f}")
    print(f"  M0_PDR_p05 (intra)  : {m0_p05_intra:.4f}")
    print(f"  M1_PDR_mean         : {m1_mean:.4f}")
    print(f"  M0_SINR_mean (dB)   : {m0_sinr:.2f}")
    print(f"  M1_SINR_mean (dB)   : {m1_sinr:.2f}")
    print(f"  M0_collision_rate   : {m0_coll:.3f}")
    print(f"  M0_collision_within_pool : {m0_coll_wp:.3f}")
    print(f"  M1_collision_within_pool : {m1_coll_wp:.3f}")

    # ── Save result ───────────────────────────────────────────────────
    pathlib.Path("results").mkdir(exist_ok=True)
    result_record = {
        "config":            CFG["config_label"],
        "N":                 CFG["n_vehicles"],
        "M":                 CFG["n_subchannels"],
        "rho":               round(CFG["n_vehicles"] / CFG["n_subchannels"], 2),
        "M_m0":              CFG["m0_subchannel_count"] if CFG["demand_separation"] else None,
        "m0_ratio":          CFG["m0_ratio"],
        "m0_pdr_mean":       m0_mean,
        "m0_pdr_p05":        m0_p05_inter,
        "m0_pdr_p05_intra":  m0_p05_intra,
        "m1_pdr_mean":       m1_mean,
        "m0_sinr_mean":      m0_sinr,
        "m1_sinr_mean":      m1_sinr,
        "m0_collision_rate": m0_coll,
        "m0_collision_rate_within_pool": m0_coll_wp,
        "m1_collision_rate_within_pool": m1_coll_wp,
        "train_m0_pdr_last100": float(np.mean(m0_pdr_history[-100:])) if len(m0_pdr_history) >= 100 else 0.0,
        "train_m1_pdr_last100": float(np.mean(m1_pdr_history[-100:])) if len(m1_pdr_history) >= 100 else 0.0,
        "actor_loss":        float(stats["actor_loss"]),
        "actor_loss_m0":     float(stats["actor_loss_m0"]),
        "actor_loss_m1":     float(stats["actor_loss_m1"]),
        "critic_loss":       float(stats["critic_loss"]),
        "entropy_m0":        float(stats["entropy_m0"]),
        "entropy_m1":        float(stats["entropy_m1"]),
        "n_episodes":        CFG["n_episodes"],
        "n_eval_episodes":   CFG["n_eval_episodes"],
        "architecture":      "path_A5_per_class_actors_with_id",
    }
    result_path = pathlib.Path(CFG["result_file"])
    existing = json.loads(result_path.read_text()) if result_path.exists() else []
    existing = [r for r in existing
                if not (r["config"] == result_record["config"]
                        and r["N"]   == result_record["N"]
                        and r["M"]   == result_record["M"])]
    existing.append(result_record)
    result_path.write_text(json.dumps(existing, indent=2))
    print(f"\nResult saved -> {CFG['result_file']}")
    print(f"  N={CFG['n_vehicles']}  rho={result_record['rho']}  "
          f"M0_PDR={m0_mean:.3f}  M1_PDR={m1_mean:.3f}")

    env.close()
    writer.close()
    print("Run complete.")


if __name__ == "__main__":
    main()
