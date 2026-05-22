#!/usr/bin/env python3
"""
eval_from_checkpoint.py - Load a pre-trained policy checkpoint and run eval episodes.

Usage:
    python scripts/eval_from_checkpoint.py --run D_N4
    python scripts/eval_from_checkpoint.py --run D_N10

Skips training and runs only the eval loop on saved policy weights. Useful
for verifying that the shipped checkpoints reproduce the canonical Phase D
numbers within single-seed sampling noise.
"""
import argparse
import json
import sys
from pathlib import Path

import torch
import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from envs.v2x_env import V2XEnv
from agents.mappo_mode0c import MAPPOTrainerMode0c


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True, choices=["D_N4", "D_N10"],
                    help="which checkpoint to evaluate")
    ap.add_argument("--n-eval-episodes", type=int, default=100,
                    help="number of eval episodes (default 100, matches production)")
    args = ap.parse_args()

    # Device selection mirrors train_mode0c.py
    if torch.cuda.is_available():
        device = torch.device("cuda:0")
        torch.backends.cudnn.benchmark = True
        print(f"GPU : {torch.cuda.get_device_name(0)}")
    else:
        device = torch.device("cpu")
        print("WARNING: CUDA not available -- falling back to CPU.")

    ckpt_path = REPO_ROOT / "checkpoints" / f"{args.run}_actors.pt"
    if not ckpt_path.is_file():
        sys.exit(f"ERROR: checkpoint not found at {ckpt_path}")

    print(f"\nLoading checkpoint: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    print(f"  config_label:      {ckpt['config_label']}")
    print(f"  N x M (M_m0):      {ckpt['N']} x {ckpt['M']} (M_m0={ckpt['M_m0']})")
    print(f"  demand_separation: {ckpt['demand_separation']}")
    print(f"  num actors:        {len(ckpt['actor_state_dicts'])}")

    # Reconstruct CFG and dims exactly as train_mode0c.py does
    CFG = dict(ckpt["cfg"])
    CFG["n_eval_episodes"] = args.n_eval_episodes

    obs_dim          = 3 + CFG["n_subchannels"] + 3
    action_dim       = CFG["n_subchannels"] * 5
    global_state_dim = CFG["n_vehicles"] * obs_dim + CFG["n_subchannels"]

    print(f"\nInitialising env and trainer...")
    env = V2XEnv(CFG)
    vc = env.vehicle_classes
    CFG["_vehicle_classes_for_logging"] = list(vc)
    trainer = MAPPOTrainerMode0c(CFG, obs_dim, action_dim, global_state_dim)

    # Load saved weights into trainer
    for actor, sd in zip(trainer.actors, ckpt["actor_state_dicts"]):
        actor.load_state_dict(sd)
    trainer.critic.load_state_dict(ckpt["critic_state_dict"])
    print("Checkpoint weights loaded into trainer.")

    # ----- Eval loop, mirroring train_mode0c.py exactly -----
    print(f"\nRunning {CFG['n_eval_episodes']} eval episodes...")
    eval_m0_pdr  = []
    eval_m1_pdr  = []
    eval_m0_p05  = []
    eval_m0_sinr = []
    eval_m1_sinr = []
    eval_m0_collision = []
    eval_m0_coll_wp = []
    eval_m1_coll_wp = []
    for ev in range(CFG["n_eval_episodes"]):
        obs, info = env.reset()
        ep_pdr_m0  = []
        ep_pdr_m1  = []
        ep_sinr_m0 = []
        ep_sinr_m1 = []
        ep_collision_steps_eval = []
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
    m0_coll      = float(np.mean(eval_m0_collision))   if eval_m0_collision else 0.0
    m0_coll_wp   = float(np.mean(eval_m0_coll_wp))     if eval_m0_coll_wp else 0.0
    m1_coll_wp   = float(np.mean(eval_m1_coll_wp))     if eval_m1_coll_wp else 0.0

    print()
    print("=" * 60)
    print(f"Eval Summary ({CFG['n_eval_episodes']} episodes) - checkpoint {args.run}")
    print("=" * 60)
    print(f"  M0_PDR_mean              : {m0_mean:.4f}")
    print(f"  M0_PDR_p05 (across)      : {m0_p05_inter:.4f}")
    print(f"  M0_PDR_p05 (intra)       : {m0_p05_intra:.4f}")
    print(f"  M1_PDR_mean              : {m1_mean:.4f}")
    print(f"  M0_SINR_mean (dB)        : {m0_sinr:.2f}")
    print(f"  M1_SINR_mean (dB)        : {m1_sinr:.2f}")
    print(f"  M0_collision_rate        : {m0_coll:.3f}")
    print(f"  M0_collision_within_pool : {m0_coll_wp:.3f}")
    print(f"  M1_collision_within_pool : {m1_coll_wp:.3f}")

    # Compare against canonical entry
    canonical_path = REPO_ROOT / "results" / "psm001_results.json"
    if canonical_path.is_file():
        canonical = json.loads(canonical_path.read_text())
        for r in canonical:
            if r["config"] == args.run:
                print()
                print(f"Canonical {args.run} reference (from psm001_results.json):")
                print(f"  M0_PDR_mean              : {r['m0_pdr_mean']:.4f}")
                print(f"  M0_collision_rate        : {r['m0_collision_rate']:.3f}")
                print(f"  M1_PDR_mean              : {r['m1_pdr_mean']:.4f}")
                print(f"  M0_SINR_mean (dB)        : {r['m0_sinr_mean']:.2f}")
                pdr_delta = m0_mean - r["m0_pdr_mean"]
                status = "PASS" if abs(pdr_delta) < 0.05 else "FAIL"
                print()
                print(f"  Delta vs canonical:")
                print(f"    M0_PDR delta           : {pdr_delta:+.4f}  [{status} at +/-0.05]")
                break

    env.close()


if __name__ == "__main__":
    main()
