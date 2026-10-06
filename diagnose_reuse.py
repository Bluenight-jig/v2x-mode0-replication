"""
Spatial-reuse diagnostic for Mode 0c at N=10.

Loads a trained checkpoint, runs eval episodes while recording per-step
subchannel assignments and vehicle positions, then compares the separation
between co-channel vehicle pairs against a random-permutation baseline
built from the SAME positions (so geometry is controlled for).
"""
import argparse, json, sys, importlib.util
from pathlib import Path
import numpy as np
import torch

sys.path.insert(0, '.')


def load_runner(py_name):
    spec = importlib.util.spec_from_file_location("runner", py_name)
    mod = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(mod)
    except SystemExit:
        pass
    return mod


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--runner", default="train_b4_c1" + chr(46) + "py")
    ap.add_argument("--episodes", type=int, default=20)
    ap.add_argument("--port", type=int, default=5899)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    from envs.v2x_env import V2XEnv
    from agents.mappo_mode0c import MAPPOTrainerMode0c

    ck = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    CFG = dict(ck["cfg"])
    CFG["ns3_port"] = args.port
    CFG["n_eval_episodes"] = args.episodes

    N = CFG["n_vehicles"]; M = CFG["n_subchannels"]
    obs_dim = 3 + M + 3
    action_dim = M * 5
    gs_dim = N * obs_dim + M

    env = V2XEnv(CFG)
    env.reset()
    vc = env.vehicle_classes
    CFG["_vehicle_classes_for_logging"] = list(vc)
    tr = MAPPOTrainerMode0c(CFG, obs_dim, action_dim, gs_dim)
    for a, sd in zip(tr.actors, ck["actor_state_dicts"]):
        a.load_state_dict(sd)
    tr.critic.load_state_dict(ck["critic_state_dict"])

    print(f"  checkpoint : {Path(args.ckpt).name}")
    print(f"  N={N}  M={M}  classes={list(vc)}")
    print(f"  running {args.episodes} eval episodes...\n")

    learned, random_bl, chan_use = [], [], []
    m0m0_learned, m0m0_random = [], []
    rng = np.random.default_rng(12345)

    for ep in range(args.episodes):
        obs, _ = env.reset()
        for _ in range(CFG["rollout_steps"]):
            acts, _ = tr.select_actions(obs, vc, deterministic=False)
            pos = env._positions().copy()
            sub = (np.asarray(acts) // 5).astype(int)
            obs, _, done, trunc, _ = env.step(np.asarray(acts))

            live = np.where((pos[:, 0] != 0) | (pos[:, 1] != 0))[0]
            if len(live) < 2:
                if done or trunc: break
                continue

            chan_use.append(len(np.unique(sub[live])))

            def pairs(assign):
                same, m0 = [], []
                for i_i, i in enumerate(live):
                    for j in live[i_i + 1:]:
                        if assign[i] == assign[j]:
                            d = float(np.linalg.norm(pos[i] - pos[j]))
                            same.append(d)
                            if vc[i] == 0 and vc[j] == 0:
                                m0.append(d)
                return same, m0

            s, m = pairs(sub)
            learned += s; m0m0_learned += m

            perm = sub.copy()
            perm[live] = rng.permutation(sub[live])
            s, m = pairs(perm)
            random_bl += s; m0m0_random += m

            if done or trunc: break
        if (ep + 1) % 5 == 0:
            print(f"    episode {ep+1}/{args.episodes}")

    env.close()

    L = np.array(learned); R = np.array(random_bl)
    print("\n" + "=" * 66)
    print("  CO-CHANNEL PAIR SEPARATION (m)")
    print("=" * 66)
    for lab, v in (("learned policy", L), ("random baseline", R)):
        if len(v):
            print(f"  {lab:<18} n={len(v):6d}  mean {v.mean():8.1f}  "
                  f"median {np.median(v):8.1f}  p05 {np.percentile(v,5):8.1f}")
    if len(L) and len(R):
        ratio = L.mean() / R.mean()
        print(f"\n  ratio learned/random : {ratio:.3f}")
        print(f"  verdict              : "
              f"{'SPATIAL REUSE — co-channel pairs are farther apart than chance' if ratio > 1.15 else 'NO EVIDENCE of spatial separation' if ratio < 1.05 else 'WEAK / inconclusive'}")

    if m0m0_learned or m0m0_random:
        a = np.array(m0m0_learned); b = np.array(m0m0_random)
        print(f"\n  M0-M0 co-channel events: learned {len(a)}  vs random {len(b)}")
        if len(a):
            print(f"    learned mean separation {a.mean():.1f} m")
        else:
            print("    learned: ZERO M0-M0 co-channel events (perfect within-class separation)")

    if chan_use:
        c = np.array(chan_use)
        print(f"\n  distinct subchannels in use: mean {c.mean():.2f} of {M}")

    if args.out:
        Path(args.out).write_text(json.dumps({
            "checkpoint": args.ckpt,
            "learned_mean": float(L.mean()) if len(L) else None,
            "random_mean": float(R.mean()) if len(R) else None,
            "ratio": float(L.mean() / R.mean()) if len(L) and len(R) else None,
            "n_pairs": int(len(L)),
            "m0m0_learned_events": int(len(m0m0_learned)),
            "channels_used_mean": float(np.mean(chan_use)) if chan_use else None,
        }, indent=2))
        print(f"\n  saved -> {args.out}")


if __name__ == "__main__":
    main()
