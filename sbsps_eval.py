#!/usr/bin/env python3
"""
sbsps_eval.py -- idealized-sensing SB-SPS baseline (minimal, for the VC revision)

SB-SPS needs no training. It is written as a policy with the same interface as
the MAPPO trainers -- select_actions(obs, vc, deterministic) -> (actions, None) --
so it runs through the SAME evaluation loop as train.py, and every metric is
computed by identical code. Nothing in the environment or the bridge changes.

Design (playbook D12, with one proposed change):
  - sensing window K = 10 steps (1 s at 100 ms per step)
  - reselection counter drawn from U[5, 15]
  - probResourceKeep = 0: always reselect when the counter expires
  - fixed 23 dBm transmit power (power index 4)
  - respects the demand-separation mask when it is on
  - candidate rule, selected with --rule:
      rsrp       (proposed) exclude subchannel k for vehicle i if another vehicle
                 reserved k inside the window with received power above the
                 threshold; raise the threshold by 3 dB until at least
                 max(1, 20% of the pool) remain; pick uniformly from what remains.
                 Received power = 23 dBm - PL(d), using the bridge's own formula,
                 mean path loss only (no shadowing, no fading).
      occupancy  (D12 as written) exclude any subchannel anyone used inside the
                 window; if none remain, fall back to the least-occupied.

Idealizations (state these in the paper -- they cut both ways):
  - perfect decoding of every reservation: no hidden terminals,
    no half-duplex blindness ............................ favours SB-SPS
  - received power from mean path loss, no measurement noise ... favours SB-SPS
  - class-blind: no priority-dependent thresholds ......... favours Mode 0
  - one reservation per 100 ms interval, M subchannels (single slot)

Usage:
  python sbsps_eval.py --N 4  --seed 1 --port 6066
  python sbsps_eval.py --N 10 --seed 1 --port 6066 --episodes 5    # smoke test
"""
import argparse
import collections
import importlib.util
import json
import math
import pathlib
import sys

import numpy as np

sys.path.insert(0, ".")

FC_GHZ = 5.9            # matches the bridge's default --fcGHz
TX_DBM = 23.0           # fixed transmit power
POWER_IDX_23DBM = 4     # index of 23 dBm in V2XEnv.POWER_DBM = [-10, 0, 10, 16, 23]


def path_loss_db(d_m, fc_ghz=FC_GHZ):
    """Same formula as v2x_bridge.cc path_loss(); distance floored at 1 m."""
    d = np.maximum(np.asarray(d_m, dtype=np.float64), 1.0)
    return 32.4 + 20.0 * np.log10(fc_ghz) + 20.0 * np.log10(d)


class SBSPSPolicy:
    """Idealized-sensing SB-SPS. One instance controls all N vehicles."""

    def __init__(self, N, M, n_pwr, vehicle_classes, rule="rsrp", K=10,
                 counter_lo=5, counter_hi=15, p_keep=0.0, th0_dbm=-80.0,
                 min_frac=0.20, step_db=3.0, demand_separation=False,
                 M_m0=None, seed=0):
        if rule not in ("rsrp", "occupancy"):
            raise ValueError("rule must be 'rsrp' or 'occupancy'")
        if demand_separation and not M_m0:
            raise ValueError("demand_separation needs M_m0")
        self.N, self.M, self.n_pwr = int(N), int(M), int(n_pwr)
        self.vc = np.asarray(vehicle_classes, dtype=int)
        self.rule, self.K = rule, int(K)
        self.counter_lo, self.counter_hi = int(counter_lo), int(counter_hi)
        self.p_keep, self.th0, self.min_frac, self.step_db = p_keep, th0_dbm, min_frac, step_db
        self.ds, self.M_m0 = bool(demand_separation), (int(M_m0) if M_m0 else None)
        self.rng = np.random.default_rng(seed)
        self.stats = collections.Counter()   # diagnostics: how often thresholds had to rise
        self.reset()

    # -- state ------------------------------------------------------------
    def reset(self):
        self.subch = np.zeros(self.N, dtype=int)
        self.counter = np.zeros(self.N, dtype=int)     # 0 -> every vehicle selects at step 1
        self.hist = collections.deque(maxlen=self.K)   # past subchannel vectors

    def _pool(self, i):
        """Subchannels vehicle i may use."""
        if not self.ds:
            return np.arange(self.M)
        if self.vc[i] == 0:
            return np.arange(self.M_m0)
        return np.arange(self.M_m0, self.M)

    # -- sensing and selection --------------------------------------------
    def _rx_dbm(self, pos):
        """Received power at every vehicle from every other vehicle, N x N."""
        diff = pos[:, None, :] - pos[None, :, :]
        d = np.sqrt((diff ** 2).sum(-1))
        rx = TX_DBM - path_loss_db(d)
        np.fill_diagonal(rx, -np.inf)                  # a vehicle does not sense itself
        return rx

    def _candidates(self, i, rx):
        pool = self._pool(i)
        if not self.hist:                              # nothing sensed yet
            return pool
        H = np.array(self.hist)                        # (window, N)
        others = np.arange(self.N) != i

        if self.rule == "occupancy":
            occ = np.array([np.sum(H[:, others] == k) for k in pool])
            free = pool[occ == 0]
            if free.size:
                return free
            self.stats["occupancy_fallback"] += 1
            return pool[occ == occ.min()]

        # rsrp rule: strongest received power per subchannel inside the window
        strongest = np.full(pool.size, -np.inf)
        for idx, k in enumerate(pool):
            users = np.any(H == k, axis=0) & others    # vehicles that reserved k in the window
            if users.any():
                strongest[idx] = rx[i, users].max()
        need = max(1, math.ceil(self.min_frac * pool.size))
        th, raised = self.th0, 0
        cand = pool[strongest <= th]
        while cand.size < need:
            th += self.step_db
            raised += 1
            cand = pool[strongest <= th]
        if raised:
            self.stats["threshold_raised"] += 1
        return cand

    def select_actions(self, obs, vc=None, deterministic=False):
        """Same signature as the MAPPO trainers. Returns (actions, None)."""
        obs = np.asarray(obs, dtype=np.float64)
        pos = obs[:, :2] * 1000.0                      # obs[:, 0:2] = [x/1000, y/1000]
        rx = self._rx_dbm(pos)

        self.counter -= 1
        due = np.where(self.counter <= 0)[0]
        new = self.subch.copy()
        # Every vehicle that reselects this step uses the SAME past window --
        # none can see another's new choice yet. Simultaneous reselection is a
        # real SB-SPS failure mode, so it is kept, not smoothed away.
        for i in due:
            if self.hist and self.rng.random() < self.p_keep:
                self.stats["kept"] += 1
            else:
                new[i] = self.rng.choice(self._candidates(i, rx))
                self.stats["reselected"] += 1
            self.counter[i] = self.rng.integers(self.counter_lo, self.counter_hi + 1)
        self.subch = new
        self.hist.append(self.subch.copy())
        return (self.subch * self.n_pwr + POWER_IDX_23DBM).astype(np.int64), None


# -------------------------------------------------------------------------
def load_cfg(train_py="train" + chr(46) + "py"):
    spec = importlib.util.spec_from_file_location("t_cfg", train_py)
    mod = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(mod)
    except SystemExit:
        pass
    return dict(mod.CFG)


def co_channel_separation(pos, subch, live):
    """Distances between every pair of live vehicles that share a subchannel."""
    out = []
    idx = np.where(live)[0]
    for a in range(len(idx)):
        for b in range(a + 1, len(idx)):
            i, j = idx[a], idx[b]
            if subch[i] == subch[j]:
                out.append(float(np.linalg.norm(pos[i] - pos[j])))
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--N", type=int, required=True)
    ap.add_argument("--M", type=int, default=5)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--port", type=int, required=True)
    ap.add_argument("--episodes", type=int, default=100)
    ap.add_argument("--rule", choices=["rsrp", "occupancy"], default="rsrp")
    ap.add_argument("--th0", type=float, default=-80.0, help="starting threshold, dBm")
    ap.add_argument("--ds", action="store_true", help="demand separation on")
    ap.add_argument("--M_m0", type=int, default=2)
    ap.add_argument("--warmup", type=int, default=0,
                    help="steps at the start of each episode excluded from the metrics")
    ap.add_argument("--counter-lo", type=int, default=5, dest="counter_lo",
                    help="lowest reselection counter; set lo=hi=1 to remove semi-persistence")
    ap.add_argument("--counter-hi", type=int, default=15, dest="counter_hi")
    ap.add_argument("--label", default=None)
    args = ap.parse_args()

    from envs.v2x_env import V2XEnv

    np.random.seed(args.seed)
    CFG = load_cfg()
    CFG.update({"n_vehicles": args.N, "n_subchannels": args.M,
                "demand_separation": bool(args.ds), "m0_subchannel_count": args.M_m0,
                "n_eval_episodes": args.episodes, "ns3_port": args.port})
    tag = args.rule + ("" if args.rule == "occupancy" else f"{int(abs(args.th0))}")
    if args.warmup:
        tag += f"w{args.warmup}"
    if (args.counter_lo, args.counter_hi) != (5, 15):
        tag += (f"c{args.counter_lo}" if args.counter_lo == args.counter_hi
                else f"c{args.counter_lo}-{args.counter_hi}")
    label = args.label or f"SBSPS_{tag}_N{args.N}{'_DS' if args.ds else ''}_seed{args.seed}"

    env = V2XEnv(CFG)
    obs, info = env.reset()
    vc = env.vehicle_classes
    pol = SBSPSPolicy(N=CFG["n_vehicles"], M=CFG["n_subchannels"], n_pwr=env.n_pwr,
                      vehicle_classes=vc, rule=args.rule, th0_dbm=args.th0,
                      demand_separation=bool(args.ds), M_m0=args.M_m0, seed=args.seed,
                      counter_lo=args.counter_lo, counter_hi=args.counter_hi)
    print(f"[SB-SPS] {label}  N={args.N} M={args.M} rule={args.rule} "
          f"th0={args.th0} DS={args.ds} seed={args.seed} warmup={args.warmup} "
          f"counter=[{args.counter_lo},{args.counter_hi}]")
    print(f"Running {args.episodes} eval episodes (no training)...\n")

    # ---- evaluation loop: copied from train.py so the metrics are identical ----
    eval_m0_pdr, eval_m1_pdr, eval_m0_p05 = [], [], []
    eval_m0_sinr, eval_m1_sinr = [], []
    eval_m0_collision, eval_m0_coll_wp, eval_m1_coll_wp = [], [], []
    learned_sep, shuffled_sep = [], []
    rng_shuffle = np.random.default_rng(12345)

    for ev in range(args.episodes):
        obs, info = env.reset()
        pol.reset()
        ep_pdr_m0, ep_pdr_m1, ep_sinr_m0, ep_sinr_m1 = [], [], [], []
        ep_coll, ep_m0_wp, ep_m1_wp = [], [], []
        for _step in range(CFG["rollout_steps"]):
            acts, _ = pol.select_actions(obs, vc, deterministic=False)
            pos = env._positions().copy()
            nobs, _, done, trunc, info = env.step(acts)
            if _step >= args.warmup:
                if len(info["pdr_m0"]) > 0:
                    ep_pdr_m0.append(float(info["pdr_m0"].mean()))
                    ep_sinr_m0.append(float(info["sinr_dB_m0"].mean()))
                if len(info["pdr_m1"]) > 0:
                    ep_pdr_m1.append(float(info["pdr_m1"].mean()))
                    ep_sinr_m1.append(float(info["sinr_dB_m1"].mean()))
                ep_coll.append(float(info.get("m0_collision_rate", 0.0)))
                ep_m0_wp.append(float(info.get("m0_collision_rate_within_pool", 0.0)))
                ep_m1_wp.append(float(info.get("m1_collision_rate_within_pool", 0.0)))
                # spatial-reuse measurement, same method as diagnose_reuse.py
                live = (pos[:, 0] != 0) | (pos[:, 1] != 0)
                sub = acts // env.n_pwr
                learned_sep += co_channel_separation(pos, sub, live)
                perm = sub.copy()
                perm[live] = rng_shuffle.permutation(sub[live])
                shuffled_sep += co_channel_separation(pos, perm, live)
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
        if ep_coll:
            eval_m0_collision.append(float(np.mean(ep_coll)))
        if ep_m0_wp:
            eval_m0_coll_wp.append(float(np.mean(ep_m0_wp)))
        if ep_m1_wp:
            eval_m1_coll_wp.append(float(np.mean(ep_m1_wp)))
        if ev % 20 == 0:
            m0p = eval_m0_pdr[-1] if eval_m0_pdr else 0.0
            m1p = eval_m1_pdr[-1] if eval_m1_pdr else 0.0
            print(f"  Eval ep {ev:3d}: M0_PDR={m0p:.3f}  M1_PDR={m1p:.3f}")

    mean = lambda v: float(np.mean(v)) if v else 0.0
    m0_mean = mean(eval_m0_pdr)
    m0_p05_inter = float(np.percentile(eval_m0_pdr, 5)) if eval_m0_pdr else 0.0
    m0_p05_intra = mean(eval_m0_p05)
    reuse_ratio = (np.mean(learned_sep) / np.mean(shuffled_sep)
                   if learned_sep and shuffled_sep else None)

    print(f"\n=== Eval Summary ({args.episodes} episodes) - SB-SPS {args.rule} ===")
    print(f"  M0_PDR_mean              : {m0_mean:.4f}")
    print(f"  M0_PDR_p05 (across)      : {m0_p05_inter:.4f}")
    print(f"  M0_PDR_p05 (intra)       : {m0_p05_intra:.4f}")
    print(f"  M1_PDR_mean              : {mean(eval_m1_pdr):.4f}")
    print(f"  M0_SINR_mean (dB)        : {mean(eval_m0_sinr):.2f}")
    print(f"  M1_SINR_mean (dB)        : {mean(eval_m1_sinr):.2f}")
    print(f"  M0_collision_rate        : {mean(eval_m0_collision):.3f}")
    print(f"  M0_collision_within_pool : {mean(eval_m0_coll_wp):.3f}")
    print(f"  M1_collision_within_pool : {mean(eval_m1_coll_wp):.3f}")
    if reuse_ratio is not None:
        print(f"  co-channel separation    : {np.mean(learned_sep):.1f} m "
              f"vs shuffled {np.mean(shuffled_sep):.1f} m  -> ratio {reuse_ratio:.3f}")
    s = pol.stats
    print(f"  reselections {s['reselected']}, threshold raised in {s['threshold_raised']}, "
          f"occupancy fallback in {s['occupancy_fallback']}")

    record = {
        "config": label, "N": CFG["n_vehicles"], "M": CFG["n_subchannels"],
        "rho": round(CFG["n_vehicles"] / CFG["n_subchannels"], 2),
        "M_m0": args.M_m0 if args.ds else None, "m0_ratio": CFG.get("m0_ratio"),
        "m0_pdr_mean": m0_mean, "m0_pdr_p05": m0_p05_inter, "m0_pdr_p05_intra": m0_p05_intra,
        "m1_pdr_mean": mean(eval_m1_pdr),
        "m0_sinr_mean": mean(eval_m0_sinr), "m1_sinr_mean": mean(eval_m1_sinr),
        "m0_collision_rate": mean(eval_m0_collision),
        "m0_collision_rate_within_pool": mean(eval_m0_coll_wp),
        "m1_collision_rate_within_pool": mean(eval_m1_coll_wp),
        # no training for SB-SPS: training fields are deliberately empty
        "train_m0_pdr_last100": None, "train_m1_pdr_last100": None,
        "actor_loss": None, "actor_loss_m0": None, "actor_loss_m1": None,
        "critic_loss": None, "entropy_m0": None, "entropy_m1": None,
        "n_episodes": 0, "n_eval_episodes": args.episodes,
        "architecture": "sbsps_idealized",
        "sbsps": {"rule": args.rule, "th0_dbm": args.th0, "K": pol.K,
                  "counter": [pol.counter_lo, pol.counter_hi], "p_keep": pol.p_keep,
                  "tx_dbm": TX_DBM, "min_frac": pol.min_frac, "warmup": args.warmup,
                  "co_channel_sep_m": float(np.mean(learned_sep)) if learned_sep else None,
                  "shuffled_sep_m": float(np.mean(shuffled_sep)) if shuffled_sep else None,
                  "reuse_ratio": reuse_ratio, "stats": dict(s)},
    }
    out = pathlib.Path(f"results/_{label}.json")
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps([record], indent=2))
    print(f"\nResult saved -> {out}")
    env.close()


if __name__ == "__main__":
    main()
