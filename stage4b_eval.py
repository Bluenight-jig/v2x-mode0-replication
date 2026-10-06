#!/usr/bin/env python3
"""stage4b_eval.py -- Stage 4b: per-vehicle, per-interval re-evaluation (aa-achievement.md; terminology v71).

  python3 stage4b_eval.py eval --label A_mode0c_N10_seed1_gaefix_g05_ent01 --port 9406
  python3 stage4b_eval.py eval --sbsps --N 10 --seed 1 --port 9406     # headline SB-SPS: rsrp, -80 dBm, warm-up 40
  python3 stage4b_eval.py report

`eval` re-runs one saved policy through the live bridge with the runner's own evaluation loop (100 episodes,
stochastic actions, 200 generation intervals each) -- or SB-SPS exactly as sbsps_eval.py configures it -- and
KEEPS every vehicle's delivery probability, subchannel and position at every interval:
    results/stage4b/<label>.npz     pdr, subch, pos [episode, interval, vehicle(, xy)], classes, lengths
    results/stage4b/_<label>.json   the metrics below, plus a validation against the run's own logged evaluation
`report` aggregates across seeds: AoI (figure 7-D), the per-vehicle counterpart of WIRC (F16), Mode 0b's
passive-versus-active split (P49-P51), and the spatial reuse ratio (Stage 4a v2).

Delivery. info["pdr"] is the PHY's delivery PROBABILITY (piecewise-linear in SINR between 0 and 20 dB). Fleet
metrics use it directly, exactly as the runner does. Age of information needs delivery EVENTS, so each message
is delivered with that probability: Bernoulli draws, K per episode (default 20), fixed seed -- exact wherever
the probability is 0 or 1. AoI is counted in generation intervals (x 100 ms): 1 = refreshed in this interval.
Read-only towards the project: writes only under results/stage4b/.
"""
import os, sys, re, json, time, pathlib, argparse, subprocess
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")          # inference on the CPU: no contention with training
import numpy as np
sys.path.insert(0, ".")

OUT = pathlib.Path("results/stage4b")
BIN = os.path.expanduser("~/v2x_clean/ns3_bridge/v2x_bridge")
REQ = 0.95

# ----------------------------------------------------------------------------------------------- metrics
def compute_metrics(pdr, subch, pos, lengths, vc, warmup, draws=20, seed=2026, passive=None, M=5):
    """Pure function of the recorded arrays (E episodes x T intervals x N vehicles); no simulator needed."""
    E, T, N = pdr.shape
    vc = np.asarray(vc, int); m0 = np.where(vc == 0)[0]; m1 = np.where(vc == 1)[0]
    rng = np.random.default_rng(seed)
    out = {"warmup": int(warmup), "episodes": int(E)}
    ep_mean, ep_p05, ep_m1 = [], [], []
    worst_v, pv_all, aoi_mean, aoi_p95, aoi_peak, aoi_worst_v, p_stale = [], [], [], [], [], [], []
    for e in range(E):
        L = int(lengths[e]); w = slice(warmup, L)
        if L <= warmup: continue
        P0 = np.nan_to_num(pdr[e, :L][:, m0], nan=0.0)                 # (L, n_m0)
        s = P0[warmup:].mean(axis=1)                                    # runner: per-interval fleet mean
        ep_mean.append(float(s.mean())); ep_p05.append(float(np.percentile(s, 5)))
        if len(m1): ep_m1.append(float(np.nan_to_num(pdr[e, w][:, m1], nan=0.0).mean()))
        V = P0[warmup:].mean(axis=0)                                    # per-vehicle delivery this episode
        worst_v.append(float(V.min())); pv_all.extend(V.tolist())
        d = rng.random((draws, L, len(m0))) < P0[None]                  # delivery events
        age = np.zeros((draws, len(m0))); ages = np.empty((draws, L, len(m0)))
        for t in range(L):
            age = np.where(d[:, t], 1.0, age + 1.0); ages[:, t] = age
        A = ages[:, warmup:]                                            # (draws, L-w, n_m0)
        aoi_mean.append(float(A.mean())); aoi_p95.append(float(np.percentile(A, 95)))
        aoi_peak.append(float(A.max(axis=(1, 2)).mean())); aoi_worst_v.append(float(A.mean(axis=(0, 1)).max()))
        p_stale.append(float((A > 1).mean()))
    f = lambda xs: float(np.mean(xs)) if xs else None
    out.update({"m0_pdr_mean": f(ep_mean), "wirc": f(ep_p05), "m1_pdr_mean": f(ep_m1),
                "worst_vehicle_delivery": f(worst_v),
                "per_vehicle_p05": float(np.percentile(pv_all, 5)) if pv_all else None,
                "share_vehicles_meeting_095": float(np.mean(np.asarray(pv_all) >= REQ)) if pv_all else None,
                "aoi_mean": f(aoi_mean), "aoi_p95": f(aoi_p95), "aoi_peak": f(aoi_peak),
                "aoi_worst_vehicle_mean": f(aoi_worst_v), "p_aoi_above_1": f(p_stale),
                "aoi_draws": int(draws)})
    if passive is not None:
        out["mode0b"] = mode0b_split(pdr, subch, lengths, vc, warmup, passive, M)
    out["reuse"] = reuse_ratio(subch, pos, lengths, warmup, seed)
    return out

def mode0b_split(pdr, subch, lengths, vc, warmup, passive, M):
    """Delivery by type within each class; within-pool rates of the passive groups (P49-P51)."""
    N = pdr.shape[2]
    groups = {"active_m0": [i for i in range(N) if vc[i] == 0 and i not in passive[0]], "passive_m0": list(passive[0]),
              "active_m1": [i for i in range(N) if vc[i] == 1 and i not in passive[1]], "passive_m1": list(passive[1])}
    res = {k: None for k in groups}
    for k, idx in groups.items():
        if idx:
            vals = [np.nan_to_num(pdr[e, warmup:int(lengths[e])][:, idx], nan=0.0).mean() for e in range(len(lengths)) if lengths[e] > warmup]
            res[k] = float(np.mean(vals))
    def within(idx):
        if len(idx) < 2: return None
        rates = []
        for e in range(len(lengths)):
            S = subch[e, warmup:int(lengths[e])][:, idx]
            if len(S) == 0: continue
            shared = np.array([[np.sum(row == row[j]) > 1 for j in range(len(idx))] for row in S])
            rates.append(shared.mean())
        return float(np.mean(rates)) if rates else None
    res["passive_m0_within_pool"] = within(groups["passive_m0"])
    res["passive_m1_within_pool"] = within(groups["passive_m1"])
    k = len(groups["passive_m1"])
    res["passive_m1_random_level"] = float(1 - (1 - 1 / M) ** (k - 1)) if k >= 2 else None
    res["groups"] = groups
    return res

def reuse_ratio(subch, pos, lengths, warmup, seed):
    """Mean co-channel separation, learned against shuffled subchannels -- the method of sbsps_eval.py."""
    rng = np.random.default_rng(seed + 7); learned, shuffled = [], []
    def sep(p, s, live):
        idx = np.where(live)[0]; out = []
        for a in range(len(idx)):
            for b in range(a + 1, len(idx)):
                i, j = idx[a], idx[b]
                if s[i] == s[j]: out.append(float(np.linalg.norm(p[i] - p[j])))
        return out
    for e in range(len(lengths)):
        for t in range(warmup, int(lengths[e])):
            p, s = pos[e, t], subch[e, t]
            live = (p[:, 0] != 0) | (p[:, 1] != 0)
            learned += sep(p, s, live)
            perm = s.copy(); perm[live] = rng.permutation(s[live]); shuffled += sep(p, perm, live)
    if not learned or not shuffled: return None
    return {"co_channel_sep_m": float(np.mean(learned)), "shuffled_sep_m": float(np.mean(shuffled)),
            "ratio": float(np.mean(learned) / np.mean(shuffled))}

# ----------------------------------------------------------------------------------------------- evaluation
def port_ok(p):
    """A port is usable only if a test bridge can bind it and is still alive after 2 s."""
    proc = subprocess.Popen([BIN, f"--port={p}", "--fcGHz=5.9"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    time.sleep(2)
    alive = proc.poll() is None
    if alive: proc.terminate(); proc.wait(timeout=5)
    return alive

def run_eval(args):
    import random, torch
    from envs.v2x_env import V2XEnv
    if not port_ok(args.port):
        sys.exit(f"Port {args.port} is not usable (a test bridge could not hold it). Choose another port; nothing was run.")
    passive, M = None, 5
    if args.sbsps:
        from sbsps_eval import SBSPSPolicy, load_cfg
        label = f"SBSPS_rsrp{int(abs(args.th0))}w{args.warmup if args.warmup is not None else 40}_N{args.N}_seed{args.seed}"
        warmup = 40 if args.warmup is None else args.warmup
        np.random.seed(args.seed)
        cfg = load_cfg()
        cfg.update({"n_vehicles": args.N, "n_subchannels": args.M, "demand_separation": False,
                    "m0_subchannel_count": 2, "n_eval_episodes": args.episodes, "ns3_port": args.port})
        env = V2XEnv(cfg); obs, info = env.reset(); vc = env.vehicle_classes; M = args.M
        pol = SBSPSPolicy(N=cfg["n_vehicles"], M=cfg["n_subchannels"], n_pwr=env.n_pwr, vehicle_classes=vc,
                          rule="rsrp", th0_dbm=args.th0, demand_separation=False, M_m0=2, seed=args.seed,
                          counter_lo=5, counter_hi=15)
        act = lambda o: pol.select_actions(o, vc, deterministic=False)[0]
        reset_policy = pol.reset
    else:
        from agents.mappo_mode0c import MAPPOTrainerMode0c
        label = args.label; warmup = 0 if args.warmup is None else args.warmup
        ck = torch.load(f"checkpoints/{label}_actors.pt", map_location="cpu", weights_only=False)
        cfg = dict(ck["cfg"]); cfg["ns3_port"] = args.port; cfg["n_eval_episodes"] = args.episodes
        seed = int(re.search(r"seed(\d+)", label).group(1)) if re.search(r"seed(\d+)", label) else 0
        random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
        Env = V2XEnv
        if label.endswith("_armS") or label.endswith("_armP"):
            from envs.v2x_env_arms import V2XEnvArmS, V2XEnvArmP
            Env = V2XEnvArmS if label.endswith("_armS") else V2XEnvArmP
        env = Env(cfg); obs, info = env.reset(); vc = env.vehicle_classes; M = int(cfg["n_subchannels"])
        trainer = MAPPOTrainerMode0c(cfg, env.obs_dim, env.M * env.n_pwr, len(info["global_state"]))
        sds = ck["actor_state_dicts"]
        if len(sds) != len(trainer.actors): sys.exit(f"checkpoint has {len(sds)} actors, environment {len(trainer.actors)}")
        for a, sd in zip(trainer.actors, sds): a.load_state_dict(sd)
        if ck.get("architecture") == "mode_0b_hybrid_actors":
            from agents.mappo_mode0b import PASSIVE_DEFAULT
            v = [int(x) for x in vc]
            passive = {c: [i for i in range(len(v)) if v[i] == c][-PASSIVE_DEFAULT[c]:] for c in (0, 1)}
        act = lambda o: trainer.select_actions(o, vc, deterministic=False)[0]
        reset_policy = lambda: None
    T, N, E = int(cfg["rollout_steps"]), int(cfg["n_vehicles"]), args.episodes
    pdr = np.full((E, T, N), np.nan, np.float32); sub = np.full((E, T, N), -1, np.int16)
    pos = np.zeros((E, T, N, 2), np.float32); lengths = np.zeros(E, np.int32)
    print(f"[Stage 4b] {label}: N={N} M={M} episodes={E} warm-up={warmup} port={args.port}  (CPU inference)")
    t0 = time.time()
    for e in range(E):
        obs, info = env.reset(); reset_policy()
        for t in range(T):
            a = act(obs)
            pos[e, t] = env._positions()
            obs, _, done, trunc, info = env.step(a)
            pdr[e, t] = info["pdr"]; sub[e, t] = info["subchannels"]; lengths[e] = t + 1
            if done or trunc: break
        if (e + 1) % 10 == 0 or e == 0:
            spe = (time.time() - t0) / (e + 1)
            print(f"  episode {e + 1:3d}/{E}   {spe:.1f} s/episode   about {spe * (E - e - 1) / 60:.0f} min left", flush=True)
    wall = time.time() - t0
    env.close()
    OUT.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(OUT / f"{label}.npz", pdr=pdr, subch=sub, pos=pos, lengths=lengths, classes=np.asarray(vc), warmup=warmup)
    own = compute_metrics(pdr, sub, pos, lengths, vc, warmup, args.draws, passive=passive, M=M)
    common = compute_metrics(pdr, sub, pos, lengths, vc, max(warmup, 40), args.draws, passive=passive, M=M)
    frac = np.nan_to_num(pdr[~np.isnan(pdr)]); frac_partial = float(np.mean((frac > 0) & (frac < 1))) if frac.size else None
    val = {}
    logged = pathlib.Path(f"results/_{label}.json")
    if logged.exists():
        r = json.loads(logged.read_text())[-1]
        val = {"logged_m0": r.get("m0_pdr_mean"), "logged_wirc": r.get("m0_pdr_p05_intra")}
        val["delta_m0"] = own["m0_pdr_mean"] - val["logged_m0"] if val["logged_m0"] is not None else None
        val["delta_wirc"] = own["wirc"] - val["logged_wirc"] if val["logged_wirc"] is not None else None
        if val["delta_m0"] is None or val["delta_wirc"] is None:
            val["status"] = "INCOMPLETE (logged result lacks a field)"
        else:
            val["status"] = "OK" if abs(val["delta_m0"]) <= 0.01 and abs(val["delta_wirc"]) <= 0.03 else "CHECK"
    rec = {"label": label, "kind": "sbsps" if args.sbsps else ck.get("architecture"), "N": N, "M": M,
           "own_window": own, "common_window_40": common, "validation": val,
           "fraction_of_partial_probabilities": frac_partial,
           "sec_per_episode": wall / E, "wall_minutes": wall / 60}
    (OUT / f"_{label}.json").write_text(json.dumps(rec, indent=2))
    print(f"\n  wall time {wall / 60:.1f} min ({wall / E:.1f} s/episode)")
    if val.get("status", "").startswith(("OK", "CHECK")):
        print(f"  validation vs the run's own evaluation: M0 {own['m0_pdr_mean']:.4f} (logged {val['logged_m0']:.4f}, "
              f"delta {val['delta_m0']:+.4f}); WIRC {own['wirc']:.4f} (logged {val['logged_wirc']:.4f}, delta {val['delta_wirc']:+.4f})"
              f"  -> {val['status']}")
    else:
        print(f"  validation: {val.get('status', 'no logged result found -- nothing to compare against')}")
    print(f"  per-vehicle: worst vehicle {own['worst_vehicle_delivery']:.4f}; 5th percentile {own['per_vehicle_p05']:.4f}; "
          f"share meeting 0.95 {own['share_vehicles_meeting_095']:.3f}")
    print(f"  AoI (intervals): mean {own['aoi_mean']:.3f}, 95th percentile {own['aoi_p95']:.2f}, peak {own['aoi_peak']:.1f}")
    if passive is not None:
        b = own["mode0b"]
        print(f"  Mode 0b: active safety {b['active_m0']:.4f}, passive safety {b['passive_m0']:.4f}; "
              f"passive M1 within-pool {b['passive_m1_within_pool']:.3f} (random {b['passive_m1_random_level']:.3f})")
    print(f"  saved {OUT / (label + '.npz')} and {OUT / ('_' + label + '.json')}")

# ----------------------------------------------------------------------------------------------- report
GROUPS = [("Mode 0c, N = 4",  [f"A_mode0c_seed{k}_5000ep_gaefix_g05_ent01" for k in range(1, 6)]),
          ("SB-SPS, N = 4",   [f"SBSPS_rsrp80w40_N4_seed{k}" for k in range(1, 6)]),
          ("Mode 0c, N = 10", [f"A_mode0c_N10_seed{k}_gaefix_g05_ent01" for k in range(1, 6)]),
          ("SB-SPS, N = 10",  [f"SBSPS_rsrp80w40_N10_seed{k}" for k in range(1, 6)]),
          ("Mode 0b, N = 10", [f"A_mode0b_N10_seed{k}_gaefix_g05_ent01" for k in (1, 2, 3)]),
          ("M = 7, shared pool", [f"A_mode0c_N10_M7_seed{k}_gaefix_g05_ent01" for k in (1, 2, 3)]),
          ("M = 7, partition",   [f"D_N10_M7_pool5_seed{k}_gaefix_g05_ent01" for k in (1, 2, 3)])]
ALL_MODE0C_AS = 0.9413          # all-Mode-0c fleet, agent-specific critic, N = 10 (record §4u) -- P50's reference

def ci(xs):
    """Mean and 95% t-interval across seeds -- ablation_stats.interval, the programme's usual intervals."""
    from ablation_stats import interval
    xs = [x for x in xs if x is not None]
    if not xs: return "-"
    m, lo, hi = interval(xs)
    return f"{m:.3f}" + (f" [{lo:.3f}, {hi:.3f}]" if lo is not None else f" (n={len(xs)})")

def run_report(_args):
    print("Stage 4b report -- safety class; AoI in generation intervals (x 100 ms). Each row is one operating point;")
    print("WIRC and AoI are never compared across densities (F16).\n")
    for name, labels in GROUPS:
        recs = [json.loads((OUT / f"_{l}.json").read_text()) for l in labels if (OUT / f"_{l}.json").exists()]
        if not recs: print(f"== {name}: no evaluations yet"); continue
        own = [r["own_window"] for r in recs]; com = [r["common_window_40"] for r in recs]
        bad = [r["label"] for r in recs if r["validation"].get("status") == "CHECK"]
        print(f"== {name}: {len(recs)} of {len(labels)} seeds   validation: {'all OK' if not bad else 'CHECK ' + ', '.join(bad)}")
        g = lambda key, src=own: ci([o.get(key) for o in src])
        print(f"   fleet: M0 {g('m0_pdr_mean')}   WIRC {g('wirc')}")
        print(f"   per vehicle: worst vehicle {g('worst_vehicle_delivery')}   5th percentile {g('per_vehicle_p05')}   share meeting 0.95 {g('share_vehicles_meeting_095')}")
        print(f"   AoI: mean {g('aoi_mean')}   95th percentile {g('aoi_p95')}   peak {g('aoi_peak')}   worst vehicle's mean {g('aoi_worst_vehicle_mean')}")
        print(f"   AoI, common window (intervals >= 40): mean {g('aoi_mean', com)}   95th percentile {g('aoi_p95', com)}   peak {g('aoi_peak', com)}")
        ru = [o["reuse"]["ratio"] for o in own if o.get("reuse")]
        print(f"   reuse ratio (co-channel separation / shuffled): {ci(ru)}")
        if name.startswith("Mode 0b"):
            b = [o["mode0b"] for o in own if o.get("mode0b")]
            if b:
                ps = [x["passive_m0"] for x in b]; ac = [x["active_m0"] for x in b]
                wp = [x["passive_m1_within_pool"] for x in b]; rl = b[0]["passive_m1_random_level"]
                print(f"   by type: active safety {ci(ac)}   passive safety {ci(ps)}   active M1 {ci([x['active_m1'] for x in b])}   passive M1 {ci([x['passive_m1'] for x in b])}")
                print(f"   passive safety pair sharing {ci([x['passive_m0_within_pool'] for x in b])}   passive M1 within-pool {ci(wp)} (random level {rl:.3f})")
                if len(b) == 3:
                    print(f"   P49 (passive safety >= 0.85 in every seed): {'CORRECT' if all(x >= 0.85 for x in ps) else 'WRONG'}  ({', '.join(f'{x:.3f}' for x in ps)})")
                    d = float(np.mean(ac)) - ALL_MODE0C_AS
                    print(f"   P50 (active safety within 0.01 of the all-Mode-0c fleet, {ALL_MODE0C_AS}): {'CORRECT' if abs(d) <= 0.01 else 'WRONG'}  (difference {d:+.4f})")
                    print(f"   P51 (passive M1 trio within 0.02 of its random level in every seed -- the density rule's tolerance): "
                          f"{'CORRECT' if all(abs(x - rl) <= 0.02 for x in wp) else 'WRONG'}  ({', '.join(f'{x:.3f}' for x in wp)})")
        print()

def main():
    ap = argparse.ArgumentParser(description="Stage 4b per-vehicle re-evaluation")
    sp = ap.add_subparsers(dest="cmd", required=True)
    e = sp.add_parser("eval")
    e.add_argument("--label"); e.add_argument("--sbsps", action="store_true")
    e.add_argument("--N", type=int); e.add_argument("--M", type=int, default=5); e.add_argument("--seed", type=int)
    e.add_argument("--th0", type=float, default=-80.0)
    e.add_argument("--port", type=int, required=True); e.add_argument("--episodes", type=int, default=100)
    e.add_argument("--warmup", type=int, default=None, help="default: 0 for trained policies, 40 for SB-SPS (the headline windows)")
    e.add_argument("--draws", type=int, default=20)
    sp.add_parser("report")
    a = ap.parse_args()
    if a.cmd == "eval":
        if a.sbsps and (a.N is None or a.seed is None): sys.exit("--sbsps needs --N and --seed")
        if not a.sbsps and not a.label: sys.exit("give --label (a checkpoint) or --sbsps")
        run_eval(a)
    else:
        run_report(a)

if __name__ == "__main__":
    main()
