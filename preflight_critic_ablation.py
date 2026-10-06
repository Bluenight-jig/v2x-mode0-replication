"""
preflight_critic_ablation.py -- run from ~/v2x_clean before batch 24 (aa-achievement.md §7c).

Checks, with real PyTorch, the real trainers and the real environment:
  1. critic widths: local = obs_dim; agent-specific = gs_dim + obs_dim
  2. same seed -> actors initialised IDENTICALLY to the paper arm (only the critic differs)
  3. a rollout-shaped sequence of value() calls feeds compute_gae; cache verified and cleared
  4. a full PPO update runs, and it trains the NEW critic
  5. live SUMO + bridge: each vehicle's slice of the global state equals its observation, exactly
"""
import sys, io, os, time, contextlib, subprocess, importlib.util
import numpy as np
import torch

sys.path.insert(0, ".")
from agents.mappo_gaefix import MAPPOTrainerGAEFix, MAPPOTrainerMode0cGAEFix
from agents.mappo_critic_ablation import (MAPPOTrainerLocalCritic, MAPPOTrainerASCritic,
                                          MAPPOTrainerMode0cLocalCritic, MAPPOTrainerMode0cASCritic)

RUNNER = "train_d5_A_mode0c_seed1_5000ep_gaefix_g05_ent01.py"   # any rebuild runner supplies CFG
results = []

def check(name, cond, detail=""):
    results.append(bool(cond))
    print(f"  [{'PASS' if cond else 'FAIL'}] {name}" + (f"  ({detail})" if detail else ""))

def load_cfg():
    spec = importlib.util.spec_from_file_location("rb", RUNNER)
    mod = importlib.util.module_from_spec(spec)
    with contextlib.redirect_stdout(io.StringIO()):
        try:
            spec.loader.exec_module(mod)
        except SystemExit:
            pass
    return dict(mod.CFG)

def modules_except_critic(tr):
    out = {}
    for name, v in vars(tr).items():
        if name == "critic":
            continue
        if isinstance(v, torch.nn.Module):
            out[name] = v
        elif isinstance(v, (list, tuple)) and v and all(isinstance(x, torch.nn.Module) for x in v):
            for j, x in enumerate(v):
                out[f"{name}[{j}]"] = x
    return out

def same_weights(a, b):
    ma, mb = modules_except_critic(a), modules_except_critic(b)
    if ma.keys() != mb.keys():
        return False, f"module sets differ: {sorted(ma)} vs {sorted(mb)}"
    for k in ma:
        sa, sb = ma[k].state_dict(), mb[k].state_dict()
        if sa.keys() != sb.keys() or any(not torch.equal(sa[p], sb[p]) for p in sa):
            return False, f"{k} differs"
    return True, f"{len(ma)} actor modules identical"

def fake_buf(tr, arch, T, N, od, gsd, A):
    dev = tr.device
    obs = torch.rand(T * N, od, device=dev)
    gs = torch.rand(T * N, gsd, device=dev)
    buf = {"obs": obs, "global_state": gs,
           "actions": torch.randint(0, A, (T * N,), device=dev),
           "log_probs": torch.full((T * N,), float(np.log(1.0 / A)), device=dev),
           "advantages": torch.randn(T * N, device=dev),
           "returns": torch.randn(T * N, device=dev)}
    idx = torch.arange(N, device=dev).repeat(T)
    if arch == "0a":
        buf["classes"] = (idx >= N // 2).long()
    else:
        buf["vehicle_idx"] = idx
    return buf

cfg0 = load_cfg()
print("1-4. Trainers (real PyTorch)")
for N in (4, 10):
    cfg = dict(cfg0); cfg["n_vehicles"] = N
    M = cfg["n_subchannels"]; od = 3 + M + 3; A = M * 5; gsd = N * od + M
    for arch, base, loc, asc in (("0a", MAPPOTrainerGAEFix, MAPPOTrainerLocalCritic, MAPPOTrainerASCritic),
                                 ("0c", MAPPOTrainerMode0cGAEFix, MAPPOTrainerMode0cLocalCritic, MAPPOTrainerMode0cASCritic)):
        torch.manual_seed(1234); ref = base(cfg, od, A, gsd)
        for cls, want in ((loc, od), (asc, gsd + od)):
            torch.manual_seed(1234); tr = cls(cfg, od, A, gsd)
            tag = f"Mode {arch}, N={N}, {tr.FEATURES}"
            first = next(m for m in tr.critic.modules() if isinstance(m, torch.nn.Linear))
            check(f"{tag}: critic width {want}", first.in_features == want, f"got {first.in_features}")
            ok, why = same_weights(ref, tr)
            check(f"{tag}: actors identical to the paper arm at the same seed", ok, why)
            T = 6
            obs = np.random.rand(T + 1, N, od).astype(np.float32)
            gs = np.concatenate([obs.reshape(T + 1, -1), np.random.rand(T + 1, M).astype(np.float32)], axis=1)
            for t in range(T + 1):
                tr.value(gs[t])
            rew = np.random.randn(T * N).astype(np.float32); dn = np.zeros(T * N, np.float32); dn[-N:] = 1
            adv, ret = tr.compute_gae(rew, np.zeros(T * N, np.float32), 0.0, dn)
            check(f"{tag}: compute_gae from cached per-vehicle values",
                  adv.shape == (T * N,) and np.isfinite(adv).all() and tr._vcache == [])
            before = [p.detach().clone() for p in tr.critic.parameters()]
            stats = tr.update_from_tensors(fake_buf(tr, arch, T, N, od, gsd, A))
            changed = any(not torch.equal(b, p) for b, p in zip(before, tr.critic.parameters()))
            check(f"{tag}: PPO update runs and trains the new critic", changed and "critic_loss" in stats,
                  f"critic loss {stats.get('critic_loss', float('nan')):.4f}")

print("\n5. Live environment: is each vehicle's slice of the global state its observation?")
from envs.v2x_env import V2XEnv
BIN = os.path.expanduser(cfg0.get("bridge_binary", "~/v2x_clean/ns3_bridge/v2x_bridge"))
def port_ok(p):
    proc = subprocess.Popen([BIN, f"--port={p}", "--fcGHz=5.9"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    time.sleep(2); alive = proc.poll() is None
    if alive:
        proc.terminate(); proc.wait(timeout=5)
    return alive
port = next((p for p in range(7936, 8036, 10) if port_ok(p)), None)
if port is None:
    check("found a free port for the live check", False)
else:
    cfg = dict(cfg0); cfg["n_vehicles"] = 4; cfg["ns3_port"] = port
    N = 4; od = 3 + cfg["n_subchannels"] + 3
    env = V2XEnv(cfg)
    try:
        obs, info = env.reset()
        worst = 0.0; steps = 0
        for _ in range(15):
            gs = np.asarray(info["global_state"], dtype=np.float32)
            worst = max(worst, float(np.abs(gs[:N * od].reshape(N, od) - obs).max()))
            obs, _, done, trunc, info = env.step(np.random.randint(0, cfg["n_subchannels"] * 5, size=N))
            steps += 1
            if done or trunc:
                break
        check(f"slices equal observations over {steps} live steps (port {port})", worst == 0.0, f"max difference {worst}")
    finally:
        with contextlib.redirect_stdout(io.StringIO()):
            env.close()

n_ok = sum(results)
print(f"\n{n_ok} of {len(results)} checks passed")
print("PRE-FLIGHT PASSED" if n_ok == len(results) else "PRE-FLIGHT FAILED — send me the output before batch 24")
