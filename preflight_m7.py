"""
preflight_m7.py -- run from ~/v2x_clean before batch 25 (aa-achievement.md §7c).
No run has used more than five channels. Checks, for M = 7 (shared pool, and demand separation with pool 5):
  1. trainer sizes: actor 13 -> 128 -> 128 -> 35 = 22,819 parameters; critic (input 137) = 101,889
  2. the environment builds, observations are (10, 13), the global state has length 137
  3. demand separation sets a safety pool of 5 and an M1 pool of 2
  4. ten live steps on channels 0-6 run with finite rewards and observations
The bridge itself only compares channel indices (bridge.cc: ch[k] != ch[i]), so channels 5 and 6 need no change there.
"""
import sys, io, os, time, contextlib, subprocess, importlib.util
import numpy as np
import torch

sys.path.insert(0, ".")
from envs.v2x_env import V2XEnv
from agents.mappo_gaefix import MAPPOTrainerMode0cGAEFix

RUNNER = "train_d3_A_mode0c_N10_seed1_gaefix_g05_ent01.py"
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

cfg0 = load_cfg()
BIN = os.path.expanduser(cfg0.get("bridge_binary", "~/v2x_clean/ns3_bridge/v2x_bridge"))
def port_ok(p):
    proc = subprocess.Popen([BIN, f"--port={p}", "--fcGHz=5.9"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    time.sleep(2); alive = proc.poll() is None
    if alive:
        proc.terminate(); proc.wait(timeout=5)
    return alive

ports = [p for p in range(8936, 9036, 10) if port_ok(p)][:2]
if len(ports) < 2:
    check("two free ports for the live checks", False)
for (ds, pool), port in zip(((False, 2), (True, 5)), ports):
    tag = "demand separation, pool 5" if ds else "shared pool"
    print(f"\n=== M = 7, {tag} (port {port}) ===")
    cfg = dict(cfg0)
    cfg.update(n_subchannels=7, demand_separation=ds, m0_subchannel_count=pool, ns3_port=port)
    N, M = cfg["n_vehicles"], 7
    od, A = 3 + M + 3, M * 5
    gsd = N * od + M
    torch.manual_seed(0)
    tr = MAPPOTrainerMode0cGAEFix(cfg, od, A, gsd)
    n_act = sum(p.numel() for p in tr.actors[0].parameters())
    n_crit = sum(p.numel() for p in tr.critic.parameters())
    check("actor parameters 22,819", n_act == 22819, f"got {n_act:,}")
    check("critic parameters 101,889", n_crit == 101889, f"got {n_crit:,}")
    env = V2XEnv(cfg)
    try:
        obs, info = env.reset()
        gs = np.asarray(info["global_state"])
        check(f"observations ({N}, {od}), global state length {gsd}", obs.shape == (N, od) and gs.shape == (gsd,),
              f"got {obs.shape}, {gs.shape}")
        if ds:
            check("safety pool 5, M1 pool 2", (env.M_m0, env.M_m1) == (5, 2), f"got {env.M_m0}, {env.M_m1}")
        ok, steps = True, 0
        for _ in range(10):
            obs, rew, done, trunc, info = env.step(np.random.randint(0, A, size=N))
            steps += 1
            ok &= bool(np.isfinite(rew).all() and np.isfinite(obs).all() and obs.shape == (N, od))
            if done or trunc:
                break
        check(f"{steps} live steps on channels 0-6, finite rewards and observations", ok)
    finally:
        with contextlib.redirect_stdout(io.StringIO()):
            env.close()

n_ok = sum(results)
print(f"\n{n_ok} of {len(results)} checks passed")
print("PRE-FLIGHT PASSED" if n_ok == len(results) else "PRE-FLIGHT FAILED — send me the output before launching batch 25")
