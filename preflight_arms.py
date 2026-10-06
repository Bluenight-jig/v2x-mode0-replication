"""
preflight_arms.py -- run from ~/v2x_clean before the Tier 4 runs (aa-achievement.md §7c).
Intercepts what the REAL bridge receives during live steps and checks, for N = 4:
  Arm S: every vehicle transmits at 23 dBm, on exactly the channel its actor chose
  Arm P: every vehicle transmits at exactly the power its actor chose, on a channel drawn at random
"""
import sys, io, os, time, contextlib, subprocess, importlib.util
import numpy as np

sys.path.insert(0, ".")
from envs.v2x_env_arms import V2XEnvArmS, V2XEnvArmP

RUNNER = "train_d5_A_mode0c_seed1_5000ep_gaefix_g05_ent01.py"
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

ports = [p for p in range(9036, 9136, 10) if port_ok(p)][:2]
if len(ports) < 2:
    check("two free ports for the live checks", False)
for (name, cls), port in zip((("Arm S", V2XEnvArmS), ("Arm P", V2XEnvArmP)), ports):
    print(f"\n=== {name} (port {port}) ===")
    cfg = dict(cfg0); cfg["ns3_port"] = port
    env = cls(cfg)
    sent = []
    original = env.bridge.step
    def intercept(positions, subch, powers, _orig=original):
        sent.append((np.array(subch).copy(), np.array(powers, dtype=float).copy()))
        return _orig(positions, subch, powers)
    env.bridge.step = intercept
    try:
        env.reset()
        sent.clear()                      # keep only what the actor-driven steps send
        N, A = cfg["n_vehicles"], env.M * env.n_pwr
        chosen = []
        for _ in range(10):
            a = np.random.randint(0, A, size=N); chosen.append(a)
            _, _, done, trunc, _ = env.step(a)
            if done or trunc:
                break
        chosen = np.concatenate(chosen)[: sum(len(c) for c, _ in sent)]
        ch = np.concatenate([c for c, _ in sent]); pw = np.concatenate([p for _, p in sent])
        check(f"{len(sent)} live steps reached the bridge", len(sent) > 0)
        if name == "Arm S":
            check("every vehicle at 23 dBm", np.all(np.isclose(pw, 23.0)), f"powers seen: {sorted(set(np.round(pw, 1)))}")
            check("channel = the actor's choice", np.array_equal(ch, chosen // env.n_pwr))
        else:
            check("power = the actor's choice", np.allclose(pw, env.POWER_DBM[chosen % env.n_pwr]))
            check("channels within 0..M-1", ch.min() >= 0 and ch.max() < env.M, f"seen {sorted(set(ch.tolist()))}")
            check("channel not the actor's choice (agrees only by chance)", np.mean(ch == chosen // env.n_pwr) < 0.6,
                  f"agreement {np.mean(ch == chosen // env.n_pwr):.2f}, chance {1 / env.M:.2f}")
    finally:
        with contextlib.redirect_stdout(io.StringIO()):
            env.close()

n_ok = sum(results)
print(f"\n{n_ok} of {len(results)} checks passed")
print("PRE-FLIGHT PASSED" if n_ok == len(results) else "PRE-FLIGHT FAILED — send me the output")
