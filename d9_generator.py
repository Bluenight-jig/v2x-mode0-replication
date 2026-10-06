"""d9_generator.py -- batch 27 (aa-achievement.md §7c, v59-v61).   Usage:  python3 d9_generator.py B   (or C)

Two derivations, each guarded so a runner is written only if its source is exactly what it should be:
  ORIGINAL -> the density sweep: Mode 0a with the agent-specific critic (D2), from the ORIGINAL runners.
              Changes: the trainer import, gamma 0.99 -> 0.5, entropy start 0.05 -> 0.01, label, result
              file, port -- plus the seed line where no original exists. Guards include N and M.
  PAPER    -> from the PAPER-ARM Mode 0c runners: Arm S (environment import only) and delta = 1.0 (R3-h2).
Option B adds Arm S at N = 4 and takes density N = 7, 15; option C takes N = 5, 6, 7 (v59 packing)."""
import re, sys, pathlib, difflib, subprocess, time, os
BIN = os.path.expanduser("~/v2x_clean/ns3_bridge/v2x_bridge")
ORIG_IMPORT = (r'^from agents\.mappo[ \t]+import MAPPOTrainer[ \t]*$',
               "from agents.mappo_critic_ablation import MAPPOTrainerASCritic as MAPPOTrainer")
ENV_IMPORT = r'^from envs\.v2x_env[ \t]+import V2XEnv[ \t]*$'
GAMMA, ENT = 0.5, 0.01
def dens(N, s, runner, label0, seed0, new_seed=None):
    return ("ORIG", (runner, label0, seed0, N, 5), f"A_N{N}_seed{s}_5000ep_gaefix_g05_ent01_as", new_seed)
DENSITY = {
    "N5":  [dens(5, s, f"train_b11_a_n5_seed{s}.py", f"A_N5_seed{s}_5000ep", s) for s in (1, 2, 3)],
    "N6":  [dens(6, s, f"train_b12_n6_s{s}_5000ep.py", f"A_N6_seed{s}_5000ep", s) for s in (1, 2, 3)],
    "N7":  [dens(7, 1, "train_b15_a_n7_seed2_5000ep.py", "A_N7_seed2_5000ep", 2, new_seed=1),
            dens(7, 2, "train_b15_a_n7_seed2_5000ep.py", "A_N7_seed2_5000ep", 2),
            dens(7, 3, "train_b15_a_n7_seed3_5000ep.py", "A_N7_seed3_5000ep", 3)],
    "N15": [dens(15, 1, "train_b10_n15_s1.py", "A_N15_seed1_5000ep", 1),
            dens(15, 2, "train_b10_n15_s2.py", "A_N15_seed2_5000ep", 2),
            dens(15, 3, "train_b10_n15_s2.py", "A_N15_seed2_5000ep", 2, new_seed=3)],
}
C10 = lambda s: (f"train_d3_A_mode0c_N10_seed{s}_gaefix_g05_ent01.py", f"A_mode0c_N10_seed{s}_gaefix_g05_ent01", s)
C4  = lambda s: (f"train_d5_A_mode0c_seed{s}_5000ep_gaefix_g05_ent01.py", f"A_mode0c_seed{s}_5000ep_gaefix_g05_ent01", s)
ARMS10 = [("PAPER", C10(s), f"A_mode0c_N10_seed{s}_gaefix_g05_ent01_armS", ({}, "V2XEnvArmS")) for s in (2, 3)]
DELTA  = [("PAPER", C10(s), f"A_mode0c_N10_seed{s}_gaefix_g05_ent01_d10", ({"delta": 1.0}, None)) for s in (1, 2, 3)]
ARMS4  = [("PAPER", C4(s), f"A_mode0c_seed{s}_5000ep_gaefix_g05_ent01_armS", ({}, "V2XEnvArmS")) for s in (1, 2, 3)]
OPTION = {"B": ARMS10 + DELTA + ARMS4 + DENSITY["N7"] + DENSITY["N15"],
          "C": ARMS10 + DELTA + DENSITY["N5"] + DENSITY["N6"] + DENSITY["N7"]}
CANDIDATES = list(range(8646, 8846, 10))      # 20 candidates, clear of every earlier batch and pre-flight
def port_ok(p):
    """Start the bridge on port p; the port is usable only if it is still alive after 2 s."""
    proc = subprocess.Popen([BIN, f"--port={p}", "--fcGHz=5.9"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    time.sleep(2)
    alive = proc.poll() is None
    if alive: proc.terminate(); proc.wait(timeout=5)
    return alive
def _get(s, pat, name):
    m = re.search(pat, s, re.M)
    if not m: raise RuntimeError(f"{name} not found")
    return m.group(1)
def _settings(s):
    return {"config_label": _get(s, r'"config_label":\s*"([^"]*)"', "config_label"),
            "_SEED": int(_get(s, r'^_SEED\s*=\s*(\d+)\s*$', "_SEED")),
            "n_episodes": int(_get(s, r'"n_episodes":\s*(\d+)', "n_episodes")),
            "gamma": float(_get(s, r'"gamma":\s*([\d.]+)', "gamma")),
            "ent_coef_start": float(_get(s, r'"ent_coef_start":\s*([\d.]+)', "ent_coef_start")),
            "delta": float(_get(s, r'"delta":\s*([\d.]+)', "delta")),
            "demand_separation": _get(s, r'"demand_separation":\s*(True|False)', "demand_separation"),
            "n_vehicles": int(_get(s, r'"n_vehicles":\s*(\d+)', "n_vehicles")),
            "n_subchannels": int(_get(s, r'"n_subchannels":\s*(\d+)', "n_subchannels"))}
def _apply(src, edits):
    s = src
    for pat, rep, flags in edits:
        s, n = re.subn(pat, rep, s, flags=flags)
        if n != 1: raise RuntimeError(f"pattern {pat[:48]!r} matched {n} times")
    diff = [l for l in difflib.unified_diff(src.splitlines(), s.splitlines(), lineterm="", n=0)
            if l[:1] in "+-" and l[:3] not in ("+++", "---")]
    if len(diff) != 2 * len(edits): raise RuntimeError(f"{len(diff)//2} lines changed, expected {len(edits)}")
    return s, len(edits)
def _common(new_label, port):
    return [(r'("config_label":\s*)"[^"]*"', rf'\g<1>"{new_label}"', 0),
            (r'("result_file":\s*)"[^"]*"', rf'\g<1>"results/_{new_label}.json"', 0),
            (r'("ns3_port":\s*)\d+,', rf'\g<1>{port},', 0)]
def derive_orig(source, new_label, new_seed, port):
    runner, label0, seed0, N, M = source
    src = pathlib.Path(runner).read_text(); st = _settings(src)
    want = {"config_label": label0, "_SEED": seed0, "n_episodes": 5000, "gamma": 0.99, "ent_coef_start": 0.05,
            "delta": 0.3, "demand_separation": "False", "n_vehicles": N, "n_subchannels": M}
    for k, v in want.items():
        if st[k] != v: raise RuntimeError(f"{k} is {st[k]!r}, expected {v!r} — not the original runner")
    edits = [(ORIG_IMPORT[0], ORIG_IMPORT[1], re.M)] + _common(new_label, port) + [
             (r'("gamma":\s*)[\d.]+', rf'\g<1>{GAMMA}', 0), (r'("ent_coef_start":\s*)[\d.]+', rf'\g<1>{ENT}', 0)]
    if new_seed is not None: edits.append((r'^(_SEED\s*=\s*)\d+(\s*)$', rf'\g<1>{new_seed}\g<2>', re.M))
    return _apply(src, edits)
def derive_paper(source, new_label, spec, port):
    runner, label0, seed0 = source; overrides, env_class = spec
    src = pathlib.Path(runner).read_text(); st = _settings(src)
    want = {"config_label": label0, "_SEED": seed0, "n_episodes": 5000, "gamma": 0.5, "ent_coef_start": 0.01,
            "delta": 0.3, "demand_separation": "False", "n_subchannels": 5}
    for k, v in want.items():
        if st[k] != v: raise RuntimeError(f"{k} is {st[k]!r}, expected {v!r} — not the paper-arm runner")
    edits = []
    if env_class: edits.append((ENV_IMPORT, f"from envs.v2x_env_arms import {env_class} as V2XEnv", re.M))
    edits += _common(new_label, port)
    for k, v in overrides.items():
        if not isinstance(v, float): raise RuntimeError(f"override {k}: only float overrides are used here")
        edits.append((rf'("{k}":\s*)[\d.]+', rf'\g<1>{v}', 0))
    return _apply(src, edits)
def main(option, check=port_ok):
    if option not in OPTION: sys.exit("usage: python3 d9_generator.py B   (or C)")
    runs = OPTION[option]; pool = iter(CANDIDATES); failed = []
    print(f"Batch 27, option {option}: {len(runs)} runs\n")
    for kind, source, new_label, extra in runs:
        try:
            port = next(p for p in pool if check(p))
        except StopIteration:
            failed.append(new_label); print(f"  FAIL {new_label}: no candidate port was free"); continue
        try:
            if kind == "ORIG":
                s, n = derive_orig(source, new_label, extra, port)
                what = "Mode 0a + agent-specific critic" + (f", seed {source[2]} -> {extra}" if extra else "")
            else:
                s, n = derive_paper(source, new_label, extra, port)
                what = extra[1] or ", ".join(f"{k} = {v}" for k, v in extra[0].items())
            pathlib.Path(f"train_d9_{new_label}.py").write_text(s)
            print(f"  OK   train_d9_{new_label + '.py':<50} port {port}  {n} lines  {what}")
        except Exception as ex:
            failed.append(new_label); print(f"  FAIL {new_label} (from {source[0]}): {ex}  — not written")
    print(f"\n{len(runs) - len(failed)} of {len(runs)} written.")
    if failed: print("Send me the FAIL lines before launching those runs.")
if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "")
