"""d8_generator.py -- batch 26 (aa-achievement.md §7c): the pivot (Mode 0a + agent-specific critic at
N = 10), the pool-4 check, Mode 0c + DS at N = 4, and Arms P and S. Every runner is derived from the
PAPER ARM's runner at the same seed, so each differs from it only in the lines listed for its kind."""
import re, pathlib, difflib, subprocess, time, os
BIN = os.path.expanduser("~/v2x_clean/ns3_bridge/v2x_bridge")
PAPER_IMPORT = {
    "0a": r'^from agents\.mappo_gaefix import MAPPOTrainerGAEFix as MAPPOTrainer[ \t]*$',
    "0c": r'^from agents\.mappo_gaefix import MAPPOTrainerMode0cGAEFix as MAPPOTrainer[ \t]*$',
}
ABLATION_CLASS = {
    ("0a", "lc"): "MAPPOTrainerLocalCritic",       ("0a", "as"): "MAPPOTrainerASCritic",
    ("0c", "lc"): "MAPPOTrainerMode0cLocalCritic", ("0c", "as"): "MAPPOTrainerMode0cASCritic",
}
ENV_IMPORT = r'^from envs\.v2x_env[ \t]+import V2XEnv[ \t]*$'
A10 = {1: ("train_d4_A_N10_seed1_5000ep_gaefix_g05_ent01.py", "A_N10_seed1_5000ep_gaefix_g05_ent01", "0a", 1),
       2: ("train_d3_A_N10_seed2_5000ep_gaefix_g05_ent01.py", "A_N10_seed2_5000ep_gaefix_g05_ent01", "0a", 2),
       3: ("train_d3_A_N10_seed3_5000ep_gaefix_g05_ent01.py", "A_N10_seed3_5000ep_gaefix_g05_ent01", "0a", 3)}
C10 = lambda s: (f"train_d3_A_mode0c_N10_seed{s}_gaefix_g05_ent01.py", f"A_mode0c_N10_seed{s}_gaefix_g05_ent01", "0c", s)
C4  = lambda s: (f"train_d5_A_mode0c_seed{s}_5000ep_gaefix_g05_ent01.py", f"A_mode0c_seed{s}_5000ep_gaefix_g05_ent01", "0c", s)
POOL4 = {"demand_separation": True, "m0_subchannel_count": 4}
DS4   = {"demand_separation": True}                       # pool stays 2: two safety vehicles, two channels
RUNS = [  # (source, new label, critic arm or None, CFG overrides, environment class or None)
    (A10[1], "A_N10_seed1_5000ep_gaefix_g05_ent01_as", "as", {}, None),
    (A10[2], "A_N10_seed2_5000ep_gaefix_g05_ent01_as", "as", {}, None),
    (A10[3], "A_N10_seed3_5000ep_gaefix_g05_ent01_as", "as", {}, None),
    (C10(1), "A_mode0c_N10_seed1_gaefix_g05_ent01_armP", None, {}, "V2XEnvArmP"),
    (C10(3), "A_mode0c_N10_seed3_gaefix_g05_ent01_armP", None, {}, "V2XEnvArmP"),
    (C10(1), "D_N10_pool4_seed1_gaefix_g05_ent01", None, POOL4, None),
    (C10(2), "D_N10_pool4_seed2_gaefix_g05_ent01", None, POOL4, None),
    (C10(3), "D_N10_pool4_seed3_gaefix_g05_ent01", None, POOL4, None),
    (C10(2), "A_mode0c_N10_seed2_gaefix_g05_ent01_armP", None, {}, "V2XEnvArmP"),
    (C10(1), "A_mode0c_N10_seed1_gaefix_g05_ent01_armS", None, {}, "V2XEnvArmS"),
    (C4(1),  "D_N4_seed1_gaefix_g05_ent01", None, DS4, None),
    (C4(2),  "D_N4_seed2_gaefix_g05_ent01", None, DS4, None),
    (C4(3),  "D_N4_seed3_gaefix_g05_ent01", None, DS4, None),
]
CANDIDATES = list(range(8436, 8636, 10))      # 20 candidates, clear of every earlier batch and pre-flight
def port_ok(p):
    """Start the bridge on port p; the port is usable only if it is still alive after 2 s."""
    proc = subprocess.Popen([BIN, f"--port={p}", "--fcGHz=5.9"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    time.sleep(2)
    alive = proc.poll() is None
    if alive: proc.terminate(); proc.wait(timeout=5)
    return alive
def fmt_value(v):
    return repr(v) if isinstance(v, bool) else str(v)
def derive(source, new_label, arm, overrides, env_class, port):
    runner, label, arch, seed = source
    src = pathlib.Path(runner).read_text(); s = src
    def get(pat, name):
        m = re.search(pat, s, re.M)
        if not m: raise RuntimeError(f"{name} not found")
        return m.group(1)
    checks = {
        "config_label":        (get(r'"config_label":\s*"([^"]*)"', "config_label"), label),
        "_SEED":               (int(get(r'^_SEED\s*=\s*(\d+)\s*$', "_SEED")), seed),
        "n_episodes":          (int(get(r'"n_episodes":\s*(\d+)', "n_episodes")), 5000),
        "gamma":               (float(get(r'"gamma":\s*([\d.]+)', "gamma")), 0.5),
        "ent_coef_start":      (float(get(r'"ent_coef_start":\s*([\d.]+)', "ent_coef_start")), 0.01),
        "delta":               (float(get(r'"delta":\s*([\d.]+)', "delta")), 0.3),
        "demand_separation":   (get(r'"demand_separation":\s*(True|False)', "demand_separation"), "False"),
        "n_subchannels":       (int(get(r'"n_subchannels":\s*(\d+)', "n_subchannels")), 5),
        "m0_subchannel_count": (int(get(r'"m0_subchannel_count":\s*(\d+)', "m0_subchannel_count")), 2),
    }
    for k, (got, want) in checks.items():
        if got != want: raise RuntimeError(f"{k} is {got!r}, expected {want!r} — not the paper-arm runner")
    edits = []
    if arm is not None:
        edits.append((PAPER_IMPORT[arch], f"from agents.mappo_critic_ablation import {ABLATION_CLASS[(arch, arm)]} as MAPPOTrainer", re.M))
    if env_class is not None:
        edits.append((ENV_IMPORT, f"from envs.v2x_env_arms import {env_class} as V2XEnv", re.M))
    edits += [(r'("config_label":\s*)"[^"]*"', rf'\g<1>"{new_label}"', 0),
              (r'("result_file":\s*)"[^"]*"', rf'\g<1>"results/_{new_label}.json"', 0),
              (r'("ns3_port":\s*)\d+,', rf'\g<1>{port},', 0)]
    for k, v in overrides.items():
        pat = rf'("{k}":\s*)(True|False)' if isinstance(v, bool) else rf'("{k}":\s*)\d+'
        edits.append((pat, rf'\g<1>{fmt_value(v)}', 0))
    for pat, rep, flags in edits:
        s, n = re.subn(pat, rep, s, flags=flags)
        if n != 1: raise RuntimeError(f"pattern {pat[:48]!r} matched {n} times")
    diff = [l for l in difflib.unified_diff(src.splitlines(), s.splitlines(), lineterm="", n=0)
            if l[:1] in "+-" and l[:3] not in ("+++", "---")]
    if len(diff) != 2 * len(edits): raise RuntimeError(f"{len(diff)//2} lines changed, expected {len(edits)}")
    return s, len(edits)
def main(check=port_ok):
    pool = iter(CANDIDATES); failed = []
    for source, new_label, arm, overrides, env_class in RUNS:
        try:
            port = next(p for p in pool if check(p))
        except StopIteration:
            failed.append(new_label); print(f"  FAIL {new_label}: no candidate port was free"); continue
        try:
            s, n = derive(source, new_label, arm, overrides, env_class, port)
            pathlib.Path(f"train_d8_{new_label}.py").write_text(s)
            what = ", ".join(x for x in [ABLATION_CLASS[(source[2], arm)] if arm else "", env_class or "",
                                         str(overrides) if overrides else ""] if x) or "paper arm"
            print(f"  OK   train_d8_{new_label + '.py':<50} port {port}  {n} lines  {what}")
        except Exception as ex:
            failed.append(new_label); print(f"  FAIL {new_label} (from {source[0]}): {ex}  — not written")
    print(f"\n{len(RUNS) - len(failed)} of {len(RUNS)} written.")
    if failed: print("Send me the FAIL lines before launching those runs.")
if __name__ == "__main__":
    main()
