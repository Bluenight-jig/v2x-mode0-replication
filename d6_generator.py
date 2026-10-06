"""d6_generator.py -- batch 24: critic-input ablation, Mode 0c (aa-achievement.md §7c).
Each runner is derived from the PAPER ARM's runner at the same seed and changes exactly
four lines: the trainer import, the label, the result file and the port."""
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
RUNS = [  # (paper-arm runner, paper-arm label, arch, seed, arm) -- chain order: T1..T5 first legs, then second legs
    ("train_d3_A_mode0c_N10_seed1_gaefix_g05_ent01.py", "A_mode0c_N10_seed1_gaefix_g05_ent01", "0c", 1, "lc"),
    ("train_d3_A_mode0c_N10_seed2_gaefix_g05_ent01.py", "A_mode0c_N10_seed2_gaefix_g05_ent01", "0c", 2, "lc"),
    ("train_d3_A_mode0c_N10_seed3_gaefix_g05_ent01.py", "A_mode0c_N10_seed3_gaefix_g05_ent01", "0c", 3, "lc"),
    ("train_d3_A_mode0c_N10_seed1_gaefix_g05_ent01.py", "A_mode0c_N10_seed1_gaefix_g05_ent01", "0c", 1, "as"),
    ("train_d3_A_mode0c_N10_seed2_gaefix_g05_ent01.py", "A_mode0c_N10_seed2_gaefix_g05_ent01", "0c", 2, "as"),
    ("train_d5_A_mode0c_seed1_5000ep_gaefix_g05_ent01.py", "A_mode0c_seed1_5000ep_gaefix_g05_ent01", "0c", 1, "as"),
    ("train_d5_A_mode0c_seed2_5000ep_gaefix_g05_ent01.py", "A_mode0c_seed2_5000ep_gaefix_g05_ent01", "0c", 2, "as"),
    ("train_d5_A_mode0c_seed3_5000ep_gaefix_g05_ent01.py", "A_mode0c_seed3_5000ep_gaefix_g05_ent01", "0c", 3, "as"),
    ("train_d5_A_mode0c_seed1_5000ep_gaefix_g05_ent01.py", "A_mode0c_seed1_5000ep_gaefix_g05_ent01", "0c", 1, "lc"),
    ("train_d5_A_mode0c_seed2_5000ep_gaefix_g05_ent01.py", "A_mode0c_seed2_5000ep_gaefix_g05_ent01", "0c", 2, "lc"),
]
CANDIDATES = list(range(8036, 8236, 10))      # 20 candidates, clear of every earlier batch and the pre-flight
def port_ok(p):
    """Start the bridge on port p; the port is usable only if it is still alive after 2 s."""
    proc = subprocess.Popen([BIN, f"--port={p}", "--fcGHz=5.9"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    time.sleep(2)
    alive = proc.poll() is None
    if alive: proc.terminate(); proc.wait(timeout=5)
    return alive
def derive(runner, label, arch, seed, arm, port):
    src = pathlib.Path(runner).read_text(); s = src
    def get(pat, name):
        m = re.search(pat, s, re.M)
        if not m: raise RuntimeError(f"{name} not found")
        return m.group(1)
    cur_label = get(r'"config_label":\s*"([^"]*)"', "config_label")
    cur_seed  = int(get(r'^_SEED\s*=\s*(\d+)\s*$', "_SEED"))
    cur_eps   = int(get(r'"n_episodes":\s*(\d+)', "n_episodes"))
    cur_gamma = float(get(r'"gamma":\s*([\d.]+)', "gamma"))
    cur_ent   = float(get(r'"ent_coef_start":\s*([\d.]+)', "ent_coef_start"))
    cur_delta = float(get(r'"delta":\s*([\d.]+)', "delta"))
    cur_ds    = get(r'"demand_separation":\s*(True|False)', "demand_separation") == "True"
    if cur_label != label:     raise RuntimeError(f"label {cur_label!r}, expected {label!r}")
    if cur_seed != seed:       raise RuntimeError(f"_SEED {cur_seed}, expected {seed}")
    if cur_eps != 5000:        raise RuntimeError(f"n_episodes {cur_eps}, expected 5000")
    if (cur_gamma, cur_ent) != (0.5, 0.01): raise RuntimeError(f"gamma {cur_gamma}, ent_coef_start {cur_ent}; expected the rebuild's 0.5, 0.01")
    if cur_delta != 0.3:       raise RuntimeError(f"delta {cur_delta}, expected 0.3")
    if cur_ds:                 raise RuntimeError("demand_separation is True; the ablation uses the shared pool")
    new_label = f"{label}_{arm}"
    cls = ABLATION_CLASS[(arch, arm)]
    edits = [(PAPER_IMPORT[arch], f"from agents.mappo_critic_ablation import {cls} as MAPPOTrainer", re.M),
             (r'("config_label":\s*)"[^"]*"', rf'\g<1>"{new_label}"', 0),
             (r'("result_file":\s*)"[^"]*"', rf'\g<1>"results/_{new_label}.json"', 0),
             (r'("ns3_port":\s*)\d+,', rf'\g<1>{port},', 0)]
    for pat, rep, flags in edits:
        s, n = re.subn(pat, rep, s, flags=flags)
        if n != 1: raise RuntimeError(f"pattern {pat[:48]!r} matched {n} times — is this a paper-arm (per-agent GAE) runner?")
    diff = [l for l in difflib.unified_diff(src.splitlines(), s.splitlines(), lineterm="", n=0)
            if l[:1] in "+-" and l[:3] not in ("+++", "---")]
    if len(diff) != 2 * len(edits): raise RuntimeError(f"{len(diff)//2} lines changed, expected {len(edits)}")
    return new_label, s, cls
def main(check=port_ok):
    pool = iter(CANDIDATES); failed = []
    for runner, label, arch, seed, arm in RUNS:
        try:
            port = next(p for p in pool if check(p))
        except StopIteration:
            failed.append(runner); print(f"  FAIL {runner}: no candidate port was free"); continue
        try:
            new_label, s, cls = derive(runner, label, arch, seed, arm, port)
            out = f"train_d6_{new_label}.py"; pathlib.Path(out).write_text(s)
            print(f"  OK   {out:<60} port {port}  {cls}")
        except Exception as ex:
            failed.append(runner); print(f"  FAIL {runner} [{arm}]: {ex}  — not written")
    print(f"\n{len(RUNS) - len(failed)} of {len(RUNS)} written, each differing from its paper-arm runner in exactly 4 lines.")
    if failed: print("Send me the FAIL lines before launching those runs.")
if __name__ == "__main__":
    main()
