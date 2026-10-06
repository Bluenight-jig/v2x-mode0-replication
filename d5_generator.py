import re, pathlib, difflib, subprocess, time, os
BIN = os.path.expanduser("~/v2x_clean/ns3_bridge/v2x_bridge")
IMPORTS = {
    "0a": (r'^from agents\.mappo[ \t]+import MAPPOTrainer[ \t]*$',
           "from agents.mappo_gaefix import MAPPOTrainerGAEFix as MAPPOTrainer"),
    "0c": (r'^from agents\.mappo_mode0c[ \t]+import MAPPOTrainerMode0c as MAPPOTrainer[ \t]*$',
           "from agents.mappo_gaefix import MAPPOTrainerMode0cGAEFix as MAPPOTrainer"),
}
GAMMA, ENT, EPISODES = 0.5, 0.01, 5000       # the rebuild configuration (D-1, D-2 strict)
RUNS = [  # (original runner, original label, arch, original seed, original episodes, demand separation, new label, new seed)
    ("train_b5_d0c1.py", "D_N10_seed1",    "0c", 1, 5000, True,  "D_N10_seed1_gaefix_g05_ent01",           None),
    ("train_b5_d0c2.py", "D_N10_seed2",    "0c", 2, 5000, True,  "D_N10_seed2_gaefix_g05_ent01",           None),
    ("train_b5_d0c3.py", "D_N10_seed3",    "0c", 3, 5000, True,  "D_N10_seed3_gaefix_g05_ent01",           None),
    ("train_b6_c0a1.py", "C_N4_seed1",     "0a", 1, 5000, True,  "C_N4_seed1_gaefix_g05_ent01",            None),
    ("train_b6_c0a2.py", "C_N4_seed2",     "0a", 2, 5000, True,  "C_N4_seed2_gaefix_g05_ent01",            None),
    ("train_b5_c0a1.py", "C_N10_seed1",    "0a", 1, 5000, True,  "C_N10_seed1_gaefix_g05_ent01",           None),
    ("train_b5_c0a2.py", "C_N10_seed2",    "0a", 2, 5000, True,  "C_N10_seed2_gaefix_g05_ent01",           None),
    ("train_b2_c2.py",   "A_mode0c_seed2", "0c", 2, 3000, False, "A_mode0c_seed1_5000ep_gaefix_g05_ent01", 1),
    ("train_b2_c2.py",   "A_mode0c_seed2", "0c", 2, 3000, False, "A_mode0c_seed2_5000ep_gaefix_g05_ent01", None),
    ("train_b2_c3.py",   "A_mode0c_seed3", "0c", 3, 3000, False, "A_mode0c_seed3_5000ep_gaefix_g05_ent01", None),
    ("train_b2_c4.py",   "A_mode0c_seed4", "0c", 4, 3000, False, "A_mode0c_seed4_5000ep_gaefix_g05_ent01", None),
    ("train_b2_c5.py",   "A_mode0c_seed5", "0c", 5, 3000, False, "A_mode0c_seed5_5000ep_gaefix_g05_ent01", None),
]
CANDIDATES = list(range(7636, 7836, 10))      # 20 candidates, clear of every earlier batch
def port_ok(p):
    """Start the bridge on port p; the port is usable only if it is still alive after 2 s."""
    proc = subprocess.Popen([BIN, f"--port={p}", "--fcGHz=5.9"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    time.sleep(2)
    alive = proc.poll() is None
    if alive: proc.terminate(); proc.wait(timeout=5)
    return alive
def derive(runner, label, arch, seed0, eps0, ds, new_label, new_seed, port):
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
    if cur_label != label:            raise RuntimeError(f"label {cur_label!r}, expected {label!r}")
    if cur_seed != seed0:             raise RuntimeError(f"_SEED {cur_seed}, expected {seed0}")
    if cur_eps != eps0:               raise RuntimeError(f"n_episodes {cur_eps}, expected {eps0}")
    if (cur_gamma, cur_ent) != (0.99, 0.05): raise RuntimeError(f"gamma {cur_gamma}, ent_coef_start {cur_ent}; expected 0.99, 0.05")
    if cur_delta != 0.3:              raise RuntimeError(f"delta {cur_delta}, expected 0.3")
    if cur_ds != ds:                  raise RuntimeError(f"demand_separation {cur_ds}, expected {ds}")
    rx, new = IMPORTS[arch]
    edits = [(rx, new, re.M),
             (r'("config_label":\s*)"[^"]*"', rf'\g<1>"{new_label}"', 0),
             (r'("result_file":\s*)"[^"]*"', rf'\g<1>"results/_{new_label}.json"', 0),
             (r'("ns3_port":\s*)\d+,', rf'\g<1>{port},', 0),
             (r'("gamma":\s*)[\d.]+', rf'\g<1>{GAMMA}', 0),
             (r'("ent_coef_start":\s*)[\d.]+', rf'\g<1>{ENT}', 0)]
    if eps0 != EPISODES: edits.append((r'("n_episodes":\s*)\d+,', rf'\g<1>{EPISODES},', 0))
    if new_seed is not None: edits.append((r'^(_SEED\s*=\s*)\d+(\s*)$', rf'\g<1>{new_seed}\g<2>', re.M))
    for pat, rep, flags in edits:
        s, n = re.subn(pat, rep, s, flags=flags)
        if n != 1: raise RuntimeError(f"pattern {pat[:40]!r} matched {n} times")
    diff = [l for l in difflib.unified_diff(src.splitlines(), s.splitlines(), lineterm="", n=0)
            if l[:1] in "+-" and l[:3] not in ("+++", "---")]
    if len(diff) != 2 * len(edits): raise RuntimeError(f"{len(diff)//2} lines changed, expected {len(edits)}")
    return s, len(edits)
def main(check=port_ok):
    pool = iter(CANDIDATES); failed = []
    for runner, label, arch, seed0, eps0, ds, new_label, new_seed in RUNS:
        try:
            port = next(p for p in pool if check(p))
        except StopIteration:
            failed.append(new_label); print(f"  FAIL {new_label}: no candidate port was free"); continue
        try:
            s, n = derive(runner, label, arch, seed0, eps0, ds, new_label, new_seed, port)
            out = f"train_d5_{new_label}.py"; pathlib.Path(out).write_text(s)
            notes = ", ".join(x for x in [f"new seed {new_seed}" if new_seed else "", f"{eps0}->{EPISODES} ep" if eps0 != EPISODES else "", "DS" if ds else ""] if x)
            print(f"  OK   {out:<56} port {port}  {n} lines  {notes}")
        except Exception as ex:
            failed.append(new_label); print(f"  FAIL {new_label} (from {runner}): {ex}  — not written")
    print(f"\n{len(RUNS) - len(failed)} of {len(RUNS)} written.")
    if failed: print("Send me the FAIL lines before launching those runs.")
if __name__ == "__main__":
    main()
