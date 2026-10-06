import re, pathlib, difflib, subprocess, time, os
BIN = os.path.expanduser("~/v2x_clean/ns3_bridge/v2x_bridge")
IMPORTS = {
    "0a": (r'^from agents\.mappo[ \t]+import MAPPOTrainer[ \t]*$',
           "from agents.mappo_gaefix import MAPPOTrainerGAEFix as MAPPOTrainer"),
    "0c": (r'^from agents\.mappo_mode0c[ \t]+import MAPPOTrainerMode0c as MAPPOTrainer[ \t]*$',
           "from agents.mappo_gaefix import MAPPOTrainerMode0cGAEFix as MAPPOTrainer"),
}
GAMMA, ENT = 0.5, 0.01                        # the rebuild configuration (D-1)
RUNS = [  # (original runner, original label, arch, original seed, episodes, new label, new seed or None)
    ("train_b4_c4.py",  "A_mode0c_N10_seed4", "0c",  4, 5000, "A_mode0c_N10_seed4_gaefix_g05_ent01", None),
    ("train_b4_c5.py",  "A_mode0c_N10_seed5", "0c",  5, 5000, "A_mode0c_N10_seed5_gaefix_g05_ent01", None),
    ("train_b3_n10.py", "A_N10_5000ep",       "0a",  1, 5000, "A_N10_seed1_5000ep_gaefix_g05_ent01", None),
    ("train_b3_s6.py",  "A_N4_seed6_5000ep",  "0a",  6, 5000, "A_N4_seed6_5000ep_gaefix_g05_ent01",  None),
    ("train_b3_s10.py", "A_N4_seed10_5000ep", "0a", 10, 5000, "A_N4_seed10_5000ep_gaefix_g05_ent01", None),
    ("train_b3_s2.py",  "A_N4_seed2_5000ep",  "0a",  2, 5000, "A_N4_seed1_5000ep_gaefix_g05_ent01",  1),
    ("train_b2_c4.py",  "A_mode0c_seed4",     "0c",  4, 3000, "A_mode0c_seed4_gaefix_g05_ent01",     None),
    ("train_b2_c5.py",  "A_mode0c_seed5",     "0c",  5, 3000, "A_mode0c_seed5_gaefix_g05_ent01",     None),
    ("train_b2_c2.py",  "A_mode0c_seed2",     "0c",  2, 3000, "A_mode0c_seed1_gaefix_g05_ent01",     1),
]
CANDIDATES = list(range(7436, 7636, 10))      # 20 candidates, clear of every earlier batch
def port_ok(p):
    """Start the bridge on port p; the port is usable only if it is still alive after 2 s."""
    proc = subprocess.Popen([BIN, f"--port={p}", "--fcGHz=5.9"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    time.sleep(2)
    alive = proc.poll() is None
    if alive: proc.terminate(); proc.wait(timeout=5)
    return alive
def derive(runner, label, arch, seed0, eps, new_label, new_seed, port):
    src = pathlib.Path(runner).read_text(); s = src
    cur = re.search(r'"config_label":\s*"([^"]*)"', s).group(1)
    if cur != label: raise RuntimeError(f"holds label {cur!r}, expected {label!r}")
    m = re.search(r'^_SEED\s*=\s*(\d+)\s*$', s, re.M)
    if not m or int(m.group(1)) != seed0: raise RuntimeError(f"seed line {m.group(0).strip() if m else 'missing'!r}, expected _SEED = {seed0}")
    g_now = float(re.search(r'"gamma":\s*([\d.]+)', s).group(1))
    e_now = float(re.search(r'"ent_coef_start":\s*([\d.]+)', s).group(1))
    n_now = int(re.search(r'"n_episodes":\s*(\d+)', s).group(1))
    if (g_now, e_now) != (0.99, 0.05): raise RuntimeError(f"gamma {g_now}, ent_coef_start {e_now}; expected 0.99, 0.05")
    if n_now != eps: raise RuntimeError(f"n_episodes {n_now}, expected {eps}")
    rx, new = IMPORTS[arch]
    edits = [(rx, new, re.M),
             (r'("config_label":\s*)"[^"]*"', rf'\g<1>"{new_label}"', 0),
             (r'("result_file":\s*)"[^"]*"', rf'\g<1>"results/_{new_label}.json"', 0),
             (r'("ns3_port":\s*)\d+,', rf'\g<1>{port},', 0),
             (r'("gamma":\s*)[\d.]+', rf'\g<1>{GAMMA}', 0),
             (r'("ent_coef_start":\s*)[\d.]+', rf'\g<1>{ENT}', 0)]
    if new_seed is not None:
        edits.append((r'^(_SEED\s*=\s*)\d+(\s*)$', rf'\g<1>{new_seed}\g<2>', re.M))
    for pat, rep, flags in edits:
        s, n = re.subn(pat, rep, s, flags=flags)
        if n != 1: raise RuntimeError(f"pattern {pat[:40]!r} matched {n} times")
    diff = [l for l in difflib.unified_diff(src.splitlines(), s.splitlines(), lineterm="", n=0)
            if l[:1] in "+-" and l[:3] not in ("+++", "---")]
    if len(diff) != 2 * len(edits): raise RuntimeError(f"{len(diff)//2} lines changed, expected {len(edits)}")
    return s, len(edits)
def main(check=port_ok):
    pool = iter(CANDIDATES); failed = []
    for runner, label, arch, seed0, eps, new_label, new_seed in RUNS:
        try:
            port = next(p for p in pool if check(p))
        except StopIteration:
            failed.append(runner); print(f"  FAIL {runner}: no candidate port was free"); continue
        try:
            s, n = derive(runner, label, arch, seed0, eps, new_label, new_seed, port)
            out = f"train_d4_{new_label}.py"; pathlib.Path(out).write_text(s)
            tag = f"new seed {new_seed}" if new_seed is not None else f"seed {seed0}"
            print(f"  OK   {out:<52} port {port}  {eps} ep  {tag:<10} {n} lines changed")
        except Exception as ex:
            failed.append(runner); print(f"  FAIL {runner}: {ex}  — not written")
    print(f"\n{len(RUNS) - len(failed)} of {len(RUNS)} written.")
    if failed: print("Send me the FAIL lines before launching those runs.")
if __name__ == "__main__":
    main()
