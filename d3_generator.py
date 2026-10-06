import re, pathlib, difflib, subprocess, time, os
BIN = os.path.expanduser("~/v2x_clean/ns3_bridge/v2x_bridge")
IMPORTS = {
    "0a": (r'^from agents\.mappo[ \t]+import MAPPOTrainer[ \t]*$',
           "from agents.mappo_gaefix import MAPPOTrainerGAEFix as MAPPOTrainer"),
    "0c": (r'^from agents\.mappo_mode0c[ \t]+import MAPPOTrainerMode0c as MAPPOTrainer[ \t]*$',
           "from agents.mappo_gaefix import MAPPOTrainerMode0cGAEFix as MAPPOTrainer"),
}
GAMMA, ENT = 0.5, 0.01                       # round 2 uses cell g05_ent01 only
RUNS = [  # (original runner, original label, arch)
    ("train_b4_c1.py",            "A_mode0c_N10_seed1", "0c"),
    ("train_b4_c2.py",            "A_mode0c_N10_seed2", "0c"),
    ("train_b4_c3.py",            "A_mode0c_N10_seed3", "0c"),
    ("train_b16_a_n10_seed2.py",  "A_N10_seed2_5000ep", "0a"),
    ("train_b16_a_n10_seed3.py",  "A_N10_seed3_5000ep", "0a"),
]
CANDIDATES = list(range(7226, 7426, 10))      # 20 candidates, clear of every earlier batch
def port_ok(p):
    """Start the bridge on port p; the port is usable only if it is still alive after 2 s."""
    proc = subprocess.Popen([BIN, f"--port={p}", "--fcGHz=5.9"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    time.sleep(2)
    alive = proc.poll() is None
    if alive: proc.terminate(); proc.wait(timeout=5)
    return alive
def derive(runner, label, arch, port):
    src = pathlib.Path(runner).read_text(); s = src
    cur = re.search(r'"config_label":\s*"([^"]*)"', s).group(1)
    if cur != label: raise RuntimeError(f"holds label {cur!r}, expected {label!r}")
    g_now = float(re.search(r'"gamma":\s*([\d.]+)', s).group(1))
    e_now = float(re.search(r'"ent_coef_start":\s*([\d.]+)', s).group(1))
    if (g_now, e_now) != (0.99, 0.05): raise RuntimeError(f"gamma {g_now}, ent_coef_start {e_now}; expected 0.99, 0.05")
    new_label = f"{label}_gaefix_g05_ent01"
    rx, new = IMPORTS[arch]
    edits = [(rx, new, re.M),
             (r'("config_label":\s*)"[^"]*"', rf'\g<1>"{new_label}"', 0),
             (r'("result_file":\s*)"[^"]*"', rf'\g<1>"results/_{new_label}.json"', 0),
             (r'("ns3_port":\s*)\d+,', rf'\g<1>{port},', 0),
             (r'("gamma":\s*)[\d.]+', rf'\g<1>{GAMMA}', 0),
             (r'("ent_coef_start":\s*)[\d.]+', rf'\g<1>{ENT}', 0)]
    for pat, rep, flags in edits:
        s, n = re.subn(pat, rep, s, flags=flags)
        if n != 1: raise RuntimeError(f"pattern {pat[:40]!r} matched {n} times")
    diff = [l for l in difflib.unified_diff(src.splitlines(), s.splitlines(), lineterm="", n=0)
            if l[:1] in "+-" and l[:3] not in ("+++", "---")]
    if len(diff) != 2 * len(edits): raise RuntimeError(f"{len(diff)//2} lines changed, expected {len(edits)}")
    return new_label, s
def main(check=port_ok):
    pool = iter(CANDIDATES); failed = []
    for runner, label, arch in RUNS:
        try:
            port = next(p for p in pool if check(p))
        except StopIteration:
            failed.append(runner); print(f"  FAIL {runner}: no candidate port was free"); continue
        try:
            new_label, s = derive(runner, label, arch, port)
            out = f"train_d3_{new_label}.py"; pathlib.Path(out).write_text(s)
            print(f"  OK   {out:<50} port {port} (bridge started there)   6 lines changed")
        except Exception as ex:
            failed.append(runner); print(f"  FAIL {runner}: {ex}  — not written")
    print(f"\n{len(RUNS) - len(failed)} of {len(RUNS)} written.")
    if failed: print("Send me the FAIL lines before launching those runs.")
if __name__ == "__main__":
    main()
