import re, pathlib, difflib
IMPORTS = {
    "0a": (r'^from agents\.mappo[ \t]+import MAPPOTrainer[ \t]*$',
           "from agents.mappo_gaefix import MAPPOTrainerGAEFix as MAPPOTrainer"),
    "0c": (r'^from agents\.mappo_mode0c[ \t]+import MAPPOTrainerMode0c as MAPPOTrainer[ \t]*$',
           "from agents.mappo_gaefix import MAPPOTrainerMode0cGAEFix as MAPPOTrainer"),
}
CELLS = {  # name: (gamma, ent_coef_start); None = keep the original value
    "base":      (None, None),
    "ent01":     (None, 0.01),
    "g05":       (0.5,  None),
    "g05_ent01": (0.5,  0.01),
}
RUNS = [  # (original runner, original label, arch, cell, port)  -- wave 1 first, then wave 2
    ("train_b2_c2.py",     "A_mode0c_seed2",    "0c", "base",      7016),
    ("train_b2_c2.py",     "A_mode0c_seed2",    "0c", "ent01",     7026),
    ("train_b2_c2.py",     "A_mode0c_seed2",    "0c", "g05",       7036),
    ("train_b2_c2.py",     "A_mode0c_seed2",    "0c", "g05_ent01", 7046),
    ("train_b2_c3.py",     "A_mode0c_seed3",    "0c", "g05_ent01", 7056),
    ("train_b2_c3.py",     "A_mode0c_seed3",    "0c", "base",      7066),
    ("train_b2_c3.py",     "A_mode0c_seed3",    "0c", "ent01",     7076),
    ("train_b2_c3.py",     "A_mode0c_seed3",    "0c", "g05",       7086),
    ("train_b3_s2.py",     "A_N4_seed2_5000ep", "0a", "g05_ent01", 7096),
    ("train_b2_anneal.py", "A_N4_seed4_5000ep", "0a", "g05_ent01", 7106),
]
def derive(runner, label, arch, cell, port):
    src = pathlib.Path(runner).read_text(); s = src
    cur = re.search(r'"config_label":\s*"([^"]*)"', s).group(1)
    if cur != label: raise RuntimeError(f"holds label {cur!r}, expected {label!r}")
    g_now = float(re.search(r'"gamma":\s*([\d.]+)', s).group(1))
    e_now = float(re.search(r'"ent_coef_start":\s*([\d.]+)', s).group(1))
    if (g_now, e_now) != (0.99, 0.05): raise RuntimeError(f"gamma {g_now}, ent_coef_start {e_now}; expected 0.99, 0.05")
    new_label = f"{label}_gaefix" + ("" if cell == "base" else f"_{cell}")
    rx, new = IMPORTS[arch]
    edits = [(rx, new, re.M),
             (r'("config_label":\s*)"[^"]*"', rf'\g<1>"{new_label}"', 0),
             (r'("result_file":\s*)"[^"]*"', rf'\g<1>"results/_{new_label}.json"', 0),
             (r'("ns3_port":\s*)\d+,', rf'\g<1>{port},', 0)]
    g, e = CELLS[cell]
    if g is not None: edits.append((r'("gamma":\s*)[\d.]+', rf'\g<1>{g}', 0))
    if e is not None: edits.append((r'("ent_coef_start":\s*)[\d.]+', rf'\g<1>{e}', 0))
    for pat, rep, flags in edits:
        s, n = re.subn(pat, rep, s, flags=flags)
        if n != 1: raise RuntimeError(f"pattern {pat[:40]!r} matched {n} times")
    diff = [l for l in difflib.unified_diff(src.splitlines(), s.splitlines(), lineterm="", n=0)
            if l[:1] in "+-" and l[:3] not in ("+++", "---")]
    if len(diff) != 2 * len(edits): raise RuntimeError(f"{len(diff)//2} lines changed, expected {len(edits)}")
    return new_label, s, len(edits)
if __name__ == "__main__":
    failed = []
    for runner, label, arch, cell, port in RUNS:
        try:
            new_label, s, n = derive(runner, label, arch, cell, port)
            out = f"train_d2_{new_label}.py"; pathlib.Path(out).write_text(s)
            print(f"  OK   {out:<52} {n} lines changed   port {port}")
        except Exception as ex:
            failed.append(runner); print(f"  FAIL {runner} [{cell}]: {ex}  — not written")
    print(f"\n{len(RUNS) - len(failed)} of {len(RUNS)} written.")
    if failed: print("Send me the FAIL lines before launching those runs.")
