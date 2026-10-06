"""d10_generator.py -- batch 28 (aa-achievement.md §7c, v65-v68).

  python3 d10_generator.py smoke   NOW: one short Mode 0b runner (3 training + 2 eval episodes) for a live
                                   check through the bridge; then run it and check_mode0b_smoke.py.
  python3 d10_generator.py batch   AT LAUNCH: the 18 runs of batch 28, so their ports are checked fresh.

Two derivations, each guarded so a runner is written only if its source is exactly what it should be:
  PAPER    -> Mode 0b from the PAPER-ARM Mode 0c runners at N = 10: the trainer import
              (MAPPOTrainerMode0bAS: hybrid actors, agent-specific critic, D5), the "architecture"
              field in BOTH the results file and the checkpoint, label, result file, port -- 6 lines.
              The smoke runner also shortens the run (episodes, eval episodes) -- 8 lines.
  ORIGINAL -> density and supply points, Mode 0a with the agent-specific critic (D2), from the ORIGINAL
              runners, exactly as in batch 27. Guards include N and M."""
import re, sys, pathlib, difflib, subprocess, time, os
BIN = os.path.expanduser("~/v2x_clean/ns3_bridge/v2x_bridge")
ORIG_IMPORT = (r'^from agents\.mappo[ \t]+import MAPPOTrainer[ \t]*$',
               "from agents.mappo_critic_ablation import MAPPOTrainerASCritic as MAPPOTrainer")
PAPER_0C_IMPORT = (r'^from agents\.mappo_gaefix import MAPPOTrainerMode0cGAEFix as MAPPOTrainer[ \t]*$',
                   "from agents.mappo_mode0b import MAPPOTrainerMode0bAS as MAPPOTrainer")
ARCH = (r'("architecture":\s*)"mode_0c_per_vehicle_actors"', r'\g<1>"mode_0b_hybrid_actors"')
GAMMA, ENT = 0.5, 0.01
C10 = lambda s: (f"train_d3_A_mode0c_N10_seed{s}_gaefix_g05_ent01.py", f"A_mode0c_N10_seed{s}_gaefix_g05_ent01", s)
def orig(runner, label0, seed, N, M, new_label):
    return ("ORIG", (runner, label0, seed, N, M), new_label, None)
MODE0B = [("PAPER", C10(s), f"A_mode0b_N10_seed{s}_gaefix_g05_ent01", {}) for s in (1, 2, 3)]
SMOKE  = [("PAPER", C10(1), "SMOKE_mode0b_N10", {"n_episodes": 3, "n_eval_episodes": 2})]
DENSITY = (
    [orig(f"train_b11_a_n5_seed{s}.py", f"A_N5_seed{s}_5000ep", s, 5, 5, f"A_N5_seed{s}_5000ep_gaefix_g05_ent01_as") for s in (1, 2, 3)] +
    [orig(f"train_b12_n6_s{s}_5000ep.py", f"A_N6_seed{s}_5000ep", s, 6, 5, f"A_N6_seed{s}_5000ep_gaefix_g05_ent01_as") for s in (1, 2, 3)] +
    [orig(f"train_b10_n20_s{s}.py", f"A_N20_seed{s}_5000ep", s, 20, 5, f"A_N20_seed{s}_5000ep_gaefix_g05_ent01_as") for s in (1, 2, 3)] +
    [orig(r, f"B_M3_seed{s}_5000ep", s, 4, 3, f"B_M3_seed{s}_5000ep_gaefix_g05_ent01_as")
     for s, r in ((1, "train_b11_b_m3_seed1.py"), (2, "train_b11_b_m3_seed2.py"), (3, "train_b13_b_m3_seed3.py"))] +
    [orig(f"train_b14_b_m4_seed{s}.py", f"B_M4_seed{s}_5000ep", s, 4, 4, f"B_M4_seed{s}_5000ep_gaefix_g05_ent01_as") for s in (1, 2, 3)])
MODES = {"smoke": SMOKE, "batch": MODE0B + DENSITY}
CANDIDATES = list(range(9106, 9306, 10))      # clear of batch 27 (8646-8836), its re-run spares (to 8876) and pre-flights
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
    """Each edit is (pattern, replacement, flags[, expected matches]); expected defaults to 1."""
    s, total = src, 0
    for e in edits:
        pat, rep, flags = e[:3]; want = e[3] if len(e) > 3 else 1
        s, n = re.subn(pat, rep, s, flags=flags)
        if n != want: raise RuntimeError(f"pattern {pat[:48]!r} matched {n} times, expected {want}")
        total += want
    diff = [l for l in difflib.unified_diff(src.splitlines(), s.splitlines(), lineterm="", n=0)
            if l[:1] in "+-" and l[:3] not in ("+++", "---")]
    if len(diff) != 2 * total: raise RuntimeError(f"{len(diff)//2} lines changed, expected {total}")
    return s, total
def _common(new_label, port):
    return [(r'("config_label":\s*)"[^"]*"', rf'\g<1>"{new_label}"', 0),
            (r'("result_file":\s*)"[^"]*"', rf'\g<1>"results/_{new_label}.json"', 0),
            (r'("ns3_port":\s*)\d+,', rf'\g<1>{port},', 0)]
def derive_orig(source, new_label, _unused, port):
    runner, label0, seed0, N, M = source
    src = pathlib.Path(runner).read_text(); st = _settings(src)
    want = {"config_label": label0, "_SEED": seed0, "n_episodes": 5000, "gamma": 0.99, "ent_coef_start": 0.05,
            "delta": 0.3, "demand_separation": "False", "n_vehicles": N, "n_subchannels": M}
    for k, v in want.items():
        if st[k] != v: raise RuntimeError(f"{k} is {st[k]!r}, expected {v!r} — not the original runner")
    edits = [(ORIG_IMPORT[0], ORIG_IMPORT[1], re.M)] + _common(new_label, port) + [
             (r'("gamma":\s*)[\d.]+', rf'\g<1>{GAMMA}', 0), (r'("ent_coef_start":\s*)[\d.]+', rf'\g<1>{ENT}', 0)]
    return _apply(src, edits)
def derive_paper(source, new_label, overrides, port):
    runner, label0, seed0 = source
    src = pathlib.Path(runner).read_text(); st = _settings(src)
    want = {"config_label": label0, "_SEED": seed0, "n_episodes": 5000, "gamma": 0.5, "ent_coef_start": 0.01,
            "delta": 0.3, "demand_separation": "False", "n_vehicles": 10, "n_subchannels": 5}
    for k, v in want.items():
        if st[k] != v: raise RuntimeError(f"{k} is {st[k]!r}, expected {v!r} — not the paper-arm runner")
    edits = [(PAPER_0C_IMPORT[0], PAPER_0C_IMPORT[1], re.M),
             (ARCH[0], ARCH[1], 0, 2)] + _common(new_label, port)   # architecture: results file AND checkpoint
    for k, v in overrides.items():
        if not isinstance(v, int): raise RuntimeError(f"override {k}: integer expected")
        edits.append((rf'("{k}":\s*)\d+,', rf'\g<1>{v},', 0))
    return _apply(src, edits)
def main(mode, check=port_ok):
    if mode not in MODES: sys.exit("usage: python3 d10_generator.py smoke   (now)   |   batch   (at launch)")
    runs = MODES[mode]; pool = iter(CANDIDATES); failed = []
    print(f"Batch 28 generator, mode '{mode}': {len(runs)} run(s)\n")
    for kind, source, new_label, extra in runs:
        try:
            port = next(p for p in pool if check(p))
        except StopIteration:
            failed.append(new_label); print(f"  FAIL {new_label}: no candidate port was free"); continue
        try:
            if kind == "ORIG":
                s, n = derive_orig(source, new_label, extra, port); what = "Mode 0a + agent-specific critic"
            else:
                s, n = derive_paper(source, new_label, extra, port)
                what = "Mode 0b (hybrid actors, agent-specific critic)" + (f", {extra}" if extra else "")
            pathlib.Path(f"train_d10_{new_label}.py").write_text(s)
            print(f"  OK   train_d10_{new_label + '.py':<50} port {port}  {n} lines  {what}")
        except Exception as ex:
            failed.append(new_label); print(f"  FAIL {new_label} (from {source[0]}): {ex}  — not written")
    print(f"\n{len(runs) - len(failed)} of {len(runs)} written.")
    if failed: print("Send me the FAIL lines before launching those runs.")
if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "")
