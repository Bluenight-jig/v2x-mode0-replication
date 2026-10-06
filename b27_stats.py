"""b27_stats.py -- batch 27 report (aa-achievement.md §7c v62; terminology v71).   Run from ~/v2x_clean.

  1. Arm S at N = 10 (P43): seeds 1-3 (seed 1 from batch 26) against Mode 0c's paper arm, matched seeds.
  2. delta = 1.0 at N = 10 (P45, answers R3-h2): against delta = 0.3 (the paper arm), matched seeds.
  3. Arm S at N = 4 (P47): against Mode 0c at N = 4 (5000 episodes), matched seeds.
  4. Density, Mode 0a + agent-specific critic (P44): N = 7 and 15 (batch 27), with N = 4 and 10 for context.
  5. The derived requirement -- PDR >= 0.95 at the mean and at WIRC, safety class -- for every new cell.
Rankings use ablation_stats.lex_rank, unchanged: M0 first (floor 0.005), M1 on a tie (floor 0.01),
coordination time last. Coordination time = episodes until delivery is sustained at >= 0.90.
"""
import re, pathlib, statistics
from rebuild_stats import res, tail, timing
from ablation_stats import interval, arm_summary, lex_rank

REQ, M = 0.95, 5
C10   = [f"A_mode0c_N10_seed{s}_gaefix_g05_ent01" for s in (1, 2, 3)]
ARMS10 = [f"A_mode0c_N10_seed{s}_gaefix_g05_ent01_armS" for s in (1, 2, 3)]
D10   = [f"A_mode0c_N10_seed{s}_gaefix_g05_ent01_d10" for s in (1, 2, 3)]
C4    = [f"A_mode0c_seed{s}_5000ep_gaefix_g05_ent01" for s in (1, 2, 3)]
ARMS4 = [f"A_mode0c_seed{s}_5000ep_gaefix_g05_ent01_armS" for s in (1, 2, 3)]
DENSITY = {4:  [f"A_N4_seed{s}_5000ep_gaefix_g05_ent01_as" for s in (1, 2, 4)],
           7:  [f"A_N7_seed{s}_5000ep_gaefix_g05_ent01_as" for s in (1, 2, 3)],
           10: [f"A_N10_seed{s}_5000ep_gaefix_g05_ent01_as" for s in (1, 2, 3)],
           15: [f"A_N15_seed{s}_5000ep_gaefix_g05_ent01_as" for s in (1, 2, 3)]}
NEW_IN_B27 = (7, 15)

def show(v):
    if v is None: return "-"
    m, lo, hi = v
    return f"{m:.4f}" + (f" [{lo:.4f}, {hi:.4f}]" if lo is not None else " (n<3)")
def ct(v): return "never" if v is None else str(v)
def row(l, r):
    w = tail(r); t = timing(l)
    return (f"      {l:<46} M0 {r['m0_pdr_mean']:.4f}  WIRC {'-' if w is None else f'{w:.4f}'}  M1 {r['m1_pdr_mean']:.4f}  "
            f"M0wp {r['m0_collision_rate_within_pool']:.3f}  M1wp {r['m1_collision_rate_within_pool']:.3f}  "
            f"M0coll {r.get('m0_collision_rate', float('nan')):.3f}  coord. time M0 {ct(t.get('m0_90_sust'))} (0.95: {ct(t.get('m0_95_sust'))})")
def compare(title, ref_labels, test_labels, ref_name, test_name):
    print(f"\n=================== {title} ===================")
    a, b = arm_summary(ref_labels, lambda r: True), arm_summary(test_labels, lambda r: True)
    for name, s in ((ref_name, a), (test_name, b)):
        print(f"  {name}: {s['n']} seeds"); [print(row(x["label"], x["r"])) for x in s["rows"]]
        print(f"      M0 {show(s['m0_all'])}   M1 {show(s['m1_all'])}")
    lr = lex_rank(a, b, ref_name, test_name)
    print(f"  LEXICOGRAPHIC {ref_name} vs {test_name}: {lr or 'incomplete -- waiting for seeds'}")
    return a, b, lr
def requirement(name, labels):
    rows = [(l, res(l)) for l in labels]; rows = [(l, r) for l, r in rows if r]
    if not rows: return
    m0 = statistics.mean(r["m0_pdr_mean"] for _, r in rows)
    ws = [tail(r) for _, r in rows if tail(r) is not None]; w = statistics.mean(ws) if ws else None
    f = lambda v: "-" if v is None else ("meets" if v >= REQ else f"short by {REQ - v:.3f}")
    print(f"  {name:<34} n = {len(rows)}   mean {m0:.4f} ({f(m0)})   WIRC {'-' if w is None else f'{w:.4f}'} ({f(w)})")
def m0_count(label):
    p = pathlib.Path(f"logs/{label}.log")
    m = re.search(r"M0 count:\s*(\d+)", p.read_text(errors="replace")) if p.exists() else None
    return int(m.group(1)) if m else None

def main():
    a, b, lr = compare("1. Arm S at N = 10 (P43)", C10, ARMS10, "Mode 0c", "Arm S")
    if lr:
        d = b["m0_all"][0] - a["m0_all"][0]
        tie = " on M0 (" not in lr
        print(f"  Arm S - Mode 0c on M0: {d:+.4f}")
        print(f"  P43 (within 0.01 of Mode 0c, a lexicographic tie on M0): {'CORRECT' if tie and abs(d) < 0.01 else 'WRONG'}")
        print(f"  F23 completed: learned power adds {-d:+.4f} of safety delivery on top of learned channel choice")

    a, b, lr = compare("2. delta = 1.0 at N = 10 (P45, R3-h2)", C10, D10, "delta 0.3", "delta 1.0")
    if lr:
        print(f"  M0 change: {b['m0_all'][0] - a['m0_all'][0]:+.4f}   M1 change: {b['m1_all'][0] - a['m1_all'][0]:+.4f}")
        print(f"  P45 (delta 0.3 ranks above delta 1.0 on M0): {'CORRECT' if lr.startswith('delta 0.3 above delta 1.0 on M0') else 'WRONG'}")

    a, b, lr = compare("3. Arm S at N = 4 (P47)", C4, ARMS4, "Mode 0c", "Arm S")
    if lr:
        both_tie = " on M0 (" not in lr and " on M1 (" not in lr
        print(f"  P47 (tie on M0 and M1 -- both at the ceiling): {'CORRECT' if both_tie else 'WRONG'}")

    print("\n=================== 4. Density, Mode 0a + agent-specific critic (P44) ===================")
    print("  Rule (v62): separates if M0 within-pool < 0.05 in every seed; on the floor if within 0.02 of")
    print("  1 - (1 - 1/M)^(m0 - 1) in every seed; otherwise intermediate.")
    for N, labels in DENSITY.items():
        rows = [(l, res(l)) for l in labels]; rows = [(l, r) for l, r in rows if r]
        tag = "batch 27" if N in NEW_IN_B27 else "context"
        if not rows: print(f"\n  N = {N} ({tag}): no results yet"); continue
        counts = {m0_count(l) for l, _ in rows}
        m0 = counts.pop() if len(counts) == 1 and None not in counts else None
        print(f"\n  N = {N} ({tag}): {len(rows)} seeds, M0 count {m0 if m0 else 'UNREADABLE'}")
        for l, r in rows: print(row(l, r))
        if m0 is None or len(rows) < 3: print("      verdict: incomplete"); continue
        floor = 1 - (1 - 1 / M) ** (m0 - 1)
        wp = [r["m0_collision_rate_within_pool"] for _, r in rows]
        v = ("SEPARATES" if all(x < 0.05 for x in wp) else
             "ON THE FLOOR" if all(abs(x - floor) <= 0.02 for x in wp) else "INTERMEDIATE -- report seed by seed")
        want = "SEPARATES" if m0 == 2 else "ON THE FLOOR"
        print(f"      floor {floor:.3f}; within-pool {', '.join(f'{x:.3f}' for x in wp)} -> {v}")
        print(f"      P44 at N = {N} (predicted {want.lower()}): {'CORRECT' if v == want else 'WRONG' if not v.startswith('INTER') else 'NOT AS PREDICTED (intermediate)'}")

    print("\n=================== 5. The derived requirement (PDR >= 0.95, safety class) ===================")
    print("  Each cell is checked at its own operating point; WIRC is never compared across densities (F16).")
    for name, labels in (("Arm S, N = 10", ARMS10), ("delta = 1.0, N = 10", D10), ("Arm S, N = 4", ARMS4),
                         ("Mode 0a + AS, N = 7", DENSITY[7]), ("Mode 0a + AS, N = 15", DENSITY[15])):
        requirement(name, labels)

if __name__ == "__main__":
    main()
