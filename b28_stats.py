"""b28_stats.py -- batch 28 report (aa-achievement.md §7c v69; terminology v71).   Run from ~/v2x_clean.

  1. Mode 0b, fleet level (P48): safety within-pool < 0.05 in all three seeds AND fleet M0 >= 0.90 (pooled
     mean; per-seed values shown). Ranked against all-Mode-0c and all-Mode-0a under the SAME agent-specific
     critic, by ablation_stats.lex_rank (unchanged). Per-type checks (P49-P51) need the Stage 4b evaluation.
  2. Density and supply, Mode 0a + agent-specific critic (P44; finalises F25): all points from batches 25-28.
     Rule (v62): separates if M0 within-pool < 0.05 in every seed; on the floor if within 0.02 of
     1 - (1 - 1/M)^(m0 - 1) in every seed; otherwise intermediate -- and seeds are always shown split.
  3. The derived requirement -- PDR >= 0.95 at the mean and at WIRC, safety class -- for every new cell,
     per seed wherever a point is split.
Coordination time = episodes until delivery is sustained at >= 0.90 (a learning measure, not a requirement).
"""
import re, pathlib, statistics
from rebuild_stats import res, tail, timing
from ablation_stats import arm_summary, lex_rank

REQ = 0.95
MODE0B = [f"A_mode0b_N10_seed{s}_gaefix_g05_ent01" for s in (1, 2, 3)]
C10_AS = [f"A_mode0c_N10_seed{s}_gaefix_g05_ent01_as" for s in (1, 2, 3)]
A10_AS = [f"A_N10_seed{s}_5000ep_gaefix_g05_ent01_as" for s in (1, 2, 3)]
D = lambda tag, seeds=(1, 2, 3): [f"{tag}_seed{s}_5000ep_gaefix_g05_ent01_as" for s in seeds]
POINTS = [  # (name, labels, M, batch)
    ("N = 4",  D("A_N4", (1, 2, 4)), 5, 25), ("N = 5", D("A_N5"), 5, 28), ("N = 6", D("A_N6"), 5, 28),
    ("N = 7",  D("A_N7"), 5, 27), ("N = 10", D("A_N10"), 5, 26), ("N = 15", D("A_N15"), 5, 27),
    ("N = 20", D("A_N20"), 5, 28), ("N = 4, M = 3", D("B_M3"), 3, 28), ("N = 4, M = 4", D("B_M4"), 4, 28)]

def show(v):
    if v is None: return "-"
    m, lo, hi = v
    return f"{m:.4f}" + (f" [{lo:.4f}, {hi:.4f}]" if lo is not None else " (n<3)")
def ct(v): return "never" if v is None else str(v)
def row(l, r):
    w = tail(r); t = timing(l)
    return (f"      {l:<44} M0 {r['m0_pdr_mean']:.4f}  WIRC {'-' if w is None else f'{w:.4f}'}  M1 {r['m1_pdr_mean']:.4f}  "
            f"M0wp {r['m0_collision_rate_within_pool']:.3f}  M1wp {r['m1_collision_rate_within_pool']:.3f}  "
            f"coord. time M0 {ct(t.get('m0_90_sust'))}")
def m0_count(label):
    p = pathlib.Path(f"logs/{label}.log")
    m = re.search(r"M0 count:\s*(\d+)", p.read_text(errors="replace")) if p.exists() else None
    return int(m.group(1)) if m else None
def req(v): return "-" if v is None else ("meets" if v >= REQ else f"short by {REQ - v:.3f}")

def mode0b():
    print("=================== 1. Mode 0b, fleet level (P48) ===================")
    b = arm_summary(MODE0B, lambda r: True)
    print(f"  Mode 0b (2 passive safety + 3 passive M1 on shared actors; agent-specific critic): {b['n']} seeds")
    for x in b["rows"]: print(row(x["label"], x["r"]))
    print(f"      M0 {show(b['m0_all'])}   M1 {show(b['m1_all'])}")
    for name, labels in (("all-Mode-0c, agent-specific", C10_AS), ("all-Mode-0a, agent-specific", A10_AS)):
        a = arm_summary(labels, lambda r: True)
        print(f"  reference {name}: M0 {show(a['m0_all'])}   M1 {show(a['m1_all'])}")
        print(f"      LEXICOGRAPHIC Mode 0b vs {name}: {lex_rank(b, a, 'Mode 0b', name) or 'incomplete'}")
    if b["n"] >= 3:
        wp = [x["r"]["m0_collision_rate_within_pool"] for x in b["rows"]]
        m0 = b["m0_all"][0]
        ok = all(w < 0.05 for w in wp) and m0 >= 0.90
        print(f"  safety within-pool by seed: {', '.join(f'{w:.3f}' for w in wp)}   fleet M0 (pooled) {m0:.4f}")
        print(f"  P48 (within-pool < 0.05 in every seed, fleet M0 >= 0.90): {'CORRECT' if ok else 'WRONG'}")
    print("  P49-P51 (passive vs active, passive M1 trio): need the per-vehicle Stage 4b evaluation")

def density():
    print("\n=================== 2. Density and supply, Mode 0a + agent-specific critic (P44 / F25) ===================")
    print(f"  {'point':<14}{'m0':>3}{'M':>3}{'floor':>7}   within-pool by seed      verdict")
    for name, labels, M, batch in POINTS:
        rows = [(l, res(l)) for l in labels]; rows = [(l, r) for l, r in rows if r]
        if not rows: print(f"  {name:<14} (batch {batch}): no results yet"); continue
        counts = {m0_count(l) for l, _ in rows}
        m0 = counts.pop() if len(counts) == 1 and None not in counts else None
        if m0 is None: print(f"  {name:<14} M0 count UNREADABLE from the logs"); continue
        floor = 1 - (1 - 1 / M) ** (m0 - 1)
        wp = [r["m0_collision_rate_within_pool"] for _, r in rows]
        if len(rows) < 3: v = "incomplete"
        elif all(x < 0.05 for x in wp): v = "SEPARATES"
        elif all(abs(x - floor) <= 0.02 for x in wp): v = "ON THE FLOOR"
        else:
            k = sum(x < 0.05 for x in wp); j = sum(abs(x - floor) <= 0.02 for x in wp)
            v = f"INTERMEDIATE ({k} separated, {j} on the floor, {len(wp) - k - j} between)"
        want = "SEPARATES" if m0 == 2 else "ON THE FLOOR"
        tag = "" if batch in (25, 26, 27) else "   <- P44: " + ("CORRECT" if v == want else "NOT AS PREDICTED" if v != "incomplete" else "incomplete")
        print(f"  {name:<14}{m0:>3}{M:>3}{floor:>7.3f}   {', '.join(f'{x:.3f}' for x in wp):<24} {v}{tag}")
    print("\n  Per-seed detail for the batch 28 points:")
    for name, labels, M, batch in POINTS:
        if batch != 28: continue
        for l in labels:
            r = res(l)
            if r: print(row(l, r))

def requirement():
    print("\n=================== 3. The derived requirement (PDR >= 0.95, safety class) ===================")
    print("  Each cell is checked at its own operating point, per seed; WIRC is never compared across densities (F16).")
    cells = [("Mode 0b, N = 10", MODE0B)] + [(f"Mode 0a + AS, {n}", ls) for n, ls, _, b in POINTS if b == 28]
    for name, labels in cells:
        for l in labels:
            r = res(l)
            if not r: continue
            w = tail(r)
            print(f"  {name:<24} {l.split('_seed')[1][:1]:>2}   mean {r['m0_pdr_mean']:.4f} ({req(r['m0_pdr_mean'])})   "
                  f"WIRC {'-' if w is None else f'{w:.4f}'} ({req(w)})")

if __name__ == "__main__":
    mode0b(); density(); requirement()
