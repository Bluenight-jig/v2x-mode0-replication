"""b26_stats.py -- batch 26 report (aa-achievement.md §7c).   Run from ~/v2x_clean:   python3 b26_stats.py

  1. The pivot (P40) -- decided in v56; printed for completeness.
  2. The low-load partition (P41): Mode 0c + demand separation at N = 4 (pool 2, three seeds) against the
     shared pool at N = 4 (n = 5). Decided by the SAME rule as ds_test_stats.py (fixed v42, extension v43),
     reproduced unchanged below as decide().
  3. The arms (P42, P43): Arm P against Mode 0c's paper arm at matched seeds 1-3, ranked by
     ablation_stats.lex_rank (M0 first, floor 0.005; M1 on a tie, floor 0.01; speed last). Arm S has only
     seed 1 in this batch, so its verdict waits for seeds 2-3 (batch 27).
The pool-4 test is decided by ds_test_stats.py -- run that as well.
"""
from rebuild_stats import res, tail, timing
from ablation_stats import interval, overlap, arm_summary, lex_rank

M0_GAP, M1_GAP = 0.005, 0.01
C10 = [f"A_mode0c_N10_seed{s}_gaefix_g05_ent01" for s in (1, 2, 3)]
ARMP = [f"A_mode0c_N10_seed{s}_gaefix_g05_ent01_armP" for s in (1, 2, 3)]
ARMS1 = "A_mode0c_N10_seed1_gaefix_g05_ent01_armS"
PIVOT = [f"A_N10_seed{s}_5000ep_gaefix_g05_ent01_as" for s in (1, 2, 3)]
S_N4 = [f"A_mode0c_seed{s}_5000ep_gaefix_g05_ent01" for s in (1, 2, 3, 4, 5)]
D_N4 = [f"D_N4_seed{s}_gaefix_g05_ent01" for s in (1, 2, 3)]

def group(labels):                       # identical to ds_test_stats.arm()
    rows = [(l, res(l)) for l in labels]
    rows = [(l, r) for l, r in rows if r]
    m0 = [r["m0_pdr_mean"] for _, r in rows]; m1 = [r["m1_pdr_mean"] for _, r in rows]
    return rows, (interval(m0) if m0 else None), (interval(m1) if m1 else None), any(v <= 0.0 for v in m1)

def clearly_above(a, b, gap):            # identical to ds_test_stats.clearly_above()
    return a is not None and b is not None and overlap(a, b) is False and a[0] - b[0] >= gap

def decide(S, D):
    """ds_test_stats.py's verdict logic, unchanged. Returns a key, 'EXTEND', or 'INCOMPLETE'."""
    if len(S[0]) < 3 or len(D[0]) < 3: return "INCOMPLETE"
    (_, s0, s1, sz), (_, d0, d1, dz) = S, D
    if clearly_above(d0, s0, M0_GAP) and not dz:   return "D_M0"
    if clearly_above(s0, d0, M0_GAP) and not sz:   return "S_M0"
    if abs(d0[0] - s0[0]) >= M0_GAP:               return "EXTEND"
    if clearly_above(s1, d1, M1_GAP) and not sz:   return "S_M1"
    if clearly_above(d1, s1, M1_GAP) and not dz:   return "D_M1"
    return "TIE"

def show(v):
    if v is None: return "-"
    m, lo, hi = v
    return f"{m:.4f}" + (f" [{lo:.4f}, {hi:.4f}]" if lo is not None else " (n<3)")

def row(l, r):
    tl = tail(r); t = timing(l)
    return (f"      {l:<44} M0 {r['m0_pdr_mean']:.4f}  tail {'-' if tl is None else f'{tl:.4f}'}  M1 {r['m1_pdr_mean']:.4f}  "
            f"M0wp {r['m0_collision_rate_within_pool']:.3f}  M1wp {r['m1_collision_rate_within_pool']:.3f}  "
            f"M0coll {r.get('m0_collision_rate', float('nan')):.3f}  to .90 M0 {t.get('m0_90_sust') or 'never'}")

def main():
    print("=================== 1. The pivot (P40, decided v56) ===================")
    w = []
    for l in PIVOT:
        r = res(l)
        if r: print(row(l, r)); w.append(r["m0_collision_rate_within_pool"])
    if len(w) == 3:
        print("  P40:", "ON THE FLOOR (confirmed)" if all(abs(x - 0.590) <= 0.02 for x in w) else "CHANGED -- inspect")

    print("\n=================== 2. Low-load partition, N = 4 (P41) ===================")
    S, D = group(S_N4), group(D_N4)
    for tag, g in (("S (shared pool)", S), ("D (partition, pool 2)", D)):
        print(f"  {tag}: {len(g[0])} seeds"); [print(row(l, r)) for l, r in g[0]]
        print(f"      M0 {show(g[1])}   M1 {show(g[2])}")
    v = decide(S, D)
    text = {"INCOMPLETE": "incomplete -- waiting for seeds",
            "EXTEND": "EXTEND -- M0 ties statistically but the point estimates differ by >= 0.005 (rule v43)",
            "D_M0": "D wins on M0", "S_M0": "S wins on M0", "S_M1": "M0 ties, S wins on M1",
            "D_M1": "M0 ties, D wins on M1", "TIE": "TIE on M0 and M1"}[v]
    print(f"  VERDICT: {text}")
    if v not in ("INCOMPLETE", "EXTEND"):
        print(f"  P41 (lexicographic tie at low load): {'CORRECT' if v == 'TIE' else 'WRONG'}")

    print("\n=================== 3. The arms (P42, P43) ===================")
    C, P = arm_summary(C10, lambda r: True), arm_summary(ARMP, lambda r: True)
    for tag, s in (("Mode 0c, paper arm, seeds 1-3", C), ("Arm P (random channel, learned power)", P)):
        print(f"  {tag}: {s['n']} seeds"); [print(row(x['label'], x['r'])) for x in s["rows"]]
        print(f"      M0 {show(s['m0_all'])}   M1 {show(s['m1_all'])}")
    lr = lex_rank(C, P, "Mode 0c", "Arm P")
    print(f"  LEXICOGRAPHIC Mode 0c vs Arm P: {lr or 'incomplete'}")
    if P["n"] >= 3:
        m = P["m0_all"][0]
        print(f"  P42 (Arm P M0 far below Mode 0c, near random level, <= 0.70): {'CORRECT' if m <= 0.70 and lr and lr.startswith('Mode 0c above') else 'WRONG'} (Arm P M0 {m:.4f})")
    s1, c1 = res(ARMS1), res(C10[0])
    if s1 and c1:
        print(f"\n  Arm S (learned channel, fixed 23 dBm), seed 1 only:"); print(row(ARMS1, s1))
        d = s1["m0_pdr_mean"] - c1["m0_pdr_mean"]
        print(f"      paired with Mode 0c seed 1: M0 {d:+.4f}, M1 {s1['m1_pdr_mean'] - c1['m1_pdr_mean']:+.4f}")
        print(f"  P43 (within 0.01 of Mode 0c): PROVISIONAL -- seed 1 is {'within' if abs(d) < 0.01 else 'outside'} 0.01; verdict after seeds 2-3 (batch 27)")
    print("\nPool-4 test: run  python3 ds_test_stats.py")

if __name__ == "__main__":
    main()
