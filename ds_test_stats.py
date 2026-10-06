"""
ds_test_stats.py -- demand-separation tests under the paper's safety-first premise (aa-achievement.md §7c).
Run from ~/v2x_clean:   python3 ds_test_stats.py

S = shared pool, D = demand separation with a partition sized to the safety demand. Mode 0c, N = 10.
  M7    : M = 7, pool 5 (the regime m0 < M < N), both arms new, seeds 1-3
  pool4 : M = 5, pool 4, against the existing shared-pool headline cell (n = 5)

DECISION RULE (fixed v42; extension fixed v43; written here before either test ran). Over ALL seeds:
  step 1  M0: an arm wins if its 95% interval does not overlap the other's AND it is higher by >= 0.005
  step 2  M1, only on an M0 tie: same test, difference >= 0.01
  floor   an arm with M1 delivery of zero in any seed cannot win
  EXTEND  if M0 ties but the point estimates differ by >= 0.005: two more seeds per arm before the verdict
"""
import statistics
from rebuild_stats import res, tail
from ablation_stats import interval, overlap

M0_GAP, M1_GAP = 0.005, 0.01
TESTS = [
    ("M = 7, pool 5 (m0 < M < N)",
     [f"A_mode0c_N10_M7_seed{s}_gaefix_g05_ent01" for s in (1, 2, 3)],
     [f"D_N10_M7_pool5_seed{s}_gaefix_g05_ent01" for s in (1, 2, 3)]),
    ("M = 5, pool 4 (the headline, critical load)",
     [f"A_mode0c_N10_seed{s}_gaefix_g05_ent01" for s in (1, 2, 3, 4, 5)],
     [f"D_N10_pool4_seed{s}_gaefix_g05_ent01" for s in (1, 2, 3)]),
]
CLAIMS = {
    "D_M0": "D wins on M0: a partition sized to the safety demand RAISES safety delivery over learned sharing; report its M1 cost.",
    "S_M0": "S wins on M0: the partition HARMS safety at this operating point; it is not recommended here.",
    "S_M1": "M0 ties, S wins on M1: the framework achieves safety-first allocation WITHOUT a partition; the partition's case rests on assurance only.",
    "D_M1": "M0 ties, D wins on M1: unexpected; report as found.",
    "TIE":  "Both tie: equivalent performance; the partition's case rests on assurance only.",
}

def arm(labels):
    rows = [(l, res(l)) for l in labels]
    rows = [(l, r) for l, r in rows if r]
    m0 = [r["m0_pdr_mean"] for _, r in rows]; m1 = [r["m1_pdr_mean"] for _, r in rows]
    return rows, (interval(m0) if m0 else None), (interval(m1) if m1 else None), any(v <= 0.0 for v in m1)

def clearly_above(a, b, gap):
    return a is not None and b is not None and overlap(a, b) is False and a[0] - b[0] >= gap

def show(v):
    if v is None: return "-"
    m, lo, hi = v
    return f"{m:.4f}" + (f" [{lo:.4f}, {hi:.4f}]" if lo is not None else " (n<3)")

for name, s_labels, d_labels in TESTS:
    print(f"\n=================== {name} ===================")
    S = arm(s_labels); D = arm(d_labels)
    for tag, (rows, m0, m1, _) in (("S (shared pool)", S), ("D (partition)", D)):
        print(f"  {tag}: {len(rows)} seeds")
        for l, r in rows:
            tl = tail(r)
            print(f"      {l:<44} M0 {r['m0_pdr_mean']:.4f}  tail {'-' if tl is None else f'{tl:.4f}'}  M1 {r['m1_pdr_mean']:.4f}  "
                  f"M0wp {r['m0_collision_rate_within_pool']:.3f}  M1wp {r['m1_collision_rate_within_pool']:.3f}  M0coll {r.get('m0_collision_rate', float('nan')):.3f}")
        print(f"      M0 {show(m0)}   M1 {show(m1)}")
    if len(S[0]) < 3 or len(D[0]) < 3:
        print("  VERDICT: incomplete — waiting for seeds"); continue
    (_, s0, s1, sz), (_, d0, d1, dz) = S, D
    if clearly_above(d0, s0, M0_GAP) and not dz:   key = "D_M0"
    elif clearly_above(s0, d0, M0_GAP) and not sz: key = "S_M0"
    elif abs(d0[0] - s0[0]) >= M0_GAP:
        print(f"  VERDICT: EXTEND — M0 ties statistically but the point estimates differ by {d0[0] - s0[0]:+.4f}; "
              f"add two seeds per arm before deciding (rule fixed v43)"); continue
    elif clearly_above(s1, d1, M1_GAP) and not sz: key = "S_M1"
    elif clearly_above(d1, s1, M1_GAP) and not dz: key = "D_M1"
    else: key = "TIE"
    print(f"  M1 cost of the partition: {d1[0] - s1[0]:+.4f}")
    print(f"  VERDICT: {CLAIMS[key]}")
