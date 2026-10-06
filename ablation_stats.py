"""
ablation_stats.py -- critic-input ablation report (aa-achievement.md §7c). Run from ~/v2x_clean:
    python3 ablation_stats.py

Arms: C = paper's critic V(gs), one team value (existing rebuild runs)
      L = local critic V(o_i)                     (label suffix _lc)
      A = agent-specific global critic V(gs + o_i) (label suffix _as)
A vs L isolates INFORMATION; A vs C isolates STRUCTURE.

Decision criteria, fixed before the runs. Two arms DIFFER if any of:
  - structure: a different number of seeds reach the cell's structural regime;
  - delivery:  95% intervals of M0 PDR, or of M1 PDR, do not overlap (in-regime seeds, n >= 3)
               AND the means differ by at least MIN_DELIVERY_GAP (0.01) -- clear and large enough to matter;
  - timing:    median sustained time to 0.90 (M0 or M1) differs by more than a factor of 2,
               or one arm reaches 0.90 in a seed where the other never does.
A verdict is printed only when both arms have all three seeds.

LEXICOGRAPHIC RANKING (added v43, before any ablation result existed). The paper's premise is that M0
is safety-critical and M1 expendable, so "better" means: higher M0 first; M1 only on an M0 tie; speed
only on an M1 tie. Computed over ALL seeds -- a seed that failed to coordinate counts against its arm.
  step 1  M0: intervals do not overlap AND difference >= M0_GAP (0.005)
  step 2  M1: intervals do not overlap AND difference >= MIN_DELIVERY_GAP (0.01)
  step 3  speed: median sustained time to M0 >= 0.90 differs by more than a factor of 2
An arm whose M1 delivery is zero in any seed cannot rank above another.
"""
import statistics
from rebuild_stats import res, tail, timing, fmt, T975

CONFIGS = [
    ("Mode 0c, N=10", [f"A_mode0c_N10_seed{s}_gaefix_g05_ent01" for s in (1, 2, 3)],
     lambda r: r['m0_collision_rate_within_pool'] <= 0.01),
    ("Mode 0c, N=4",  [f"A_mode0c_seed{s}_5000ep_gaefix_g05_ent01" for s in (1, 2, 3)],
     lambda r: r['m0_collision_rate_within_pool'] <= 0.01 and r['m1_collision_rate_within_pool'] <= 0.01),
    ("Mode 0a, N=4",  [f"A_N4_seed{s}_5000ep_gaefix_g05_ent01" for s in (1, 2, 4)],
     lambda r: r['m0_collision_rate_within_pool'] < 0.05),
]
ARMS = (("C", ""), ("L", "_lc"), ("A", "_as"))
MIN_DELIVERY_GAP = 0.01   # fixed before the runs
M0_GAP = 0.005            # lexicographic step 1 (v43), fixed before any ablation result

def interval(xs):
    n = len(xs); m = statistics.mean(xs)
    if n < 3:
        return m, None, None
    h = T975[n] * statistics.stdev(xs) / n ** 0.5
    return m, m - h, m + h

def arm_summary(labels, regime):
    rows = []
    for l in labels:
        r = res(l)
        if r:
            t = timing(l)
            rows.append({"label": l, "r": r, "t": t, "ok": regime(r)})
    ok = [x for x in rows if x["ok"]]
    s = {"n": len(rows), "n_ok": len(ok), "rows": rows,
         "m0": interval([x["r"]["m0_pdr_mean"] for x in ok]) if ok else None,
         "m1": interval([x["r"]["m1_pdr_mean"] for x in ok]) if ok else None,
         "m0_all": interval([x["r"]["m0_pdr_mean"] for x in rows]) if rows else None,
         "m1_all": interval([x["r"]["m1_pdr_mean"] for x in rows]) if rows else None,
         "m1_zero": any(x["r"]["m1_pdr_mean"] <= 0.0 for x in rows)}
    for key in ("m0_90_sust", "m1_90_sust"):
        vals = [x["t"][key] for x in rows]
        got = [v for v in vals if v is not None]
        s[key] = (statistics.median(got) if got else None, len(vals) - len(got))
    return s

def overlap(a, b):
    if a is None or b is None or a[1] is None or b[1] is None:
        return None
    return max(a[1], b[1]) <= min(a[2], b[2])

def compare(x, y):
    """Return (verdict text, differs?) or None if either arm is incomplete."""
    if x["n"] < 3 or y["n"] < 3:
        return None
    reasons = []
    if x["n_ok"] != y["n_ok"]:
        reasons.append(f"structure: {x['n_ok']} vs {y['n_ok']} of 3 seeds in regime")
    for k, name in (("m0", "M0 PDR"), ("m1", "M1 PDR")):
        ov = overlap(x[k], y[k])
        if ov is False and abs(x[k][0] - y[k][0]) >= MIN_DELIVERY_GAP:
            reasons.append(f"{name}: {x[k][0]:.4f} vs {y[k][0]:.4f}, intervals do not overlap")
    for k, name in (("m0_90_sust", "M0 time to 0.90"), ("m1_90_sust", "M1 time to 0.90")):
        (mx, nx), (my, ny) = x[k], y[k]
        if nx != ny:
            reasons.append(f"{name}: reached in {3 - nx} vs {3 - ny} seeds")
        elif mx and my and max(mx, my) / min(mx, my) > 2:
            reasons.append(f"{name}: medians {mx} vs {my} (x{max(mx, my) / min(mx, my):.1f})")
    return ("DIFFER — " + "; ".join(reasons)) if reasons else "no difference by the pre-registered criteria", bool(reasons)

def lex_rank(x, y, nx, ny):
    """Lexicographic ranking of arm x against arm y over all seeds; None if either is incomplete."""
    if x["n"] < 3 or y["n"] < 3:
        return None
    def clearly_above(a, b, gap):
        return overlap(a, b) is False and a[0] - b[0] >= gap
    for key, gap, name in (("m0_all", M0_GAP, "M0"), ("m1_all", MIN_DELIVERY_GAP, "M1")):
        a, b = x[key], y[key]
        if clearly_above(a, b, gap):
            return f"{nx} above {ny} on {name} ({a[0]:.4f} vs {b[0]:.4f})" + ("" if not x["m1_zero"] else f" -- but {nx} has M1 = 0 in a seed: cannot rank above")
        if clearly_above(b, a, gap):
            return f"{ny} above {nx} on {name} ({b[0]:.4f} vs {a[0]:.4f})" + ("" if not y["m1_zero"] else f" -- but {ny} has M1 = 0 in a seed: cannot rank above")
    (mx, nnx), (my, nny) = x["m0_90_sust"], y["m0_90_sust"]
    if nnx != nny:
        faster = nx if nnx < nny else ny
        return f"M0 and M1 tie; {faster} above on speed (reaches M0 0.90 in more seeds)"
    if mx and my and max(mx, my) / min(mx, my) > 2:
        faster = nx if mx < my else ny
        return f"M0 and M1 tie; {faster} above on speed (median {min(mx, my)} vs {max(mx, my)})"
    return "tie on M0, M1 and speed"

def show(v):
    m, lo, hi = v if v else (None, None, None)
    if m is None: return "-"
    return f"{m:.4f}" + (f" [{lo:.4f}, {hi:.4f}]" if lo is not None else " (n<3)")

if __name__ == "__main__":
    for name, base, regime in CONFIGS:
        print(f"\n=================== {name} ===================")
        summ = {}
        for arm, suf in ARMS:
            s = arm_summary([b + suf for b in base], regime); summ[arm] = s
            print(f"  arm {arm}: {s['n']} of 3 seeds done, {s['n_ok']} in regime")
            for x in s["rows"]:
                r, t = x["r"], x["t"]; tl = tail(r)
                print(f"      {x['label']:<46} M0 {r['m0_pdr_mean']:.4f}  tail {'-' if tl is None else f'{tl:.4f}'}  "
                      f"M1 {r['m1_pdr_mean']:.4f}  M0wp {r['m0_collision_rate_within_pool']:.3f}  "
                      f"M1wp {r['m1_collision_rate_within_pool']:.3f}  to .90 M0 {fmt(t['m0_90_sust'])} M1 {fmt(t['m1_90_sust'])}"
                      + ("" if x["ok"] else "  [out of regime]") + ("" if t["done"] else "  [no 'Run complete.']"))
            print(f"      pooled: M0 {show(s['m0'])}   M1 {show(s['m1'])}   "
                  f"median to .90: M0 {fmt(s['m0_90_sust'][0])}, M1 {fmt(s['m1_90_sust'][0])}")
        for (a, b), what in ((("A", "L"), "INFORMATION (A vs L)"), (("A", "C"), "STRUCTURE   (A vs C)")):
            v = compare(summ[a], summ[b])
            print(f"  {what}: {'incomplete — waiting for seeds' if v is None else v[0]}")
        for a, b in (("A", "L"), ("A", "C"), ("L", "C")):
            r = lex_rank(summ[a], summ[b], a, b)
            print(f"  LEXICOGRAPHIC {a} vs {b}: {'incomplete — waiting for seeds' if r is None else r}")
        if name == "Mode 0a, N=4":
            print("  P35 (Mode 0a M1 stays on its floor, within-pool near 0.333, under every critic):")
            for arm, _ in ARMS:
                w = [x['r']['m1_collision_rate_within_pool'] for x in summ[arm]['rows']]
                print(f"      arm {arm}: " + (", ".join(f"{v:.3f}" for v in w) if w else "no results yet"))
