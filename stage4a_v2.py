#!/usr/bin/env python3
"""
stage4a_v2.py -- Stage 4a v2: every interval the paper reports, from the rebuild runs.
Run from ~/v2x_clean:   python3 stage4a_v2.py          (read-only; writes only results/_ci_summary_v2.*)

The method is stage4a_ci.py's (v1, aa-achievement.md §4n), unchanged:
  two-sided 95% t-intervals on the SAMPLE standard deviation (ddof = 1); bounded metrics are clipped at their
  bounds and marked *; bimodal cells are SPLIT before averaging; n = 2 cells are reported as their two seed values
  (rule v24); an unsplit cell whose M0 or M1 delivery spans more than 0.05 is flagged.
New in v2:
  * the cells are the rebuild's (standard MAPPO): figures 7-A to 7-D, table T-1, Mode 0b, the delta test;
  * metric names follow the C&L terminology (v71): WIRC = m0_pdr_p05_intra;
  * the 0.95 requirement (safety class) is judged on the INTERVAL as well as the mean:
    MEETS (interval wholly >= 0.95), SHORT (wholly below) or STRADDLES;
  * coordination time (episodes until delivery is sustained at >= 0.90) is a median and range, never an interval;
  * Stage 4b's per-vehicle, information-age and reuse metrics get the same intervals;
  * a reconciliation against the numbers the record already cites -- v2's values are the ones the paper uses.
Writes results/_ci_summary_v2.csv and results/_ci_summary_v2.json, which the figure tools read.
"""
import csv, json, math, pathlib, statistics as st

T975 = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571, 6: 2.447, 7: 2.365, 8: 2.306, 9: 2.262, 10: 2.228}  # by df
REQ, SPREAD_FLAG = 0.95, 0.05
METRICS = [("m0_pdr_mean", "M0 PDR"), ("m0_pdr_p05_intra", "WIRC"), ("m1_pdr_mean", "M1 PDR"),
           ("m0_collision_rate", "M0 coll"), ("m0_collision_rate_within_pool", "M0 wp"),
           ("m1_collision_rate_within_pool", "M1 wp")]
S4B = [("worst_vehicle_delivery", "worst vehicle", (0, 1)), ("per_vehicle_p05", "per-vehicle p05", (0, 1)),
       ("share_vehicles_meeting_095", "share meeting 0.95", (0, 1)), ("aoi_mean", "AoI mean", (1, None)),
       ("aoi_p95", "AoI p95", (1, None)), ("aoi_peak", "AoI peak", (1, None)),
       ("aoi_worst_vehicle_mean", "AoI worst vehicle", (1, None))]
REQ_METRICS = ("M0 PDR", "WIRC")

S = lambda pre, ss, suf="": [f"{pre}{s}{suf}" for s in ss]
G = "_gaefix_g05_ent01"
SPLIT_M0 = ("m0_collision_rate_within_pool", 0.05, "not separated", "separated")
# (figure, cell, labels, split, not_run)   not_run = labels deliberately never run (with the decision)
GROUPS = [
    ("7-B", "SB-SPS, N=4",                    S("SBSPS_rsrp80w40_N4_seed", range(1, 6)), None, {}),
    ("7-B", "Mode 0a, N=4, paper critic",     S("A_N4_seed", (1, 2, 4, 6, 10), "_5000ep" + G), None, {}),
    ("7-B", "Mode 0a, N=4, agent-specific",   S("A_N4_seed", (1, 2, 4), "_5000ep" + G + "_as"), None, {}),
    ("7-B", "Arm S, N=4",                     S("A_mode0c_seed", (1, 2, 3), "_5000ep" + G + "_armS"), None, {}),
    ("7-B", "Mode 0c, N=4",                   S("A_mode0c_seed", range(1, 6), "_5000ep" + G), None, {}),
    ("7-B", "SB-SPS, N=10",                   S("SBSPS_rsrp80w40_N10_seed", range(1, 6)), None, {}),
    ("7-B", "Mode 0a, N=10, paper critic",    S("A_N10_seed", (1, 2, 3), "_5000ep" + G), None, {}),
    ("7-B", "Mode 0a, N=10, agent-specific",  S("A_N10_seed", (1, 2, 3), "_5000ep" + G + "_as"), None, {}),
    ("7-B", "Arm P, N=10",                    S("A_mode0c_N10_seed", (1, 2, 3), G + "_armP"), None, {}),
    ("7-B", "Arm S, N=10",                    S("A_mode0c_N10_seed", (1, 2, 3), G + "_armS"), None, {}),
    ("7-B", "Mode 0c, N=10",                  S("A_mode0c_N10_seed", range(1, 6), G), None, {}),
    ("T-1", "Mode 0c, N=10, team critic",     S("A_mode0c_N10_seed", (1, 2, 3), G), None, {}),
    ("T-1", "Mode 0c, N=10, local critic",    S("A_mode0c_N10_seed", (1, 2, 3), G + "_lc"), None, {}),
    ("T-1", "Mode 0c, N=10, agent-specific",  S("A_mode0c_N10_seed", (1, 2, 3), G + "_as"), None, {}),
    ("T-1", "Mode 0c, N=4, team critic",      S("A_mode0c_seed", (1, 2, 3), "_5000ep" + G), None, {}),
    ("T-1", "Mode 0c, N=4, local critic",     S("A_mode0c_seed", (1, 2, 3), "_5000ep" + G + "_lc"), None, {}),
    ("T-1", "Mode 0c, N=4, agent-specific",   S("A_mode0c_seed", (1, 2, 3), "_5000ep" + G + "_as"), None, {}),
    ("T-1", "Mode 0a, N=4, team critic",      S("A_N4_seed", (1, 2, 4), "_5000ep" + G), None, {}),
    ("T-1", "Mode 0a, N=4, local critic",     S("A_N4_seed", (1, 2, 4), "_5000ep" + G + "_lc"), None, {}),
    ("T-1", "Mode 0a, N=4, agent-specific",   S("A_N4_seed", (1, 2, 4), "_5000ep" + G + "_as"), None, {}),
    ("T-1", "Mode 0c, N=10, delta=1.0",       S("A_mode0c_N10_seed", (1, 2, 3), G + "_d10"), None, {}),
    ("7-C", "M=7, shared pool",               S("A_mode0c_N10_M7_seed", (1, 2, 3), G), None, {}),
    ("7-C", "M=7, partition (pool 5)",        S("D_N10_M7_pool5_seed", (1, 2, 3), G), None, {}),
    ("7-C", "N=10, partition (pool 2)",       S("D_N10_seed", (1, 2, 3), G), None, {}),
    ("7-C", "N=10, partition (pool 4)",       S("D_N10_pool4_seed", (1, 2, 3), G), None, {}),
    ("7-C", "N=4, partition (pool 2)",        S("D_N4_seed", (1, 2, 3), G), None, {}),
    ("7-C", "Mode 0a + DS, N=10 (suppl.)",    S("C_N10_seed", (1, 2), G), None, {f"C_N10_seed3{G}": "D4"}),
    ("7-C", "Mode 0a + DS, N=4 (suppl.)",     S("C_N4_seed", (1, 2), G), None, {f"C_N4_seed3{G}": "D4"}),
    ("R1-b3", "Mode 0b, N=10",                S("A_mode0b_N10_seed", (1, 2, 3), G), None, {}),
    ("7-A", "N=4, m0=2",                      S("A_N4_seed", (1, 2, 4), "_5000ep" + G + "_as"), None, {}),
    ("7-A", "N=5, m0=2",                      S("A_N5_seed", (1, 2, 3), "_5000ep" + G + "_as"), None, {}),
    ("7-A", "N=6, m0=3",                      S("A_N6_seed", (1, 2, 3), "_5000ep" + G + "_as"), None, {}),
    ("7-A", "N=7, m0=3",                      S("A_N7_seed", (1, 2, 3), "_5000ep" + G + "_as"), SPLIT_M0, {}),
    ("7-A", "N=10, m0=5",                     S("A_N10_seed", (1, 2, 3), "_5000ep" + G + "_as"), None, {}),
    ("7-A", "N=15, m0=7",                     S("A_N15_seed", (1, 2, 3), "_5000ep" + G + "_as"), None, {}),
    ("7-A", "N=20, m0=10",                    S("A_N20_seed", (1, 2, 3), "_5000ep" + G + "_as"), None, {}),
    ("7-A", "N=4, M=3, m0=2",                 S("B_M3_seed", (1, 2, 3), "_5000ep" + G + "_as"), None, {}),
    ("7-A", "N=4, M=4, m0=2",                 S("B_M4_seed", (1, 2, 3), "_5000ep" + G + "_as"), None, {}),
]
S4B_GROUPS = [("Mode 0c, N=4", S("A_mode0c_seed", range(1, 6), "_5000ep" + G), False),
              ("SB-SPS, N=4", S("SBSPS_rsrp80w40_N4_seed", range(1, 6)), True),
              ("Mode 0c, N=10", S("A_mode0c_N10_seed", range(1, 6), G), True),
              ("SB-SPS, N=10", S("SBSPS_rsrp80w40_N10_seed", range(1, 6)), True),
              ("Mode 0b, N=10", S("A_mode0b_N10_seed", (1, 2, 3), G), True),
              ("M=7, shared pool", S("A_mode0c_N10_M7_seed", (1, 2, 3), G), True),
              ("M=7, partition (pool 5)", S("D_N10_M7_pool5_seed", (1, 2, 3), G), False)]   # reuse ratio meaningful?
# numbers the record and coverage review already cite: (cell, metric) -> (mean, low, high)
RECORD = {("Mode 0c, N=10", "M0 PDR"): (0.9405, 0.9281, 0.9529), ("Mode 0c, N=10", "WIRC"): (0.7921, 0.7815, 0.8028),
          ("Mode 0c, N=10", "M1 PDR"): (0.9628, 0.9568, 0.9689), ("SB-SPS, N=10", "M0 PDR"): (0.8967, 0.8900, 0.9034),
          ("SB-SPS, N=10", "WIRC"): (0.6736, 0.6578, 0.6894), ("SB-SPS, N=10", "M1 PDR"): (0.9437, 0.9418, 0.9455),
          ("SB-SPS, N=4", "M0 PDR"): (0.9811, 0.9800, 0.9823), ("SB-SPS, N=4", "WIRC"): (0.8573, 0.8465, 0.8681),
          ("Mode 0a, N=10, paper critic", "M0 PDR"): (0.6161, 0.6120, 0.6201), ("Mode 0c, N=4", "M0 PDR"): (0.9997, 0.9995, 0.9998),
          ("Mode 0a, N=4, paper critic", "M0 PDR"): (0.9999, 0.9998, 1.0000), ("M=7, shared pool", "M0 PDR"): (0.9732, 0.9502, 0.9963),
          ("N=10, partition (pool 4)", "M0 PDR"): (0.8785, 0.8772, 0.8799), ("Arm P, N=10", "M0 PDR"): (0.6149, 0.6126, 0.6172),
          ("Arm S, N=10", "M0 PDR"): (0.9375, 0.9097, 0.9652), ("Mode 0c, N=10, delta=1.0", "M0 PDR"): (0.9393, 0.9056, 0.9729),
          ("Mode 0b, N=10", "M0 PDR"): (0.9491, 0.9098, 0.9884)}

def load(label):
    f = pathlib.Path(f"results/_{label}.json")
    return json.loads(f.read_text())[-1] if f.exists() else None

def load4b(label):
    f = pathlib.Path(f"results/stage4b/_{label}.json")
    return json.loads(f.read_text()) if f.exists() else None

def interval(values, bounds=None):
    """n, mean, sample SD, low, high, clipped -- v1's method."""
    n = len(values); m = st.mean(values)
    if n < 2: return n, m, None, None, None, False
    s = st.stdev(values); half = T975.get(n - 1, 1.96) * s / math.sqrt(n)
    lo, hi, clipped = m - half, m + half, False
    if bounds:
        b_lo, b_hi = bounds
        if b_lo is not None and lo < b_lo: lo, clipped = b_lo, True
        if b_hi is not None and hi > b_hi: hi, clipped = b_hi, True
    return n, m, s, lo, hi, clipped

def requirement(n, m, lo, hi, vals):
    if n >= 3: return "MEETS" if lo >= REQ else "SHORT" if hi < REQ else "STRADDLES"
    return ("MEETS" if all(v >= REQ for v in vals) else "SHORT" if all(v < REQ for v in vals) else "MIXED") + " (by seed)"

def coord(labels):
    try:
        from rebuild_stats import timing
    except Exception:
        return None
    tim = [timing(l) for l in labels]
    if not any(t.get("log") for t in tim):
        return None                                   # untrained (SB-SPS) or no logs: coordination time does not apply
    out = {}
    for cls in ("m0", "m1"):
        ts = [t.get(f"{cls}_90_sust") for t in tim]
        reached = sorted(t for t in ts if t is not None)
        out[cls] = {"median": st.median(reached) if reached else None, "min": reached[0] if reached else None,
                    "max": reached[-1] if reached else None, "never": sum(t is None for t in ts), "n": len(ts)}
    return out

def fmt(n, m, lo, hi, clipped, vals):
    if n >= 3: return f"{m:.4f} [{lo:.4f}, {hi:.4f}]{'*' if clipped else ''}"
    if n == 2: return f"{m:.4f} (seeds {vals[0]:.4f}, {vals[1]:.4f})"
    return f"{m:.4f} (n=1)"

def summarise(fig, cell, recs, labels, rows, flag=True):
    if not recs:
        print(f"  {fig:<5} {cell:<36} (no results found)"); return
    out = []
    for key, short in METRICS:
        vals = [r[key] for r in recs if r.get(key) is not None]
        if not vals: continue
        n, m, s, lo, hi, cl = interval(vals, (0, 1))
        row = {"figure": fig, "cell": cell, "metric": short, "n": n, "mean": m, "sd_sample": s, "ci95_low": lo,
               "ci95_high": hi, "clipped": cl, "seeds": vals, "requirement_095": ""}
        if short in REQ_METRICS: row["requirement_095"] = requirement(n, m, lo, hi, vals)
        rows.append(row)
        out.append(f"{short} {fmt(n, m, lo, hi, cl, vals)}" + (f" {row['requirement_095']}" if row["requirement_095"] else ""))
    c = coord(labels)
    print(f"  {fig:<5} {cell:<36} n={len(recs)}")
    for i in range(0, len(out), 2): print("          " + "   ".join(out[i:i + 2]))
    if c:
        def ct(d): return (f"median {d['median']:g} [{d['min']:g}-{d['max']:g}]" if d["median"] is not None else "never") + (f", never in {d['never']} of {d['n']}" if d["never"] and d["median"] is not None else "")
        print(f"          coordination time (episodes to sustained 0.90): M0 {ct(c['m0'])};  M1 {ct(c['m1'])}")
        rows.append({"figure": fig, "cell": cell, "metric": "coordination time", "n": len(labels), "mean": None,
                     "sd_sample": None, "ci95_low": None, "ci95_high": None, "clipped": False, "seeds": c, "requirement_095": ""})
    if flag:
        for key in ("m0_pdr_mean", "m1_pdr_mean"):
            v = [r[key] for r in recs if r.get(key) is not None]
            if len(v) > 1 and max(v) - min(v) > SPREAD_FLAG:
                print(f"          ⚠ {key} spans {min(v):.4f}-{max(v):.4f}: a second mode or continuous spread? Check before quoting the mean")

def main():
    rows, missing, not_run = [], [], []
    print("Stage 4a v2 -- 95% t-intervals (sample SD) across seeds; * = clipped at a bound; n = 2 cells as seed values.")
    print("Requirement (safety class, PDR >= 0.95) judged on the interval: MEETS / SHORT / STRADDLES.\n")
    for fig, cell, labels, split, nr in GROUPS:
        got = [(l, load(l)) for l in labels]
        missing += [l for l, r in got if r is None]; not_run += [f"{l} ({why})" for l, why in nr.items()]
        recs = [r for _, r in got if r is not None]; labs = [l for l, r in got if r is not None]
        if split is None:
            summarise(fig, cell, recs, labs, rows); continue
        key, thr, above, below = split
        hi_g = [(l, r) for l, r in zip(labs, recs) if r.get(key, 0) > thr]
        lo_g = [(l, r) for l, r in zip(labs, recs) if r.get(key, 0) <= thr]
        print(f"  {fig:<5} {cell:<36} -- bimodal, split on {key} at {thr}: {len(lo_g)} {below}, {len(hi_g)} {above}")
        for name, grp in ((below, lo_g), (above, hi_g)):
            if grp: summarise(fig, f"{cell}: {name}", [r for _, r in grp], [l for l, _ in grp], rows, flag=False)
    print("\n  Stage 4b -- per vehicle, information age (generation intervals; >= 1) and spatial reuse")
    for cell, labels, reuse_ok in S4B_GROUPS:
        recs = [load4b(l) for l in labels]; recs = [r["own_window"] for r in recs if r]
        if not recs: print(f"  4b    {cell:<36} (no Stage 4b results found)"); continue
        out = []
        for key, short, bounds in S4B:
            vals = [r[key] for r in recs if r.get(key) is not None]
            if not vals: continue
            n, m, s, lo, hi, cl = interval(vals, bounds)
            rows.append({"figure": "4b", "cell": cell, "metric": short, "n": n, "mean": m, "sd_sample": s, "ci95_low": lo,
                         "ci95_high": hi, "clipped": cl, "seeds": vals, "requirement_095": ""})
            out.append(f"{short} {fmt(n, m, lo, hi, cl, vals)}")
        ru = [r["reuse"]["ratio"] for r in recs if r.get("reuse")]
        if ru and reuse_ok:
            n, m, s, lo, hi, cl = interval(ru)
            rows.append({"figure": "4b", "cell": cell, "metric": "reuse ratio", "n": n, "mean": m, "sd_sample": s,
                         "ci95_low": lo, "ci95_high": hi, "clipped": False, "seeds": ru, "requirement_095": ""})
            out.append(f"reuse ratio {fmt(n, m, lo, hi, False, ru)}")
        elif ru:
            out.append("reuse ratio n/a (F27: no sharing to measure, or the shuffle ignores a partition)")
        print(f"  4b    {cell:<36} n={len(recs)}")
        for i in range(0, len(out), 2): print("          " + "   ".join(out[i:i + 2]))
    print("\n  Reconciliation against the numbers the record cites (v2's values are the ones to use)")
    idx = {(r["cell"], r["metric"]): r for r in rows}
    worst = 0.0
    for (cell, metric), (m0, lo0, hi0) in RECORD.items():
        r = idx.get((cell, metric))
        if not r or r["ci95_low"] is None: print(f"          {cell} / {metric}: not computed"); continue
        d = max(abs(r["mean"] - m0), abs(r["ci95_low"] - lo0), abs(r["ci95_high"] - hi0)); worst = max(worst, d)
        tag = "same" if d < 0.00005 else "rounding" if d <= 0.00015 else "DIFFERS"
        print(f"          {cell + ' / ' + metric:<44} record {m0:.4f} [{lo0:.4f}, {hi0:.4f}]   v2 {r['mean']:.4f} [{r['ci95_low']:.4f}, {r['ci95_high']:.4f}]{'*' if r['clipped'] else ''}   {tag}")
    out = pathlib.Path("results")
    flat = [{**r, "seeds": json.dumps(r["seeds"])} for r in rows]
    with open(out / "_ci_summary_v2.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(flat[0].keys())); w.writeheader(); w.writerows(flat)
    (out / "_ci_summary_v2.json").write_text(json.dumps(rows, indent=2))
    nint = sum(1 for r in rows if r["ci95_low"] is not None)
    print(f"\n{len(rows)} rows ({nint} intervals) written to results/_ci_summary_v2.csv and .json; "
          f"{sum(r['clipped'] for r in rows)} clipped at a bound (*).")
    if not_run: print(f"Not run by decision: {', '.join(not_run)}")
    print(f"⚠ {len(missing)} expected result file(s) not found: {', '.join(missing)}" if missing else "All expected result files found.")

if __name__ == "__main__":
    main()
