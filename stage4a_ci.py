#!/usr/bin/env python3
"""
stage4a_ci.py -- 95% confidence intervals for every configuration Section 7 reports.

Reads results/_<label>.json (last record of each), and for each metric computes the
mean, the SAMPLE standard deviation (ddof = 1) and a two-sided 95% t-interval.

Rules it enforces:
  * bimodal configurations are SPLIT, never averaged (coverage review §8d);
  * any unsplit group whose M0 or M1 delivery spans more than 0.05 is flagged, so
    an unexpected second mode cannot be averaged silently;
  * n = 1 gets no interval; n = 2 intervals are wide by construction (t = 12.706).

Writes results/_ci_summary.csv and results/_ci_summary.json.

Note: the ± values in aa-achievement.md are POPULATION standard deviations
(ddof = 0). The paper should report the intervals produced here.
"""
import csv
import json
import math
import pathlib
import statistics as st

T975 = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571,
        6: 2.447, 7: 2.365, 8: 2.306, 9: 2.262, 10: 2.228}
METRICS = [("m0_pdr_mean", "M0 PDR"), ("m0_pdr_p05_intra", "M0 tail"),
           ("m1_pdr_mean", "M1 PDR"), ("m0_collision_rate", "M0 coll"),
           ("m0_collision_rate_within_pool", "M0 wp")]
SPREAD_FLAG = 0.05


def seeds(prefix, ss, suffix=""):
    return [f"{prefix}{s}{suffix}" for s in ss]


# (figure, name, labels, split)   split = None or (key, threshold, name_if_above, name_if_below)
GROUPS = [
    ("7-B", "Mode 0a, N=4, 5000 ep",      seeds("A_N4_seed", [2, 4, 6, 10], "_5000ep"), None),
    ("7-B", "Mode 0c, N=4, 3000 ep",      ["A_mode0c_check"] + seeds("A_mode0c_seed", [2, 3, 4, 5]), None),
    ("7-B", "SB-SPS, N=4, warm-up 40",    seeds("SBSPS_rsrp80w40_N4_seed", range(1, 6)), None),
    ("7-B", "Mode 0a, N=10, 5000 ep",     ["A_N10_5000ep"] + seeds("A_N10_seed", [2, 3], "_5000ep"), None),
    ("7-B", "Mode 0c, N=10, 5000 ep",     seeds("A_mode0c_N10_seed", range(1, 6)), None),
    ("7-B", "SB-SPS, N=10, warm-up 40",   seeds("SBSPS_rsrp80w40_N10_seed", range(1, 6)), None),
    ("7-A", "N=4, M=3",                   seeds("B_M3_seed", range(1, 6), "_5000ep"),
        ("m1_collision_rate_within_pool", 0.4, "arrangement A (M1 crowds)", "arrangement B (M1 separates)")),
    ("7-A", "N=4, M=4",                   seeds("B_M4_seed", range(1, 6), "_5000ep"), None),
    ("7-A", "N=5, 5000 ep",               seeds("A_N5_seed", range(1, 6), "_5000ep"),
        ("m0_collision_rate_within_pool", 0.05, "on the floor", "separated")),
    ("7-A", "N=5, 10000 ep",              seeds("A_N5_seed", [1, 4], "_10000ep"), None),
    ("7-A", "N=6, 5000 ep",               seeds("A_N6_seed", [1, 2, 3], "_5000ep"), None),
    ("7-A", "N=6, 10000 ep",              seeds("A_N6_seed", [1, 2], "_10000ep"), None),
    ("7-A", "N=7, 5000 ep",               ["A_N7_5000ep"] + seeds("A_N7_seed", [2, 3], "_5000ep"), None),
    ("7-A", "N=15, 5000 ep",              seeds("A_N15_seed", [1, 2], "_5000ep"), None),
    ("7-A", "N=20, 5000 ep",              seeds("A_N20_seed", [1, 2, 3], "_5000ep"), None),
    ("7-C", "Mode 0a + DS, N=4",          seeds("C_N4_seed", [1, 2, 3]), None),
    ("7-C", "Mode 0c + DS, N=4",          seeds("D_N4_seed", [1, 2, 3]), None),
    ("7-C", "Mode 0a + DS, N=10",         seeds("C_N10_seed", [1, 2, 3]), None),
    ("7-C", "Mode 0c + DS, N=10",         seeds("D_N10_seed", [1, 2, 3]), None),
    ("7-C", "Mode 0a + DS, N=4, 3000 ep", seeds("C_N4_3000ep_seed", [3, 4, 5, 6, 10]), None),
    ("T-2", "DS, delta=0.6, N=4",         seeds("C_N4_delta06_seed", [1, 2]),
        ("m1_collision_rate_within_pool", 0.1, "M1 on the floor", "M1 coordinated")),
    ("T-2", "DS, delta=1.0, N=4",         seeds("C_N4_delta10_seed", [1, 2]), None),
    ("T-2", "DS, delta=2.0, N=4",         ["C_N4_delta20_seed1"], None),
    ("T-2", "shared, delta=1.0, N=4",     seeds("A_N4_delta10_seed", [2, 4]), None),
    ("T-2", "shared, delta=1.0, N=10",    seeds("A_N10_delta10_seed", [1, 2, 3]), None),
]


def load(label):
    f = pathlib.Path(f"results/_{label}.json")
    return json.loads(f.read_text())[-1] if f.exists() else None


BOUNDED = {"M0 PDR", "M0 tail", "M1 PDR", "M0 coll", "M0 wp"}   # all lie in [0, 1]


def interval(values, bounded=False):
    """Returns n, mean, sample SD, low, high, clipped."""
    n = len(values)
    m = st.mean(values)
    if n < 2:
        return n, m, None, None, None, False
    s = st.stdev(values)                      # sample SD, ddof = 1
    half = T975.get(n - 1, 1.96) * s / math.sqrt(n)
    lo, hi, clipped = m - half, m + half, False
    if bounded and (lo < 0.0 or hi > 1.0):    # a t-interval can overshoot a physical bound
        lo, hi, clipped = max(lo, 0.0), min(hi, 1.0), True
    return n, m, s, lo, hi, clipped


def summarise(fig, name, recs, rows, flag=True):
    if not recs:
        print(f"  {fig:<4} {name:<38} (no results found)")
        return
    cells = []
    for key, short in METRICS:
        vals = [r[key] for r in recs if r.get(key) is not None]
        if not vals:
            continue
        n, m, s, lo, hi, clipped = interval(vals, bounded=short in BOUNDED)
        rows.append({"figure": fig, "config": name, "metric": short, "n": n, "mean": m,
                     "sd_sample": s, "ci95_low": lo, "ci95_high": hi, "clipped": clipped})
        cells.append(f"{short} {m:.4f}" + (f" [{lo:.4f}, {hi:.4f}]{'*' if clipped else ''}"
                                           if lo is not None else " (n=1)"))
    print(f"  {fig:<4} {name:<38} n={len(recs)}")
    for i in range(0, len(cells), 2):
        print("        " + "   ".join(cells[i:i + 2]))
    if flag:
        for key in ("m0_pdr_mean", "m1_pdr_mean"):
            v = [r[key] for r in recs if r.get(key) is not None]
            if len(v) > 1 and max(v) - min(v) > SPREAD_FLAG:
                print(f"        ⚠ {key} spans {min(v):.4f}-{max(v):.4f}: "
                      f"check whether this is a second mode or continuous spread before reporting a mean")


def main():
    rows, missing = [], []
    print("95% confidence intervals (t-distribution, sample SD). Brackets are [low, high].\n")
    for fig, name, labels, split in GROUPS:
        got = [(lab, load(lab)) for lab in labels]
        missing += [lab for lab, r in got if r is None]
        recs = [r for _, r in got if r is not None]
        if split is None:
            summarise(fig, name, recs, rows)
            continue
        key, thr, above, below = split
        hi_grp = [r for r in recs if r.get(key, 0) > thr]
        lo_grp = [r for r in recs if r.get(key, 0) <= thr]
        print(f"  {fig:<4} {name:<38} — bimodal, split on {key} at {thr}")
        summarise(fig, f"{name}: {above}", hi_grp, rows, flag=False)
        summarise(fig, f"{name}: {below}", lo_grp, rows, flag=False)

    # spatial-reuse ratios, where measured
    print("\n  Spatial-reuse ratio (co-channel distance against shuffled)")
    for name, vals in [
        ("Mode 0c, N=10", [json.loads(pathlib.Path(f"results/_reuse_seed{s}.json").read_text()).get("ratio")
                           for s in range(1, 6) if pathlib.Path(f"results/_reuse_seed{s}.json").exists()]),
        ("SB-SPS, N=10", [((load(f"SBSPS_rsrp80w40_N10_seed{s}") or {}).get("sbsps") or {}).get("reuse_ratio")
                          for s in range(1, 6)]),
    ]:
        vals = [v for v in vals if v is not None]
        if vals:
            n, m, s, lo, hi, _ = interval(vals)
            rows.append({"figure": "7-B", "config": name, "metric": "reuse ratio", "n": n, "mean": m,
                         "sd_sample": s, "ci95_low": lo, "ci95_high": hi, "clipped": False})
            print(f"        {name:<16} {m:.4f}" + (f" [{lo:.4f}, {hi:.4f}]  n={n}" if lo is not None else ""))

    out = pathlib.Path("results")
    with open(out / "_ci_summary.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    (out / "_ci_summary.json").write_text(json.dumps(rows, indent=2))
    nclip = sum(1 for r in rows if r["clipped"])
    print(f"\n{len(rows)} intervals written to results/_ci_summary.csv and results/_ci_summary.json")
    if nclip:
        print(f"* {nclip} interval(s) clipped at 0 or 1. The t-interval fits poorly this close to a physical "
              f"bound; report these as clipped, or give the range of the seeds instead")
    if missing:
        print(f"⚠ {len(missing)} expected result file(s) not found: {', '.join(missing)}")
    else:
        print("All expected result files found.")


if __name__ == "__main__":
    main()
