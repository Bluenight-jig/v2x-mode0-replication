"""
rebuild_stats.py -- analysis helpers for the rebuild batches (aa-achievement.md §7c).

Usage (from ~/v2x_clean):
    python3 rebuild_stats.py tier1     headline cells, with time to coordinate
    python3 rebuild_stats.py b23       batch 23: Mode 0c N=4 at 5000 ep, demand separation

Time to coordinate is read from the training curves (logged every 50 episodes,
exploring policy, so values sit a little below evaluation). "first" is the first
logged episode at or above the threshold; "sust" is the first from which three
consecutive logged points stay at or above it (held for about 100 episodes).
Regimes are STRUCTURAL only (within-pool rates), never the outcome being measured;
a delivery far from its cell's median is flagged for inspection, not excluded.
"""
import json, pathlib, re, statistics, sys

T975 = {2: 12.706, 3: 4.303, 4: 3.182, 5: 2.776, 6: 2.571}

def res(label):
    f = pathlib.Path(f"results/_{label}.json")
    return json.loads(f.read_text())[-1] if f.exists() else None

def tail(r):
    for k in ("m0_pdr_p05_intra", "m0_pdr_p05", "m0_tail"):
        if k in r:
            return r[k]
    return None

def ci(xs):
    n = len(xs); m = statistics.mean(xs)
    if n == 1:
        return f"{m:.4f} (n=1)"
    if n == 2:
        return f"{m:.4f} (seeds {xs[0]:.4f}, {xs[1]:.4f})"
    h = T975[n] * statistics.stdev(xs) / n ** 0.5
    return f"{m:.4f} [{max(0.0, m - h):.4f}, {min(1.0, m + h):.4f}]"

def curve(label):
    p = pathlib.Path(f"logs/{label}.log")
    if not p.exists():
        return None, False
    text = p.read_text(); pts = []
    for ln in text.splitlines():
        m = re.search(r'^Ep\s+(\d+).*?M0_PDR\s+([\d.]+).*?M1_PDR\s+([\d.]+)', ln)
        if m:
            pts.append((int(m.group(1)), float(m.group(2)), float(m.group(3))))
    return pts, "Run complete." in text[-600:]

def reach(pts, k, thr, n):
    for i in range(len(pts) - n + 1):
        if all(pts[j][k] >= thr for j in range(i, i + n)):
            return pts[i][0]
    return None

def timing(label):
    pts, done = curve(label)
    out = {"done": done, "log": pts is not None}
    for k, cls in ((1, "m0"), (2, "m1")):
        for thr in (0.90, 0.95):
            for n, tag in ((1, "first"), (3, "sust")):
                out[f"{cls}_{int(thr * 100)}_{tag}"] = reach(pts, k, thr, n) if pts else None
    return out

def fmt(v):
    return "never" if v is None else str(v)

def report(name, labels, regime=None, ref=None):
    print(f"\n=== {name} ===")
    print(f"    {'run':<44}{'M0':>7}{'tail':>7}{'M1':>7}{'M0wp':>6}{'M1wp':>6}   "
          f"{'M0 .90/.95 sust':>16}   {'M1 .90/.95 sust':>16}   status")
    rows = []
    for l in labels:
        r = res(l)
        if not r:
            print(f"    {l:<44} (no result)"); continue
        t = timing(l); tl = tail(r)
        ok = True if regime is None else regime(r)
        rows.append((l, r, t, ok))
        tl_s = "   -  " if tl is None else f"{tl:7.4f}"
        m0t = f"{fmt(t['m0_90_sust'])}/{fmt(t['m0_95_sust'])}"
        m1t = f"{fmt(t['m1_90_sust'])}/{fmt(t['m1_95_sust'])}"
        status = ("in regime" if ok else "*** STRUCTURE DIFFERS - split ***") + ("" if t["done"] else "  [no 'Run complete.']")
        print(f"    {l:<44}{r['m0_pdr_mean']:7.4f}{tl_s}{r['m1_pdr_mean']:7.4f}"
              f"{r['m0_collision_rate_within_pool']:6.3f}{r['m1_collision_rate_within_pool']:6.3f}   "
              f"{m0t:>16}   {m1t:>16}   {status}")
    keep = [x for x in rows if x[3]]
    if keep:
        med = statistics.median(x[1]['m0_pdr_mean'] for x in keep)
        for l, r, _, _ in keep:
            if abs(r['m0_pdr_mean'] - med) > 0.05:
                print(f"    note: {l} M0 PDR is {r['m0_pdr_mean'] - med:+.3f} from the cell median -> inspect structure; not excluded")
        m0 = [x[1]['m0_pdr_mean'] for x in keep]; m1 = [x[1]['m1_pdr_mean'] for x in keep]
        ts = [tail(x[1]) for x in keep if tail(x[1]) is not None]
        print(f"  pooled n={len(keep)}:  M0 PDR {ci(m0)}   M1 PDR {ci(m1)}" + (f"   M0 tail {ci(ts)}" if ts else ""))
        if ref:
            print(f"  reference: {ref}")
    return keep

def median_time(labels, key):
    vals = [timing(l)[key] for l in labels]
    got = [v for v in vals if v is not None]
    return (statistics.median(got) if got else None), len(vals) - len(got)

WP0 = lambda r: r['m0_collision_rate_within_pool']
WP1 = lambda r: r['m1_collision_rate_within_pool']
CELL = {
    "0c_N4_3000":  [f"A_mode0c_seed{s}_gaefix_g05_ent01" for s in (1, 2, 3, 4, 5)],
    "0c_N4_5000":  [f"A_mode0c_seed{s}_5000ep_gaefix_g05_ent01" for s in (1, 2, 3, 4, 5)],
    "0a_N4":       [f"A_N4_seed{s}_5000ep_gaefix_g05_ent01" for s in (1, 2, 4, 6, 10)],
    "0c_N10":      [f"A_mode0c_N10_seed{s}_gaefix_g05_ent01" for s in (1, 2, 3, 4, 5)],
    "0a_N10":      [f"A_N10_seed{s}_5000ep_gaefix_g05_ent01" for s in (1, 2, 3)],
    "0cDS_N10":    [f"D_N10_seed{s}_gaefix_g05_ent01" for s in (1, 2, 3)],
    "0aDS_N10":    [f"C_N10_seed{s}_gaefix_g05_ent01" for s in (1, 2, 3)],
    "0aDS_N4":     [f"C_N4_seed{s}_gaefix_g05_ent01" for s in (1, 2, 3)],
    "0cDS_N4":     [f"D_N4_seed{s}_gaefix_g05_ent01" for s in (1, 2, 3)],
}

def tier1():
    report("Mode 0c, N=4, 3000 ep (supplementary)", CELL["0c_N4_3000"], lambda r: WP0(r) <= 0.01 and WP1(r) <= 0.01)
    report("Mode 0a, N=4, 5000 ep", CELL["0a_N4"], lambda r: WP0(r) < 0.05)
    report("Mode 0c, N=10, 5000 ep", CELL["0c_N10"], lambda r: WP0(r) <= 0.01)
    report("Mode 0a, N=10, 5000 ep", CELL["0a_N10"], lambda r: abs(WP0(r) - 0.590) <= 0.02)

def b23():
    report("Mode 0c, N=4, 5000 ep (standardised)", CELL["0c_N4_5000"], lambda r: WP0(r) <= 0.01 and WP1(r) <= 0.01,
           ref="3000-ep cell M0 0.9997, M1 0.9970 (P29: within 0.003)")
    report("Mode 0c + DS, N=10", CELL["0cDS_N10"], None, ref="shared pool M0 0.9405, M1 0.9628 (P30: DS far below)")
    report("Mode 0a + DS, N=10", CELL["0aDS_N10"], None, ref="shared pool M0 0.6161, M1 0.6879 (P30: DS far below)")
    report("Mode 0a + DS, N=4", CELL["0aDS_N4"], lambda r: WP0(r) < 0.05, ref="shared pool M0 0.9999, M1 0.7977")
    sh, sh_never = median_time(CELL["0a_N4"], "m0_90_sust")
    ds, ds_never = median_time([l for l in CELL["0aDS_N4"] if res(l)], "m0_90_sust")
    print("\n=== P31: demand separation's convergence advantage at N=4, Mode 0a (sustained M0 >= 0.90) ===")
    print(f"    shared pool median {fmt(sh)} (never: {sh_never})   DS median {fmt(ds)} (never: {ds_never})")
    if sh and ds:
        print(f"    shared / DS = {sh / ds:.2f}x   (F12 claimed ~4x; P31 predicts under 2x)")

if __name__ == "__main__":
    {"tier1": tier1, "b23": b23}.get(sys.argv[1] if len(sys.argv) > 1 else "", lambda: print(__doc__))()
