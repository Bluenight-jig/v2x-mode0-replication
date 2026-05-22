#!/usr/bin/env python3
"""
compare_results.py - Compare a freshly-generated results JSON against the
canonical psm001_results.json shipped with the package.

Usage:
    python scripts/compare_results.py [--canonical results/psm001_results.json]
                                       [--candidate results/_my_rerun.json]
                                       [--tolerance 0.02]

Reports per-row deltas across all matching (config, N, M) tuples. A row
"PASSES" if every numeric metric is within tolerance.
"""
import argparse
import json
from pathlib import Path

NUMERIC_FIELDS = [
    "m0_pdr_mean",
    "m1_pdr_mean",
    "m0_pdr_p05",
    "m0_pdr_p05_intra",
    "m0_collision_rate",
    "m0_collision_rate_within_pool",
    "m1_collision_rate_within_pool",
    "m0_sinr_mean",
    "m1_sinr_mean",
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--canonical", default="results/psm001_results.json")
    ap.add_argument("--candidate", default="results/psm001_results.json")
    ap.add_argument("--tolerance", type=float, default=0.02,
                    help="absolute tolerance for PDR/collision; SINR uses 10x this")
    args = ap.parse_args()

    canonical = json.loads(Path(args.canonical).read_text())
    candidate = json.loads(Path(args.candidate).read_text())

    canonical_idx = {(r["config"], r["N"], r["M"]): r for r in canonical}
    candidate_idx = {(r["config"], r["N"], r["M"]): r for r in candidate}

    all_keys = sorted(set(canonical_idx) | set(candidate_idx))
    failed = []

    for key in all_keys:
        config, N, M = key
        c_row = canonical_idx.get(key)
        r_row = candidate_idx.get(key)
        if c_row is None:
            print(f"[{config}, N={N}, M={M}]  NOT IN CANONICAL")
            continue
        if r_row is None:
            print(f"[{config}, N={N}, M={M}]  NOT IN CANDIDATE")
            continue

        row_ok = True
        deltas = []
        for f in NUMERIC_FIELDS:
            cv = c_row.get(f)
            rv = r_row.get(f)
            if cv is None or rv is None:
                continue
            tol = args.tolerance * (10.0 if "sinr" in f else 1.0)
            d = rv - cv
            status = "OK" if abs(d) <= tol else "FAIL"
            if status == "FAIL":
                row_ok = False
            deltas.append((f, cv, rv, d, status))

        overall = "PASS" if row_ok else "FAIL"
        if not row_ok:
            failed.append(key)
        print(f"[{config:10s} N={N:2d} M={M:2d}]  {overall}")
        for f, cv, rv, d, st in deltas:
            mark = " " if st == "OK" else "!"
            print(f"  {mark} {f:35s}  canonical={cv:9.4f}  candidate={rv:9.4f}  delta={d:+9.4f}  [{st}]")

    print()
    print(f"Summary: {len(all_keys) - len(failed)}/{len(all_keys)} rows pass")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
