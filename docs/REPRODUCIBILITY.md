# Reproducibility Documentation

This document records the reproducibility verification performed before
package release.

## Code-state continuity

The fifteen production runs in `results/psm001_results.json` were produced
under a single coherent code state. Specifically:

- **Phase A (rows 0-5)** used the v3.5 Path A.5 architecture (per-class
  actors + vehicle_id feature). All env, trainer, and reward fixes from the
  iteration journey (Fix A1-A4) were already in place.
- **Phase A Mode 0c (row 6)** used the Mode 0c trainer in
  `agents/mappo_mode0c.py`, which is structurally a sibling of `mappo.py`
  (per-vehicle actors instead of per-class shared). The env code path is
  identical.
- **Phase B (rows 7-9)** added no env or trainer code changes - CFG-only
  modifications (vary `n_subchannels`).
- **Phase C (rows 10-12)** added the `demand_separation` flag and
  within-pool collision diagnostics in `v2x_env.py`. With
  `demand_separation=False`, the env steps through exactly the pre-Phase-C
  code path. The within-pool diagnostic fields in `info` are pure
  observability - no effect on rewards, observations, or gradients.
- **Phase D (rows 13-14)** used `train_mode0c.py` and
  `agents/mappo_mode0c.py`, both siblings of the Mode 0a code. `train.py`
  and `agents/mappo.py` were not touched for Phase D.

## Reproducibility smoke test

After Phase D was complete, a smoke test re-ran the Phase A N=4
configuration under the current (Phase D-era) code, with the result file
pointed to a separate `_reproducibility_smoke.json` so as not to pollute the
canonical record. The deltas against the original row 0 of
`psm001_results.json`:

| Metric | Canonical | Smoke | Delta | Tolerance |
|---|---|---|---|---|
| M0 PDR mean | 0.7763 | 0.7743 | -0.0020 | +/- 0.02 |
| M0 collision rate | 0.4855 | 0.490 | +0.0045 | +/- 0.05 |
| M1 PDR mean | 0.890 | 0.8927 | +0.0027 | +/- 0.02 |
| M0 SINR mean (dB) | 19.7 | 19.66 | -0.04 | +/- 1.0 dB |

All deltas are roughly an order of magnitude inside their tolerances -
within single-seed evaluation noise for MAPPO at 3000 episodes.

A bonus confirmation came from the within-pool diagnostic that the smoke
test produced (the original row 0 predates this metric):
`M0_collision_within_pool = 0.202`. The analytical prediction for intra-M0
random collision with `m0_count=2` and `M=5` is `1 - (4/5)^1 = 0.20`.
Match to within 0.002 - independent validation that the within-pool
diagnostic is correctly implemented.

## Checkpoint sanity

After the smoke test, D_N4 and D_N10 were re-run with checkpoint saving
enabled (the patch added a `torch.save(...)` call between the JSON-save and
`env.close()` in `train_mode0c.py`). The reruns' eval scalars against the
canonical rows:

| Config | Metric | Canonical | Rerun | Delta |
|---|---|---|---|---|
| D_N4 | M0 PDR mean | 0.9993 | 0.9996 | +0.0003 |
| D_N4 | M0 collision rate | 0.0001 | 0.0000 | -0.0001 |
| D_N10 | M0 PDR mean | 0.7748 | 0.7772 | +0.0024 |
| D_N10 | M0 collision rate | 0.99999 | 1.0000 | +0.00001 |

The shipped checkpoints in `checkpoints/D_N4_actors.pt` and `D_N10_actors.pt`
correspond to these rerun policies. They reproduce the canonical D-row
numbers within single-seed noise.

## Tested environment

| Component | Version |
|---|---|
| OS | Ubuntu 22.04.5 LTS (WSL2) |
| Python | 3.10 |
| PyTorch | 2.6.0+cu124 |
| CUDA | 12.4 |
| GPU | NVIDIA RTX 3080 Laptop, 8 GB |
| SUMO | 1.26.0 |
| libzmq3-dev | 4.3.4-2 |
| g++ | 11.4.0 |

## Verifying reproducibility on your system

```bash
# Quick smoke (~2-3 minutes)
python scripts/quickstart_smoke.py

# Phase A N=4 reproduction (~35 min)
# (Edit train.py CFG to N=4, demand_separation=False, full 3000 episodes,
#  then redirect result_file to results/_my_repro.json)
python train.py

# Compare against canonical
python scripts/compare_results.py \
    --canonical results/psm001_results.json \
    --candidate results/_my_repro.json
```

Or skip training and verify the shipped Phase D checkpoints directly:

```bash
python scripts/eval_from_checkpoint.py --run D_N4
python scripts/eval_from_checkpoint.py --run D_N10
```
