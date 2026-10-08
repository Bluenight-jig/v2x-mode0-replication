# Mode 0 V2X — Reproducibility Package (v2.0.1)

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.20338167.svg)](https://doi.org/10.5281/zenodo.20338167)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

Code, simulation environment, results and figures for:

> **Mode 0: Architecture, Risk Taxonomy, and Standardization Pathway for RCU-Assisted V2X Safety Communication**
> Dewei Jiang
> Preprint, 2026. Manuscript under review.

**v2.0.1 accompanies the revised manuscript** (v2.0.0 had the same code and results; v2.0.1 corrects the run register's labels, this README and one figure label). Releases v1.0, v1.0.1 and v1.1 accompanied the original
submission; they remain tagged for traceability, but **their results are superseded** (see *Corrections*).

## Corrections since v1.x

Two implementation defects were found during the revision. Every learning-based result has been regenerated.

1. **Incomplete fleet at episode start.** The environment's reset used a fixed warm-up of 60 steps; at the configured
   traffic flow not every vehicle had entered the road by then, so episodes could start with fewer vehicles than
   configured. The warm-up now runs until all N vehicles are present. This affected the results of the original
   submission.
2. **Advantages computed across vehicles.** The original trainers (`agents/mappo.py`, `agents/mappo_mode0c.py`)
   store each step's transitions interleaved by vehicle and computed GAE over that buffer as one time sequence, so a
   vehicle's advantages mixed in other vehicles' rewards and values. The fix is `agents/mappo_gaefix.py`: per-agent
   GAE, in subclasses that override only the advantage computation. The original files are deliberately unchanged,
   so that every earlier run stays reproducible from the code that produced it.

All 200 training runs are listed in `RUN_REGISTER.csv`, whose `role` column gives each run's place in the revised
manuscript: **88 reported** (every result in Section 7, all at the final settings and 5,000 episodes); 11 supplementary
(the same settings, not cited: five 3,000-episode runs of Mode 0c at N = 4, two extra seeds of Mode 0a with the
team-value critic at N = 4, and four runs of Mode 0a with a safety partition); 15 diagnostic (per-agent GAE at other
settings, used to choose the final ones); 85 superseded (defect 2); and 1 excluded (defect 1).

## What is in this release

| Path | Contents |
|---|---|
| `agents/`, `envs/` | Trainers and environments. Rebuild: `mappo_gaefix.py` (per-agent GAE), `mappo_critic_ablation.py` (local and agent-specific critics), `mappo_mode0b.py` (hybrid fleet), `envs/v2x_env_arms.py` (capability ablations) |
| `train_*.py` | The exact runner behind every run (see `RUN_REGISTER.csv`); `d*_generator.py` derived each rebuild batch's runners from earlier ones, changing only stated lines |
| `results/_*.json` | Every run's evaluation (100 episodes), and the SB-SPS baseline evaluations |
| `results/_ci_summary_v2.{csv,json}` | **Every interval the paper reports** (Stage 4a v2: two-sided 95% t-intervals across seeds) |
| `results/stage4b/` | Per-vehicle evaluation: information age, per-vehicle delivery, spatial reuse, Mode 0b by vehicle type |
| `logs/` | Training logs (coordination times are computed from them) |
| `checkpoints/` | Trained actors for every policy that Stage 4b re-evaluates |
| `figures/`, `tables/` | The revised manuscript's Section 7 figures (PDF, PNG) and tables (LaTeX) |

Not included: Stage 4b's per-interval arrays (about 190 MB), which `stage4b_eval.py` regenerates from the included checkpoints.

## Reproducing the paper's numbers and figures

```bash
python3 stage4a_v2.py      # every interval  -> results/_ci_summary_v2.{csv,json}   (seconds)
python3 make_figures.py    # every figure and table -> figures/, tables/              (seconds)
```

`REPRODUCE.md` maps each figure and table to its inputs, and explains how to re-evaluate a checkpoint or retrain any run.

## Method in brief

Standard MAPPO with per-agent GAE (λ = 0.95), discount γ = 0.5, entropy coefficient annealed 0.01 → 0.001, 5,000
training episodes, then 100 evaluation episodes. Actors: two hidden layers of 128 units; critics: two hidden layers of
256 units with LayerNorm after the first. Five seeds on the headline cells, three elsewhere; intervals are two-sided
95% t-intervals on the sample standard deviation; comparisons use a lexicographic rule (safety delivery first) fixed
before each batch.

## Environment used

| Component | Version |
|---|---|
| OS | Linux-6.18.33.2-microsoft-standard-WSL2-x86_64-with-glibc2.35 |
| Python | 3.10.21 |
| torch | 2.14.0+cu130 |
| numpy | 2.2.6 |
| matplotlib | 3.10.9 |
| zmq | 27.2.0 |
| GPU | NVIDIA GeForce RTX 3060 |
| SUMO | Eclipse SUMO sumo 1.27.1 |

Hardware: AMD Ryzen 5 5600X, NVIDIA RTX 3060 (12 GB), Ubuntu 22.04 under WSL2; five runs in parallel. Wall time per
5,000-episode run with five in parallel: about 13–14.5 h (Mode 0c, N = 10), 8.3 h (Mode 0a, N = 10), 7 h (Mode 0c, N = 4).

## Citation

Please cite the paper (manuscript under review) and this software through its concept DOI,
[10.5281/zenodo.20338167](https://doi.org/10.5281/zenodo.20338167), which always resolves to the latest version. See `CITATION.cff`.

## License

MIT — see `LICENSE`.
