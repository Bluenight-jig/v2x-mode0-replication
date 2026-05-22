# Mode 0 V2X — Reproducibility Package

[![DOI](https://zenodo.org/badge/DOI/[Zenodo DOI].svg)](https://doi.org/[Zenodo DOI])
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

Code, simulation environment, and empirical results for:

> **Mode 0: A New 3GPP V2X Resource Allocation Category for Roadside Computing Unit-Assisted Safety Communication**
> Dewei Jiang
> *IEEE Transactions on Intelligent Transportation Systems*, 2026

This repository enables full reproduction of the fifteen-run MAPPO simulation
programme reported in Section 6 of the paper, spanning Phase A (Mode 0a
baseline + Mode 0c architectural comparison), Phase B (supply-axis sweep),
Phase C (Mode 0a + demand separation), and Phase D (Mode 0c + demand
separation).

## Hardware requirements

- CUDA-capable NVIDIA GPU (developed on RTX 3080 Laptop, 8 GB VRAM)
- ~10 GB free disk space (Python env + SUMO + TensorBoard logs)
- Per-run wall time: ~30-40 min for a 3000-episode training run

Full programme reproduction (all 15 runs sequentially): ~8 hours of GPU time.

## Installation

### System requirements

- Ubuntu 22.04 LTS (or WSL2 equivalent on Windows)
- Python 3.10 or 3.11
- CUDA 12.4 (other versions require alternate PyTorch wheels)

### Step 1 - System packages

```bash
sudo apt update
sudo apt install sumo sumo-tools sumo-doc \
                 libzmq3-dev libjsoncpp-dev \
                 build-essential
```

Installs SUMO 1.26+, ZeroMQ C++ libraries, JsonCpp, and g++ 11+.

### Step 2 - Python environment

```bash
conda create -n v2x python=3.10
conda activate v2x
pip install -r requirements.txt \
    --extra-index-url https://download.pytorch.org/whl/cu124
```

For other CUDA versions, see <https://pytorch.org/get-started/locally/>.

### Step 3 - Compile the channel-simulation bridge

```bash
cd ns3_bridge
make
cd ..
```

This compiles `v2x_bridge` against `libzmq` and `libjsoncpp`.

### Step 4 - Sanity check

```bash
python -c "
from envs.v2x_env import V2XEnv
from envs.ns3_bridge import NS3Bridge
from agents.mappo import MAPPOTrainer
from agents.mappo_mode0c import MAPPOTrainerMode0c
print('all imports OK')
"
```

### Tested configuration

The simulation programme was executed on Ubuntu 22.04.5 LTS, SUMO 1.26.0,
libzmq3-dev 4.3.4-2, g++ 11.4.0, PyTorch 2.6.0+cu124. A reproducibility
smoke test (see `docs/REPRODUCIBILITY.md`) confirmed equivalent Phase A N=4
results under this exact stack, to within single-seed sampling noise.

## Quickstart

Run a 30-episode smoke test to verify the install works end-to-end:

```bash
python scripts/quickstart_smoke.py
```

Expected output: an Eval Summary block printing M0_PDR around 0.7-0.8 and
M0_collision_rate around 0.4-0.5 (noisy partial-training values; the full
3000-episode canonical numbers in `results/psm001_results.json` are tighter).

## Reproducibility recipe

| Phase | Description                                              | Command                            | Wall time | JSON rows |
|-------|----------------------------------------------------------|------------------------------------|-----------|-----------|
| A     | Mode 0a baseline sweep, N in {2,3,4,5,7,10}, shared pool | `bash scripts/run_phase_A.sh`      | ~3 h      | 0-5       |
| A-0c  | Mode 0c architectural comparison at N=4                  | `bash scripts/run_phase_A_mode0c.sh` | ~30 min | 6         |
| B     | Supply-axis sweep, M in {3,7,10} at N=4                  | `bash scripts/run_phase_B.sh`      | ~1.5 h    | 7-9       |
| C     | Mode 0a + demand separation, N in {4,7,10}, M_m0=2       | `bash scripts/run_phase_C.sh`      | ~1.5 h    | 10-12     |
| D     | Mode 0c + demand separation, N in {4,10}, M_m0=2         | `bash scripts/run_phase_D.sh`      | ~1 h      | 13-14     |

After running, compare against the canonical record:

```bash
python scripts/compare_results.py
```

For Phase D, pre-trained policy checkpoints are included so you can skip
training and run only evaluation:

```bash
python scripts/eval_from_checkpoint.py --run D_N4
python scripts/eval_from_checkpoint.py --run D_N10
```

## Repository layout

```
v2x-mode0-replication/
+-- README.md
+-- LICENSE                    MIT
+-- CITATION.cff
+-- requirements.txt
+-- envs/                      Gymnasium environment + ZMQ bridge client
+-- agents/                    MAPPO trainer (Mode 0a) + Mode 0c trainer
+-- ns3_bridge/                Channel-simulation bridge (C++ source + Makefile)
+-- scenarios/highway/         SUMO highway scenario files
+-- scripts/                   Phase run scripts + comparison utilities
+-- train.py                   Mode 0a runner
+-- train_mode0c.py            Mode 0c runner
+-- results/
|   +-- psm001_results.json    Canonical 15-entry eval ledger
+-- checkpoints/               Pre-trained D_N4 and D_N10 policies
+-- figures/                   Matplotlib source + PDF/PNG outputs
+-- docs/
    +-- SCHEMA.md              JSON field reference
    +-- ARCHITECTURE.md        Brief architecture notes
    +-- REPRODUCIBILITY.md     Smoke-test recipe + interpretation
    +-- KNOWN_GOTCHAS.md       Development pitfalls
```

## JSON schema (psm001_results.json)

15 entries, one per simulation run, with a uniform 26-field schema.
See `docs/SCHEMA.md` for full field documentation. In brief:

- **Configuration**: `config`, `N`, `M`, `M_m0`, `m0_ratio`, `rho`, `architecture`
- **Performance metrics**: `m0_pdr_mean`, `m1_pdr_mean`, `m0_pdr_p05`,
  `m0_pdr_p05_intra`, `m0_collision_rate`, `m0_collision_rate_within_pool`,
  `m1_collision_rate_within_pool`, `m0_sinr_mean`, `m1_sinr_mean`
- **Training diagnostics**: `entropy_m0`, `entropy_m1`, `critic_loss`,
  `actor_loss`, `actor_loss_m0`, `actor_loss_m1`,
  `train_m0_pdr_last100`, `train_m1_pdr_last100`,
  `n_episodes`, `n_eval_episodes`

Note: `m0_collision_rate_within_pool` and `m1_collision_rate_within_pool` are
`null` for Phase A/B entries (rows 0-9) because these diagnostics were added
during Phase C development. For shared-pool runs, the within-pool rate can be
computed analytically as `1 - ((M-1)/M)^(m0_count-1)`.

## Citation

If you use this code or data, please cite the paper:

```bibtex
@article{Jiang2026mode0,
  title   = {Mode 0: A New 3GPP V2X Resource Allocation Category for Roadside Computing Unit-Assisted Safety Communication},
  author  = {Jiang, Dewei},
  journal = {IEEE Transactions on Intelligent Transportation Systems},
  year    = {2026},
  doi     = {[DOI]}
}
```

and the Zenodo archive of this repository:

```bibtex
@software{Jiang2026mode0_repo,
  title     = {Mode 0 V2X --- Reproducibility Package},
  author    = {Jiang, Dewei},
  year      = {2026},
  publisher = {Zenodo},
  doi       = {[Zenodo DOI]}
}
```

## License

MIT - see `LICENSE` for the full text.

## Known gotchas

Several pitfalls are worth noting for anyone modifying the code. See
`docs/KNOWN_GOTCHAS.md` for the complete list. Highlights:

- **sed substring matching**: when changing numeric configs via sed, always
  anchor the trailing comma with `[0-9]\+,`. Otherwise `s/10/100/` matches
  the `10` inside `100` and corrupts the file.
- **NS-3 ZMQ linking**: linking ZMQ inside NS-3's scratch directories fails
  reliably. The bridge is intentionally a standalone C++ binary compiled with
  `g++` directly, no NS-3 dependency at runtime.
- **Chat-client autolinking**: some chat interfaces autolink `.py` filenames
  during paste. For patch scripts that touch source files, construct paths via
  `"v2x_env" + chr(46) + "py"` or `os.path.join` to defeat autolinking.
