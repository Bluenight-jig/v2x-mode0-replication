# Reproducing the revised manuscript's results

## Figures and tables (seconds, no simulation)

| Paper item (internal label) | Command | Reads |
|---|---|---|
| Every interval and requirement verdict | `python3 stage4a_v2.py` | `results/_*.json`, `logs/*.log`, `results/stage4b/_*.json` |
| Fig. 7-A — separation against class-group size | `python3 make_figures.py` | `results/_ci_summary_v2.json` |
| Fig. 7-B — the capability ladder | `python3 make_figures.py` | `results/_ci_summary_v2.json` |
| Fig. 7-C — the safety-priority partition | `python3 make_figures.py` | `results/_ci_summary_v2.json` |
| Fig. 7-D — information age | `python3 make_figures.py` | `results/_ci_summary_v2.json` |
| Table: critic-input ablation and reward weight | `python3 make_figures.py` | `results/_ci_summary_v2.json` |
| Table: Mode 0b by vehicle type | `python3 make_figures.py` | `results/stage4b/_A_mode0b_*.json` |
| Table: requirement verdicts | `python3 make_figures.py` | `results/_ci_summary_v2.json` |

The internal labels map to the manuscript's figure and table numbers, which are set at typesetting.

## Per-vehicle evaluation (minutes per policy; needs SUMO and the ns-3 bridge)

`python3 stage4b_eval.py eval --label <label> --port <free port>` re-evaluates a saved policy (or SB-SPS) for 100
episodes, keeping every vehicle's delivery, subchannel and position at every interval, and checks the result against the
run's own logged evaluation. `python3 stage4b_eval.py report` summarises all evaluations.

## Retraining any run (hours per run)

Each run's runner is listed in `RUN_REGISTER.csv`: `OMP_NUM_THREADS=1 python -u <runner>`. It writes
`results/_<label>.json` and, for per-vehicle architectures, `checkpoints/<label>_actors.pt`. Build the ns-3 bridge
from `ns3_bridge/` first. The 88 runs whose `role` is `reported` are the ones behind every result in the revised manuscript.
