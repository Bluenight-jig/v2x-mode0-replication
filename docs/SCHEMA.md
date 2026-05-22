# JSON Schema Reference

`results/psm001_results.json` is a JSON array of 15 entries, one per
simulation run. All entries share a uniform 26-field schema.

## Configuration fields

| Field | Type | Description |
|---|---|---|
| `config` | str | Run identifier. `"A"` for Phase A entries (distinguished by `N`); `"A_mode0c"`, `"B_M{M}"`, `"C_N{N}"`, `"D_N{N}"` for other phases. |
| `N` | int | Vehicle count. |
| `M` | int | Total subchannel count. |
| `M_m0` | int or null | M0-dedicated subchannel count. `null` when demand separation is off (Phase A, A_mode0c, B). |
| `m0_ratio` | float | Fraction of M0-class vehicles. Fixed at 0.5 throughout. |
| `rho` | float | Demand-supply ratio `N/M`, rounded to 2 decimal places. |
| `architecture` | str | `"path_A5_per_class_actors_with_id"` (Mode 0a, used for Phase A/B/C); `"mode_0c_per_vehicle_actors"` (Mode 0c, used for A_mode0c and Phase D). |

## Performance metrics

| Field | Type | Description |
|---|---|---|
| `m0_pdr_mean` | float | Mean M0 packet-delivery ratio over eval episodes. |
| `m1_pdr_mean` | float | Mean M1 packet-delivery ratio over eval episodes. |
| `m0_pdr_p05` | float | 5th-percentile M0 PDR across episodes (between-episode tail). |
| `m0_pdr_p05_intra` | float | 5th-percentile M0 PDR within episodes (worst-TTI tail; primary latency-safety indicator). |
| `m0_collision_rate` | float | Probability an M0 vehicle shares its subchannel with at least one other vehicle (intra-class M0-M0 + cross-class M0-M1). |
| `m0_collision_rate_within_pool` | float or null | Intra-class M0-M0 collision rate only. `null` for Phase A/B entries (rows 0-9) because the diagnostic was added during Phase C development. For shared-pool runs, the within-pool rate can be computed analytically as `1 - ((M-1)/M)^(m0_count-1)`. |
| `m1_collision_rate_within_pool` | float or null | Intra-class M1-M1 collision rate. Same null pattern as above. |
| `m0_sinr_mean` | float | Mean M0 SINR in dB. |
| `m1_sinr_mean` | float | Mean M1 SINR in dB. |

## Training diagnostics

| Field | Type | Description |
|---|---|---|
| `entropy_m0` | float | Terminal M0-actor policy entropy (training-end, in nats). |
| `entropy_m1` | float | Terminal M1-actor policy entropy. |
| `critic_loss` | float | Final critic loss (MSE on advantage targets). |
| `actor_loss` | float | Final pooled actor PPO loss. |
| `actor_loss_m0` | float | Final M0-actor PPO loss component. |
| `actor_loss_m1` | float | Final M1-actor PPO loss component. |
| `train_m0_pdr_last100` | float | Mean training-time M0 PDR over last 100 training episodes. |
| `train_m1_pdr_last100` | float | Mean training-time M1 PDR over last 100 training episodes. |
| `n_episodes` | int | Number of training episodes (3000 for all production runs). |
| `n_eval_episodes` | int | Number of evaluation episodes (100 for all production runs; Phase A N=2 used 1000 due to an earlier sed-substring incident, the over-precision is benign). |

## Phase-to-row mapping

| Index | config | N | M | M_m0 | demand_separation | architecture |
|---|---|---|---|---|---|---|
| 0 | A | 4 | 5 | - | False | Mode 0a |
| 1 | A | 2 | 5 | - | False | Mode 0a |
| 2 | A | 3 | 5 | - | False | Mode 0a |
| 3 | A | 5 | 5 | - | False | Mode 0a |
| 4 | A | 7 | 5 | - | False | Mode 0a |
| 5 | A | 10 | 5 | - | False | Mode 0a |
| 6 | A_mode0c | 4 | 5 | - | False | Mode 0c |
| 7 | B_M3 | 4 | 3 | - | False | Mode 0a |
| 8 | B_M7 | 4 | 7 | - | False | Mode 0a |
| 9 | B_M10 | 4 | 10 | - | False | Mode 0a |
| 10 | C_N4 | 4 | 5 | 2 | True | Mode 0a |
| 11 | C_N7 | 7 | 5 | 2 | True | Mode 0a |
| 12 | C_N10 | 10 | 5 | 2 | True | Mode 0a |
| 13 | D_N4 | 4 | 5 | 2 | True | Mode 0c |
| 14 | D_N10 | 10 | 5 | 2 | True | Mode 0c |
