# Architecture Notes

This document summarises the simulation architecture and the key design
choices behind the Mode 0a (Path A.5) and Mode 0c trainers. For deeper
discussion see Section 5 of the paper.

## System diagram

```
   Python                                    C++
+---------+   ZMQ REQ/REP TCP:5556   +---------------+
|         |  -- subchannels --->     |               |
|  V2XEnv |  -- powers ----->        |  v2x_bridge   |
|         |  <-- SINR/PDR --         |  (channel sim)|
+---------+                          +---------------+
     |                                       |
     | TraCI                                 | TR 37.885 V2V
     v                                       | path loss + shadowing
+---------+                                  | + Rayleigh fading
|  SUMO   |
| highway |
+---------+
```

## Mode 0a (Path A.5)

Two per-class actor networks with intra-class parameter sharing:

- `actor_m0`: 2-layer MLP, hidden=128, ~21,300 parameters. Receives observations from M0-class vehicles only.
- `actor_m1`: 2-layer MLP, hidden=128, ~21,300 parameters. Receives observations from M1-class vehicles only.
- Centralized critic on global state, hidden=256 with LayerNorm, ~70-94k parameters depending on N.

Observation vector (`obs_dim = 3 + M + 3`):
`[x/1000, y/1000, speed/33.3, EMA_chan_quality * M, last_pdr, vehicle_class, vehicle_id_norm]`

Action space (`action_dim = M * 5`): joint (subchannel x power_level), with
M subchannels and 5 power levels.

Within a class, all vehicles share parameters. The vehicle_id_norm feature
was added (Path A -> Path A.5) to allow potential per-agent specialisation,
but empirically did not break symmetric-Nash equilibria - the within-class
collision rate at the uniform-random ceiling for `m0_count >= 2` is the
empirical signature of this constraint.

## Mode 0c (per-vehicle architecture)

N independent actor networks, one per vehicle. No parameter sharing across
vehicles even within a class. Centralized critic and all environment
components identical to Mode 0a.

This eliminates the within-class symmetric-policy phenomenon by removing
parameter sharing entirely. At N=4 with shared pool: M0 PDR jumps from
0.776 (Mode 0a) to 0.982 (Mode 0c); entropy drops from 2.27 to 0.46.

## Reward function (class-differentiated)

M0 reward (safety-critical, primary objective):

    r_M0 = alpha * PDR_i + beta * clip(SINR_dB_i / 20, 0, 1) + gamma_team * mean(PDR_M0)

with `alpha=1.0`, `beta=0.3`, `gamma_team=0.5`.

M1 reward (non-safety, with cooperative penalty):

    r_M1 = delta * clip(SINR_dB_i / 20, 0, 1) - eta * (1 - mean(PDR_M0))

with `delta=0.3`, `eta=0.3`. The eta term realigns M1's gradient to penalise
M1 throughput gains that come at the cost of M0 reliability.

## Channel model

3GPP TR 37.885 V2V highway scenario implemented in `v2x_bridge.cc`:
- Path loss: `PL = 32.4 + 20 * log10(f_c [GHz]) + 20 * log10(d [m])`
- Log-normal shadowing with 3 dB standard deviation
- Rayleigh fast fading per TTI
- Sigmoid BLER-SINR mapping calibrated from TR 38.901

The bridge is computationally independent of NS-3. ZMQ REQ/REP TCP on port
5556 with JSON payloads. Each TTI sends (positions, subchannels, powers)
and receives (SINR_lin, SINR_dB, PDR) per vehicle.

## Training schedule

- 3000 training episodes per run (200 TTIs per episode).
- PPO with gamma=0.99, gae_lambda=0.95, clip_eps=0.2, ppo_epochs=4.
- lr_actor = lr_critic = 3e-4 with cosine decay to lr_min = 1e-5.
- Entropy schedule: linear anneal from 0.05 to 0.001 (owned by trainer's
  `step_entropy_schedule()`).
- 100 eval episodes at training completion, stochastic sampling at terminal
  entropy (not argmax - prevents coordination collapse).
