#!/usr/bin/env python3
"""
stage4a_obu_timing.py -- the per-decision inference cost of a Mode 0c actor (R1-c2).

Loads a saved Mode 0c checkpoint, rebuilds each VehicleActor exactly as trained
(obs 3+M+3 -> 128 -> 128 -> 5M, ReLU), and times ONE vehicle's decision on ONE CPU
thread at batch size 1 -- the situation on a vehicle's on-board unit. Three forms:

  1. network forward (PyTorch)
  2. deployed decision: argmax of the logits (PyTorch)
  3. the same decision in plain NumPy -- no deep-learning framework at all

The NumPy decisions are checked against PyTorch for exact agreement. A dense MLP
does the same arithmetic whatever its input, so random inputs time it faithfully.

Measured on this machine's CPU. An embedded OBU processor is slower; the MAC count
printed is hardware-independent, which is what lets the margin be judged.
"""
import argparse
import pathlib
import statistics as st
import sys
import time

import numpy as np
import torch

sys.path.insert(0, ".")
from agents.actor import VehicleActor  # the trained architecture itself

BUDGET_US = 100_000  # one decision per 100 ms interval


def load_ckpt(path):
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:                     # older PyTorch without weights_only
        return torch.load(path, map_location="cpu")


def bench(fn, n_inputs, iters, warmup=2000):
    for i in range(warmup):
        fn(i % n_inputs)
    t = []
    for i in range(iters):
        a = time.perf_counter_ns()
        fn(i % n_inputs)
        t.append(time.perf_counter_ns() - a)
    t.sort()
    return st.median(t) / 1000.0, t[int(0.99 * len(t))] / 1000.0   # microseconds


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="checkpoints/A_mode0c_N10_seed1_actors.pt")
    ap.add_argument("--iters", type=int, default=20000)
    args = ap.parse_args()

    torch.set_num_threads(1)               # one core, as on an OBU
    ck = load_ckpt(args.ckpt)
    M, N = ck["M"], ck["N"]
    obs_dim, act_dim = 3 + M + 3, M * 5
    actors = []
    for sd in ck["actor_state_dicts"]:
        a = VehicleActor(obs_dim, act_dim)
        a.load_state_dict(sd)
        a.eval()
        actors.append(a)
    actor = actors[0]
    n_params = sum(p.numel() for p in actor.parameters())
    lin = [m for m in actor.net if isinstance(m, torch.nn.Linear)]
    macs = sum(m.in_features * m.out_features for m in lin)

    rng = np.random.default_rng(0)
    X = rng.standard_normal((4096, obs_dim)).astype(np.float32)
    Xt = [torch.from_numpy(X[i:i + 1]) for i in range(len(X))]
    W = [(m.weight.detach().numpy().T.copy(), m.bias.detach().numpy().copy()) for m in lin]

    def np_decide(x):
        h = np.maximum(x @ W[0][0] + W[0][1], 0.0)
        h = np.maximum(h @ W[1][0] + W[1][1], 0.0)
        return int(np.argmax(h @ W[2][0] + W[2][1]))

    # correctness: NumPy must choose exactly what PyTorch chooses
    with torch.no_grad():
        agree = sum(np_decide(X[i:i + 1]) == int(actor.net(Xt[i]).argmax(-1)) for i in range(2000))

    with torch.no_grad():
        fwd = bench(lambda i: actor.net(Xt[i]), len(X), args.iters)
        dec = bench(lambda i: int(actor.net(Xt[i]).argmax(-1)), len(X), args.iters)
        allN = bench(lambda i: [int(a.net(Xt[i]).argmax(-1)) for a in actors], len(X), args.iters // 10)
    npd = bench(lambda i: np_decide(X[i:i + 1]), len(X), args.iters)

    print(f"Checkpoint        : {args.ckpt}")
    print(f"Configuration     : N={N}, M={M}, {len(actors)} per-vehicle actors")
    print(f"One actor         : {obs_dim} -> 128 -> 128 -> {act_dim}, ReLU")
    print(f"Parameters        : {n_params:,}")
    print(f"Multiply-adds     : {macs:,} per decision (hardware-independent)")
    print(f"Threads           : {torch.get_num_threads()}  (PyTorch {torch.__version__})")
    print(f"NumPy = PyTorch   : {agree} of 2000 decisions identical")
    print()
    print(f"{'one vehicle, one decision':<38}{'median':>10}{'p99':>10}{'share of 100 ms':>18}")
    for name, (med, p99) in [("network forward (PyTorch)", fwd),
                             ("deployed decision, argmax (PyTorch)", dec),
                             ("deployed decision, plain NumPy", npd)]:
        print(f"  {name:<36}{med:>8.1f} µs{p99:>8.1f} µs{100 * p99 / BUDGET_US:>16.4f} %")
    med, p99 = allN
    print(f"\n  all {len(actors)} actors in sequence (if one unit computed every vehicle's decision)")
    print(f"  {'':<36}{med:>8.1f} µs{p99:>8.1f} µs{100 * p99 / BUDGET_US:>16.4f} %")
    worst = max(fwd[1], dec[1], npd[1])
    print(f"\nHeadroom          : the slowest single-decision p99 ({worst:.1f} µs) fits "
          f"{BUDGET_US / worst:,.0f} times into the 100 ms interval.")


if __name__ == "__main__":
    main()
