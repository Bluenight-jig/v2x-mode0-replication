# Known Gotchas

Development pitfalls accumulated over the simulation programme. If you're
modifying the code or running on a fresh system, these are the things most
likely to bite.

## sed substring matching

When changing numeric CFG values via sed, always anchor on the trailing comma:

    # BAD - "10" matches inside "100", doubling the value
    sed -i 's/"n_eval_episodes": 10/"n_eval_episodes": 100/' train.py

    # GOOD - the comma anchor prevents substring matching
    sed -i 's/"n_eval_episodes": [0-9]\+,/"n_eval_episodes": 100,/' train.py

The Phase A N=2 entry in `psm001_results.json` was originally run with
`n_eval_episodes=1000` due to this exact substring incident (10 -> 100 ->
1000). The over-precision is benign and was left in place.

The phase-run scripts in `scripts/` use Python regex via `python3 - << EOF`
heredocs instead of sed for the same reason - regex with `\d+,` and
`\g<1>...` is harder to corrupt.

## TraCI version compatibility

Older TraCI bindings (pre-1.15) don't accept `stdout=` or `stderr=` keyword
arguments to `traci.start()`. The env handles this via `io.StringIO()`
redirection in `reset()`. If you upgrade TraCI and want to use the newer
kwargs, the `reset()` block can be simplified.

## NS-3 ZMQ linking inside scratch directories

Linking ZMQ inside NS-3's CMake/scratch system failed reliably during
development. The fix is to compile the bridge entirely outside NS-3, as a
standalone C++ binary with `g++` directly:

    cd ns3_bridge && make

The bridge does not need to link against NS-3 at runtime - it only needs
ZMQ and JsonCpp. The TR 37.885 channel model is implemented directly in
`v2x_bridge.cc`.

## Chat-client autolinking on .py paths

Some chat interfaces autolink `.py` filenames during paste, transforming
strings like `'envs/v2x_env.py'` into `'envs/v2x_[env.py](http://env.py)'`.
Linux accepts these as valid filenames, so the autolinked version coexists
with the real file silently.

If you ever see `dpkg -l` or `grep` output containing markdown link syntax
mixed into a filename, autolinking is the culprit. To defeat it in patch
scripts:

    # Reconstruct the path so the chat client cannot recognise '.py'
    fname = "envs/v2x_env" + chr(46) + "py"
    p = pathlib.Path(fname)

This is used throughout the patch-management scripts in development history.

## Argmax evaluation causes coordination collapse

In v2 of the trainer, evaluation used argmax on the policy distribution.
When all agents observe similar global state (especially at low N), argmax
collapses to identical actions across vehicles, producing degenerate
coordination (M0 PDR=0, SINR=-42 dB).

Fix (Fix A2): use stochastic sampling at the terminal entropy coefficient
(`ent_coef_end = 0.001`). The eval-time entropy is low enough to commit to
the learned policy but high enough to break ties between symmetric vehicles.

## Reward function: log1p collapse

The original v1 reward used `log1p(SINR_dB + 20)`, which inflates
artificially to ~241 per episode regardless of policy quality. This made
advantages essentially uninformative; the critic could not learn a useful
value function.

Fix (Fix A3): use bounded normalised rewards. M0 uses
`clip(SINR_dB/20, 0, 1)` for the SINR term, plus PDR-based components. M1's
SINR ceiling was also lowered from 40 dB to 20 dB to match M0 - otherwise
M1 has an incentive to grab the cleanest subchannels even when M0 needs
them.

## Observation space herding

v1 included a shared subchannel occupancy map as observation. This caused
all agents to react to the same observation in the same way, producing
herding oscillation (every agent fleeing the "crowded" channel simultaneously).

Fix: per-vehicle EMA channel quality maps (`ema_alpha=0.3`). Each vehicle
maintains its own private channel quality history, decoupling observations
across agents.

## Entropy schedule silently overridden

The v2 entropy fix originally lived in `train.py`, which set `cfg["ent_coef"]`
via a schedule each episode. But `MAPPOTrainer.__init__` read `cfg["ent_coef"]`
once at construction and cached it - so the schedule had no effect.

Fix (Fix A4-corrected): entropy schedule is owned by the trainer. The runner
calls `trainer.step_entropy_schedule()` once per episode. This makes the
schedule visible to the training loop's PPO update without any cfg-mutation
indirection.

## Filename autolinking in heredocs

Related to the autolinking gotcha above, but specifically for patches that
embed file paths INSIDE the heredoc text. The string `'envs/v2x_env.py'`
inside a heredoc gets mangled when pasted into a terminal from some chat
clients. Solution: same as above, use `chr(46)` instead of `.` in the
filename string.

## SUMO version drift via apt-get upgrade

`apt-get upgrade` can update SUMO and its Python bindings (`sumolib`,
`traci`) underneath your conda environment without warning. If you find
that pip-installed `sumolib` and `traci` versions don't match what you
expected, an apt-side update is the most likely cause. The reproducibility
smoke test confirmed Phase A N=4 results are stable across SUMO 1.19 ->
1.26, but in general it's worth checking `sumo --version` if a re-run
produces unexpected numbers.
