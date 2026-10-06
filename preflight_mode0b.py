"""preflight_mode0b.py -- the Mode 0b trainer with real PyTorch and the real repository modules.
Run from ~/v2x_clean:   python3 preflight_mode0b.py
CPU only, so every comparison can demand EXACT equality. No bridge needed (synthetic rollouts in the
runner's interleaved layout); it can run alongside a batch. A live check follows with the runners."""
import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""
import numpy as np
import torch
torch.set_num_threads(1)
from agents.mappo_mode0b import MAPPOTrainerMode0bAS
from agents.mappo_critic_ablation import MAPPOTrainerMode0cASCritic

N, M = 10, 5
OBS, ACT = 3 + M + 3, M * 5
GS = N * OBS + M
T = 40
BASE = dict(n_vehicles=N, n_subchannels=M, gamma=0.5, gae_lambda=0.95, clip_eps=0.2, ppo_epochs=4,
            ent_coef_start=0.01, ent_coef_end=0.001, n_episodes=5000, lr_actor=3e-4, lr_critic=3e-4)
BLOCK, INTER = [0] * 5 + [1] * 5, [0, 1] * 5
results = []

def check(name, ok, detail=""):
    results.append(bool(ok))
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f"  ({detail})" if detail else ""))

def make(cls, vc=BLOCK, **kw):
    torch.manual_seed(0)
    return cls(dict(BASE, _vehicle_classes_for_logging=list(vc), **kw), OBS, ACT, GS)

def flat(module):
    return torch.cat([p.detach().reshape(-1).clone() for p in module.parameters()])

def maxdiff(a, b):
    return float((flat(a) - flat(b)).abs().max())

def opt_params(tr, j):
    return [p.detach().clone() for g in tr.opts_a[j].param_groups for p in g["params"]]

def synthetic(tr, seed=1):
    """One rollout in the runner's layout (index t*N + i); log-probs from tr's own actors."""
    rng = np.random.default_rng(seed)
    obs = rng.random((T, N, OBS)).astype(np.float32)
    gs = np.concatenate([obs.reshape(T, N * OBS), rng.random((T, M)).astype(np.float32)], axis=1)
    act = rng.integers(0, ACT, size=(T, N))
    lp = np.zeros((T, N), np.float32)
    with torch.no_grad():
        for i in range(N):
            d = tr.actors[i].forward(torch.as_tensor(obs[:, i]))
            lp[:, i] = d.log_prob(torch.as_tensor(act[:, i])).numpy()
    buf = {"obs": torch.as_tensor(obs.reshape(T * N, OBS)),
           "global_state": torch.as_tensor(np.repeat(gs, N, axis=0)),
           "actions": torch.as_tensor(act.reshape(-1), dtype=torch.long),
           "log_probs": torch.as_tensor(lp.reshape(-1)),
           "advantages": torch.as_tensor(rng.normal(size=T * N).astype(np.float32)),
           "returns": torch.as_tensor(rng.normal(size=T * N).astype(np.float32)),
           "vehicle_idx": torch.as_tensor(np.tile(np.arange(N), T), dtype=torch.long)}
    return buf, gs

print("=== 1. Structure ===")
b = make(MAPPOTrainerMode0bAS)
check("passive: safety [3, 4], M1 [7, 8, 9]; shared slots 3 and 7", b.passive == {0: [3, 4], 1: [7, 8, 9]} and b.shared_slot == {0: 3, 1: 7})
check("passive slots share their class's actor", b.actors[4] is b.actors[3] and b.actors[8] is b.actors[7] and b.actors[9] is b.actors[7])
check("unique actors: 7 (5 active + 2 shared)", len({id(a) for a in b.actors}) == 7)
check("optimisers: 10 distinct objects", len({id(o) for o in b.opts_a}) == 10)
shared = {id(p) for c in (3, 7) for p in b.actors[c].parameters()}
check("unused slots' optimisers hold none of the shared actors' parameters",
      all(id(p) not in shared for j in (4, 8, 9) for g in b.opts_a[j].param_groups for p in g["params"]))
check("critic is agent-specific (global state + own observation)", b.critic_in_dim == GS + OBS, f"input {b.critic_in_dim}")
bi = make(MAPPOTrainerMode0bAS, vc=INTER)
check("interleaved classes: passive safety [6, 8], M1 [5, 7, 9]", bi.passive == {0: [6, 8], 1: [5, 7, 9]})

print("\n=== 2. With no passive vehicles it is exactly Mode 0c with the agent-specific critic ===")
z = make(MAPPOTrainerMode0bAS, mode0b_passive_m0=0, mode0b_passive_m1=0); c = make(MAPPOTrainerMode0cASCritic)
d0 = max([maxdiff(z.actors[i], c.actors[i]) for i in range(N)] + [maxdiff(z.critic, c.critic)])
check("identical at construction", d0 == 0, f"max difference {d0:.1e}")
buf, _ = synthetic(z); z.update_from_tensors(buf); c.update_from_tensors(buf)
d1 = max([maxdiff(z.actors[i], c.actors[i]) for i in range(N)] + [maxdiff(z.critic, c.critic)])
check("identical after one full update", d1 == 0, f"max difference {d1:.1e}")

print("\n=== 3. The update is the Mode 0c update with passive transitions routed to the shared actors ===")
b = make(MAPPOTrainerMode0bAS); c = make(MAPPOTrainerMode0cASCritic)
check("shared actors start identical to the reference's slots 3 and 7", maxdiff(b.actors[3], c.actors[3]) == 0 and maxdiff(b.actors[7], c.actors[7]) == 0)
buf, _ = synthetic(b)
orphans = {j: opt_params(b, j) for j in (4, 8, 9)}
trained = b.active + [3, 7]
before = {i: flat(b.actors[i]) for i in trained}
route = torch.as_tensor([0, 1, 2, 3, 3, 5, 6, 7, 7, 7], dtype=torch.long)
rbuf = dict(buf); rbuf["vehicle_idx"] = route[buf["vehicle_idx"]]
sb = b.update_from_tensors(buf); sc = c.update_from_tensors(rbuf)
d2 = max([maxdiff(b.actors[i], c.actors[i]) for i in trained] + [maxdiff(b.critic, c.critic)])
check("active actors, shared actors and critic identical to the reference", d2 == 0, f"max difference {d2:.1e}")
check("every active and shared actor actually trained", all(float((flat(b.actors[i]) - before[i]).abs().max()) > 0 for i in trained))
check("unused slots' original actors never stepped", all(all(torch.equal(x, y) for x, y in zip(orphans[j], opt_params(b, j))) for j in orphans))
for key, k in (("entropy", 10 / 7), ("entropy_m0", 5 / 4), ("entropy_m1", 5 / 3), ("actor_loss", 10 / 7)):
    check(f"logging: {key} averages only the slots in use", abs(sb[key] - sc[key] * k) < 1e-9, f"{sb[key]:.5f}")

print("\n=== 4. Routing cannot reach the advantages ===")
b = make(MAPPOTrainerMode0bAS); c = make(MAPPOTrainerMode0cASCritic)
_, gs = synthetic(b, seed=5)
rew = np.random.default_rng(7).normal(size=T * N).astype(np.float32); dn = np.zeros(T * N, np.float32)
for t in range(T):
    b.value(gs[t]); c.value(gs[t])
b.value(gs[-1]); c.value(gs[-1])
ab, rb = b.compute_gae(rew, None, None, dn); ac, rc = c.compute_gae(rew, None, None, dn)
check("advantages and returns identical to the reference", np.array_equal(ab, ac) and np.array_equal(rb, rc))

print("\n=== 5. Acting and saving ===")
b = make(MAPPOTrainerMode0bAS)
obs = np.random.default_rng(3).random((N, OBS)).astype(np.float32)
obs[4] = obs[3]; obs[8] = obs[7]; obs[9] = obs[7]
acts, _ = b.select_actions(obs, np.array(BLOCK), deterministic=True)
check("passive vehicles of a class, same observation -> same action", acts[3] == acts[4] and acts[7] == acts[8] == acts[9])
sd = [a.state_dict() for a in b.actors]
check("checkpoint: passive slots save their class's shared policy", all(torch.equal(sd[4][k], sd[3][k]) for k in sd[3]) and all(torch.equal(sd[9][k], sd[7][k]) for k in sd[7]))

print(f"\n{sum(results)} of {len(results)} checks passed")
print("PRE-FLIGHT PASSED" if all(results) else "PRE-FLIGHT FAILED -- send me the output")
