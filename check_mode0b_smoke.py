"""check_mode0b_smoke.py -- checks the live Mode 0b smoke run (d10_generator.py smoke), then removes its
result and checkpoint so they cannot be mistaken for real runs. Run from ~/v2x_clean after the smoke run."""
import json, pathlib, re
import torch
from agents.mappo_mode0b import PASSIVE_DEFAULT

LABEL = "SMOKE_mode0b_N10"
log, res, ckp = pathlib.Path(f"logs/{LABEL}.log"), pathlib.Path(f"results/_{LABEL}.json"), pathlib.Path(f"checkpoints/{LABEL}_actors.pt")
results = []
def check(name, ok, detail=""):
    results.append(bool(ok)); print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f"  ({detail})" if detail else ""))

print("=== 1. The run, from its log ===")
txt = log.read_text(errors="replace") if log.exists() else ""
check("log exists", log.exists())
check("training and evaluation completed", "Training complete" in txt and "Run complete." in txt)
check("agent-specific critic built (99,073 parameters)", re.search(r"Critic params\s*:\s*99,073", txt) is not None)
check("five safety and five M1 vehicles", re.search(r"M0 count:\s*5\s+M1 count:\s*5", txt) is not None)
check("no traceback", "Traceback" not in txt)

print("\n=== 2. The results file ===")
r = json.loads(res.read_text())[-1] if res.exists() else {}
check("results file written", bool(r))
check("labelled as Mode 0b", r.get("architecture") == "mode_0b_hybrid_actors", r.get("architecture"))
check("3 training and 2 evaluation episodes", r.get("n_episodes") == 3 and r.get("n_eval_episodes") == 2)

print("\n=== 3. The checkpoint, against the REAL environment's class layout ===")
ck = torch.load(ckp, map_location="cpu", weights_only=False) if ckp.exists() else {}
check("checkpoint written and labelled as Mode 0b", ck.get("architecture") == "mode_0b_hybrid_actors", ck.get("architecture"))
vc = [int(v) for v in ck.get("cfg", {}).get("_vehicle_classes_for_logging", [])]
check("vehicle classes stored (10)", len(vc) == 10, str(vc))
passive = {c: [i for i in range(len(vc)) if vc[i] == c][-PASSIVE_DEFAULT[c]:] for c in (0, 1)}
shared = {c: passive[c][0] for c in (0, 1)}
active = [i for i in range(len(vc)) if not any(i in p for p in passive.values())]
print(f"      passive safety {passive[0]}, passive M1 {passive[1]}; shared slots {shared}; active {active}")
sd = ck.get("actor_state_dicts", [])
same = lambda a, b: a.keys() == b.keys() and all(torch.equal(a[k], b[k]) for k in a)
if len(sd) == 10:
    check("every passive vehicle saves its class's shared policy", all(same(sd[j], sd[shared[vc[j]]]) for c in (0, 1) for j in passive[c]))
    distinct = active + [shared[0], shared[1]]
    check("the 7 distinct policies (5 active + 2 shared) all differ", all(not same(sd[a], sd[b]) for i, a in enumerate(distinct) for b in distinct[i + 1:]))
    w = next(t for t in ck["critic_state_dict"].values() if t.dim() == 2)
    check("critic input = global state + own observation (126)", w.shape[1] == 126, f"input {w.shape[1]}")
else:
    check("10 actor state dicts saved", False, str(len(sd)))

ok = all(results)
print(f"\n{sum(results)} of {len(results)} checks passed")
if ok:
    for f in (res, ckp): f.unlink()
    print(f"SMOKE RUN PASSED -- its result and checkpoint removed; log kept as {log}")
else:
    print("SMOKE RUN FAILED -- files kept for inspection; send me this output")
