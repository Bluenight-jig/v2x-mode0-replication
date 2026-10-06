"""batch_eta.py -- finish-time estimate for a running batch, measured from its own logs.

For each chain: finished runs are timed from their result files; the running run's speed is measured
from its log (episodes done / hours elapsed); runs not yet started use planned durations, scaled by
how fast this batch's finished runs actually went. Read-only: touches nothing."""
import os, re, sys, time, subprocess

CHAINS = {"B26": [
    ["A_N10_seed1_5000ep_gaefix_g05_ent01_as", "D_N10_pool4_seed1_gaefix_g05_ent01", "D_N4_seed1_gaefix_g05_ent01"],
    ["A_N10_seed2_5000ep_gaefix_g05_ent01_as", "D_N10_pool4_seed2_gaefix_g05_ent01", "D_N4_seed2_gaefix_g05_ent01"],
    ["A_N10_seed3_5000ep_gaefix_g05_ent01_as", "D_N10_pool4_seed3_gaefix_g05_ent01", "D_N4_seed3_gaefix_g05_ent01"],
    ["A_mode0c_N10_seed1_gaefix_g05_ent01_armP", "A_mode0c_N10_seed2_gaefix_g05_ent01_armP"],
    ["A_mode0c_N10_seed3_gaefix_g05_ent01_armP", "A_mode0c_N10_seed1_gaefix_g05_ent01_armS"],
]}
EPISODES = 5000
def planned_hours(label):
    """Planned durations (PSM003_rebuild_scope_v2.md, measured where available)."""
    if label.startswith("A_N10_"):  return 8.35     # Mode 0a, N = 10: 8.3-8.4 h measured
    if label.startswith("D_N4_"):   return 7.1      # Mode 0c, N = 4: ~7.1 h
    return 13.9                                      # Mode 0c, N = 10 (pool 4, arms): 13.4-14.5 h measured
def birth(path):
    """Creation time of a file, or None if the filesystem does not record it."""
    try:
        t = int(subprocess.run(["stat", "-c", "%W", path], capture_output=True, text=True).stdout.strip() or 0)
        return t or None
    except Exception:
        return None
def last_episode(log):
    txt = open(log, errors="replace").read()
    if "Training complete" in txt: return EPISODES, True
    eps = re.findall(r'^Ep\s*(\d+) \|', txt, re.M)
    return (int(eps[-1]) if eps else 0), False
def hm(t): return time.strftime("%a %d %b %H:%M", time.gmtime(t + 8 * 3600)) + " BJT"   # Beijing, whatever WSL's zone
def main(batch="B26", now=None, birth_fn=birth):
    now = now or time.time()
    chains = CHAINS[batch]
    # pass 1: how fast did this batch's finished runs actually go, against plan?
    ratios = []
    for chain in chains:
        prev_end = None
        for lab in chain:
            res, log = f"results/_{lab}.json", f"logs/{lab}.log"
            if not os.path.exists(res): break
            end = os.path.getmtime(res); start = prev_end or (birth_fn(log) if os.path.exists(log) else None)
            if start: ratios.append(((end - start) / 3600) / planned_hours(lab))
            prev_end = end
    factor = sum(ratios) / len(ratios) if ratios else 1.0
    print(f"{batch}: {len(ratios)} finished run(s) timed; actual/planned = {factor:.2f}\n")
    ends = []
    for k, chain in enumerate(chains, 1):
        prev_end, t, lines = None, now, []
        for lab in chain:
            res, log = f"results/_{lab}.json", f"logs/{lab}.log"
            if os.path.exists(res):
                prev_end = os.path.getmtime(res); lines.append(f"    done     {lab}  ({hm(prev_end)})"); continue
            if os.path.exists(log):
                ep, evaluating = last_episode(log)
                start = prev_end or birth_fn(log)
                if start and ep > 0:
                    h_per_ep = (now - start) / 3600 / ep
                    left = (EPISODES - ep) * h_per_ep + 100 * h_per_ep * 0.5      # eval: 100 episodes, no updates
                    note = f"ep {ep}, {h_per_ep * 3600:.1f} s/ep"
                else:
                    left = planned_hours(lab) * factor * (1 - ep / EPISODES); note = f"ep {ep}, speed not measurable yet"
                t = now + left * 3600
                lines.append(f"    RUNNING  {lab}  {note}{' (evaluating)' if evaluating else ''} -> ~{hm(t)}")
            else:
                t += planned_hours(lab) * factor * 3600
                lines.append(f"    queued   {lab}  -> ~{hm(t)}")
        ends.append(t)
        print(f"  T{k}: ends ~{hm(t)}"); print("\n".join(lines))
    print(f"\n{batch} estimated end: ~{hm(max(ends))}   (now {hm(now)})")
if __name__ == "__main__":
    main(*(sys.argv[1:2] or ["B26"]))
