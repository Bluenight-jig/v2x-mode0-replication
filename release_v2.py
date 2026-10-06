#!/usr/bin/env python3
"""
release_v2.py -- prepares release v2.0 of github.com/Bluenight-jig/v2x-mode0-replication from ~/v2x_clean.

  python3 release_v2.py stage --concept-doi 10.5281/zenodo.NNNNNNN
        writes README.md, CITATION.cff, .zenodo.json, REPRODUCE.md, RELEASE_NOTES.md; checks RUN_REGISTER.csv;
        stages an EXPLICIT list of files (never `git add -A`), refuses any file over 50 MB, prints a summary.
  python3 release_v2.py verify
        after you commit: clones the local repository into /tmp, reruns Stage 4a v2 and make_figures.py from the
        clone alone, and checks that every interval and every table is identical to yours.
Nothing is pushed by this script.
"""
import argparse, json, platform, re, shutil, subprocess, sys, datetime
from pathlib import Path

REPO = Path(__file__).resolve().parent
PAPER = "Mode 0: Architecture, Risk Taxonomy, and Standardization Pathway for RCU-Assisted V2X Safety Communication"
AUTHOR = ("Jiang", "Dewei")
REPO_URL = "https://github.com/Bluenight-jig/v2x-mode0-replication"
VERSION, TAG = "2.0.0", "v2.0"
TODAY = datetime.date.today().isoformat()
MAX_BYTES = 50 * 1024 * 1024
CODE = ["agents/*.py", "envs/*.py", "ns3_bridge/*.cc", "ns3_bridge/*.cpp", "ns3_bridge/*.h", "ns3_bridge/Makefile",
        "ns3_bridge/CMakeLists.txt", "ns3_bridge/*.sh", "*.py", "*.sh", "scenarios/**/*.xml", "scenarios/**/*.sumocfg",
        "requirements*.txt", "environment*.yml"]
RESULTS = ["results/*.json", "results/*.csv", "results/*.txt", "results/stage4b/*.json"]
LOGS = ["logs/*.log"]
FIGS = ["figures/*.pdf", "figures/*.png", "figures/captions.md", "tables/*.tex"]
G = "_gaefix_g05_ent01"
CKPTS = ([f"checkpoints/A_mode0c_seed{s}_5000ep{G}_actors.pt" for s in range(1, 6)] +
         [f"checkpoints/A_mode0c_N10_seed{s}{G}_actors.pt" for s in range(1, 6)] +
         [f"checkpoints/A_mode0b_N10_seed{s}{G}_actors.pt" for s in (1, 2, 3)] +
         [f"checkpoints/A_mode0c_N10_M7_seed{s}{G}_actors.pt" for s in (1, 2, 3)] +
         [f"checkpoints/D_N10_M7_pool5_seed{s}{G}_actors.pt" for s in (1, 2, 3)])
DOCS = ["README.md", "CITATION.cff", ".zenodo.json", "REPRODUCE.md", "RELEASE_NOTES.md", "RUN_REGISTER.csv", "LICENSE"]
EXCLUDE = re.compile(r"(SMOKE|__pycache__|\.pyc$|\.npz$|\.tar\.gz$|\.bak$)")

def sh(cmd, **kw):
    return subprocess.run(cmd, cwd=kw.pop("cwd", REPO), capture_output=True, text=True, **kw)

def versions():
    v = {"OS": platform.platform(), "Python": platform.python_version()}
    for mod in ("torch", "numpy", "matplotlib", "zmq"):
        try: v[mod] = __import__(mod).__version__
        except Exception: v[mod] = "not found"
    try:
        import torch
        if torch.cuda.is_available(): v["GPU"] = torch.cuda.get_device_name(0)
    except Exception: pass
    try:
        r = sh(["sumo", "--version"]); v["SUMO"] = (r.stdout.splitlines() or ["not found"])[0].strip() if r.returncode == 0 else "not found"
    except FileNotFoundError:
        v["SUMO"] = "not found on PATH"
    return v

def readme(doi, v):
    env = "\n".join(f"| {k} | {val} |" for k, val in v.items())
    return f"""# Mode 0 V2X — Reproducibility Package (v{VERSION})

[![DOI](https://zenodo.org/badge/DOI/{doi}.svg)](https://doi.org/{doi})
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

Code, simulation environment, results and figures for:

> **{PAPER}**
> {AUTHOR[1]} {AUTHOR[0]}
> Preprint, 2026. Manuscript under review.

**v{VERSION} accompanies the revised manuscript.** Releases v1.0, v1.0.1 and v1.1 accompanied the original
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

All 200 training runs are listed in `RUN_REGISTER.csv` with their era: run 1 (excluded, defect 1), runs 2–86
(original programme, superseded by defect 2), runs 87–110 (impact checks and diagnostics) and **runs 111–200 (the
rebuild, which the revised manuscript reports)**.

## What is in this release

| Path | Contents |
|---|---|
| `agents/`, `envs/` | Trainers and environments. Rebuild: `mappo_gaefix.py` (per-agent GAE), `mappo_critic_ablation.py` (local and agent-specific critics), `mappo_mode0b.py` (hybrid fleet), `envs/v2x_env_arms.py` (capability ablations) |
| `train_*.py` | The exact runner behind every run (see `RUN_REGISTER.csv`); `d*_generator.py` derived each rebuild batch's runners from earlier ones, changing only stated lines |
| `results/_*.json` | Every run's evaluation (100 episodes), and the SB-SPS baseline evaluations |
| `results/_ci_summary_v2.{{csv,json}}` | **Every interval the paper reports** (Stage 4a v2: two-sided 95% t-intervals across seeds) |
| `results/stage4b/` | Per-vehicle evaluation: information age, per-vehicle delivery, spatial reuse, Mode 0b by vehicle type |
| `logs/` | Training logs (coordination times are computed from them) |
| `checkpoints/` | Trained actors for every policy that Stage 4b re-evaluates |
| `figures/`, `tables/` | The revised manuscript's Section 7 figures (PDF, PNG) and tables (LaTeX) |

Not included: Stage 4b's per-interval arrays (about 190 MB), which `stage4b_eval.py` regenerates from the included checkpoints.

## Reproducing the paper's numbers and figures

```bash
python3 stage4a_v2.py      # every interval  -> results/_ci_summary_v2.{{csv,json}}   (seconds)
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
{env}

Hardware: AMD Ryzen 5 5600X, NVIDIA RTX 3060 (12 GB), Ubuntu 22.04 under WSL2; five runs in parallel. Wall time per
5,000-episode run with five in parallel: about 13–14.5 h (Mode 0c, N = 10), 8.3 h (Mode 0a, N = 10), 7 h (Mode 0c, N = 4).

## Citation

Please cite the paper (manuscript under review) and this software through its concept DOI,
[{doi}](https://doi.org/{doi}), which always resolves to the latest version. See `CITATION.cff`.

## License

MIT — see `LICENSE`.
"""

def citation(doi):
    return f"""cff-version: 1.2.0
message: "If you use this software, please cite it as below."
title: "Mode 0 V2X --- Reproducibility Package"
authors:
  - family-names: "{AUTHOR[0]}"
    given-names: "{AUTHOR[1]}"
    # orcid: https://orcid.org/0009-0000-3698-0248
type: software
license: MIT
repository-code: "{REPO_URL}"
url: "https://doi.org/{doi}"
version: "{VERSION}"
date-released: "{TODAY}"
keywords:
  - V2X
  - vehicle-to-everything
  - resource scheduling
  - multi-agent reinforcement learning
  - MAPPO
  - reproducibility
preferred-citation:
  type: article
  title: "{PAPER}"
  authors:
    - family-names: "{AUTHOR[0]}"
      given-names: "{AUTHOR[1]}"
  status: submitted
  year: 2026
"""

def zenodo():
    return json.dumps({
        "title": f"Mode 0 V2X — Reproducibility Package (v{VERSION})",
        "upload_type": "software", "version": VERSION, "license": "MIT",
        "creators": [{"name": f"{AUTHOR[0]}, {AUTHOR[1]}"}],
        "description": (f"Code, simulation environment, results and figures for the revised manuscript \"{PAPER}\" "
                        "(manuscript under review). Release 2.0 regenerates every learning-based result after correcting two "
                        "implementation defects found during the revision (see the README); releases 1.x are superseded."),
        "keywords": ["V2X", "vehicle-to-everything", "resource scheduling", "multi-agent reinforcement learning", "MAPPO", "reproducibility"],
        "notes": "Manuscript under review."}, indent=2, ensure_ascii=False) + "\n"

REPRODUCE = """# Reproducing the revised manuscript's results

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
from `ns3_bridge/` first. Runs 111–200 are the rebuild that the revised manuscript reports.
"""

def notes():
    return f"""Release {TAG} accompanies the revised manuscript "{PAPER}" (manuscript under review).

- Regenerates every learning-based result after correcting two implementation defects (incomplete fleet at episode
  start; advantages computed across vehicles) — see the README. Results of v1.x are superseded.
- 200 training runs in `RUN_REGISTER.csv`; runs 111–200 (standard MAPPO with per-agent GAE) are the ones reported.
- `stage4a_v2.py` and `make_figures.py` regenerate every interval, figure and table from this release alone.
"""

def expand(globs):
    out = set()
    for g in globs:
        out.update(p for p in REPO.glob(g) if p.is_file() and not EXCLUDE.search(str(p.relative_to(REPO))))
    return sorted(out)

def stage(doi):
    if not re.fullmatch(r"10\.5281/zenodo\.\d+", doi): sys.exit(f"--concept-doi must look like 10.5281/zenodo.1234567, got {doi!r}")
    if sh(["git", "rev-parse", "--is-inside-work-tree"]).stdout.strip() != "true": sys.exit("not inside the git repository")
    branch = sh(["git", "branch", "--show-current"]).stdout.strip()
    if branch != "main": sys.exit(f"on branch {branch!r}; switch to main first")
    reg = REPO / "RUN_REGISTER.csv"
    if not reg.exists(): sys.exit("RUN_REGISTER.csv not found in ~/v2x_clean -- copy it in first (see the instructions)")
    n_reg = len(reg.read_text(encoding="utf-8").splitlines()) - 1
    if n_reg != 200: sys.exit(f"RUN_REGISTER.csv has {n_reg} runs, expected 200")
    v = versions()
    for name, text in (("README.md", readme(doi, v)), ("CITATION.cff", citation(doi)), (".zenodo.json", zenodo()),
                       ("REPRODUCE.md", REPRODUCE), ("RELEASE_NOTES.md", notes())):
        (REPO / name).write_text(text, encoding="utf-8")
    groups = {"code": expand(CODE), "results": expand(RESULTS), "logs": expand(LOGS), "figures and tables": expand(FIGS),
              "checkpoints": [REPO / c for c in CKPTS if (REPO / c).exists()], "documentation": [REPO / d for d in DOCS if (REPO / d).exists()]}
    missing_ck = [c for c in CKPTS if not (REPO / c).exists()]
    big = [(p, p.stat().st_size) for g in groups.values() for p in g if p.stat().st_size > MAX_BYTES]
    if big: sys.exit("refusing: files over 50 MB:\n  " + "\n  ".join(f"{p.relative_to(REPO)} ({s / 2**20:.0f} MB)" for p, s in big))
    files = sorted({str(p.relative_to(REPO)) for g in groups.values() for p in g})
    for i in range(0, len(files), 200):
        r = sh(["git", "add", "-f", "--"] + files[i:i + 200])
        if r.returncode: sys.exit(r.stderr)
    print(f"Release {TAG}: {len(files)} files staged (explicit list; nothing else added)\n")
    for name, g in groups.items():
        print(f"  {name:<20} {len(g):4d} files  {sum(p.stat().st_size for p in g) / 2**20:7.1f} MB")
    print(f"  {'total':<20} {len(files):4d} files  {sum((REPO / f).stat().st_size for f in files) / 2**20:7.1f} MB")
    if missing_ck: print("\n⚠ checkpoints not found (not staged):\n  " + "\n  ".join(missing_ck))
    big5 = sorted(((REPO / f).stat().st_size, f) for f in files)[-5:]
    print("\n  largest: " + ", ".join(f"{f} ({s / 2**20:.1f} MB)" for s, f in reversed(big5)))
    print("\n  environment recorded in README:", "; ".join(f"{k} {val}" for k, val in v.items()))
    left = len([l for l in sh(["git", "status", "--porcelain"]).stdout.splitlines() if not l[:1] in "AMDR" or l[1:2] != " "])
    print(f"\n  {left} other changes in the working tree stay uncommitted.")
    print("\nNext: review with `git diff --cached --stat | tail -3`, then commit (see the instructions).")

def verify():
    tmp = Path("/tmp/v2x_release_check"); shutil.rmtree(tmp, ignore_errors=True)
    r = sh(["git", "clone", "--quiet", str(REPO), str(tmp)])
    if r.returncode: sys.exit("clone failed: " + r.stderr)
    head = sh(["git", "log", "-1", "--format=%h %s"], cwd=tmp).stdout.strip()
    print(f"Fresh clone of the committed state ({head}) in {tmp}\n")
    ok = []
    def check(name, cond, detail=""):
        ok.append(bool(cond)); print(f"  [{'PASS' if cond else 'FAIL'}] {name}" + (f"  ({detail})" if detail else ""))
    check("documentation present", all((tmp / d).exists() for d in DOCS))
    check("no per-interval arrays in the release", not list(tmp.rglob("*.npz")))
    a = sh([sys.executable, "stage4a_v2.py"], cwd=tmp) if (tmp / "stage4a_v2.py").exists() else subprocess.CompletedProcess([], 1, "", "")
    check("Stage 4a v2 runs from the clone alone", a.returncode == 0 and "All expected result files found." in a.stdout,
          (a.stdout.strip().splitlines() or ["no output"])[-1][:80])
    a_mine, a_theirs = REPO / "results/_ci_summary_v2.json", tmp / "results/_ci_summary_v2.json"
    if a_mine.exists() and a_theirs.exists():
        mine, theirs = json.loads(a_mine.read_text()), json.loads(a_theirs.read_text())
        check("every interval identical to yours", mine == theirs, f"{len(theirs)} rows")
    else:
        check("every interval identical to yours", False, "summary file missing in " + ("the clone" if a_mine.exists() else "your tree"))
    f = sh([sys.executable, "make_figures.py"], cwd=tmp) if (tmp / "make_figures.py").exists() else subprocess.CompletedProcess([], 1, "", "")
    check("make_figures.py runs from the clone alone", f.returncode == 0 and "All inputs found." in f.stdout)
    tabs = sorted((REPO / "tables").glob("*.tex"))
    same = [t.name for t in tabs if (tmp / "tables" / t.name).exists() and (tmp / "tables" / t.name).read_text() == t.read_text()]
    check("every table identical to yours", len(same) == len(tabs) and tabs, f"{len(same)} of {len(tabs)}")
    figs = [p for p in (tmp / "figures").glob("fig7*.pdf")]
    check("all four figures regenerated", len(figs) == 4, ", ".join(sorted(p.stem for p in figs)))
    size = sum(p.stat().st_size for p in tmp.rglob("*") if p.is_file() and ".git" not in p.parts) / 2**20
    print(f"\n  release size (working files): {size:.0f} MB")
    print(f"\n{sum(ok)} of {len(ok)} checks passed")
    print("VERIFIED -- safe to push and release" if all(ok) else "NOT VERIFIED -- do not push; send me this output")

if __name__ == "__main__":
    ap = argparse.ArgumentParser(); sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("stage"); s.add_argument("--concept-doi", required=True)
    sub.add_parser("verify")
    args = ap.parse_args()
    stage(args.concept_doi) if args.cmd == "stage" else verify()
