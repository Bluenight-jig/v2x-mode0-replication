#!/usr/bin/env python3
"""
release_v2_0_1.py -- prepares release v2.0.1 of github.com/Bluenight-jig/v2x-mode0-replication. Run from ~/v2x_clean.

v2.0.1 corrects documentation and one figure; the code and every result are unchanged from v2.0:
  * RUN_REGISTER.csv gains a `role` column -- 88 reported, 11 supplementary, 15 diagnostic, 85 superseded, 1 excluded --
    computed here from the cell definitions in stage4a_v2.py; its era column no longer calls runs 111-200 "the runs reported".
  * README.md and REPRODUCE.md: the run-range sentences are replaced by the roles.
  * make_figures.py: Fig. 4 (fig7A) loses the wrong "capacity" label at M = 3; "within-pool" becomes "within-class";
    Table IV (T_mode0b) gains the fleet rows (Mode 0b safety and M1, all-Mode-0c M1); captions match the manuscript;
    PDFs no longer embed a creation date, so regenerating them is byte-stable.
  * CITATION.cff, .zenodo.json and RELEASE_NOTES.md: version 2.0.1.

  python3 release_v2_0_1.py        prepares and stages these files only. Nothing is committed or pushed.
"""
import csv, datetime, hashlib, io, json, pathlib, re, subprocess, sys

REPO = pathlib.Path(__file__).resolve().parent
VERSION, TODAY = "2.0.1", datetime.date.today().isoformat()
EXPECT_MAKE_FIGURES = "1befc9691ad48451"            # the v2.0 make_figures.py
UNCITED_CELLS = {"Mode 0a, N=4, paper critic", "Mode 0a + DS, N=10 (suppl.)", "Mode 0a + DS, N=4 (suppl.)"}
EXPECT_ROLES = {"reported": 88, "supplementary": 11, "diagnostic": 15, "superseded": 85, "excluded": 1}

def fail(msg): sys.exit("STOPPED: " + msg + "\nNothing has been staged.")
def sh(*cmd): return subprocess.run(cmd, cwd=REPO, capture_output=True, text=True)
def fp(path): return hashlib.sha256(path.read_bytes()).hexdigest()[:16]
def patch(text, old, new, what):
    n = text.count(old)
    if n != 1: fail(f"{what}: expected text found {n} times (file edited since v2.0?)")
    return text.replace(old, new, 1)

NEW_CAPTIONS = '# Figure captions (as in the revised manuscript)\n\n**Fig. 3.** (`fig7B_ladder`) The capability ladder at (a) N = 4 and (b) N = 10, the critical safety load. Filled markers: mean safety delivery; hollow markers: WIRC (m0_pdr_p05_intra); bars: 95% confidence intervals over seeds (n = 5 for SB-SPS and Mode 0c, n = 3 otherwise; * clipped at 1). Mode 0a uses the agent-specific critic, its best. Arm P draws subchannels at random with learned power; Arm S learns subchannels with power fixed at 23 dBm. The dashed line is the 0.95 requirement. Inset: the means nearest the requirement, zoomed — Mode 0c\'s interval includes 0.95. The panels use different vertical scales and are not compared with each other.\n\n**Fig. 4.** (`fig7A_separation`) Shared actors\' separation of safety vehicles against class-group size m0 (Mode 0a, agent-specific critic; three seeds per point). (a) Within-class collision rate of the safety vehicles, each seed shown; dashed ticks give each point\'s random-choice floor, 1 − (1 − 1/M)^(m0 − 1); diamonds show Mode 0c at N = 4 and 10. (b) Coordination time — episodes until safety delivery is sustained at ≥ 0.90 — median and range over the seeds that reached it; "+1 never" marks a further seed that did not reach it within the 5,000-episode budget. At N = 4 with M = 3, four vehicles share three subchannels, and the learner pairs one safety vehicle with an M1 vehicle in every interval rather than letting the two M1 vehicles share; safety delivery stays near 0.76 and never reaches the threshold, although the safety vehicles separate completely (Section 7.4). Two safety vehicles always separate; three usually separate, slowly; five or more do not within 5,000 episodes.\n\n**Fig. 5.** (`fig7C_partition`) The safety-priority partition at three operating points: (a) N = 4, (b) the critical load N = 10 with M = 5, (c) spare spectrum, N = 10 with M = 7. Top: safety delivery (filled: mean; hollow: WIRC (m0_pdr_p05_intra)) with the 0.95 requirement; bottom: M1 delivery. Circles: shared pool (Mode 0c); crosses: a partition reserving the indicated number of subchannels for safety traffic (pool 2 and pool 4 at the critical load; pool 5, sized to the safety demand by the RCU, at M = 7). Only with spare spectrum does a partition sized to the safety demand meet the requirement at both levels; at the critical load every partition loses to the shared pool.\n\n**Fig. 6.** (`fig7D_aoi`) Information age of safety messages at (a) N = 4 and (b) N = 10, Mode 0c against SB-SPS (five seeds each; 95% intervals). Age is counted in generation intervals (right axis: ms), from delivery events drawn with each interval\'s delivery probability; 1 means refreshed in the current interval. The peak is each episode\'s longest staleness, averaged over episodes. Information age measures staleness, not latency.\n'

FILES = ["RUN_REGISTER.csv", "README.md", "REPRODUCE.md", "CITATION.cff", ".zenodo.json", "RELEASE_NOTES.md", "make_figures.py"]

# ------------------------------------------------------------------------------------------------ checks
if sh("git", "branch", "--show-current").stdout.strip() != "main": fail("not on branch main")
if "v2.0" not in sh("git", "tag").stdout.split(): fail("tag v2.0 not found -- is this the released repository?")
dirty = [f for f in FILES if sh("git", "diff", "--quiet", "--", f).returncode]
if dirty: fail("uncommitted changes in " + ", ".join(dirty) + " -- commit or restore them first")
if fp(REPO / "make_figures.py") != EXPECT_MAKE_FIGURES: fail("make_figures.py is not the v2.0 version")

# ------------------------------------------------------------------------------------------------ 1. register roles
src = (REPO / "stage4a_v2.py").read_text(encoding="utf-8")
ns = {}; exec(compile(src[:src.index("S4B_GROUPS")], "stage4a_v2_head", "exec"), ns)
cells = {}
for g in ns["GROUPS"]:
    for lab in (g[2] if not isinstance(g[2], str) else [g[2]]):
        cells.setdefault(lab, []).append(g[1])
reported = {l for l, cs in cells.items() if any(c not in UNCITED_CELLS for c in cs) and not l.startswith("SBSPS")}
rows = list(csv.DictReader(open(REPO / "RUN_REGISTER.csv", encoding="utf-8")))
if len(rows) != 200: fail(f"RUN_REGISTER.csv has {len(rows)} rows, expected 200")
def era(n):
    if n == 1: return "G0 -- warm-up defect (defect 1)"
    if n <= 86: return "original program -- advantages computed across vehicles (defect 2)"
    if n <= 95: return "per-agent GAE -- impact check at the original settings"
    if n <= 110: return "per-agent GAE -- settings screen (batches 19-21)"
    return "per-agent GAE -- rebuild (batches 22-28)"
def role(n, lab):
    if n == 1: return "excluded"
    if n <= 86: return "superseded"
    if lab in reported: return "reported"
    if "_g05_ent01" in lab: return "supplementary"
    return "diagnostic"
cols = ["run", "label", "era", "role"] + [c for c in rows[0] if c not in ("run", "label", "era", "role")]
for r in rows: n = int(r["run"]); r["era"] = era(n); r["role"] = role(n, r["label"])
counts = {k: sum(r["role"] == k for r in rows) for k in EXPECT_ROLES}
if counts != EXPECT_ROLES: fail(f"role counts {counts} differ from the expected {EXPECT_ROLES}")
missing = reported - {r["label"] for r in rows}
if missing: fail(f"reported runs missing from the register: {sorted(missing)[:3]}")
buf = io.StringIO(); w = csv.DictWriter(buf, fieldnames=cols, lineterminator="\n"); w.writeheader(); w.writerows(rows)
(REPO / "RUN_REGISTER.csv").write_text(buf.getvalue(), encoding="utf-8")
print("RUN_REGISTER.csv  role column added:", ", ".join(f"{v} {k}" for k, v in counts.items()))

# ------------------------------------------------------------------------------------------------ 2. README, REPRODUCE
p = REPO / "README.md"; t = p.read_text(encoding="utf-8")
t = patch(t, "Reproducibility Package (v2.0.0)", f"Reproducibility Package (v{VERSION})", "README title")
t = patch(t, "**v2.0.0 accompanies the revised manuscript.**",
          f"**v{VERSION} accompanies the revised manuscript** (v2.0.0 had the same code and results; v{VERSION} corrects the run register's labels, this README and one figure label).", "README version line")
t = patch(t, """All 200 training runs are listed in `RUN_REGISTER.csv` with their era: run 1 (excluded, defect 1), runs 2–86
(original programme, superseded by defect 2), runs 87–110 (impact checks and diagnostics) and **runs 111–200 (the
rebuild, which the revised manuscript reports)**.""",
"""All 200 training runs are listed in `RUN_REGISTER.csv`, whose `role` column gives each run's place in the revised
manuscript: **88 reported** (every result in Section 7, all at the final settings and 5,000 episodes); 11 supplementary
(the same settings, not cited: five 3,000-episode runs of Mode 0c at N = 4, two extra seeds of Mode 0a with the
team-value critic at N = 4, and four runs of Mode 0a with a safety partition); 15 diagnostic (per-agent GAE at other
settings, used to choose the final ones); 85 superseded (defect 2); and 1 excluded (defect 1).""", "README run roles")
p.write_text(t, encoding="utf-8")
p = REPO / "REPRODUCE.md"; t = p.read_text(encoding="utf-8")
t = patch(t, "Runs 111–200 are the rebuild that the revised manuscript reports.",
          "The 88 runs whose `role` is `reported` are the ones behind every result in the revised manuscript.", "REPRODUCE run range")
p.write_text(t, encoding="utf-8")
print("README.md, REPRODUCE.md  run-range sentences replaced by roles")

# ------------------------------------------------------------------------------------------------ 3. metadata
p = REPO / "CITATION.cff"; t = p.read_text(encoding="utf-8")
t = patch(t, 'version: "2.0.0"', f'version: "{VERSION}"', "CITATION version")
t, k = re.subn(r'date-released: "[0-9-]+"', f'date-released: "{TODAY}"', t)
if k != 1: fail("CITATION.cff date-released line not found")
p.write_text(t, encoding="utf-8")
p = REPO / ".zenodo.json"; z = json.loads(p.read_text(encoding="utf-8"))
z["version"] = VERSION; z["title"] = re.sub(r"\(v[0-9.]+\)", f"(v{VERSION})", z["title"])
if f"v{VERSION}" not in z["title"]: fail(".zenodo.json title has no version to update")
z["description"] = z["description"].rstrip() + f" Release {VERSION} corrects the run register's role labels, the README and one figure label; code and results are unchanged from 2.0.0."
p.write_text(json.dumps(z, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
(REPO / "RELEASE_NOTES.md").write_text(f"""Release v{VERSION} corrects documentation and one figure; the code and every result are unchanged from v2.0.

- `RUN_REGISTER.csv` gains a `role` column: 88 reported (every result in Section 7 of the revised manuscript), 11 supplementary,
  15 diagnostic, 85 superseded, 1 excluded. v2.0's era column wrongly implied that runs 111-200 were the reported set.
- `README.md` and `REPRODUCE.md`: the run-range sentences now point to the roles.
- `make_figures.py`: Fig. 4 (`fig7A_separation`) no longer labels the M = 3 point "capacity" (the cap there comes from the
  learner's placement, not from capacity); "within-pool" is now "within-class"; Table IV (`T_mode0b`) adds the fleet-level
  rows; captions match the manuscript; PDFs no longer embed a creation date.
- Version metadata updated to {VERSION}.
""", encoding="utf-8")
print("CITATION.cff, .zenodo.json, RELEASE_NOTES.md  version", VERSION)

# ------------------------------------------------------------------------------------------------ 4. make_figures.py
p = REPO / "make_figures.py"; t = p.read_text(encoding="utf-8")
t = patch(t, 'fig.savefig(out / f"{name}.{ext}", bbox_inches="tight", pad_inches=0.02)',
          'fig.savefig(out / f"{name}.{ext}", bbox_inches="tight", pad_inches=0.02,\n                    metadata={"CreationDate": None} if ext == "pdf" else None)', "savefig")
t = patch(t, 'a.set_ylabel("Safety within-pool collision rate")', 'a.set_ylabel("Safety within-class collision rate")', "7-A axis label")
t = patch(t, 'txt = "capacity" if "M=3" in cell else f"+{never} never" if plotted else "never"',
          'txt = f"+{never}\\nnever" if plotted else "never"', "7-A M = 3 label")
t = patch(t, "across densities (7-A plots within-pool)", "across densities (7-A plots within-class collision)", "docstring")
t = patch(t, """                "Passive safety pair, within-pool & " + ", ".join(f"{r['passive_m0_within_pool']:.3f}" for r in recs) + " \\\\\\\\",
                "Passive M1 trio, within-pool (random 0.360) & " + ", ".join(f"{r['passive_m1_within_pool']:.3f}" for r in recs) + " \\\\\\\\",
                "\\\\midrule", f"All-Mode-0c fleet, safety (agent-specific) & {ci_tex(row('Mode 0c, N=10, agent-specific', 'M0 PDR'), 3)} \\\\\\\\",
                f"All-Mode-0a fleet, safety (agent-specific) & {ci_tex(row('Mode 0a, N=10, agent-specific', 'M0 PDR'), 3)} \\\\\\\\"]""",
"""                "Passive safety pair, within-class collision & " + ", ".join(f"{r['passive_m0_within_pool']:.3f}" for r in recs) + " \\\\\\\\",
                "Passive M1 trio, within-class collision (random 0.360) & " + ", ".join(f"{r['passive_m1_within_pool']:.3f}" for r in recs) + " \\\\\\\\",
                "\\\\midrule", f"Mode 0b fleet, safety & {ci_tex(row('Mode 0b, N=10', 'M0 PDR'))} \\\\\\\\",
                f"Mode 0b fleet, M1 & {ci_tex(row('Mode 0b, N=10', 'M1 PDR'))} \\\\\\\\",
                f"All-Mode-0c fleet, safety (agent-specific) & {ci_tex(row('Mode 0c, N=10, agent-specific', 'M0 PDR'))} \\\\\\\\",
                f"All-Mode-0c fleet, M1 (agent-specific) & {ci_tex(row('Mode 0c, N=10, agent-specific', 'M1 PDR'))} \\\\\\\\",
                f"All-Mode-0a fleet, safety (agent-specific) & {ci_tex(row('Mode 0a, N=10, agent-specific', 'M0 PDR'))} \\\\\\\\"]""", "Table IV rows")
t = patch(t, '"Vehicles & Delivery [95\\\\% CI] or within-pool by seed"', '"Vehicles & Delivery [95\\\\% CI], or within-class collision by seed"', "Table IV header")
i, j = t.index('CAPTIONS = """'), t.index('"""\n\ndef main')
t = t[:i] + 'CAPTIONS = """' + NEW_CAPTIONS + t[j:]
p.write_text(t, encoding="utf-8")
print("make_figures.py  patched (fingerprint " + fp(p) + ")")

# ------------------------------------------------------------------------------------------------ 5. regenerate, check, stage
r = sh(sys.executable, "make_figures.py")
if r.returncode or "All inputs found." not in r.stdout: fail("make_figures.py did not finish cleanly:\n" + r.stdout[-800:] + r.stderr[-800:])
tb = (REPO / "tables" / "T_mode0b.tex").read_text()
if "Mode 0b fleet, M1" not in tb or "within-class collision" not in tb: fail("Table IV was not regenerated with the fleet rows")
if "capacity" in (REPO / "figures" / "captions.md").read_text(): fail("captions still mention capacity")
stage = FILES + ["release_v2_0_1.py"] + [str(q.relative_to(REPO)) for q in sorted((REPO / "figures").glob("fig7*")) + [REPO / "figures" / "captions.md"] + sorted((REPO / "tables").glob("*.tex"))]
r = sh("git", "add", "--", *stage)
if r.returncode: fail(r.stderr)
print("\nStaged for v" + VERSION + ":")
print(sh("git", "diff", "--cached", "--stat").stdout)
print("Table IV fleet rows:")
for line in tb.splitlines():
    if "fleet" in line: print("   ", line.replace("\\\\", "").strip())
print("\nNext: commit, then `python3 release_v2.py verify` (see the instructions).")
