#!/usr/bin/env python3
"""
make_figures.py -- Section 7's figures and tables from the rebuild (aa-achievement.md §4ab; coverage review §8a, §8d).
Run from ~/v2x_clean:   python3 make_figures.py          (read-only; writes figures/ and tables/)

Inputs:  results/_ci_summary_v2.json (Stage 4a v2 -- every interval) and results/stage4b/_<label>.json (Mode 0b by type).
Outputs: figures/fig7A_separation.{pdf,png}   shared actors' separation against class-group size (F25)
         figures/fig7B_ladder.{pdf,png}       the capability ladder at N = 4 and N = 10 (F15, F21, F23)
         figures/fig7C_partition.{pdf,png}    the safety-priority partition at three operating points (F9, F22)
         figures/fig7D_aoi.{pdf,png}          information age, Mode 0c against SB-SPS (F26)
         tables/T1_critic.tex, tables/T_mode0b.tex, tables/T_requirement.tex     (booktabs)
         figures/captions.md                  draft captions in the C&L terminology
Style rules (coverage review §8d): 95% intervals on every n >= 3 point; n = 2 as the two seeds; clipped intervals
marked *; the 0.95 requirement line on safety panels only; WIRC labelled "WIRC (m0_pdr_p05_intra)"; never WIRC
across densities (7-A plots within-class collision); bimodal cells shown split; "generation interval", never "TTI".
Format: Elsevier double-column width (190 mm), vector PDF with embedded TrueType fonts, >= 7 pt text,
Okabe-Ito colours plus a distinct marker per scheme so every figure reads in greyscale.
"""
import json, pathlib, statistics as st, sys
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

REQ = 0.95
W2 = 190 / 25.4                                           # double column, inches
plt.rcParams.update({"font.family": "sans-serif", "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
                     "font.size": 8, "axes.titlesize": 8, "axes.labelsize": 8, "xtick.labelsize": 7, "ytick.labelsize": 7,
                     "legend.fontsize": 7, "pdf.fonttype": 42, "ps.fonttype": 42, "axes.linewidth": 0.6,
                     "xtick.major.width": 0.6, "ytick.major.width": 0.6, "savefig.dpi": 300})
STYLE = {"SB-SPS": ("#D55E00", "s"), "Mode 0a": ("#555555", "v"), "Arm P": ("#E69F00", "^"), "Arm S": ("#56B4E9", "D"),
         "Mode 0c": ("#0072B2", "o"), "Mode 0b": ("#009E73", "P"), "Partition": ("#CC79A7", "X")}
WIRC_LABEL = "WIRC (m0_pdr_p05_intra)"

# ------------------------------------------------------------------------------------------- data access
ROWS = json.loads(pathlib.Path("results/_ci_summary_v2.json").read_text())
IDX = {(r["cell"], r["metric"]): r for r in ROWS}
MISSING = []
def row(cell, metric):
    r = IDX.get((cell, metric))
    if r is None: MISSING.append(f"{cell} / {metric}")
    return r

def point(ax, x, r, color, marker, filled=True, size=4.5, label=None):
    """Mean with its 95% interval (n >= 3), or the two seeds (n = 2); a clipped interval is marked *."""
    if r is None: return
    mfc = color if filled else "white"
    if r["n"] >= 3 and r["ci95_low"] is not None:
        lo, hi = r["mean"] - r["ci95_low"], r["ci95_high"] - r["mean"]
        ax.errorbar([x], [r["mean"]], yerr=[[lo], [hi]], fmt=marker, color=color, mfc=mfc, mec=color, ms=size,
                    mew=0.9, elinewidth=0.8, capsize=2, label=label, zorder=3)
        if r["clipped"]:
            ax.annotate("*", (x, r["ci95_high"]), xytext=(2.5, 1), textcoords="offset points", fontsize=7, color=color)
    else:
        ax.plot([x] * len(r["seeds"]), r["seeds"], marker, color=color, mfc=mfc, mec=color, ms=size, mew=0.9,
                ls="none", label=label, zorder=3)

def req_line(ax, label=True):
    ax.axhline(REQ, ls=(0, (4, 2)), lw=0.8, color="#333333", zorder=1)
    if label: ax.text(1.0, REQ, " 0.95", transform=ax.get_yaxis_transform(), va="center", ha="left", fontsize=6.5, color="#333333")

def save(fig, name):
    out = pathlib.Path("figures"); out.mkdir(exist_ok=True)
    for ext in ("pdf", "png"): fig.savefig(out / f"{name}.{ext}", bbox_inches="tight", pad_inches=0.02,
                    metadata={"CreationDate": None} if ext == "pdf" else None)
    plt.close(fig); print(f"  wrote figures/{name}.pdf and .png")

# ------------------------------------------------------------------------------------------- 7-A
def fig7A():
    pts = [("N=4", "N=4, m0=2", 2, 5), ("N=5", "N=5, m0=2", 2, 5), ("N=4\nM=4", "N=4, M=4, m0=2", 2, 4),
           ("N=4\nM=3", "N=4, M=3, m0=2", 2, 3), ("N=6", "N=6, m0=3", 3, 5), ("N=7", "N=7, m0=3", 3, 5),
           ("N=10", "N=10, m0=5", 5, 5), ("N=15", "N=15, m0=7", 7, 5), ("N=20", "N=20, m0=10", 10, 5)]
    fig, (a, b) = plt.subplots(1, 2, figsize=(W2, 2.55), gridspec_kw={"width_ratios": [1.15, 1]})
    c0a, m0a = STYLE["Mode 0a"]
    for i, (lab, cell, m0, M) in enumerate(pts):
        if cell.startswith("N=7"):
            seeds = (row(cell + ": separated", "M0 wp") or {"seeds": []})["seeds"] + (row(cell + ": not separated", "M0 wp") or {"seeds": []})["seeds"]
        else:
            seeds = (row(cell, "M0 wp") or {"seeds": []})["seeds"]
        jit = [(-0.12, 0.0, 0.12)[k % 3] for k in range(len(seeds))]
        a.plot([i + j for j in jit], seeds, m0a, color=c0a, mfc="white", ms=4.2, mew=0.9, ls="none", zorder=3)
        f = 1 - (1 - 1 / M) ** (m0 - 1)
        a.plot([i - 0.3, i + 0.3], [f, f], color="#999999", lw=1.0, ls=(0, (2, 1.5)), zorder=2)
    for x, cell in ((0, "Mode 0c, N=4"), (6, "Mode 0c, N=10")):
        r = row(cell, "M0 wp")
        if r: a.plot([x + 0.32], [r["mean"]], STYLE["Mode 0c"][1], color=STYLE["Mode 0c"][0], ms=4.2, zorder=4)
    a.set_xticks(range(len(pts))); a.set_xticklabels([p[0] for p in pts])
    a.set_ylabel("Safety within-class collision rate"); a.set_ylim(-0.04, 1.0); a.set_xlim(-0.6, len(pts) - 0.4)
    for lo, hi, txt in ((0, 3, "m$_0$ = 2"), (4, 5, "3"), (6, 6, "5"), (7, 7, "7"), (8, 8, "10")):
        a.annotate("", xy=(lo - 0.3, -0.24), xytext=(hi + 0.3, -0.24), xycoords=("data", "axes fraction"),
                   arrowprops=dict(arrowstyle="-", lw=0.6, color="#333333"), annotation_clip=False)
        a.text((lo + hi) / 2, -0.30, txt, transform=a.get_xaxis_transform(), ha="center", va="top", fontsize=7)
    a.legend(handles=[Line2D([], [], marker=m0a, color=c0a, mfc="white", ls="none", ms=4.2, label="Mode 0a, each seed"),
                      Line2D([], [], marker=STYLE["Mode 0c"][1], color=STYLE["Mode 0c"][0], ls="none", ms=4.2, label="Mode 0c (mean)"),
                      Line2D([], [], color="#999999", ls=(0, (2, 1.5)), label="random-choice floor")],
             loc="upper left", frameon=False, handletextpad=0.3)
    a.set_title("(a) Separation of safety vehicles on one shared actor", loc="left")
    for i, (lab, cell, m0, M) in enumerate(pts):
        cells = [cell + ": separated", cell + ": not separated"] if cell.startswith("N=7") else [cell]
        never = 0; plotted = False
        for c in cells:
            r = row(c, "coordination time")
            if not r: continue
            d = r["seeds"]["m0"]; never += d["never"]
            if d["median"] is not None:
                b.errorbar([i], [d["median"]], yerr=[[d["median"] - d["min"]], [d["max"] - d["median"]]], fmt=m0a, color=c0a,
                           mfc=c0a, ms=4.2, elinewidth=0.8, capsize=2, zorder=3); plotted = True
        if never:
            txt = f"+{never}\nnever" if plotted else "never"
            b.text(i, 6300, txt, ha="center", va="bottom", fontsize=6.3, color="#333333")
    b.set_yscale("log"); b.set_ylim(40, 14000); b.set_xlim(-0.6, len(pts) - 0.4)
    b.set_xticks(range(len(pts))); b.set_xticklabels([p[0] for p in pts])
    b.axhline(5000, color="#bbbbbb", lw=0.6, zorder=1)
    b.text(-0.5, 4700, "training budget (5,000)", fontsize=6.3, va="top", ha="left", color="#777777")
    b.set_ylabel("Coordination time (episodes)")
    b.set_title("(b) Episodes until safety delivery is sustained at ≥ 0.90", loc="left")
    fig.tight_layout(w_pad=1.2); save(fig, "fig7A_separation")

# ------------------------------------------------------------------------------------------- 7-B
def fig7B():
    fig, axes = plt.subplots(1, 2, figsize=(W2, 2.45))
    ladders = [("N = 4 (λ$_0$ = 0.4)", [("SB-SPS", "SB-SPS, N=4"), ("Mode 0a", "Mode 0a, N=4, agent-specific"),
                                        ("Arm S", "Arm S, N=4"), ("Mode 0c", "Mode 0c, N=4")], (0.80, 1.012)),
               ("N = 10 (λ$_0$ = 1, the critical load)", [("SB-SPS", "SB-SPS, N=10"), ("Mode 0a", "Mode 0a, N=10, agent-specific"),
                                        ("Arm P", "Arm P, N=10"), ("Arm S", "Arm S, N=10"), ("Mode 0c", "Mode 0c, N=10")], (0.15, 1.02))]
    for k, (ax, (title, schemes, ylim)) in enumerate(zip(axes, ladders)):
        for i, (name, cell) in enumerate(schemes):
            c, m = STYLE[name]
            point(ax, i - 0.13, row(cell, "M0 PDR"), c, m, filled=True)
            point(ax, i + 0.13, row(cell, "WIRC"), c, m, filled=False)
        req_line(ax); ax.set_ylim(*ylim); ax.set_xlim(-0.5, len(schemes) - 0.5)
        ax.set_xticks(range(len(schemes))); ax.set_xticklabels([s[0] for s in schemes])
        ax.set_title(f"({'ab'[k]}) {title}", loc="left"); ax.set_ylabel("Safety delivery (PDR)")
        if k == 0:
            ax.set_yticks([0.80, 0.85, 0.90, 0.95, 1.00])
        else:                                   # the means nearest the requirement, zoomed: Mode 0c's interval straddles 0.95
            ins = ax.inset_axes([0.21, 0.15, 0.28, 0.25])
            for j, (name, cell) in enumerate([s_ for s_ in schemes if s_[0] in ("SB-SPS", "Arm S", "Mode 0c")]):
                c, m = STYLE[name]; point(ins, j, row(cell, "M0 PDR"), c, m, filled=True, size=3.8)
            ins.axhline(REQ, ls=(0, (4, 2)), lw=0.7, color="#333333")
            ins.set_ylim(0.88, 0.975); ins.set_xlim(-0.6, 2.6); ins.set_xticks([]); ins.set_yticks([0.90, 0.95])
            ins.yaxis.tick_right(); ins.tick_params(labelsize=6, length=2); ins.set_title("means, zoomed", fontsize=6, pad=1.5)
    axes[0].legend(handles=[Line2D([], [], marker="o", color="#333333", ls="none", ms=4.5, label="mean"),
                            Line2D([], [], marker="o", color="#333333", mfc="white", ls="none", ms=4.5, label=WIRC_LABEL),
                            Line2D([], [], color="#333333", ls=(0, (4, 2)), lw=0.8, label="requirement 0.95")],
                   loc="lower right", frameon=False, handletextpad=0.3)
    fig.tight_layout(w_pad=1.6); save(fig, "fig7B_ladder")

# ------------------------------------------------------------------------------------------- 7-C
def fig7C():
    cols = [("N = 4, M = 5 (λ$_0$ = 0.4)", [("shared", "Mode 0c, N=4"), ("pool 2", "N=4, partition (pool 2)")]),
            ("N = 10, M = 5 (λ$_0$ = 1)", [("shared", "Mode 0c, N=10"), ("pool 2", "N=10, partition (pool 2)"),
                                           ("pool 4", "N=10, partition (pool 4)")]),
            ("N = 10, M = 7 (λ$_0$ ≈ 0.71)", [("shared", "M=7, shared pool"), ("pool 5", "M=7, partition (pool 5)")])]
    fig, axes = plt.subplots(2, 3, figsize=(W2, 3.5), gridspec_kw={"width_ratios": [2, 3, 2], "height_ratios": [1.5, 1]}, sharey="row")
    for j, (title, cats) in enumerate(cols):
        top, bot = axes[0, j], axes[1, j]
        for i, (lab, cell) in enumerate(cats):
            c, m = STYLE["Mode 0c"] if lab == "shared" else STYLE["Partition"]
            point(top, i - 0.13, row(cell, "M0 PDR"), c, m, True); point(top, i + 0.13, row(cell, "WIRC"), c, m, False)
            point(bot, i, row(cell, "M1 PDR"), c, m, True)
        req_line(top, label=(j == 2))
        for ax in (top, bot):
            ax.set_xlim(-0.5, len(cats) - 0.5); ax.set_xticks(range(len(cats)))
        top.set_xticklabels([]); bot.set_xticklabels([c[0] for c in cats])
        top.set_title(f"({'abc'[j]}) {title}", loc="left")
    axes[0, 0].set_ylim(0.15, 1.03); axes[1, 0].set_ylim(-0.03, 1.05)
    axes[0, 0].set_ylabel("Safety delivery (PDR)"); axes[1, 0].set_ylabel("M1 delivery (PDR)")
    axes[0, 0].legend(handles=[Line2D([], [], marker="o", color=STYLE["Mode 0c"][0], ls="none", ms=4.5, label="shared pool (Mode 0c)"),
                               Line2D([], [], marker="X", color=STYLE["Partition"][0], ls="none", ms=4.5, label="partition (subchannels reserved for safety)"),
                               Line2D([], [], marker="o", color="#333333", mfc="white", ls="none", ms=4.5, label="hollow: " + WIRC_LABEL)],
                      loc="lower left", frameon=False, handletextpad=0.3)
    fig.tight_layout(h_pad=0.6, w_pad=0.8); save(fig, "fig7C_partition")

# ------------------------------------------------------------------------------------------- 7-D
def fig7D():
    metrics = [("mean", "AoI mean"), ("95th pct.", "AoI p95"), ("peak", "AoI peak"), ("worst\nvehicle's mean", "AoI worst vehicle")]
    fig, axes = plt.subplots(1, 2, figsize=(W2, 2.4), sharey=True)
    for k, (ax, N) in enumerate(zip(axes, (4, 10))):
        for i, (lab, metric) in enumerate(metrics):
            for dx, name in ((-0.13, "Mode 0c"), (0.13, "SB-SPS")):
                c, m = STYLE[name]; point(ax, i + dx, row(f"{name}, N={N}", metric), c, m, True)
        ax.set_xticks(range(len(metrics))); ax.set_xticklabels([m[0] for m in metrics]); ax.set_xlim(-0.5, len(metrics) - 0.5)
        ax.set_title(f"({'ab'[k]}) N = {N}" + (" (λ$_0$ = 1)" if N == 10 else " (λ$_0$ = 0.4)"), loc="left")
    axes[0].set_ylabel("Information age\n(generation intervals)"); axes[0].set_ylim(0.8, 9.5)
    sec = axes[1].secondary_yaxis("right", functions=(lambda v: v * 100, lambda v: v / 100)); sec.set_ylabel("Age (ms)")
    axes[0].legend(handles=[Line2D([], [], marker=STYLE[n][1], color=STYLE[n][0], ls="none", ms=4.5, label=n) for n in ("Mode 0c", "SB-SPS")],
                   loc="upper left", frameon=False, handletextpad=0.3)
    fig.tight_layout(w_pad=1.0); save(fig, "fig7D_aoi")

# ------------------------------------------------------------------------------------------- tables
def ci_tex(r, nd=4):
    if r is None: return "--"
    if r["n"] >= 3 and r["ci95_low"] is not None:
        return f"{r['mean']:.{nd}f} [{r['ci95_low']:.{nd}f}, {r['ci95_high']:.{nd}f}]" + ("$^{*}$" if r["clipped"] else "")
    return f"{r['mean']:.{nd}f} (seeds " + ", ".join(f"{v:.{nd}f}" for v in r["seeds"]) + ")"
def ct_tex(r, cls):
    if r is None: return "--"
    d = r["seeds"][cls]
    if d["median"] is None: return "never"
    s = f"{d['median']:g} [{d['min']:g}--{d['max']:g}]"
    return s + (f", never in {d['never']} of {d['n']}" if d["never"] else "")
def table(name, header, body, caption, label, colspec):
    out = pathlib.Path("tables"); out.mkdir(exist_ok=True)
    tex = ["\\begin{table}[t]", "\\centering", f"\\caption{{{caption}}}", f"\\label{{{label}}}", "\\footnotesize",
           f"\\begin{{tabular}}{{{colspec}}}", "\\toprule", header + " \\\\", "\\midrule"] + body + ["\\bottomrule", "\\end{tabular}", "\\end{table}"]
    (out / f"{name}.tex").write_text("\n".join(tex) + "\n"); print(f"  wrote tables/{name}.tex")

def tables():
    body = []
    for group, cells in (("Mode 0c, $N = 10$", [("team value $V(\\mathbf{s})$", "Mode 0c, N=10, team critic"), ("local $V(o_i)$", "Mode 0c, N=10, local critic"),
                                               ("agent-specific $V(\\mathbf{s}\\oplus o_i)$", "Mode 0c, N=10, agent-specific")]),
                         ("Mode 0c, $N = 4$", [("team value", "Mode 0c, N=4, team critic"), ("local", "Mode 0c, N=4, local critic"), ("agent-specific", "Mode 0c, N=4, agent-specific")]),
                         ("Mode 0a, $N = 4$", [("team value", "Mode 0a, N=4, team critic"), ("local", "Mode 0a, N=4, local critic"), ("agent-specific", "Mode 0a, N=4, agent-specific")]),
                         ("Mode 0c, $N = 10$, $\\delta = 1.0$", [("team value", "Mode 0c, N=10, delta=1.0")])):
        body.append(f"\\multicolumn{{5}}{{l}}{{\\textit{{{group}}}}} \\\\")
        for crit, cell in cells:
            c = row(cell, "coordination time")
            body.append(f"\\quad {crit} & {ci_tex(row(cell, 'M0 PDR'))} & {ci_tex(row(cell, 'M1 PDR'))} & {ct_tex(c, 'm0')} & {ct_tex(c, 'm1')} \\\\")
    table("T1_critic", "Critic & M0 PDR [95\\% CI] & M1 PDR [95\\% CI] & \\multicolumn{2}{c}{Coordination time (episodes)} \\\\ \\cmidrule(lr){4-5} & & & M0 & M1",
          body, "Critic-input ablation (three seeds per cell) and the reward-weight test ($\\delta$). Coordination time: episodes until delivery is sustained at $\\geq 0.90$, median [range]. $^{*}$ interval clipped at 1.",
          "tab:critic", "lllll")
    from statistics import mean
    recs = [json.loads(p.read_text())["own_window"]["mode0b"] for p in sorted(pathlib.Path("results/stage4b").glob("_A_mode0b_N10_seed*_gaefix_g05_ent01.json"))]
    if len(recs) == 3:
        import math
        T = 4.303
        def iv(key):
            xs = [r[key] for r in recs]; m = mean(xs); s = st.stdev(xs); h = T * s / math.sqrt(3)
            lo, hi = max(0.0, m - h), min(1.0, m + h); clip = (m - h < 0) or (m + h > 1)
            return f"{m:.3f} [{lo:.3f}, {hi:.3f}]" + ("$^{*}$" if clip else "")
        body = [f"Safety, active (3 vehicles) & {iv('active_m0')} \\\\", f"Safety, passive (2, one shared actor) & {iv('passive_m0')} \\\\",
                f"M1, active (2 vehicles) & {iv('active_m1')} \\\\", f"M1, passive (3, one shared actor) & {iv('passive_m1')} \\\\", "\\midrule",
                "Passive safety pair, within-class collision & " + ", ".join(f"{r['passive_m0_within_pool']:.3f}" for r in recs) + " \\\\",
                "Passive M1 trio, within-class collision (random 0.360) & " + ", ".join(f"{r['passive_m1_within_pool']:.3f}" for r in recs) + " \\\\",
                "\\midrule", f"Mode 0b fleet, safety & {ci_tex(row('Mode 0b, N=10', 'M0 PDR'))} \\\\",
                f"Mode 0b fleet, M1 & {ci_tex(row('Mode 0b, N=10', 'M1 PDR'))} \\\\",
                f"All-Mode-0c fleet, safety (agent-specific) & {ci_tex(row('Mode 0c, N=10, agent-specific', 'M0 PDR'))} \\\\",
                f"All-Mode-0c fleet, M1 (agent-specific) & {ci_tex(row('Mode 0c, N=10, agent-specific', 'M1 PDR'))} \\\\",
                f"All-Mode-0a fleet, safety (agent-specific) & {ci_tex(row('Mode 0a, N=10, agent-specific', 'M0 PDR'))} \\\\"]
        table("T_mode0b", "Vehicles & Delivery [95\\% CI], or within-class collision by seed", body,
              "Mode 0b at $N = 10$ by vehicle type (three seeds; agent-specific critic). $^{*}$ interval clipped at 1.", "tab:mode0b", "ll")
    else:
        MISSING.append("results/stage4b/_A_mode0b_N10_seed{1,2,3}_gaefix_g05_ent01.json (Mode 0b by type)")
    req = [("$N = 4$", "Mode 0c", "Mode 0c, N=4"), ("", "SB-SPS", "SB-SPS, N=4"), ("$N = 10$", "Mode 0c", "Mode 0c, N=10"),
           ("", "Mode 0b", "Mode 0b, N=10"), ("", "Arm S", "Arm S, N=10"), ("", "SB-SPS", "SB-SPS, N=10"),
           ("", "Mode 0a (agent-specific)", "Mode 0a, N=10, agent-specific"), ("", "partition, pool 4", "N=10, partition (pool 4)"),
           ("$N = 10$, $M = 7$", "shared pool", "M=7, shared pool"), ("", "partition, pool 5 (RCU-sized)", "M=7, partition (pool 5)")]
    body = []
    for op, name, cell in req:
        a, b_ = row(cell, "M0 PDR"), row(cell, "WIRC")
        v = lambda r: (r or {}).get("requirement_095", "--").replace("MEETS", "meets").replace("SHORT", "short").replace("STRADDLES", "straddles")
        body.append(f"{op} & {name} & {ci_tex(a)} & {v(a)} & {ci_tex(b_)} & {v(b_)} \\\\")
    table("T_requirement", "Operating point & Scheme & Mean M0 PDR [95\\% CI] & Verdict & WIRC [95\\% CI] & Verdict", body,
          "Safety delivery against the derived requirement (PDR $\\geq 0.95$), judged on the 95\\% interval: \\textit{meets} (interval wholly above), \\textit{short} (wholly below), \\textit{straddles}. Each operating point is judged separately; WIRC is not compared across densities.",
          "tab:requirement", "llllll")

CAPTIONS = """# Figure captions (as in the revised manuscript)

**Fig. 3.** (`fig7B_ladder`) The capability ladder at (a) N = 4 and (b) N = 10, the critical safety load. Filled markers: mean safety delivery; hollow markers: WIRC (m0_pdr_p05_intra); bars: 95% confidence intervals over seeds (n = 5 for SB-SPS and Mode 0c, n = 3 otherwise; * clipped at 1). Mode 0a uses the agent-specific critic, its best. Arm P draws subchannels at random with learned power; Arm S learns subchannels with power fixed at 23 dBm. The dashed line is the 0.95 requirement. Inset: the means nearest the requirement, zoomed — Mode 0c's interval includes 0.95. The panels use different vertical scales and are not compared with each other.

**Fig. 4.** (`fig7A_separation`) Shared actors' separation of safety vehicles against class-group size m0 (Mode 0a, agent-specific critic; three seeds per point). (a) Within-class collision rate of the safety vehicles, each seed shown; dashed ticks give each point's random-choice floor, 1 − (1 − 1/M)^(m0 − 1); diamonds show Mode 0c at N = 4 and 10. (b) Coordination time — episodes until safety delivery is sustained at ≥ 0.90 — median and range over the seeds that reached it; "+1 never" marks a further seed that did not reach it within the 5,000-episode budget. At N = 4 with M = 3, four vehicles share three subchannels, and the learner pairs one safety vehicle with an M1 vehicle in every interval rather than letting the two M1 vehicles share; safety delivery stays near 0.76 and never reaches the threshold, although the safety vehicles separate completely (Section 7.4). Two safety vehicles always separate; three usually separate, slowly; five or more do not within 5,000 episodes.

**Fig. 5.** (`fig7C_partition`) The safety-priority partition at three operating points: (a) N = 4, (b) the critical load N = 10 with M = 5, (c) spare spectrum, N = 10 with M = 7. Top: safety delivery (filled: mean; hollow: WIRC (m0_pdr_p05_intra)) with the 0.95 requirement; bottom: M1 delivery. Circles: shared pool (Mode 0c); crosses: a partition reserving the indicated number of subchannels for safety traffic (pool 2 and pool 4 at the critical load; pool 5, sized to the safety demand by the RCU, at M = 7). Only with spare spectrum does a partition sized to the safety demand meet the requirement at both levels; at the critical load every partition loses to the shared pool.

**Fig. 6.** (`fig7D_aoi`) Information age of safety messages at (a) N = 4 and (b) N = 10, Mode 0c against SB-SPS (five seeds each; 95% intervals). Age is counted in generation intervals (right axis: ms), from delivery events drawn with each interval's delivery probability; 1 means refreshed in the current interval. The peak is each episode's longest staleness, averaged over episodes. Information age measures staleness, not latency.
"""

def main():
    print("Section 7 figures and tables from results/_ci_summary_v2.json\n")
    fig7A(); fig7B(); fig7C(); fig7D(); tables()
    pathlib.Path("figures").mkdir(exist_ok=True); (pathlib.Path("figures") / "captions.md").write_text(CAPTIONS)
    print("  wrote figures/captions.md")
    if MISSING:
        print(f"\n⚠ {len(MISSING)} input(s) not found -- those points are absent from the output:\n  " + "\n  ".join(sorted(set(MISSING))))
        sys.exit(1)
    print("\nAll inputs found.")

if __name__ == "__main__":
    main()
