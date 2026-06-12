#!/usr/bin/env python3
"""Figure 3: per-game stacked verification bars for the schedule sweep.
Rows = discount-erasure snapshots (Opus, Sonnet); columns = the four multi-failure
schedules (2-strike, 3-strike, mid2strike, recur). One stacked bar per game:
Q4-verify (targeting D, dark) + non-Q4 (pale); total height = all-verify/game.
Dashed line = memoryless i.i.d. all-verify anchor. y-axis shared within a row.
"""
import csv, json, os, glob
from collections import defaultdict
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.rcParams.update({"font.size":13,"axes.titlesize":14,"axes.labelsize":13,"xtick.labelsize":12,"ytick.labelsize":12,"legend.fontsize":12,"figure.titlesize":15})

SD = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(SD, "..", "results")
IMG = os.path.join(SD, "figures"); os.makedirs(IMG, exist_ok=True)
SKIP = ("APIERROR", "CONTAMINATED", "to_be_deleted", "default_experiment", "SUPERSEDED")
MODELS = ["Opus", "Sonnet", "GPT-5.1", "GPT-5.4-mini", "Gemini Pro", "Gemini Flash"]
SCHED = ["2strike", "3strike", "mid2strike", "recur"]
SLAB = {"2strike": "2-strike (clustered)", "3strike": "3-strike (clustered)",
        "mid2strike": "mid2strike (clustered mid)", "recur": "recur (spread)"}
# games (1-based) in which D gave a wrong answer, per schedule
FAIL = {"2strike": {1, 2}, "3strike": {1, 2, 3}, "mid2strike": {4, 5}, "recur": {1, 5}}

def clean(pm):
    pm = pm.split(":")[-1].split("/")[-1]
    return {"claude-opus-4-6": "Opus", "claude-sonnet-4-6": "Sonnet",
            "gemini-3.1-pro-preview": "Gemini Pro", "gemini-2.5-flash": "Gemini Flash",
            "gpt-5.4-mini-2026-03-17": "GPT-5.4-mini", "gpt-5.1": "GPT-5.1"}.get(pm, pm)

def classify(f):
    try: s = json.load(open(os.path.join(f, "experiment_settings.json")))
    except Exception: return None
    model = clean(s.get("player_models", ["?"])[0])
    mc = s.get("memory_config", {}).get("A", {})
    ml = (mc.get("persist_across_games") is False) or (mc.get("mode") == "current_game_only")
    dummy = s.get("dummy_config", {}).get("D", {}).get("correctness_list", [])
    nw = sum(1 for x in dummy if not x)
    if ml: cond = "iid_dcorrect" if nw == 0 else "iid_dwrong"
    elif nw == 0: cond = "smooth"
    elif nw == 1 and dummy and not dummy[0]: cond = "1strike"
    elif nw == 2 and dummy[:2] == [False, False]: cond = "2strike"
    elif nw == 3 and dummy[:3] == [False, False, False]: cond = "3strike"
    elif dummy == [True, True, True, False, False, True, True, True, True, True, True]: cond = "mid2strike"
    elif dummy == [False, True, True, True, False, True, True, True, True, True, True]: cond = "recur"
    else: cond = "other"
    return model, cond

acc = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
ANCH = defaultdict(lambda: {"dc_all": [], "dc_q4": [], "dw_all": [], "dw_q4": []})
for f in sorted(glob.glob(os.path.join(RES, "*/"))):
    if os.path.basename(f.rstrip("/")).startswith(SKIP):
        continue
    c = classify(f)
    if not c: continue
    model, cond = c
    if model not in MODELS or cond not in SCHED + ["iid_dcorrect", "iid_dwrong"]: continue
    p = os.path.join(f, "game_data_agent_decisions.csv")
    if not os.path.exists(p): continue
    rows = list(csv.DictReader(open(p)))
    if not rows: continue
    gids = sorted(set(r["game_id"] for r in rows)); idx = {g: i + 1 for i, g in enumerate(gids)}
    q4 = defaultdict(int); nq = defaultdict(int)
    for r in rows:
        d = r.get("agent_decision", "")
        if d.startswith("Verify:"):
            g = idx[r["game_id"]]
            if "Q4" in d: q4[g] += 1
            else: nq[g] += 1
    if cond == "iid_dcorrect":
        for g in idx.values():
            ANCH[model]["dc_q4"].append(q4.get(g, 0)); ANCH[model]["dc_all"].append(q4.get(g, 0) + nq.get(g, 0))
    elif cond == "iid_dwrong":
        for g in idx.values():
            ANCH[model]["dw_q4"].append(q4.get(g, 0)); ANCH[model]["dw_all"].append(q4.get(g, 0) + nq.get(g, 0))
    else:
        for g in idx.values():
            acc[model][cond][g].append((q4.get(g, 0), nq.get(g, 0)))

def mean_qn(model, cond):
    gd = acc[model][cond]; gs = sorted(gd)
    return gs, [np.mean([t[0] for t in gd[g]]) for g in gs], [np.mean([t[1] for t in gd[g]]) for g in gs]

DQ, DN = "#5b2c83", "#cdbbe0"
ANCH_LINES = [("dc_all", "#1f3a93", "--", "i.i.d. D-correct: all-verify"),
              ("dc_q4",  "#1f3a93", ":",  "i.i.d. D-correct: Q4-verify"),
              ("dw_all", "#cc6600", "--", "i.i.d. D-wrong: all-verify"),
              ("dw_q4",  "#cc6600", ":",  "i.i.d. D-wrong: Q4-verify")]
fig, axes = plt.subplots(len(MODELS), len(SCHED), figsize=(3.4 * len(SCHED), 1.95 * len(MODELS)),
                         squeeze=False, sharey="row")
for r, model in enumerate(MODELS):
    for cidx, cond in enumerate(SCHED):
        ax = axes[r][cidx]
        # shade the games where D gave a wrong answer (the failure schedule)
        for fg in sorted(FAIL[cond]):
            ax.axvspan(fg - 0.5, fg + 0.5, color="#e74c3c", alpha=0.16, zorder=0,
                       label="D gave wrong answer")
        gs, mq, mn = mean_qn(model, cond)
        if gs:
            ax.bar(gs, mq, 0.78, color=DQ, label="Q4-verify (targets D)", zorder=3)
            ax.bar(gs, mn, 0.78, bottom=mq, color=DN, label="non-Q4 verify", zorder=3)
        ax.set_title(f"{model} — {SLAB[cond]}", fontsize=12.2)
        ax.set_xticks(range(1, 12, 2)); ax.margins(x=0.02)
        if cidx == 0: ax.set_ylabel("verifies / game")
        if r == len(MODELS) - 1: ax.set_xlabel("game")
h0, l0 = axes[0][0].get_legend_handles_labels()
seen = set(); h, l = [], []
for hh, ll in zip(h0, l0):
    if ll not in seen:
        seen.add(ll); h.append(hh); l.append(ll)
fig.legend(h, l, loc="lower center", ncol=2, fontsize=12.2, frameon=False)
fig.tight_layout(rect=[0, 0.03, 1, 1])
fig.savefig(os.path.join(IMG, "fig3_stacked_schedules.pdf"))
print("wrote fig3_stacked_schedules.pdf")
