#!/usr/bin/env python3
"""Figure 1: per-game stacked verification bars, all six snapshots.
Per game index, two grouped bars (smooth | 1-strike); each bar stacked into
Q4-verify (targeting D, dark) + non-Q4-verify (rest of team, pale).
Total bar height = all-verify per game (suspicion volume).
Faint dashed line = memoryless i.i.d. D-correct all-verify rate (no-history anchor).
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
MODEL_ORDER = ["Opus", "Sonnet", "GPT-5.1", "GPT-5.4-mini", "Gemini Pro", "Gemini Flash"]

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
    else: cond = "other"
    return model, cond

# acc[model][cond][game] = list over runs of (q4, nonq4)
acc = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
# four memoryless anchors per model (unit = game): {dc,dw} x {all,q4}
ANCH = defaultdict(lambda: {"dc_all": [], "dc_q4": [], "dw_all": [], "dw_q4": []})
for f in sorted(glob.glob(os.path.join(RES, "*/"))):
    if os.path.basename(f.rstrip("/")).startswith(SKIP):
        continue
    c = classify(f)
    if not c: continue
    model, cond = c
    if cond == "other": continue
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
    elif cond in ("smooth", "1strike"):
        for g in idx.values():
            acc[model][cond][g].append((q4.get(g, 0), nq.get(g, 0)))

def mean_qn(model, cond):
    gd = acc[model][cond]
    gs = sorted(gd)
    mq = [np.mean([t[0] for t in gd[g]]) for g in gs]
    mn = [np.mean([t[1] for t in gd[g]]) for g in gs]
    return gs, mq, mn

COL = {"smooth": ("#1b6f3b", "#aedcbb"), "1strike": ("#9b1d3e", "#e9aebd")}
LAB = {"smooth": ("smooth: Q4-verify", "smooth: non-Q4"),
       "1strike": ("1-strike: Q4-verify", "1-strike: non-Q4")}
W = 0.4
# (key, color, linestyle, legend label) for the four i.i.d. anchor lines
ANCH_LINES = [("dc_all", "#1f3a93", "--", "i.i.d. D-correct: all-verify"),
              ("dc_q4",  "#1f3a93", ":",  "i.i.d. D-correct: Q4-verify"),
              ("dw_all", "#cc6600", "--", "i.i.d. D-wrong: all-verify"),
              ("dw_q4",  "#cc6600", ":",  "i.i.d. D-wrong: Q4-verify")]

ncol = 3; nrow = 2
fig, axes = plt.subplots(nrow, ncol, figsize=(4.5 * ncol, 3.1 * nrow), squeeze=False, sharey=True)
handles = {}
for i, model in enumerate(MODEL_ORDER):
    ax = axes[i // ncol][i % ncol]
    for cond, off in (("smooth", -W / 2 - 0.02), ("1strike", W / 2 + 0.02)):
        gs, mq, mn = mean_qn(model, cond)
        if not gs: continue
        dq, dn = COL[cond]
        x = np.array(gs) + off
        b1 = ax.bar(x, mq, W, color=dq, label=LAB[cond][0])
        b2 = ax.bar(x, mn, W, bottom=mq, color=dn, label=LAB[cond][1])
        handles[LAB[cond][0]] = b1; handles[LAB[cond][1]] = b2
    ax.set_title(model, fontsize=13.5)
    ax.set_xlabel("game"); ax.set_ylabel("verifies / game")
    ax.set_xticks(range(1, 12, 2)); ax.margins(x=0.02)
# shared legend: 4 bar swatches
order = [LAB["smooth"][0], LAB["smooth"][1], LAB["1strike"][0], LAB["1strike"][1]]
fig.legend([handles[k] for k in order], order,
           loc="lower center", ncol=4, fontsize=11.9, frameon=False)
fig.tight_layout(rect=[0, 0.06, 1, 1])
fig.savefig(os.path.join(IMG, "fig1_stacked_verify.pdf"))
print("wrote fig1_stacked_verify.pdf")
