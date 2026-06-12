#!/usr/bin/env python3
"""Memoryless (i.i.d.) verification broken out BY ROUND within a game.
One panel per model. Within a panel: x = round (1..4); per round two stacked bars
(neutral = D-correct [light], reactive = D-wrong [full color]); each bar stacked by
puzzle Q1..Q4 (Q4 = D). Bar height = mean verifications in that round per game.
Reveals the within-game temporal scan: round 1 ~ Q1-first, later rounds shift.
"""
import csv, json, os, glob
from collections import defaultdict
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.rcParams.update({"font.size":19.5,"axes.titlesize":21,"axes.labelsize":19.5,"xtick.labelsize":18,"ytick.labelsize":18,"legend.fontsize":18,"figure.titlesize":22.5})
import matplotlib.colors as mcolors

SD = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(SD, "..", "results")
IMG = os.path.join(SD, "figures"); os.makedirs(IMG, exist_ok=True)
SKIP = ("APIERROR", "CONTAMINATED", "to_be_deleted", "default_experiment", "SUPERSEDED")
MODEL_ORDER = ["Opus", "Sonnet", "GPT-5.1", "GPT-5.4-mini", "Gemini Pro", "Gemini Flash"]
ROUNDS = [1, 2, 3, 4]
QS = ["Q1", "Q2", "Q3", "Q4"]
QCOL = {"Q1": "#4c72b0", "Q2": "#55a868", "Q3": "#c9a227", "Q4": "#c44e52"}

def clean(pm):
    pm = pm.split(":")[-1].split("/")[-1]
    return {"claude-opus-4-6": "Opus", "claude-sonnet-4-6": "Sonnet",
            "gemini-3.1-pro-preview": "Gemini Pro", "gemini-2.5-flash": "Gemini Flash",
            "gpt-5.4-mini-2026-03-17": "GPT-5.4-mini", "gpt-5.1": "GPT-5.1"}.get(pm, pm)

def iidkind(f):
    try: s = json.load(open(os.path.join(f, "experiment_settings.json")))
    except Exception: return None
    mc = s.get("memory_config", {}).get("A", {})
    if not ((mc.get("persist_across_games") is False) or (mc.get("mode") == "current_game_only")):
        return None
    dummy = s.get("dummy_config", {}).get("D", {}).get("correctness_list", [])
    return clean(s.get("player_models", ["?"])[0]), ("dcorrect" if sum(1 for x in dummy if not x) == 0 else "dwrong")

def shade(hexc, f):
    r, g, b = mcolors.to_rgb(hexc)
    if f <= 1: return (r * f, g * f, b * f)
    t = f - 1; return (r + (1 - r) * t, g + (1 - g) * t, b + (1 - b) * t)
SHADE = {"dcorrect": 1.45, "dwrong": 1.0}

# cnt[(model,kind,round)][Qx]=count ; games[(model,kind)] = total games
cnt = defaultdict(lambda: defaultdict(int))
games = defaultdict(int)
for f in sorted(glob.glob(os.path.join(RES, "*/"))):
    if os.path.basename(f.rstrip("/")).startswith(SKIP): continue
    k = iidkind(f)
    if not k: continue
    model, kind = k
    p = os.path.join(f, "game_data_agent_decisions.csv")
    if not os.path.exists(p): continue
    rows = list(csv.DictReader(open(p)))
    if not rows: continue
    games[(model, kind)] += len(set(r["game_id"] for r in rows))
    for r in rows:
        d = r.get("agent_decision", "")
        if d.startswith("Verify:"):
            try: rd = int(r.get("round", 0))
            except ValueError: continue
            rd = min(rd, 4)
            q = d.split(":")[1].strip()[:2]
            if q in QS: cnt[(model, kind, rd)][q] += 1

W = 0.36
fig, axes = plt.subplots(1, 6, figsize=(21, 4.8), squeeze=False, sharey=True)
for i, model in enumerate(MODEL_ORDER):
    ax = axes[0][i]
    for kind, off in (("dcorrect", -W / 2 - 0.02), ("dwrong", W / 2 + 0.02)):
        g = games.get((model, kind), 0)
        if not g: continue
        x = np.array(ROUNDS, dtype=float) + off
        bottom = np.zeros(len(ROUNDS))
        for q in QS:
            seg = np.array([cnt[(model, kind, rd)][q] / g for rd in ROUNDS])
            ax.bar(x, seg, W, bottom=bottom, color=shade(QCOL[q], SHADE[kind]),
                   edgecolor="white", linewidth=0.3,
                   label=(f"{q}" + (" = D" if q == "Q4" else "")) if (i == 0 and kind == "dwrong") else None)
            bottom += seg
    ax.set_title(model, fontsize=20)
    ax.set_xticks(ROUNDS); ax.set_xlabel("round")
    if i == 0: ax.set_ylabel("verifies / game")
    ax.margins(x=0.04)
h, l = axes[0][0].get_legend_handles_labels()
fig.legend(h, l, loc="lower center", ncol=4, fontsize=18.3, frameon=False)
fig.suptitle("Memoryless verification by round (within game): left bar = neutral (light), "
             "right bar = reactive (full color), stacked by puzzle", fontsize=22.4)
fig.tight_layout(rect=[0, 0.12, 1, 0.90])
fig.savefig(os.path.join(IMG, "iid_rounds.pdf"))
print("wrote iid_rounds.pdf")
