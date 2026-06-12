#!/usr/bin/env python3
"""Scenario x model game-score matrix.
For every (model, scenario) cell, the team score = mean coins banked per agent
(A/B/C) per game, averaged over runs (from final_scores.csv). Produces a heatmap
coloured by WITHIN-SCENARIO rank (so reshuffles across scenarios pop), annotated
with the raw coins and run count. Missing cells (not yet run) show 'n/a'.

Supports the conclusion: no absolute-best trust default (column winners change),
while some dynamics are inefficient across the board (a model low in every column).
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
SCEN = ["smooth", "1strike", "2strike", "3strike", "mid2strike", "recur"]
SCEN_LAB = {"smooth": "smooth", "1strike": "1-strike", "2strike": "2-strike",
            "3strike": "3-strike", "mid2strike": "mid2strike", "recur": "recur"}

def clean(pm):
    pm = pm.split(":")[-1].split("/")[-1]
    return {"claude-opus-4-6": "Opus", "claude-sonnet-4-6": "Sonnet",
            "gemini-3.1-pro-preview": "Gemini Pro", "gemini-2.5-flash": "Gemini Flash",
            "gpt-5.4-mini-2026-03-17": "GPT-5.4-mini", "gpt-5.1": "GPT-5.1"}.get(pm, pm)

def classify(f):
    try: s = json.load(open(os.path.join(f, "experiment_settings.json")))
    except Exception: return None
    mc = s.get("memory_config", {}).get("A", {})
    if (mc.get("persist_across_games") is False): return None   # exclude memoryless
    model = clean(s.get("player_models", ["?"])[0])
    dummy = s.get("dummy_config", {}).get("D", {}).get("correctness_list", [])
    nw = sum(1 for x in dummy if not x)
    if nw == 0: cond = "smooth"
    elif nw == 1 and dummy and not dummy[0]: cond = "1strike"
    elif nw == 2 and dummy[:2] == [False, False]: cond = "2strike"
    elif nw == 3 and dummy[:3] == [False, False, False]: cond = "3strike"
    elif dummy == [True, True, True, False, False, True, True, True, True, True, True]: cond = "mid2strike"
    elif dummy == [False, True, True, True, False, True, True, True, True, True, True]: cond = "recur"
    else: return None
    return model, cond

# Game-count normalization: every cell is scored over STRETCH games. Runs that ran
# fewer games (smooth Gemini Pro = 7, GPT-5.1 = 10) are stretched to STRETCH by
# imputing perfect clean-escape games (0 verify, immediate volunteer) worth
# MAX_ESCAPE_COINS coins/agent -- justified because those models had converged
# (stopped verifying, escaping at once) by the time their runs ended.
STRETCH = 11
MAX_ESCAPE_COINS = 4.0

scores = defaultdict(list)   # (model,scen) -> list of per-run team coins/agent/game (game-count-normalized)
for f in sorted(glob.glob(os.path.join(RES, "*/"))):
    if os.path.basename(f.rstrip("/")).startswith(SKIP): continue
    c = classify(f)
    if not c: continue
    model, scen = c
    p = os.path.join(f, "final_scores.csv")
    if not os.path.exists(p): continue
    rows = {r["player"]: r for r in csv.DictReader(open(p))}
    if not all(a in rows for a in ("A", "B", "C")): continue
    per_agent = []
    for a in ("A", "B", "C"):
        avg = float(rows[a]["avg_coins"]); ng = int(rows[a]["num_games"])
        if ng < STRETCH:   # impute clean-escape games for the converged tail
            avg = (avg * ng + MAX_ESCAPE_COINS * (STRETCH - ng)) / STRETCH
        per_agent.append(avg)
    scores[(model, scen)].append(np.mean(per_agent))

M = np.full((len(MODELS), len(SCEN)), np.nan)   # mean score
N = np.zeros((len(MODELS), len(SCEN)), dtype=int)
for i, m in enumerate(MODELS):
    for j, sc in enumerate(SCEN):
        v = scores.get((m, sc))
        if v: M[i, j] = np.mean(v); N[i, j] = len(v)

# colour by within-column rank (0=worst .. 1=best) so per-scenario winners pop
C = np.full_like(M, np.nan)
for j in range(len(SCEN)):
    col = M[:, j]; ok = ~np.isnan(col)
    if ok.sum() >= 2:
        r = col[ok].argsort().argsort().astype(float)
        C[ok, j] = r / (ok.sum() - 1)
    elif ok.sum() == 1:
        C[ok, j] = 0.5

# cluster-bootstrap 95% CI over runs, per cell, for the bar chart
RNG = np.random.default_rng(11)
CI = {}
for i, m in enumerate(MODELS):
    for j, sc in enumerate(SCEN):
        v = np.array(scores.get((m, sc), []))
        if len(v) >= 2:
            bs = [v[RNG.integers(0, len(v), len(v))].mean() for _ in range(3000)]
            CI[(i, j)] = np.percentile(bs, [2.5, 97.5])
        else:
            CI[(i, j)] = (np.nan, np.nan)

# ---------- Figure 1: heatmap ----------
fig, ax = plt.subplots(figsize=(9.5, 5.2))
im = ax.imshow(C, cmap="RdYlGn", vmin=0, vmax=1, aspect="auto")
ax.set_xticks(range(len(SCEN))); ax.set_xticklabels([SCEN_LAB[s] for s in SCEN], fontsize=12.2)
ax.set_yticks(range(len(MODELS))); ax.set_yticklabels(MODELS, fontsize=13.5)
for i in range(len(MODELS)):
    for j in range(len(SCEN)):
        if np.isnan(M[i, j]):
            ax.text(j, i, "n/a", ha="center", va="center", fontsize=10.8, color="0.5")
        else:
            ax.text(j, i, f"{M[i,j]:.2f}", ha="center", va="center", fontsize=12.2,
                    color="0.1", fontweight="bold")
            ax.text(j, i + 0.30, f"n={N[i,j]}", ha="center", va="center", fontsize=8.1, color="0.35")
ax.set_title("Game score by scenario and model (coins/agent/game, normalized to 11 games)\n"
             "colour = rank within each scenario (green = best); reshuffles across columns "
             "= no absolute-best default", fontsize=13.5)
fig.colorbar(im, ax=ax, label="within-scenario rank (green = best)", shrink=0.8)
fig.tight_layout()
fig.savefig(os.path.join(IMG, "scenario_score_matrix.pdf"))
print("wrote scenario_score_matrix.pdf")

# ---------- Figure 2: grouped bars (x = scenario, bars = models) ----------
MCOL = {"Opus": "#1f6f3e", "Sonnet": "#3a9b62", "GPT-5.1": "#b03060",
        "GPT-5.4-mini": "#d98a3d", "Gemini Pro": "#2e6da4", "Gemini Flash": "#7f7f7f"}
fig2, ax2 = plt.subplots(figsize=(14, 6.0))
nM = len(MODELS); w = 0.8 / nM
x = np.arange(len(SCEN))
for k, m in enumerate(MODELS):
    means = [M[MODELS.index(m), j] for j in range(len(SCEN))]
    lo = [M[MODELS.index(m), j] - CI[(MODELS.index(m), j)][0] for j in range(len(SCEN))]
    hi = [CI[(MODELS.index(m), j)][1] - M[MODELS.index(m), j] for j in range(len(SCEN))]
    pos = x + (k - (nM - 1) / 2) * w
    means_plot = [0 if np.isnan(v) else v for v in means]
    err = np.array([[0 if np.isnan(l) else l for l in lo], [0 if np.isnan(h) else h for h in hi]])
    ax2.bar(pos, means_plot, w, color=MCOL[m], label=m, yerr=err, capsize=2,
            error_kw=dict(lw=0.9, ecolor="0.35"))
ax2.set_xticks(x); ax2.set_xticklabels([SCEN_LAB[s] for s in SCEN], fontsize=16)
ax2.tick_params(axis="y", labelsize=15)
ax2.set_ylabel("game score (coins/agent/game, normalized to 11 games)", fontsize=15)
ax2.set_title("Game score by D-failure scenario and model\n"
              "(grouped by scenario; cluster-bootstrap 95% CIs)", fontsize=16)
ax2.legend(ncol=6, fontsize=14, frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.11))
ax2.margins(x=0.02); ax2.set_axisbelow(True); ax2.grid(axis="y", color="0.9")
fig2.tight_layout()
fig2.savefig(os.path.join(IMG, "scenario_score_bars.pdf"))
print("wrote scenario_score_bars.pdf")
print(f"\n{'model':14}" + "".join(f"{SCEN_LAB[s]:>12}" for s in SCEN))
for i, m in enumerate(MODELS):
    print(f"{m:14}" + "".join((f"{M[i,j]:>8.2f}(n{N[i,j]})" if not np.isnan(M[i,j]) else f"{'n/a':>12}") for j in range(len(SCEN))))
