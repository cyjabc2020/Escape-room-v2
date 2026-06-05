"""
Generate paper figures beyond the clustering plots.

Produces:
  - scar_by_cell.pdf: scar rate per (model, schedule) cell, with Wilson 95% CI
  - scar_threshold_sensitivity.pdf: scar rate sweep across thresholds 0.3 -> 0.8
  - survival_vs_scar.pdf: scatter of cell-level scar rate vs mean survival
  - trajectory_q4_verifies.pdf: per-game Q4 verification rate over time (selected cells)
"""
import csv, json, os, collections, math
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
CSV_PATH = os.path.join(SCRIPT_DIR, "per_rep_metrics.csv")
IMG_DIR = os.path.join(SCRIPT_DIR, "..", "..", "paper", "images")
os.makedirs(IMG_DIR, exist_ok=True)

ROOT = os.environ.get("ESCAPE_RESULTS") or os.path.join(SCRIPT_DIR, "..", "results")

def wilson(x, n, z=1.96):
    if n == 0: return 0, 0
    p = x / n
    denom = 1 + z*z/n
    center = (p + z*z/(2*n)) / denom
    halfw = z * math.sqrt(p*(1-p)/n + z*z/(4*n*n)) / denom
    return max(0, center - halfw), min(1, center + halfw)

def normal_ci(xs, z=1.96):
    n = len(xs)
    if n == 0: return 0, 0, 0
    mean = sum(xs)/n
    var = sum((x-mean)**2 for x in xs)/(n-1) if n>1 else 0
    se = (var**0.5)/(n**0.5) if n>1 else float('inf')
    if se == float('inf'): return mean, mean, mean
    return mean, mean - z*se, mean + z*se

rows = list(csv.DictReader(open(CSV_PATH)))

# ======================
# Fig 1: Scar by cell with Wilson 95% CI
# ======================
cells_order = [
    ("gpt5.1-high",         "1strike",    "GPT-5.1 V1 / 1-strike"),
    ("gpt-5.4-mini-2026-03-17","1strike",   "GPT-5.4-mini / 1-strike"),
    ("claude-opus-4-6",       "1strike",    "Opus 4.6 / 1-strike"),
    ("claude-opus-4-6",       "2strike",    "Opus 4.6 / 2-strike"),
    ("claude-opus-4-6",       "3strike",    "Opus 4.6 / 3-strike"),
    ("claude-opus-4-6",       "mid2strike", "Opus 4.6 / mid2strike"),
    ("claude-opus-4-6",       "recur",      "Opus 4.6 / recur"),
    ("claude-sonnet-4-6",     "1strike",    "Sonnet 4.6 / 1-strike"),
    ("claude-sonnet-4-6",     "2strike",    "Sonnet 4.6 / 2-strike"),
    ("claude-sonnet-4-6",     "3strike",    "Sonnet 4.6 / 3-strike"),
    ("claude-sonnet-4-6",     "mid2strike", "Sonnet 4.6 / mid2strike"),
    ("claude-sonnet-4-6",     "recur",      "Sonnet 4.6 / recur"),
    ("gemini-3.1-pro-preview","1strike",    "Gemini 3.1 Pro / 1-strike"),
    ("gemini-2.5-flash",      "1strike",    "Gemini 2.5 Flash / 1-strike"),
]
fig, ax = plt.subplots(figsize=(8, 5))
labels, scar_rates, lo_errs, hi_errs, ns = [], [], [], [], []
for model, sched, lbl in cells_order:
    cell_rows = [r for r in rows if r["model"]==model and r["schedule"]==sched and r["refl"]=="none"]
    n = len(cell_rows)
    scarred = sum(1 for r in cell_rows if int(r["scarred"]) == 1)
    if n == 0: continue
    rate = scarred / n
    lo, hi = wilson(scarred, n)
    labels.append(f"{lbl}\n(n={n})")
    scar_rates.append(rate*100)
    lo_errs.append((rate - lo)*100)
    hi_errs.append((hi - rate)*100)
    ns.append(n)

# Colors by model
colors = []
for model, sched, lbl in cells_order:
    if "opus" in model: colors.append("#ad5500")
    elif "sonnet" in model: colors.append("#e09000")
    elif "gpt5.1" in model.lower() or "gpt-5.1" in model.lower(): colors.append("#0a7b85")
    elif "mini" in model: colors.append("#5cbac4")
    elif "pro" in model: colors.append("#4d4daf")
    elif "flash" in model: colors.append("#8a8acf")
    else: colors.append("gray")
colors = colors[:len(scar_rates)]

x = np.arange(len(labels))
ax.bar(x, scar_rates, yerr=[lo_errs, hi_errs], color=colors, capsize=4, edgecolor="black", linewidth=0.5)
ax.set_ylabel("Scar rate (% of runs with TARG > 0.5) ± Wilson 95% CI", fontsize=10)
ax.set_xticks(x)
ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=7.5)
ax.set_ylim(0, 110)
ax.axhline(50, color="gray", linestyle=":", alpha=0.5, linewidth=0.5)
ax.set_title("Scar rate across (snapshot, schedule) cells", fontsize=11)
ax.grid(axis="y", alpha=0.3, linewidth=0.5)
fig.tight_layout()
fig.savefig(os.path.join(IMG_DIR, "scar_by_cell.pdf"))
plt.close(fig)
print("Saved scar_by_cell.pdf")

# ======================
# Fig 2: Threshold sensitivity
# ======================
thresholds = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
fig, ax = plt.subplots(figsize=(7, 4.5))

key_cells = [
    ("claude-opus-4-6",   "1strike",    "Opus 1-strike",   "-",  "#ad5500"),
    ("claude-opus-4-6",   "2strike",    "Opus 2-strike",   "-",  "#ad5500"),
    ("claude-sonnet-4-6", "1strike",    "Sonnet 1-strike", "--", "#e09000"),
    ("claude-sonnet-4-6", "2strike",    "Sonnet 2-strike", "--", "#e09000"),
    ("claude-sonnet-4-6", "recur",      "Sonnet recur",    ":",  "#e09000"),
    ("claude-sonnet-4-6", "mid2strike", "Sonnet mid2strike", "-.", "#e09000"),
    ("gemini-2.5-flash",  "1strike",    "Flash 1-strike",  "-",  "#8a8acf"),
]
for model, sched, lbl, ls, color in key_cells:
    cell_rows = [r for r in rows if r["model"]==model and r["schedule"]==sched and r["refl"]=="none"]
    n = len(cell_rows)
    if n == 0: continue
    rates = []
    for t in thresholds:
        scarred = sum(1 for r in cell_rows if float(r["targ"]) > t)
        rates.append(scarred/n*100)
    ax.plot(thresholds, rates, ls, label=f"{lbl} (n={n})", color=color, marker="o", markersize=5)

ax.axvline(0.5, color="gray", linestyle=":", alpha=0.5, linewidth=0.5)
ax.text(0.51, 95, "default threshold", fontsize=7, color="gray")
ax.set_xlabel("Scar threshold (TARG)", fontsize=10)
ax.set_ylabel("Scar rate (%)", fontsize=10)
ax.set_title("Sensitivity of scar classification to threshold choice", fontsize=11)
ax.set_ylim(0, 105)
ax.legend(fontsize=7.5, loc="upper right")
ax.grid(alpha=0.3)
fig.tight_layout()
fig.savefig(os.path.join(IMG_DIR, "scar_threshold_sensitivity.pdf"))
plt.close(fig)
print("Saved scar_threshold_sensitivity.pdf")

# ======================
# Fig 3: Survival vs scar (cell-level scatter)
# ======================
fig, ax = plt.subplots(figsize=(6, 4.2))
cell_scar = {}
cell_surv = {}
cell_n = {}
cell_meta = {}
for r in rows:
    if r["schedule"] == "smooth": continue
    if r["refl"] != "none": continue
    key = (r["model"], r["schedule"])
    cell_scar.setdefault(key, []).append(int(r["scarred"]))
    cell_surv.setdefault(key, []).append(float(r["survival"]))

for key in cell_scar:
    n = len(cell_scar[key])
    if n < 3: continue
    sr = sum(cell_scar[key]) / n * 100
    surv = sum(cell_surv[key]) / n
    model, sched = key
    if "opus" in model: c = "#ad5500"
    elif "sonnet" in model: c = "#e09000"
    elif "mini" in model: c = "#5cbac4"
    elif "pro" in model: c = "#4d4daf"
    elif "flash" in model: c = "#8a8acf"
    elif "gpt5.1" in model.lower() or "gpt-5.1" in model.lower(): c = "#0a7b85"
    else: c = "gray"
    ax.scatter(sr, surv, s=80, color=c, alpha=0.7, edgecolors="black", linewidths=0.5)
    label_short = f"{model.split('-')[0][:6]}/{sched[:6]}"
    ax.annotate(label_short, (sr, surv), fontsize=7, xytext=(4,4), textcoords="offset points")

ax.set_xlabel("Scar rate (%)", fontsize=10)
ax.set_ylabel("Mean team survival (%)", fontsize=10)
ax.set_title("Scar rate vs team survival across (snapshot, schedule) cells", fontsize=11)
ax.grid(alpha=0.3)
fig.tight_layout()
fig.savefig(os.path.join(IMG_DIR, "survival_vs_scar.pdf"))
plt.close(fig)
print("Saved survival_vs_scar.pdf")

# ======================
# Fig 4: Per-game Q4-verify trajectory (Sonnet schedules)
# ======================
def load_per_game_q4(folder, schedule):
    """Return list of (game_idx, q4_verify_count_in_that_game) for the run."""
    R = os.path.join(ROOT, folder)
    try:
        sm = list(csv.DictReader(open(f"{R}/game_data_summary.csv")))
        dec = list(csv.DictReader(open(f"{R}/game_data_agent_decisions.csv")))
    except: return []
    out = []
    for i, r in enumerate(sm):
        gid = r["game_id"]
        q4 = sum(1 for d in dec if d["game_id"]==gid and d["agent_decision"].startswith("Verify:") and "Q4" in d["agent_decision"])
        out.append((i+1, q4))  # 1-indexed game
    return out

# Pick illustrative schedules in Sonnet
illustrative = [
    ("claude-sonnet-4-6", "smooth",     "Sonnet smooth",          "#7eb6c0"),
    ("claude-sonnet-4-6", "1strike",    "Sonnet 1-strike",        "#bbbb55"),
    ("claude-sonnet-4-6", "2strike",    "Sonnet 2-strike",        "#d05050"),
    ("claude-sonnet-4-6", "mid2strike", "Sonnet mid2strike",      "#9b30bf"),
    ("claude-sonnet-4-6", "recur",      "Sonnet recur (spread)",  "#6d995e"),
]
fig, ax = plt.subplots(figsize=(7.5, 4.5))
for model, sched, lbl, color in illustrative:
    cell_rows = [r for r in rows if r["model"]==model and r["schedule"]==sched and r["refl"]=="none"]
    if not cell_rows: continue
    # Aggregate q4 per game across reps
    per_game = collections.defaultdict(list)
    for r in cell_rows:
        traj = load_per_game_q4(r["folder"], sched)
        for g, q4 in traj:
            per_game[g].append(q4)
    games = sorted(per_game.keys())
    means = [np.mean(per_game[g]) for g in games]
    # Shade fail games
    ax.plot(games, means, marker="o", markersize=4.5, label=f"{lbl} (n={len(cell_rows)})",
            color=color, linewidth=1.5)
ax.set_xlabel("Game number", fontsize=10)
ax.set_ylabel("Mean Q4 verifications per game", fontsize=10)
ax.set_title("Q4 verification trajectory across schedules (Claude Sonnet 4.6)", fontsize=11)
ax.legend(fontsize=8, loc="upper right")
ax.grid(alpha=0.3)
ax.set_xticks(range(1, 12))
fig.tight_layout()
fig.savefig(os.path.join(IMG_DIR, "trajectory_q4_sonnet.pdf"))
plt.close(fig)
print("Saved trajectory_q4_sonnet.pdf")
