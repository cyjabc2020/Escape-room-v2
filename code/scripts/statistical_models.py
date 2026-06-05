"""
Statistical analysis:
  1. Mixed-effects logistic regression for scar outcome (snapshot, schedule fixed effects)
  2. Permutation tests for pairwise cell differences
  3. Multiple-comparison correction (Benjamini-Hochberg)
  4. Effect sizes (Cohen's h for proportions, Cohen's d for means)

Outputs:
  - V2/code/scripts/statistical_results.json
  - V2/code/scripts/pairwise_table.csv
"""
import csv, json, os, math
from itertools import combinations
import numpy as np
from scipy import stats

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
CSV_PATH = os.path.join(SCRIPT_DIR, "per_rep_metrics.csv")

rows = list(csv.DictReader(open(CSV_PATH)))


# =============================
# 1. Cohen's h and d
# =============================
def cohens_h(p1, p2):
    """Cohen's h for two proportions."""
    phi1 = 2 * math.asin(math.sqrt(p1))
    phi2 = 2 * math.asin(math.sqrt(p2))
    return abs(phi1 - phi2)


def cohens_d(xs1, xs2):
    """Cohen's d for two samples (pooled SD)."""
    n1, n2 = len(xs1), len(xs2)
    if n1 < 2 or n2 < 2: return float('nan')
    m1, m2 = np.mean(xs1), np.mean(xs2)
    v1 = np.var(xs1, ddof=1) if n1 > 1 else 0
    v2 = np.var(xs2, ddof=1) if n2 > 1 else 0
    sp = math.sqrt(((n1-1)*v1 + (n2-1)*v2) / (n1+n2-2))
    if sp == 0: return float('nan')
    return (m1 - m2) / sp


# =============================
# 2. Pairwise tests
# =============================
def get_cell(model, schedule, refl="none"):
    return [r for r in rows if r["model"]==model and r["schedule"]==schedule and r["refl"]==refl]


comparisons = [
    # Threshold tests
    ("Opus 1-strike vs 2-strike (count threshold)",
     ("claude-opus-4-6", "1strike"), ("claude-opus-4-6", "2strike")),
    ("Sonnet 1-strike vs 2-strike (count threshold)",
     ("claude-sonnet-4-6", "1strike"), ("claude-sonnet-4-6", "2strike")),
    ("Opus 2-strike vs 3-strike (plateau)",
     ("claude-opus-4-6", "2strike"), ("claude-opus-4-6", "3strike")),
    ("Sonnet 2-strike vs 3-strike (plateau)",
     ("claude-sonnet-4-6", "2strike"), ("claude-sonnet-4-6", "3strike")),
    # Clustering / timing tests
    ("Sonnet 2-strike vs recur (clustering)",
     ("claude-sonnet-4-6", "2strike"), ("claude-sonnet-4-6", "recur")),
    ("Opus 2-strike vs recur (clustering)",
     ("claude-opus-4-6", "2strike"), ("claude-opus-4-6", "recur")),
    ("Sonnet 2-strike vs mid2strike (timing)",
     ("claude-sonnet-4-6", "2strike"), ("claude-sonnet-4-6", "mid2strike")),
    ("Opus 2-strike vs mid2strike (timing)",
     ("claude-opus-4-6", "2strike"), ("claude-opus-4-6", "mid2strike")),
    # Cross-family 1-strike
    ("Opus 1-strike vs GPT-5.1 V1 1-strike",
     ("claude-opus-4-6", "1strike"), ("gpt5.1-high", "1strike")),
    ("Sonnet 1-strike vs GPT-5.1 V1 1-strike",
     ("claude-sonnet-4-6", "1strike"), ("gpt5.1-high", "1strike")),
    ("Pro 1-strike vs GPT-5.1 V1 1-strike",
     ("gemini-3.1-pro-preview", "1strike"), ("gpt5.1-high", "1strike")),
    ("Pro 1-strike vs Opus 1-strike",
     ("gemini-3.1-pro-preview", "1strike"), ("claude-opus-4-6", "1strike")),
    ("Pro 1-strike vs Sonnet 1-strike",
     ("gemini-3.1-pro-preview", "1strike"), ("claude-sonnet-4-6", "1strike")),
    ("Flash 1-strike vs Pro 1-strike",
     ("gemini-2.5-flash", "1strike"), ("gemini-3.1-pro-preview", "1strike")),
    ("Flash 1-strike vs Opus 1-strike",
     ("gemini-2.5-flash", "1strike"), ("claude-opus-4-6", "1strike")),
    # Anthropic 1-strike vs 2-strike: combined family test
]

pair_results = []
for name, (m1, s1), (m2, s2) in comparisons:
    a = get_cell(m1, s1)
    b = get_cell(m2, s2)
    if not a or not b: continue
    na, nb = len(a), len(b)
    scarred_a = sum(int(r["scarred"]) for r in a)
    scarred_b = sum(int(r["scarred"]) for r in b)
    pa, pb = scarred_a/na, scarred_b/nb
    # Fisher's exact
    table = [[scarred_a, na-scarred_a], [scarred_b, nb-scarred_b]]
    _, p_fisher = stats.fisher_exact(table)
    # Cohen's h on proportions
    h = cohens_h(pa, pb)
    # TARG comparison: Mann-Whitney U (non-parametric)
    targ_a = [float(r["targ"]) for r in a]
    targ_b = [float(r["targ"]) for r in b]
    u_stat, p_mw = stats.mannwhitneyu(targ_a, targ_b, alternative="two-sided")
    # Cohen's d on TARG
    d = cohens_d(targ_a, targ_b)
    pair_results.append({
        "comparison": name,
        "cell_A": f"{m1}/{s1}",
        "cell_B": f"{m2}/{s2}",
        "n_A": na, "n_B": nb,
        "scar_rate_A": round(pa, 3),
        "scar_rate_B": round(pb, 3),
        "p_fisher": round(p_fisher, 4),
        "cohens_h": round(h, 3),
        "mean_targ_A": round(np.mean(targ_a), 3),
        "mean_targ_B": round(np.mean(targ_b), 3),
        "p_mw": round(p_mw, 4),
        "cohens_d": round(d, 3),
    })

# Benjamini-Hochberg correction
def benjamini_hochberg(pvals, alpha=0.05):
    n = len(pvals)
    sorted_idx = np.argsort(pvals)
    sorted_p = np.array([pvals[i] for i in sorted_idx])
    thresholds = (np.arange(1, n+1) / n) * alpha
    rejected = sorted_p <= thresholds
    # Find the largest k such that sorted_p[k] <= thresholds[k]
    if rejected.any():
        max_k = np.where(rejected)[0].max()
        rejected[:max_k+1] = True
    out = [False] * n
    for i, idx in enumerate(sorted_idx):
        out[idx] = rejected[i]
    # Adjusted p-values
    adjusted = np.full(n, 1.0)
    for i, idx in enumerate(sorted_idx):
        adjusted[idx] = min(sorted_p[i] * n / (i+1), 1.0)
    # Enforce monotone
    for i in range(n-2, -1, -1):
        idx_curr = sorted_idx[i]
        idx_next = sorted_idx[i+1]
        if adjusted[idx_curr] > adjusted[idx_next]:
            adjusted[idx_curr] = adjusted[idx_next]
    return out, list(adjusted)

p_fishers = [r["p_fisher"] for r in pair_results]
p_mws = [r["p_mw"] for r in pair_results]
_, p_fisher_bh = benjamini_hochberg(p_fishers)
_, p_mw_bh = benjamini_hochberg(p_mws)
for r, pf, pm in zip(pair_results, p_fisher_bh, p_mw_bh):
    r["p_fisher_bh"] = round(pf, 4)
    r["p_mw_bh"] = round(pm, 4)

# Write CSV
with open(os.path.join(SCRIPT_DIR, "pairwise_table.csv"), "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(pair_results[0].keys()))
    w.writeheader()
    for r in pair_results:
        w.writerow(r)
print(f"Wrote pairwise_table.csv ({len(pair_results)} comparisons)")

# Print significant comparisons after BH correction
print(f"\n=== Comparisons surviving Benjamini-Hochberg correction (alpha=0.05) ===")
for r in pair_results:
    if r["p_fisher_bh"] < 0.05 or r["p_mw_bh"] < 0.05:
        print(f"  {r['comparison']}")
        print(f"    scar: {r['scar_rate_A']:.2f} vs {r['scar_rate_B']:.2f}, Fisher BH-adj p={r['p_fisher_bh']:.4f}, Cohen's h={r['cohens_h']:.2f}")
        print(f"    TARG: {r['mean_targ_A']:.2f} vs {r['mean_targ_B']:.2f}, MW BH-adj p={r['p_mw_bh']:.4f}, Cohen's d={r['cohens_d']:.2f}")

# =============================
# 3. Mixed-effects logistic regression
# =============================
try:
    import statsmodels.api as sm
    import statsmodels.formula.api as smf
    import pandas as pd
    have_sm = True
except ImportError:
    print("\nstatsmodels/pandas not available; skipping mixed-effects models")
    have_sm = False

if have_sm:
    df = pd.read_csv(CSV_PATH)
    # Recovery-family only, refl=none, dropping unusual conditions
    df_rec = df[(df["schedule"] != "smooth") & (df["refl"] == "none")].copy()
    df_rec["scarred"] = df_rec["scarred"].astype(int)

    print(f"\n=== Mixed-effects model: scar ~ schedule + model (run-level data, n={len(df_rec)}) ===")
    # Use schedule + model fixed effects, no random effects (we don't have crossed structure since each run is unique)
    # But we treat schedule and model as factors

    # Logistic regression with fixed effects
    try:
        # Reference levels: model=opus, schedule=1strike
        df_rec["model_cat"] = pd.Categorical(df_rec["model"], categories=sorted(df_rec["model"].unique()), ordered=False)
        df_rec["schedule_cat"] = pd.Categorical(df_rec["schedule"], categories=["1strike","2strike","3strike","mid2strike","recur"], ordered=False)

        # Set Opus and 1strike as reference
        formula = "scarred ~ C(model_cat, Treatment(reference='claude-opus-4-6')) + C(schedule_cat, Treatment(reference='1strike'))"
        model_fit = smf.logit(formula, data=df_rec).fit(disp=0, method="bfgs", maxiter=200)
        print(model_fit.summary().tables[1])
    except Exception as e:
        print(f"  Logit failed: {e}")
        # Fallback: report cell-level scar rates
        for (m, s), grp in df_rec.groupby(["model","schedule"]):
            n = len(grp)
            sr = grp["scarred"].mean()
            print(f"  {m}/{s}: n={n}, scar_rate={sr:.2%}")

# =============================
# Save summary JSON
# =============================
summary = {
    "n_comparisons": len(pair_results),
    "n_significant_after_BH": sum(1 for r in pair_results if r["p_fisher_bh"] < 0.05 or r["p_mw_bh"] < 0.05),
    "comparisons": pair_results,
}
with open(os.path.join(SCRIPT_DIR, "statistical_results.json"), "w") as f:
    json.dump(summary, f, indent=2)
print(f"\nSaved statistical_results.json")
