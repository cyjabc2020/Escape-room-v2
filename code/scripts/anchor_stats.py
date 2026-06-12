#!/usr/bin/env python3
"""
Anchor-relative trust statistics for the V2 paper.

Per (model, condition) we extract per-RUN values (run = replication unit):
  q4    = # Verify:Q4 decisions in the run
  allv  = # Verify:* decisions in the run
  games = # games in the run
Cell statistics (pooled over runs):
  share      = sum(q4)/sum(allv)        # targeting: how much of verification hits D
  suspicion  = sum(allv)/sum(games)     # volume:    verifies per game

CIs are CLUSTER bootstraps: resample runs with replacement (B=5000), recompute
the pooled statistic. n<3 cells get no CI (flagged) -> handled in the size plan.

Anchor-relative deltas per model:
  Delta_trust = smooth   - iid_dcorrect   (negative => trust formed as a discount)
  Delta_scar  = 1strike  - iid_dcorrect   (~0 => discount erasure; >0 => suspicion accrual)
Significance of a delta = bootstrap 95% CI excludes 0 (anchor uncertainty included
when the anchor cell has >=2 runs; otherwise anchor treated as a fixed point and FLAGGED).

Sample-size plan: from per-run dispersion, n needed for a target CI half-width.
"""
import csv, json, glob, os
from collections import defaultdict
import numpy as np

RNG = np.random.default_rng(20260608)
B = 5000
RES = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results")

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
    return model, cond, dummy

# Replication UNIT depends on the cell:
#   memoryless (iid_*) cells: games are i.i.d. (persist_across_games=False) -> unit = GAME
#   with-memory cells (smooth/perturbed): games are sequentially dependent -> unit = RUN (cluster)
# per_run[(model,cond)] = list of units, each (q4, allv, n_games) where n_games=1 for a game-unit.
per_run = defaultdict(list)
SKIP_PREFIXES = ("APIERROR", "CONTAMINATED", "to_be_deleted", "default_experiment", "SUPERSEDED")
for f in sorted(glob.glob(os.path.join(RES, "*/"))):
    if os.path.basename(f.rstrip("/")).startswith(SKIP_PREFIXES):
        continue
    c = classify(f)
    if not c: continue
    model, cond, dummy = c
    p = os.path.join(f, "game_data_agent_decisions.csv")
    if not os.path.exists(p): continue
    rows = list(csv.DictReader(open(p)))
    if not rows: continue
    is_iid = cond.startswith("iid_")
    by_game = defaultdict(lambda: [0, 0])  # game_id -> [q4, allv]
    for r in rows:
        d = r.get("agent_decision", "")
        if d.startswith("Verify:"):
            by_game[r["game_id"]][1] += 1
            if "Q4" in d: by_game[r["game_id"]][0] += 1
        else:
            by_game[r["game_id"]]  # ensure game registered even with no verifies
    if not is_iid and any(x is False or x == False for x in dummy):
        # Exclude games where D's scripted answer was WRONG: metrics are computed only
        # over D-correct games, so they match the D-correct memoryless anchor's
        # within-game information environment. game_id timestamps sort chronologically.
        order = sorted(by_game.keys())
        for i, gid in enumerate(order):
            if i < len(dummy) and not dummy[i]:
                del by_game[gid]
    if is_iid:
        for g, (q4, allv) in by_game.items():
            per_run[(model, cond)].append((q4, allv, 1))   # one unit per game
    else:
        q4 = sum(v[0] for v in by_game.values()); allv = sum(v[1] for v in by_game.values())
        per_run[(model, cond)].append((q4, allv, len(by_game)))  # one unit per run

def pooled(runs, which):
    q4 = sum(r[0] for r in runs); allv = sum(r[1] for r in runs); g = sum(r[2] for r in runs)
    if which == "share": return q4 / allv if allv else np.nan
    return allv / g if g else np.nan  # suspicion

def boot_cell(runs, which):
    n = len(runs)
    if n < 3: return (np.nan, np.nan)
    arr = np.array(runs, dtype=float)
    stats = np.empty(B)
    for b in range(B):
        idx = RNG.integers(0, n, n)
        s = arr[idx]
        if which == "share":
            allv = s[:, 1].sum(); stats[b] = s[:, 0].sum() / allv if allv else np.nan
        else:
            g = s[:, 2].sum(); stats[b] = s[:, 1].sum() / g if g else np.nan
    return tuple(np.nanpercentile(stats, [2.5, 97.5]))

def boot_delta(runs_a, runs_b, which):
    """statistic(a) - statistic(b); resample each cell's runs independently.
    Returns (point, lo, hi, anchor_flag) where anchor_flag set if a cell has <2 runs."""
    flag = (len(runs_a) < 2) or (len(runs_b) < 2)
    pa, pb = pooled(runs_a, which), pooled(runs_b, which)
    point = pa - pb
    A = np.array(runs_a, float); Bn = np.array(runs_b, float)
    d = np.empty(B)
    for b in range(B):
        sa = A[RNG.integers(0, len(A), len(A))] if len(A) > 1 else A
        sb = Bn[RNG.integers(0, len(Bn), len(Bn))] if len(Bn) > 1 else Bn
        if which == "share":
            va = sa[:, 0].sum() / sa[:, 1].sum() if sa[:, 1].sum() else np.nan
            vb = sb[:, 0].sum() / sb[:, 1].sum() if sb[:, 1].sum() else np.nan
        else:
            va = sa[:, 1].sum() / sa[:, 2].sum() if sa[:, 2].sum() else np.nan
            vb = sb[:, 1].sum() / sb[:, 2].sum() if sb[:, 2].sum() else np.nan
        d[b] = va - vb
    lo, hi = np.nanpercentile(d, [2.5, 97.5])
    return point, lo, hi, flag

MO = ["Opus", "Sonnet", "GPT-5.1", "GPT-5.4-mini", "Gemini Pro", "Gemini Flash"]
CO = ["iid_dcorrect", "iid_dwrong", "smooth", "1strike", "2strike", "3strike", "mid2strike", "recur"]

print("=" * 96)
print("TABLE A. Per-cell estimates (pooled over runs) with cluster-bootstrap 95% CI")
print("=" * 96)
print(f"{'model':13}{'condition':13}{'n':>3}{'games':>6}{'Q4-share':>10}{'  95% CI':>16}{'suspicion':>11}{'  95% CI':>16}")
for m in MO:
    for c in CO:
        runs = per_run.get((m, c))
        if not runs: continue
        n = len(runs); g = sum(r[2] for r in runs)
        sh = pooled(runs, "share"); su = pooled(runs, "suspicion")
        shlo, shhi = boot_cell(runs, "share"); sulo, suhi = boot_cell(runs, "suspicion")
        ci = lambda lo, hi: "  (n<3)" if np.isnan(lo) else f"[{lo:.3f},{hi:.3f}]"
        shs = " nan" if np.isnan(sh) else f"{sh:.3f}"
        print(f"{m:13}{c:13}{n:>3}{g:>6}{shs:>10}{ci(shlo,shhi):>16}{su:>11.2f}{ci(sulo,suhi):>16}")

print("\n" + "=" * 96)
print("TABLE B. Anchor-relative deltas (vs i.i.d. D-correct).  * = 95% CI excludes 0")
print("  Delta_trust = smooth - anchor (neg => trust formed);  Delta_scar = 1strike - anchor")
print("=" * 96)
print(f"{'model':13}{'metric':10}{'Δ_trust':>9}{'  95% CI':>16}{'sig':>4}{'Δ_scar':>9}{'  95% CI':>16}{'sig':>4}{'  anchor_n'}")
delta_rows = [["model","metric","delta_trust","trust_lo","trust_hi","trust_sig","delta_scar","scar_lo","scar_hi","scar_sig","anchor_n","anchor_flag"]]
for m in MO:
    anc = per_run.get((m, "iid_dcorrect"), [])
    sm = per_run.get((m, "smooth"), []); st = per_run.get((m, "1strike"), [])
    if not anc or not sm or not st:
        print(f"{m:13}{'(missing cells - skipped)'}")
        continue
    for which, lab in [("share", "Q4-share"), ("suspicion", "suspicion")]:
        dt, tlo, thi, tflag = boot_delta(sm, anc, which)
        ds, slo, shi, sflag = boot_delta(st, anc, which)
        tsig = "*" if (tlo > 0 or thi < 0) else " "
        ssig = "*" if (slo > 0 or shi < 0) else " "
        fl = "(anchor n<2!)" if (tflag or sflag) else ""
        print(f"{m:13}{lab:10}{dt:>9.3f}[{tlo:>6.3f},{thi:>6.3f}]{tsig:>4}{ds:>9.3f}[{slo:>6.3f},{shi:>6.3f}]{ssig:>4}  {len(anc)} {fl}")
        delta_rows.append([m,lab,round(dt,4),round(tlo,4),round(thi,4),tsig.strip(),round(ds,4),round(slo,4),round(shi,4),ssig.strip(),len(anc),int(tflag or sflag)])

with open(os.path.join(os.path.dirname(__file__), "anchor_deltas.csv"), "w", newline="") as fh:
    csv.writer(fh).writerows(delta_rows)

print("\n" + "=" * 96)
print("TABLE C. Sample-size plan for Q4-share (target 95% CI half-width = 0.05)")
print("  per-run SD estimated within cell when n>=3; else borrowed pooled SD (median of estimable cells)")
print("=" * 96)
# estimate per-run share SD per cell (n>=3); borrow median for thin cells
def run_shares(runs): return [r[0]/r[1] for r in runs if r[1] > 0]
sds = {}
for (m, c), runs in per_run.items():
    rs = run_shares(runs)
    if len(rs) >= 3: sds[(m, c)] = np.std(rs, ddof=1)
borrow = np.median(list(sds.values())) if sds else 0.12
TARGET_H = 0.05
print(f"(borrowed SD for thin cells = {borrow:.3f}; target half-width h = {TARGET_H})")
print(f"{'model':13}{'condition':13}{'n_now':>6}{'run_SD':>8}{'n_need':>8}{'add':>6}{'  priority'}")
plan_rows = [["model","condition","n_now","run_sd_used","n_needed","add_runs","priority"]]
PLAN_CONDS = ["iid_dcorrect", "iid_dwrong", "smooth", "1strike", "2strike", "3strike", "mid2strike", "recur"]
for m in MO:
    for c in PLAN_CONDS:
        runs = per_run.get((m, c))
        if runs is None:
            # only flag missing anchors that the headline needs
            if c == "iid_dcorrect":
                pass
            continue
        n_now = len(runs)
        sd = sds.get((m, c), borrow)
        n_need = int(np.ceil((1.96 * sd / TARGET_H) ** 2))
        n_need = max(n_need, 3)  # need >=3 just to estimate a CI
        add = max(0, n_need - n_now)
        # priority: anchors feeding the two-architecture split, then thin headline cells
        if c == "iid_dcorrect" and add > 0: pri = "HIGH (anchor)"
        elif c == "iid_dcorrect": pri = "ok"
        elif add > 0 and n_now < 5: pri = "med"
        elif add > 0: pri = "low"
        else: pri = "ok"
        used = "borrow" if (m, c) not in sds else "in-cell"
        print(f"{m:13}{c:13}{n_now:>6}{sd:>8.3f}{n_need:>8}{add:>6}  {pri} [{used}]")
        plan_rows.append([m, c, n_now, round(sd,4), n_need, add, pri])
with open(os.path.join(os.path.dirname(__file__), "sample_size_plan.csv"), "w", newline="") as fh:
    csv.writer(fh).writerows(plan_rows)
print("\nwrote anchor_deltas.csv, sample_size_plan.csv")
