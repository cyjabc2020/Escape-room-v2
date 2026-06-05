"""
Canonical per-rep metrics extraction. Produces a single CSV (one row per rep)
with TARG, SPILL, raw non-Q4 rate, scarred flag, survival, g1 severity, and
metadata (model, schedule, refl, reasoning). All downstream analyses (clustering,
mixed-effects, figures) consume this CSV.

Usage:
    python3 compute_metrics.py
    # writes V2/code/scripts/per_rep_metrics.csv
"""
import csv, json, os, sys, re

# Results live at V2/code/results, i.e. one level up from this scripts/ dir.
# Allow an explicit override via the ESCAPE_RESULTS env var.
ROOT = os.environ.get("ESCAPE_RESULTS") or os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "results")

OUT_CSV = os.path.join(os.path.dirname(__file__), "per_rep_metrics.csv")

SCAR_THRESHOLD = 0.5  # default; sensitivity analysis sweeps this


def analyze(folder):
    R = os.path.join(ROOT, folder)
    try:
        s = json.load(open(f"{R}/experiment_settings.json"))
        sm = list(csv.DictReader(open(f"{R}/game_data_summary.csv")))
        dec = list(csv.DictReader(open(f"{R}/game_data_agent_decisions.csv")))
        fs = list(csv.DictReader(open(f"{R}/final_scores.csv")))
    except Exception:
        return None
    if not sm or not fs:
        return None

    # Detect API-error contamination: count decisions where reasoning text contains
    # API failure markers ("ERROR:", "API call failed", etc.). Defaulted-to-pass is
    # a related but distinct measure — defaulted is set when the parser couldn't
    # extract a decision; API errors can appear in reasoning even when a decision
    # parses successfully (fallback string returned by helpers).
    api_error_keywords = [
        "API call failed", "API error",
        "OverloadedError", "BadRequestError",
        "RateLimitError", "InvalidRequestError",
        "Unauthenticated",
    ]
    n_api_errors = 0
    # Scan per-game JSON reasoning fields (richer than CSV)
    for fn in os.listdir(R):
        if not (fn.startswith("game_") and fn.endswith(".json")):
            continue
        try:
            gd = json.load(open(os.path.join(R, fn)))
        except Exception:
            continue
        for rd in gd.get("rounds", []):
            reasoning_d = rd.get("reasoning", {})
            for agent_id, text in reasoning_d.items():
                if agent_id == "D" or not isinstance(text, str):
                    continue
                if any(kw in text for kw in api_error_keywords):
                    n_api_errors += 1
    # Also check decision field
    for d in dec:
        if any(kw in d.get("agent_decision", "") for kw in api_error_keywords):
            n_api_errors += 1
    # Compute error rate
    n_decisions_made = sum(1 for d in dec if d.get("agent_decision", "") != "N/A")
    api_error_rate = n_api_errors / max(n_decisions_made, 1)
    # EXCLUDE runs with >5% API errors (severe contamination)
    if api_error_rate > 0.05:
        print(f"[EXCLUDED] {folder}: API error rate {api_error_rate*100:.0f}% ({n_api_errors}/{n_decisions_made})")
        return None

    pm = s.get("player_models", ["?"])[0]
    if pm == "openai/gpt-5.1":
        model = "gpt5.1-high"
    elif ":" in pm:
        model = pm.split(":")[-1]
    else:
        model = pm

    # Exclude pre-fix Flash (truncation bug)
    date_part = folder.split("_")[-2]
    if "gemini-2.5-flash" in pm and date_part < "20260519":
        return None

    refl = s.get("memory_config", {}).get("A", {}).get("reflection", "none")
    reasoning = s.get("reasoning_effort_config", {}).get("A", "?")
    dummy = s.get("dummy_config", {}).get("D", {}).get("correctness_list", [])
    n_wrong = sum(1 for x in dummy if not x)

    # Schedule classification
    if n_wrong == 0:
        schedule = "smooth"
    elif n_wrong == 1 and dummy[0] is False:
        schedule = "1strike"
    elif n_wrong == 2 and dummy[:2] == [False, False]:
        schedule = "2strike"
    elif n_wrong == 3 and dummy[:3] == [False, False, False]:
        schedule = "3strike"
    elif n_wrong == 2 and dummy == [True, True, True, False, False, True, True, True, True, True, True]:
        schedule = "mid2strike"
    elif n_wrong == 2 and dummy == [False, True, True, True, False, True, True, True, True, True, True]:
        schedule = "recur"
    else:
        schedule = f"unknown_{n_wrong}wrong"

    # Post-failure window
    last_fail_idx = max([i for i, x in enumerate(dummy) if not x], default=-1)
    post_start = last_fail_idx + 1
    if schedule == "smooth":
        later = [r["game_id"] for r in sm]
    else:
        later = [r["game_id"] for r in sm[post_start:]]
    n_post = len(later)
    if n_post == 0:
        return None

    # Verification counts
    q4_count = sum(1 for d in dec if d["game_id"] in later and d["agent_decision"].startswith("Verify:") and "Q4" in d["agent_decision"])
    nonq4_count = sum(1 for d in dec if d["game_id"] in later and d["agent_decision"].startswith("Verify:") and "Q4" not in d["agent_decision"])

    targ = q4_count / n_post
    nonq4_raw = nonq4_count / n_post

    # Game 1 severity
    g1 = sm[0]["game_id"]
    g1_out = sm[0]["outcome"]
    deaths_q4d = 0
    for d in dec:
        if d["game_id"] != g1:
            continue
        if not d["agent_decision"].startswith("Volunteer:"):
            continue
        for piece in d["agent_decision"].split(","):
            piece = piece.strip()
            idx = piece.find("Q4-")
            if idx >= 0:
                q4 = piece[idx:].split()[0]
                if q4.startswith("Q4-D") and d["agent_status_end_of_round"] == "DIED":
                    deaths_q4d += 1
                break
    if g1_out == "ALL_DEAD":
        g1_sev = "CATAS"
    elif deaths_q4d:
        g1_sev = "MEMBER"
    elif schedule == "smooth":
        g1_sev = "NA"
    else:
        g1_sev = "CAUGHT"

    # Survival (mean per-player survival %)
    survival = sum(float(r["avg_survival_pct"]) for r in fs) / len(fs)

    # Decision count (for sanity / mixed-effects)
    total_decisions = sum(1 for d in dec if d["agent_decision"] != "N/A")
    defaulted = sum(1 for d in dec if d.get("defaulted_to_pass") == "True")

    return dict(
        folder=folder,
        model=model,
        reasoning=reasoning,
        refl=refl,
        schedule=schedule,
        g1_sev=g1_sev,
        n_post=n_post,
        targ=round(targ, 4),
        nonq4_raw=round(nonq4_raw, 4),
        scarred=int(targ > SCAR_THRESHOLD),
        survival=round(survival, 2),
        total_decisions=total_decisions,
        defaulted_decisions=defaulted,
        api_errors=n_api_errors,
        api_error_rate=round(api_error_rate, 4),
    )


def main():
    rows = []
    skipped = 0
    for f in sorted(os.listdir(ROOT)):
        if "2025" not in f and "2026" not in f:
            continue
        if not any(f.startswith(p) for p in ("main_", "ablation_",
                                                 "rough_start_high_none_max_", "trust_converge_high_none_max_",
                                                 "rough_start_high_reflection_max_", "trust_converge_high_reflection_max_")):
            continue
        r = analyze(f)
        if r is None:
            skipped += 1
            continue
        rows.append(r)

    # === Analysis-set construction (standardization decisions) ===
    # (a) Restrict to the deliberation-default condition (refl=none); reflection
    #     conditions are not part of the analysis.
    # (b) Exclude the gpt-5.1-2025-11-13 direct-API cell (underpowered).
    # (c) Standardize sampling, keeping the earliest runs by timestamp: smooth
    #     (baseline) cells target n=5; rough-start (non-smooth) cells target n=10.
    def _ts(f):
        m = re.search(r"(\d{8})_(\d{6})$", f)
        return (m.group(1) + m.group(2)) if m else ""
    def _cap(schedule):
        return 5 if schedule == "smooth" else 10
    rows = [r for r in rows
            if r["refl"] == "none" and r["model"] != "gpt-5.1-2025-11-13"]
    by_cell = {}
    for r in rows:
        by_cell.setdefault((r["model"], r["schedule"]), []).append(r)
    capped = []
    for cell, rs in by_cell.items():
        capped.extend(sorted(rs, key=lambda r: _ts(r["folder"]))[:_cap(cell[1])])
    rows = capped

    # Compute SPILL (relative to same-snapshot smooth baseline)
    smooth_baseline = {}
    for r in rows:
        if r["schedule"] == "smooth":
            key = (r["model"], r["reasoning"], r["refl"])
            smooth_baseline.setdefault(key, []).append(r["nonq4_raw"])
    smooth_avg = {k: sum(v) / len(v) for k, v in smooth_baseline.items()}

    for r in rows:
        key = (r["model"], r["reasoning"], r["refl"])
        baseline = smooth_avg.get(key)
        # Fallback: same model, any reasoning/refl smooth
        if baseline is None:
            for k, v in smooth_avg.items():
                if k[0] == r["model"]:
                    baseline = v
                    break
        if baseline is None:
            r["spill"] = ""
        else:
            r["spill"] = round(r["nonq4_raw"] - baseline, 4)
        r["smooth_baseline_nonq4"] = round(baseline, 4) if baseline is not None else ""

    # Write CSV
    fieldnames = ["folder", "model", "reasoning", "refl", "schedule", "g1_sev",
                  "n_post", "targ", "nonq4_raw", "smooth_baseline_nonq4", "spill",
                  "scarred", "survival", "total_decisions", "defaulted_decisions",
                  "api_errors", "api_error_rate"]
    with open(OUT_CSV, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow(r)

    print(f"Wrote {len(rows)} rows to {OUT_CSV}")
    print(f"Skipped {skipped} folders (incomplete/excluded)")


if __name__ == "__main__":
    main()
