"""
Bayesian normative comparator for trust recovery.

For each (schedule, game) condition, computes what a Bayes-optimal verifier
would do given:
  - Prior: Beta(alpha=1, beta=1) over D's per-game reliability (uniform)
  - Likelihood: observed D-correctness in prior games (revealed through successful escapes)
  - Cost structure: verify costs 1 coin; trusting a wrong D-capsule means the volunteer dies
  - Decision rule: verify Q4 if expected utility of verifying > expected utility of trusting

This provides a normative baseline TARG curve (verification rate per game) under
optimal Bayesian updating. We then compare observed LLM TARG to this baseline:
  - LLM TARG above Bayes-optimal -> over-reactive (irrational scarring)
  - LLM TARG at Bayes-optimal -> calibrated
  - LLM TARG below Bayes-optimal -> under-reactive (irrational forgetting)

Game cost structure (approximate):
  - Verify cost = 1 coin (out of 4 starting coins)
  - Wrong-volunteer death = total payoff loss
  - Random elimination probability (no-volunteer) approximately 25% per round
  - Pass cost (no info gain, exposes to random death)

We treat each game as: agent decides whether to spend 1 coin verifying Q4 before
the team's eventual volunteer attempt. The decision pivots on:
  P(D wrong) * V(detected wrong via verify) > cost(verify)

where V(...) is the expected utility delta from catching D's mistake.

Outputs:
  - bayesian_baseline.csv: per-schedule per-game expected verify rate
  - V2/paper/images/bayesian_vs_llm.pdf: comparison plot
"""
import csv, json, os, math
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
CSV_PATH = os.path.join(SCRIPT_DIR, "per_rep_metrics.csv")
IMG_DIR = os.path.join(SCRIPT_DIR, "..", "..", "paper", "images")

# Schedules studied
SCHEDULES = {
    "smooth":     [True]*11,
    "1strike":    [False] + [True]*10,
    "2strike":    [False, False] + [True]*9,
    "3strike":    [False]*3 + [True]*8,
    "mid2strike": [True]*3 + [False]*2 + [True]*6,
    "recur":      [False] + [True]*3 + [False] + [True]*6,
}

# Cost parameters (in coins; one volunteer death = "many coins" loss)
COST_VERIFY = 1.0  # cost of one verification action
DEATH_PENALTY = 4.0  # coin-equivalent loss if volunteer dies (could be tuned)
# Without verification, an agent volunteering an unverified Q4-D capsule that's wrong dies
# But the agent has the OPTION to pass instead, so cost of "not verifying" depends on
# downstream decisions. For simplicity we treat: an agent that doesn't verify Q4 will
# face a coin-flip on volunteering vs passing. We approximate the expected loss from
# NOT verifying as: P(D wrong) * P(team eventually volunteers Q4-D) * DEATH_PENALTY


def posterior(n_correct, n_failed, prior_a=1.0, prior_b=1.0):
    """Beta posterior over D's reliability. Returns (mean, alpha, beta)."""
    a = prior_a + n_correct
    b = prior_b + n_failed
    return a / (a + b), a, b


def expected_verify_utility(p_d_wrong, p_team_volunteers_q4d=0.5):
    """Expected utility of verifying Q4 (vs. not).
    If verify: spend COST_VERIFY, learn truth, can switch to correct capsule.
    If not verify: face P(D wrong) * P(team uses Q4-D) * DEATH_PENALTY in expectation.
    """
    expected_loss_not_verify = p_d_wrong * p_team_volunteers_q4d * DEATH_PENALTY
    expected_loss_verify = COST_VERIFY
    # Verify if loss-from-not-verifying > loss-from-verifying
    return expected_loss_not_verify - expected_loss_verify


def bayes_optimal_verify(schedule, observe_game_outcomes=True):
    """For each game in the schedule, compute Bayes-optimal verify decision.

    Assumption: agent observes prior D-game outcomes (game success/failure reveals
    whether D's answer was correct, IF the team caught and corrected it, OR the
    team trusted D and outcome reflects D's correctness).

    For simplicity we assume: after each game, the agent learns whether D was correct
    that game (true Bayesian update). This is the upper-bound on information.

    Returns: list of (game_idx, p_d_wrong, expected_utility, verify_decision)
    """
    decisions = []
    # Prior: uniform Beta(1,1)
    n_correct_observed = 0
    n_failed_observed = 0
    for game_idx, d_correct_this_game in enumerate(schedule):
        # Decide BEFORE observing this game's outcome
        p_d_correct, _, _ = posterior(n_correct_observed, n_failed_observed)
        p_d_wrong = 1 - p_d_correct
        eu = expected_verify_utility(p_d_wrong)
        verify = eu > 0
        decisions.append({
            "game_idx": game_idx + 1,
            "p_d_wrong_prior": round(p_d_wrong, 4),
            "expected_utility": round(eu, 4),
            "verify": int(verify),
            "n_correct_seen": n_correct_observed,
            "n_failed_seen": n_failed_observed,
        })
        # Now observe outcome
        if observe_game_outcomes:
            if d_correct_this_game:
                n_correct_observed += 1
            else:
                n_failed_observed += 1
    return decisions


# Compute Bayes-optimal verify rate per schedule (averaged over post-failure games)
print("=== Bayes-optimal verify decisions by schedule ===")
print(f"(Assumes uniform Beta(1,1) prior, COST_VERIFY={COST_VERIFY}, DEATH_PENALTY={DEATH_PENALTY})")
print()
results = {}
for sched_name, sched in SCHEDULES.items():
    decisions = bayes_optimal_verify(sched)
    # Post-failure window: games after last D-failure
    last_fail = max((i for i, c in enumerate(sched) if not c), default=-1)
    post_window = decisions[last_fail + 1:] if last_fail >= 0 else decisions
    verify_rate = np.mean([d["verify"] for d in post_window]) if post_window else 0
    # Average P(D wrong) in post-failure window
    avg_p_wrong = np.mean([d["p_d_wrong_prior"] for d in post_window]) if post_window else 0
    results[sched_name] = {
        "schedule": sched,
        "decisions": decisions,
        "post_failure_verify_rate": verify_rate,
        "avg_p_d_wrong_post": avg_p_wrong,
    }
    print(f"{sched_name:<12s}: post-failure verify rate = {verify_rate*100:.0f}%  "
          f"avg P(D wrong) = {avg_p_wrong:.3f}  "
          f"n post-failure games = {len(post_window)}")

# Compare to observed LLM TARG
print()
print("=== Bayesian baseline vs observed LLM TARG (per schedule, recovery family) ===")
rows = list(csv.DictReader(open(CSV_PATH)))

import collections
by_schedule = collections.defaultdict(list)
for r in rows:
    if r["schedule"] == "smooth": continue
    if r["refl"] != "none": continue
    by_schedule[(r["model"], r["schedule"])].append(float(r["targ"]))

print(f"{'cell':<55s} {'n':<4s} {'LLM TARG':<12s} {'Bayes verify rate':<18s} {'LLM/Bayes ratio'}")
comparisons = []
for (model, sched), targs in sorted(by_schedule.items()):
    n = len(targs)
    if n < 3: continue
    llm_mean = np.mean(targs)
    bayes_rate = results.get(sched, {}).get("post_failure_verify_rate", float("nan"))
    if math.isnan(bayes_rate) or bayes_rate == 0:
        ratio_str = "N/A"
    else:
        # bayes_rate is per-game prob of verify; LLM TARG can be > 1 (multiple agents verify)
        ratio_str = f"{llm_mean / bayes_rate:.2f}x"
    cell = f"{model}/{sched}"
    print(f"{cell:<55s} {n:<4d} {llm_mean:<12.2f} {bayes_rate:<18.2f} {ratio_str}")
    comparisons.append({"model": model, "schedule": sched, "n": n, "llm_targ": round(llm_mean, 3),
                        "bayes_verify_rate": round(bayes_rate, 3),
                        "ratio": round(llm_mean / bayes_rate, 3) if bayes_rate > 0 else None})

# Save outputs
with open(os.path.join(SCRIPT_DIR, "bayesian_baseline.json"), "w") as f:
    json.dump({
        "params": {"cost_verify": COST_VERIFY, "death_penalty": DEATH_PENALTY,
                   "prior": "Beta(1,1)", "p_team_uses_q4d_assumption": 0.5},
        "per_schedule": {k: {"post_failure_verify_rate": v["post_failure_verify_rate"],
                             "avg_p_d_wrong_post": v["avg_p_d_wrong_post"]}
                         for k, v in results.items()},
        "comparisons": comparisons,
    }, f, indent=2)
print(f"\nSaved bayesian_baseline.json")

# Plot
fig, ax = plt.subplots(figsize=(8, 5))
schedules_order = ["1strike", "2strike", "3strike", "mid2strike", "recur"]
models_to_show = ["claude-opus-4-6", "claude-sonnet-4-6", "gemini-3.1-pro-preview",
                  "gemini-2.5-flash", "gpt-5.4-mini-2026-03-17", "gpt5.1-high"]
colors = {"claude-opus-4-6": "#ad5500", "claude-sonnet-4-6": "#e09000",
          "gpt-5.4-mini-2026-03-17": "#5cbac4", "gemini-3.1-pro-preview": "#4d4daf",
          "gemini-2.5-flash": "#8a8acf", "gpt5.1-high": "#0a7b85"}

x_pos = np.arange(len(schedules_order))
bar_width = 0.11
for i, model in enumerate(models_to_show):
    means = []
    errs_lo = []
    errs_hi = []
    for sched in schedules_order:
        targs = [float(r["targ"]) for r in rows if r["model"] == model and r["schedule"] == sched and r["refl"] == "none"]
        if not targs:
            means.append(0); errs_lo.append(0); errs_hi.append(0)
            continue
        m = np.mean(targs)
        sd = np.std(targs, ddof=1) if len(targs) > 1 else 0
        se = sd / math.sqrt(len(targs)) if len(targs) > 1 else 0
        means.append(m); errs_lo.append(1.96*se); errs_hi.append(1.96*se)
    offset = (i - len(models_to_show)/2) * bar_width
    ax.bar(x_pos + offset, means, bar_width, label=model.split("-")[0]+"/"+(model.split("-")[1] if len(model.split("-")) > 1 else ""),
           yerr=[errs_lo, errs_hi], capsize=2, color=colors.get(model, "gray"), alpha=0.85,
           edgecolor="black", linewidth=0.3, error_kw={"linewidth": 0.5})

# Bayes-optimal line
bayes_rates = [results[s]["post_failure_verify_rate"] for s in schedules_order]
ax.plot(x_pos, bayes_rates, "k--", marker="*", markersize=11, linewidth=1.5,
        label="Bayes-optimal (single agent)", zorder=5)

ax.set_xticks(x_pos)
ax.set_xticklabels(schedules_order)
ax.set_ylabel("TARG (Q4 verifies per post-failure game)", fontsize=10)
ax.set_title(f"Observed LLM TARG vs Bayes-optimal verify rate\n"
             f"(prior Beta(1,1), verify cost=1, death penalty={DEATH_PENALTY})",
             fontsize=10)
ax.legend(fontsize=8, loc="upper right", ncol=2)
ax.grid(alpha=0.3, axis="y")
fig.tight_layout()
fig.savefig(os.path.join(IMG_DIR, "bayesian_vs_llm.pdf"))
plt.close(fig)
print(f"Saved bayesian_vs_llm.pdf")
