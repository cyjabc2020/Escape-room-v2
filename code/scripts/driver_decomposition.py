#!/usr/bin/env python3
"""Decompose game score into its concrete drivers, per model (pooled over the
perturbed scenarios). Explains WHY some models score high and others low.

Columns:
  verif_per_game   verifications by A/B/C per game (coin cost)
  rounds_per_game  rounds the game lasts (longer = more dithering)
  vol_per_game     Volunteer actions by A/B/C per game (commitment)
  redundancy       verifications per DISTINCT puzzle verified (>1 = repeated work)
  pct_die_novol    % of A/B/C deaths from no-volunteer random elimination (indecision)
  pct_die_wrong    % of A/B/C deaths from volunteering a wrong password (decisive error)
  survival_pct     A/B/C agent-game survival (from simulation_scores.csv)
  coins_per_agent  A/B/C coins banked per agent per game (the score)
Writes driver_table.csv.
"""
import csv, json, glob, os
from collections import defaultdict

SD = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(SD, "..", "results")
SKIP = ("APIERROR", "SUPERSEDED", "CONTAMINATED", "to_be_deleted", "default_experiment")

def clean(pm):
    pm = pm.split(":")[-1].split("/")[-1]
    return {"claude-opus-4-6": "Opus", "claude-sonnet-4-6": "Sonnet",
            "gemini-3.1-pro-preview": "Gemini Pro", "gemini-2.5-flash": "Gemini Flash",
            "gpt-5.4-mini-2026-03-17": "GPT-5.4-mini", "gpt-5.1": "GPT-5.1"}.get(pm, pm)

def perturbed(f):
    try: s = json.load(open(os.path.join(f, "experiment_settings.json")))
    except Exception: return None
    mc = s.get("memory_config", {}).get("A", {})
    if mc.get("persist_across_games") is False: return None
    d = s.get("dummy_config", {}).get("D", {}).get("correctness_list", [])
    if sum(1 for x in d if not x) == 0: return None   # perturbed only
    return clean(s.get("player_models", ["?"])[0])

A = defaultdict(lambda: {"games": 0, "verif": 0, "rounds": 0, "vol": 0, "rn": 0, "rd": 0,
                         "die_nov": 0, "die_wrong": 0, "surv": 0, "sn": 0, "coins": 0.0})
for f in glob.glob(os.path.join(RES, "*/")):
    if os.path.basename(f.rstrip("/")).startswith(SKIP): continue
    m = perturbed(f)
    if not m: continue
    dp = os.path.join(f, "game_data_agent_decisions.csv")
    if not os.path.exists(dp): continue
    rows = list(csv.DictReader(open(dp)))
    if not rows: continue
    a = A[m]
    bygame = defaultdict(list)
    for r in rows: bygame[r["game_id"]].append(r)
    for gid, rs in bygame.items():
        a["games"] += 1
        a["rounds"] += max(int(r["round"]) for r in rs)
        pv = defaultdict(int)
        for r in rs:
            if r["agent_id"] not in ("A", "B", "C"): continue
            d = r["agent_decision"]
            if d.startswith("Verify:"):
                a["verif"] += 1; pv[d.split(":")[1].strip()[:2]] += 1
            elif d.startswith("Volunteer"):
                a["vol"] += 1
            if r["agent_status_end_of_round"] == "DIED":
                if r["death_reason"] == "no_volunteer": a["die_nov"] += 1
                else: a["die_wrong"] += 1
        if pv:
            a["rn"] += sum(pv.values()); a["rd"] += len(pv)
    sp = os.path.join(f, "simulation_scores.csv")
    if os.path.exists(sp):
        for r in csv.DictReader(open(sp)):
            if r["player"] in ("A", "B", "C"):
                a["sn"] += 1; a["coins"] += float(r["coins"]); a["surv"] += (r["survival"] == "True")

MO = ["GPT-5.1", "Opus", "Sonnet", "GPT-5.4-mini", "Gemini Pro", "Gemini Flash"]
hdr = ["model", "verif_per_game", "rounds_per_game", "vol_per_game", "redundancy",
       "pct_die_novol", "pct_die_wrong", "survival_pct", "coins_per_agent"]
out = [hdr]
print(f"{'model':14}{'verif/g':>8}{'rnds/g':>7}{'vol/g':>7}{'redund':>8}{'%noVol':>8}{'%wrong':>8}{'surv%':>7}{'coins':>7}")
for m in MO:
    a = A[m]; g = a["games"]; dt = a["die_nov"] + a["die_wrong"]
    row = [m, round(a["verif"]/g, 2), round(a["rounds"]/g, 2), round(a["vol"]/g, 2),
           round(a["rn"]/a["rd"], 2), round(100*a["die_nov"]/dt) if dt else 0,
           round(100*a["die_wrong"]/dt) if dt else 0, round(100*a["surv"]/a["sn"], 1),
           round(a["coins"]/a["sn"], 2)]
    out.append(row)
    print(f"{m:14}{row[1]:>8}{row[2]:>7}{row[3]:>7}{row[4]:>8}{row[5]:>8}{row[6]:>8}{row[7]:>7}{row[8]:>7}")
with open(os.path.join(SD, "driver_table.csv"), "w", newline="") as fh:
    csv.writer(fh).writerows(out)
print("\nwrote driver_table.csv")
