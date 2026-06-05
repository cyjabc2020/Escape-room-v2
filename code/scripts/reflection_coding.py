"""
Systematic coding of post-game reflection text.

For each reflection record, count mentions of:
  - RELATIONAL keywords: D-specific distrust language (D, partner, unreliable, untrustworthy, mistake, verify D, Q4)
  - PROCEDURAL keywords: timing / coordination / volunteer-faster language (volunteer, faster, slow, too cautious, time, round, immediately)

A reflection is classified as RELATIONAL-dominant, PROCEDURAL-dominant, or MIXED based on counts.

This provides a quantitative mechanism comparison for H4 (reflection direction
asymmetry between GPT-5.1 and Opus).
"""
import csv, json, os, re, collections

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT = os.environ.get("ESCAPE_RESULTS") or os.path.join(SCRIPT_DIR, "..", "results")

RELATIONAL_KEYWORDS = [
    r"\bD\b", r"\bplayer D\b", r"\bD's\b",
    r"\bunreliable\b", r"\buntrust", r"\bdistrust\b",
    r"\bmistake\b", r"\berror\b", r"\bwrong\b", r"\bincorrect\b",
    r"\bdoubt\b", r"\bsuspicious\b",
    r"\bQ4\b", r"\bquestion 4\b",
    r"\bverify D\b", r"\bcheck D\b",
]

PROCEDURAL_KEYWORDS = [
    r"\bvolunteer (?:faster|sooner|earlier)\b",
    r"\btoo slow\b", r"\btoo cautious\b", r"\btoo careful\b",
    r"\bnobody volunteered\b", r"\bno one volunteered\b",
    r"\bover.{0,5}verif", r"\bexcessive verif",
    r"\binformation without action\b",
    r"\bact (?:sooner|faster|now)\b",
    r"\bwasted? (?:round|time|coin)",
    r"\brandom (?:elimination|death)\b",
    r"\bdrag(?:ged|ging)? on\b",
]

REL_PATTERNS = [re.compile(k, re.IGNORECASE) for k in RELATIONAL_KEYWORDS]
PROC_PATTERNS = [re.compile(k, re.IGNORECASE) for k in PROCEDURAL_KEYWORDS]


def count_matches(text, patterns):
    return sum(len(p.findall(text)) for p in patterns)


def classify(text):
    if not text or len(text.strip()) < 20:
        return None, 0, 0
    rel = count_matches(text, REL_PATTERNS)
    proc = count_matches(text, PROC_PATTERNS)
    if rel == 0 and proc == 0:
        return "EMPTY", 0, 0
    # Margin-based classification
    if rel >= proc * 1.5:
        return "RELATIONAL", rel, proc
    elif proc >= rel * 1.5:
        return "PROCEDURAL", rel, proc
    else:
        return "MIXED", rel, proc


def analyze_run(folder):
    R = os.path.join(ROOT, folder)
    refl_path = os.path.join(R, "game_data_reflections.csv")
    settings_path = os.path.join(R, "experiment_settings.json")
    if not os.path.isfile(refl_path) or not os.path.isfile(settings_path):
        return None
    try:
        s = json.load(open(settings_path))
        refls = list(csv.DictReader(open(refl_path)))
    except Exception:
        return None
    if not refls:
        return None
    refl_mode = s.get("memory_config", {}).get("A", {}).get("reflection", "none")
    if refl_mode != "evolving":
        return None  # only analyze runs where reflections were actually generated
    pm = s.get("player_models", ["?"])[0]
    model = "gpt5.1-high" if pm == "openai/gpt-5.1" else pm.split(":")[-1] if ":" in pm else pm
    dummy = s.get("dummy_config", {}).get("D", {}).get("correctness_list", [])
    scenario = "smooth" if all(dummy) else "recovery"

    out = []
    for r in refls:
        if r.get("agent_id") == "D":
            continue  # dummy has no reflection
        text = r.get("reflection", "")
        cat, rel_n, proc_n = classify(text)
        if cat is None:
            continue
        out.append({
            "folder": folder,
            "model": model,
            "scenario": scenario,
            "game_id": r.get("game_id"),
            "agent_id": r.get("agent_id"),
            "category": cat,
            "rel_count": rel_n,
            "proc_count": proc_n,
            "text_len": len(text),
        })
    return out


all_records = []
for f in sorted(os.listdir(ROOT)):
    recs = analyze_run(f)
    if recs:
        all_records.extend(recs)

print(f"Total reflection records analyzed: {len(all_records)}")
print(f"Across {len(set(r['folder'] for r in all_records))} runs")

# By model
print(f"\n=== Reflection content classification by model (recovery scenario only) ===")
by_model = collections.defaultdict(lambda: collections.Counter())
for r in all_records:
    if r["scenario"] != "recovery":
        continue
    by_model[r["model"]][r["category"]] += 1
print(f"\n{'model':<28s} {'n_refl':<8s} {'RELATIONAL':<12s} {'PROCEDURAL':<12s} {'MIXED':<8s} {'EMPTY':<8s}")
for model in sorted(by_model.keys()):
    cats = by_model[model]
    total = sum(cats.values())
    rel_pct = cats["RELATIONAL"] / total * 100 if total else 0
    proc_pct = cats["PROCEDURAL"] / total * 100 if total else 0
    mixed_pct = cats["MIXED"] / total * 100 if total else 0
    empty_pct = cats["EMPTY"] / total * 100 if total else 0
    print(f"{model:<28s} {total:<8d} {cats['RELATIONAL']:>3d} ({rel_pct:>4.0f}%)  {cats['PROCEDURAL']:>3d} ({proc_pct:>4.0f}%)  {cats['MIXED']:>3d}     {cats['EMPTY']:>3d}")

# Per-run by game position (do later-game reflections look different from early?)
print(f"\n=== Reflection content by game position (recovery only) ===")
by_pos = collections.defaultdict(lambda: collections.Counter())
for r in all_records:
    if r["scenario"] != "recovery": continue
    # game_id is timestamp; we need ordinal. Use folder + sort
    pass  # skipping for brevity; can be added if needed

# Save full record CSV
out_csv = os.path.join(SCRIPT_DIR, "reflection_coded.csv")
with open(out_csv, "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=["folder","model","scenario","game_id","agent_id","category","rel_count","proc_count","text_len"])
    w.writeheader()
    for r in all_records: w.writerow(r)
print(f"\nSaved reflection_coded.csv ({len(all_records)} records)")

# Summary JSON
summary = {}
for model, cats in by_model.items():
    total = sum(cats.values())
    summary[model] = {
        "n_reflections": total,
        "relational_count": cats["RELATIONAL"],
        "procedural_count": cats["PROCEDURAL"],
        "mixed_count": cats["MIXED"],
        "empty_count": cats["EMPTY"],
        "relational_pct": round(cats["RELATIONAL"] / total * 100, 1) if total else 0,
        "procedural_pct": round(cats["PROCEDURAL"] / total * 100, 1) if total else 0,
    }
with open(os.path.join(SCRIPT_DIR, "reflection_summary.json"), "w") as f:
    json.dump(summary, f, indent=2)
print(f"Saved reflection_summary.json")
