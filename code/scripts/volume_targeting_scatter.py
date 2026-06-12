#!/usr/bin/env python3
"""Summary landscape: trust-formation (volume) vs culprit-targeting.
x = Delta_trust on verification VOLUME (smooth - anchor, verifies/game):
    more negative = stronger trust formation (checks a reliable partner much less).
y = Delta_scar on Q4-SHARE (1-strike - anchor): more positive = after a failure,
    verification concentrates on the culprit (D = Q4).
Reads anchor_deltas.csv (rows per model x metric).
"""
import csv, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.rcParams.update({"font.size":13,"axes.titlesize":14,"axes.labelsize":13,"xtick.labelsize":12,"ytick.labelsize":12,"legend.fontsize":12,"figure.titlesize":15})

SD = os.path.dirname(os.path.abspath(__file__))
IMG = os.path.join(SD, "figures"); os.makedirs(IMG, exist_ok=True)
rows = list(csv.DictReader(open(os.path.join(SD, "anchor_deltas.csv"))))
vol = {r["model"]: r for r in rows if r["metric"] == "suspicion"}
tgt = {r["model"]: r for r in rows if r["metric"] == "Q4-share"}

MODELS = ["Opus", "Sonnet", "GPT-5.1", "Gemini Pro", "GPT-5.4-mini", "Gemini Flash"]
# grouping for color: forms-trust+targets / forms-trust-only / fails-to-form
GRP = {"Opus": "form", "Sonnet": "form", "GPT-5.1": "both", "Gemini Pro": "both",
       "GPT-5.4-mini": "none", "Gemini Flash": "none"}
COL = {"form": "#1f6f3e", "both": "#b03060", "none": "#7f7f7f"}
LAB = {"form": "Forms trust (volume), broad re-checking",
       "both": "Forms trust + targets culprit",
       "none": "Weak / no trust formation"}

fig, ax = plt.subplots(figsize=(8.2, 6.0))
ax.axhline(0, color="0.7", lw=1, zorder=0)
ax.axvline(0, color="0.7", lw=1, zorder=0)
seen = set()
for m in MODELS:
    x = float(vol[m]["delta_trust"]); xlo, xhi = float(vol[m]["trust_lo"]), float(vol[m]["trust_hi"])
    y = float(tgt[m]["delta_scar"]); ylo, yhi = float(tgt[m]["scar_lo"]), float(tgt[m]["scar_hi"])
    g = GRP[m]
    lab = LAB[g] if g not in seen else None; seen.add(g)
    ax.errorbar(x, y, xerr=[[x - xlo], [xhi - x]], yerr=[[y - ylo], [yhi - y]],
                fmt="o", ms=9, color=COL[g], ecolor=COL[g], elinewidth=1.2, capsize=3, alpha=0.9,
                label=lab, zorder=3)
    ax.annotate(m, (x, y), textcoords="offset points", xytext=(8, 6),
                fontsize=13.5, fontweight="bold", color=COL[g])

ax.text(-3.4, -0.10, "strong trust formation\n(checks reliable D much less)",
        fontsize=10.8, color="0.4", style="italic", ha="left")
ax.text(-0.2, 0.34, "culprit-specific\ntargeting after failure", fontsize=10.8, color="0.4", style="italic")
ax.set_xlabel(r"trust formation:  $\Delta$ verification volume, smooth $-$ no-history "
              "(verifies/game)\n(more negative $=$ verifies a reliable partner much less)", fontsize=13.5)
ax.set_ylabel(r"culprit targeting:  $\Delta$ Q4-share, 1-strike $-$ no-history" "\n"
              r"(positive $=$ singles out the once-failed partner)", fontsize=13.5)
ax.set_title("Two axes of trust behavior across frontier models:\n"
             "forming trust (verifying less) vs. targeting a once-failed partner", fontsize=14.9)
ax.legend(loc="upper left", fontsize=11.5, frameon=True, framealpha=0.9)
ax.margins(0.16)
fig.tight_layout()
fig.savefig(os.path.join(IMG, "volume_targeting_scatter.pdf"))
print("wrote volume_targeting_scatter.pdf")
