"""
Formal clustering analysis on run-level (TARG, SPILL) data.

Addresses reviewer critique that "clusters" were asserted without analysis.

Procedure:
  1. K-means with k=2..6 on standardized (TARG, SPILL) features
  2. Silhouette score selects best k
  3. Bootstrap stability via Adjusted Rand Index across resamples
  4. Per-cluster composition by model and schedule

Outputs:
  - clustering_results.json (cluster assignments, silhouettes, bootstrap ARI)
  - V2/paper/images/clustering_silhouette.pdf
  - V2/paper/images/clustering_scatter.pdf
  - V2/paper/images/clustering_bootstrap.pdf
"""
import csv, json, os, collections
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, adjusted_rand_score
from sklearn.preprocessing import StandardScaler

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
CSV_PATH = os.path.join(SCRIPT_DIR, "per_rep_metrics.csv")
IMG_DIR = os.path.join(SCRIPT_DIR, "..", "..", "paper", "images")
os.makedirs(IMG_DIR, exist_ok=True)

# Load
rows = list(csv.DictReader(open(CSV_PATH)))

# Filter to recovery-family runs (have meaningful TARG signal) with valid SPILL
recovery = [r for r in rows
            if r["schedule"] != "smooth"
            and r["spill"] != ""]
print(f"Loaded {len(recovery)} recovery-family runs with valid SPILL")

# Feature matrix
X_raw = np.array([[float(r["targ"]), float(r["spill"])] for r in recovery])
scaler = StandardScaler()
X = scaler.fit_transform(X_raw)

# === 1. K-means with k=2..6, silhouette ===
ks = [2, 3, 4, 5, 6]
silhouettes = []
inertias = []
labels_by_k = {}
for k in ks:
    km = KMeans(n_clusters=k, n_init=20, random_state=42)
    labels = km.fit_predict(X)
    sil = silhouette_score(X, labels)
    silhouettes.append(sil)
    inertias.append(km.inertia_)
    labels_by_k[k] = labels
    print(f"  k={k}: silhouette={sil:.3f}  inertia={km.inertia_:.2f}")

best_k = ks[int(np.argmax(silhouettes))]
print(f"\nBest k by silhouette: k={best_k}  (silhouette={max(silhouettes):.3f})")

# === 2. Silhouette plot ===
fig, ax = plt.subplots(figsize=(5, 3.2))
bars = ax.bar(ks, silhouettes, color=["#a8a8a8"]*len(ks))
bars[ks.index(best_k)].set_color("#2b5d8a")
ax.set_xlabel("k (number of clusters)")
ax.set_ylabel("Silhouette score")
ax.set_title("K-means cluster quality on (TARG, SPILL)", fontsize=10)
ax.set_ylim(0, max(silhouettes) * 1.2)
for k, s in zip(ks, silhouettes):
    ax.annotate(f"{s:.3f}", (k, s), ha="center", va="bottom", fontsize=8)
ax.set_xticks(ks)
fig.tight_layout()
fig.savefig(os.path.join(IMG_DIR, "clustering_silhouette.pdf"))
plt.close(fig)
print(f"Saved silhouette plot to clustering_silhouette.pdf")

# === 3. Bootstrap stability (Adjusted Rand Index) ===
B = 200
np.random.seed(42)
aris_by_k = {k: [] for k in ks}
ref_labels = {k: labels_by_k[k] for k in ks}
for b in range(B):
    idx = np.random.choice(len(X), size=len(X), replace=True)
    Xb = X[idx]
    for k in ks:
        km_b = KMeans(n_clusters=k, n_init=20, random_state=b)
        labels_b = km_b.fit_predict(Xb)
        # Compare on bootstrap-sampled subset of original labels
        # Use ARI of labels_b vs original-labels[idx]
        ari = adjusted_rand_score(ref_labels[k][idx], labels_b)
        aris_by_k[k].append(ari)

mean_aris = {k: np.mean(aris_by_k[k]) for k in ks}
print(f"\nBootstrap stability (mean ARI across {B} resamples):")
for k in ks:
    print(f"  k={k}: mean ARI={mean_aris[k]:.3f} ± {np.std(aris_by_k[k]):.3f}")

# Bootstrap stability plot
fig, ax = plt.subplots(figsize=(5, 3.2))
bp = ax.boxplot([aris_by_k[k] for k in ks], positions=ks, widths=0.5,
                patch_artist=True)
for patch, k in zip(bp["boxes"], ks):
    patch.set_facecolor("#2b5d8a" if k == best_k else "#a8a8a8")
    patch.set_alpha(0.6)
ax.set_xlabel("k (number of clusters)")
ax.set_ylabel("Adjusted Rand Index (bootstrap stability)")
ax.set_title(f"Cluster assignment stability ({B} bootstrap resamples)", fontsize=10)
ax.set_xticks(ks)
ax.set_ylim(0, 1.05)
ax.axhline(y=0.75, color="green", linestyle="--", alpha=0.4, linewidth=0.8)
ax.text(ks[-1]+0.3, 0.76, "ARI=0.75\n(stable)", fontsize=7, color="green")
fig.tight_layout()
fig.savefig(os.path.join(IMG_DIR, "clustering_bootstrap.pdf"))
plt.close(fig)
print(f"Saved bootstrap plot to clustering_bootstrap.pdf")

# === 4. Scatter plot with cluster colors at best_k ===
best_labels = labels_by_k[best_k]
cluster_colors = ["#e07b00", "#0073b0", "#7b3294", "#008837", "#c51b7d", "#5e8b00"][:best_k]
model_markers = {
    "claude-opus-4-6": "o",
    "claude-sonnet-4-6": "s",
    "gpt-5.4-mini-2026-03-17": "^",
    "gemini-3.1-pro-preview": "D",
    "gemini-2.5-flash": "v",
    "gpt5.1-high": "P",
    "gpt-5.1-2025-11-13": "X",
}
schedule_labels = {"1strike":"1","2strike":"2","3strike":"3","mid2strike":"M","recur":"R"}

fig, ax = plt.subplots(figsize=(7, 5.5))
for i, r in enumerate(recovery):
    color = cluster_colors[best_labels[i]]
    marker = model_markers.get(r["model"], "*")
    ax.scatter(float(r["targ"]), float(r["spill"]),
               c=color, marker=marker, s=55, alpha=0.7,
               edgecolors="black", linewidths=0.4)
ax.axhline(y=0, color="gray", linewidth=0.5, linestyle="--", alpha=0.5)
ax.axvline(x=0.5, color="gray", linewidth=0.5, linestyle="--", alpha=0.5)
ax.text(0.55, ax.get_ylim()[1]*0.95, "TARG=0.5 (scar threshold)", fontsize=7, color="gray")
ax.set_xlabel("TARG (Q4 verifications per post-failure game)", fontsize=10)
ax.set_ylabel("SPILL (non-Q4 verifications vs smooth baseline)", fontsize=10)
ax.set_title(f"Empirical clusters in (TARG, SPILL) space at k={best_k}\n"
             f"silhouette={silhouettes[ks.index(best_k)]:.3f}, "
             f"bootstrap ARI={mean_aris[best_k]:.3f}",
             fontsize=10)

# Legend for clusters
cluster_handles = [plt.Line2D([0],[0], marker="o", color="w",
                              markerfacecolor=cluster_colors[i], markersize=8,
                              label=f"Cluster {i+1}")
                   for i in range(best_k)]
# Legend for models
model_handles = [plt.Line2D([0],[0], marker=m, color="gray",
                            markersize=7, linestyle="",
                            label=name)
                 for name, m in model_markers.items()]
leg1 = ax.legend(handles=cluster_handles, loc="upper left", fontsize=8, title="Cluster")
ax.add_artist(leg1)
ax.legend(handles=model_handles, loc="upper right", fontsize=7, title="Model")
fig.tight_layout()
fig.savefig(os.path.join(IMG_DIR, "clustering_scatter.pdf"))
plt.close(fig)
print(f"Saved scatter plot to clustering_scatter.pdf")

# === 5. Per-cluster composition ===
print(f"\n=== Per-cluster composition at k={best_k} ===")
cluster_compo = collections.defaultdict(lambda: collections.Counter())
for r, label in zip(recovery, best_labels):
    cluster_compo[label][(r["model"], r["schedule"])] += 1

# Centroid TARG and SPILL per cluster
centroids_raw = scaler.inverse_transform(KMeans(n_clusters=best_k, n_init=20, random_state=42).fit(X).cluster_centers_)
for c_id in range(best_k):
    tg, sp = centroids_raw[c_id]
    n_in = sum(1 for l in best_labels if l == c_id)
    print(f"\nCluster {c_id+1}: centroid TARG={tg:.2f}, SPILL={sp:+.2f}, n={n_in}")
    for (m, s), c in sorted(cluster_compo[c_id].items()):
        print(f"  {m:<28s} {s:<12s} {c}")

# === Save results JSON ===
results = {
    "k_swept": ks,
    "silhouettes": [round(s, 4) for s in silhouettes],
    "inertias": [round(i, 2) for i in inertias],
    "bootstrap_ari_mean": {k: round(mean_aris[k], 4) for k in ks},
    "bootstrap_ari_std": {k: round(float(np.std(aris_by_k[k])), 4) for k in ks},
    "best_k": int(best_k),
    "best_k_silhouette": round(silhouettes[ks.index(best_k)], 4),
    "n_runs_clustered": len(recovery),
    "centroids_raw": [{"cluster_id": i+1, "targ": round(float(centroids_raw[i, 0]), 3),
                       "spill": round(float(centroids_raw[i, 1]), 3),
                       "n": sum(1 for l in best_labels if l == i)} for i in range(best_k)],
}
with open(os.path.join(SCRIPT_DIR, "clustering_results.json"), "w") as f:
    json.dump(results, f, indent=2)
print(f"\nSaved results to clustering_results.json")
