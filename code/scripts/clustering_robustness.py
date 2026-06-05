"""
Alternative clustering methods as robustness check on the k-means result.

Tests:
  1. Gaussian Mixture Model (GMM) — relaxes spherical cluster assumption
  2. Hierarchical (Ward linkage) — different distance metric, gives dendrogram

For each method, compute Adjusted Rand Index against the k-means assignments
to test whether the cluster structure is method-robust.
"""
import csv, json, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.mixture import GaussianMixture
from sklearn.metrics import adjusted_rand_score, silhouette_score
from sklearn.preprocessing import StandardScaler
from scipy.cluster.hierarchy import dendrogram, linkage

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
CSV_PATH = os.path.join(SCRIPT_DIR, "per_rep_metrics.csv")
IMG_DIR = os.path.join(SCRIPT_DIR, "..", "..", "paper", "images")

rows = list(csv.DictReader(open(CSV_PATH)))
recovery = [r for r in rows if r["schedule"] != "smooth" and r["spill"] != ""]
X_raw = np.array([[float(r["targ"]), float(r["spill"])] for r in recovery])
X = StandardScaler().fit_transform(X_raw)

print(f"Robustness analysis on n={len(recovery)} recovery runs in 2D (TARG, SPILL)")

import json as _json
K = _json.load(open(os.path.join(SCRIPT_DIR,"clustering_results.json")))["best_k"]
# Reference: k-means at best k
km_labels = KMeans(n_clusters=K, n_init=20, random_state=42).fit_predict(X)
km_sil = silhouette_score(X, km_labels)

# GMM
gmm = GaussianMixture(n_components=K, n_init=20, random_state=42, covariance_type="full")
gmm_labels = gmm.fit_predict(X)
gmm_sil = silhouette_score(X, gmm_labels)
gmm_ari = adjusted_rand_score(km_labels, gmm_labels)

# Hierarchical (Ward)
hier_labels = AgglomerativeClustering(n_clusters=K, linkage="ward").fit_predict(X)
hier_sil = silhouette_score(X, hier_labels)
hier_ari = adjusted_rand_score(km_labels, hier_labels)

# Hierarchical (average)
hier_avg_labels = AgglomerativeClustering(n_clusters=K, linkage="average").fit_predict(X)
hier_avg_sil = silhouette_score(X, hier_avg_labels)
hier_avg_ari = adjusted_rand_score(km_labels, hier_avg_labels)

print(f"\n=== Alternative clustering methods, k={K} ===")
print(f"{'method':<28s} {'silhouette':<12s} {'ARI vs k-means':<15s}")
print(f"{'k-means (reference)':<28s} {km_sil:<12.3f} {'1.000 (self)':<15s}")
print(f"{'Gaussian Mixture (full cov)':<28s} {gmm_sil:<12.3f} {gmm_ari:<15.3f}")
print(f"{'Hierarchical (Ward)':<28s} {hier_sil:<12.3f} {hier_ari:<15.3f}")
print(f"{'Hierarchical (average)':<28s} {hier_avg_sil:<12.3f} {hier_avg_ari:<15.3f}")

# Save
results = {
    "n_runs": len(recovery),
    "k": K,
    "kmeans_silhouette": round(km_sil, 4),
    "gmm_silhouette": round(gmm_sil, 4),
    "gmm_ari_vs_kmeans": round(gmm_ari, 4),
    "hier_ward_silhouette": round(hier_sil, 4),
    "hier_ward_ari_vs_kmeans": round(hier_ari, 4),
    "hier_avg_silhouette": round(hier_avg_sil, 4),
    "hier_avg_ari_vs_kmeans": round(hier_avg_ari, 4),
}
with open(os.path.join(SCRIPT_DIR, "clustering_robustness.json"), "w") as f:
    json.dump(results, f, indent=2)
print(f"\nSaved clustering_robustness.json")

# Dendrogram for hierarchical
Z = linkage(X, method="ward")
fig, ax = plt.subplots(figsize=(7, 4))
dendrogram(Z, no_labels=True, color_threshold=Z[-4, 2], ax=ax)
ax.set_title("Hierarchical clustering dendrogram (Ward linkage)", fontsize=10)
ax.set_ylabel("Distance", fontsize=9)
ax.set_xlabel(f"runs (n={len(recovery)})", fontsize=9)
fig.tight_layout()
fig.savefig(os.path.join(IMG_DIR, "clustering_dendrogram.pdf"))
plt.close(fig)
print(f"Saved clustering_dendrogram.pdf")

# Comparison scatter: k-means vs GMM vs Hierarchical labels
fig, axes = plt.subplots(1, 3, figsize=(13, 4))
methods = [("k-means", km_labels), ("GMM (full cov)", gmm_labels), ("Hierarchical (Ward)", hier_labels)]
colors_palette = ["#e07b00", "#0073b0", "#7b3294", "#008837", "#c51b7d"]
for ax, (name, labels) in zip(axes, methods):
    for i, r in enumerate(recovery):
        c = colors_palette[labels[i] % 5]
        ax.scatter(float(r["targ"]), float(r["spill"]), c=c, s=35, alpha=0.7, edgecolors="black", linewidths=0.3)
    ax.set_title(f"{name}\nsilhouette={silhouette_score(X, labels):.3f}", fontsize=10)
    ax.set_xlabel("TARG")
    ax.set_ylabel("SPILL")
    ax.grid(alpha=0.3)
fig.tight_layout()
fig.savefig(os.path.join(IMG_DIR, "clustering_method_comparison.pdf"))
plt.close(fig)
print(f"Saved clustering_method_comparison.pdf")
