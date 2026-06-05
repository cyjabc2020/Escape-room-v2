#!/usr/bin/env bash
#
# regenerate.sh — rebuild every analysis artifact and the paper from raw run data.
#
# Run this after adding new experiment runs (e.g. top-up batches). It rebuilds the
# canonical metrics file, all statistics, clustering, and figures, then recompiles
# the paper PDF. The metrics step applies the standard sampling rules automatically
# (rough-start cells capped at the earliest 10 runs, smooth cells at the earliest 5;
# gpt-5.1-2025-11-13 and reflection cells excluded), so simply launching more runs
# and re-running this script keeps everything consistent.
#
# Usage:
#   cd V2/code/scripts
#   ./regenerate.sh
#
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PAPER_DIR="$(cd "${SCRIPT_DIR}/../../paper" && pwd)"
cd "${SCRIPT_DIR}"

# Optionally activate the shared venv if present (harmless if already active).
VENV="${SCRIPT_DIR}/../../../venv/bin/activate"
if [ -f "${VENV}" ]; then
  # shellcheck disable=SC1090
  source "${VENV}"
fi

PY="${PYTHON:-python3}"

echo "=== [1/6] compute_metrics.py (rebuild per_rep_metrics.csv) ==="
"${PY}" compute_metrics.py

echo "=== [2/6] statistical_models.py (pairwise tests + logit) ==="
"${PY}" statistical_models.py

echo "=== [3/6] clustering_analysis.py (k-sweep, clusters, scatter) ==="
"${PY}" clustering_analysis.py

echo "=== [4/6] clustering_robustness.py (alt methods at best k) ==="
"${PY}" clustering_robustness.py

echo "=== [5/6] paper_figures.py (survival, threshold, trajectory) ==="
"${PY}" paper_figures.py

echo "=== [6/6] bayesian_baseline.py (normative reference + figure) ==="
"${PY}" bayesian_baseline.py

echo
echo "=== Analysis artifacts rebuilt. Key numbers: ==="
"${PY}" - <<'PYEOF'
import csv, json, os
from collections import Counter
here = os.path.dirname(os.path.abspath("compute_metrics.py"))
rows = list(csv.DictReader(open(os.path.join(here, "per_rep_metrics.csv"))))
cells = Counter((r["model"], r["schedule"]) for r in rows)
print(f"  total runs: {len(rows)}   cells: {len(cells)}")
below = [f"{m}/{s}={n}" for (m, s), n in sorted(cells.items())
         if (n < 5 if s == "smooth" else n < 10)]
print("  cells below target:", ", ".join(below) if below else "none")
try:
    cj = json.load(open(os.path.join(here, "clustering_results.json")))
    print(f"  clustering: n={cj['n_runs_clustered']} best_k={cj['best_k']} "
          f"silhouette={cj['best_k_silhouette']}")
except Exception as e:
    print("  clustering summary unavailable:", e)
PYEOF

echo
echo "=== Recompiling paper (${PAPER_DIR}/preprint.pdf) ==="
if command -v pdflatex >/dev/null 2>&1; then
  cd "${PAPER_DIR}"
  pdflatex -interaction=nonstopmode preprint.tex >/dev/null 2>&1 || true
  if command -v bibtex >/dev/null 2>&1; then
    bibtex preprint >/dev/null 2>&1 || true
  fi
  pdflatex -interaction=nonstopmode preprint.tex >/dev/null 2>&1 || true
  pdflatex -interaction=nonstopmode preprint.tex >/dev/null 2>&1 || true
  if grep -qiE 'undefined (citation|reference)' preprint.log 2>/dev/null; then
    echo "  WARNING: undefined citations/references — check preprint.log"
  fi
  echo "  Done: $(ls -1 preprint.pdf)"
else
  echo "  pdflatex not found; skipped PDF recompile. Numbers/figures are updated;"
  echo "  recompile the paper manually where LaTeX is available."
fi

echo
echo "All done."
