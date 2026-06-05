#!/usr/bin/env bash
# Run N replicate sessions of a single settings file.
#
# Usage:
#   ./run_batch.sh <N> <path_to_settings.json>
#
# Example:
#   ./run_batch.sh 20 settings/main/main_gemini_recovery.json
#
# Each session writes to its own timestamped folder under results/.
# Per-session stdout logs are saved under /tmp/escape_logs/.
# Press Ctrl-C to stop the loop gracefully (the current session finishes,
# then the loop exits without starting the next one).

set -uo pipefail

N="${1:-}"
CFG="${2:-}"

if [[ -z "$N" || -z "$CFG" ]]; then
  echo "Usage: $0 <N> <path_to_settings.json>"
  echo "Example: $0 20 settings/main/main_gemini_recovery.json"
  exit 1
fi

if [[ ! -f "$CFG" ]]; then
  echo "ERROR: settings file not found: $CFG"
  exit 1
fi

TAG=$(basename "$CFG" .json)
LOG_DIR="/tmp/escape_logs"
mkdir -p "$LOG_DIR"

# Trap Ctrl-C: let the current `python` finish, then exit the loop.
# Without this, Ctrl-C kills python AND continues to the next iteration.
ABORT=0
trap 'ABORT=1; echo ""; echo "[run_batch] Ctrl-C received. Will exit after the current session finishes."' INT

echo "===== run_batch: $N x $TAG ====="
echo "Settings:  $CFG"
echo "Logs to:   $LOG_DIR/${TAG}_runNN.log"
echo "Results:   results/${TAG}_<timestamp>/"
echo ""

START_TS=$(date +%s)
for i in $(seq 1 "$N"); do
  if [[ "$ABORT" -eq 1 ]]; then
    echo "[run_batch] Aborted after run $((i-1))/$N."
    break
  fi
  echo "----- Run $i/$N starting at $(date '+%Y-%m-%d %H:%M:%S') -----"
  python -u src/run_experiment.py "$CFG" 2>&1 | tee "$LOG_DIR/${TAG}_run${i}.log"
  echo "----- Run $i/$N done at $(date '+%Y-%m-%d %H:%M:%S') -----"
done

END_TS=$(date +%s)
ELAPSED=$((END_TS - START_TS))
echo ""
echo "===== run_batch complete: $i/$N runs, ${ELAPSED}s total ====="
echo "Inspect a results folder:"
echo "  ls -lt results/ | grep '$TAG' | head"
