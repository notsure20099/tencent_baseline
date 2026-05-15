#!/bin/bash
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
export PYTHONPATH="${SCRIPT_DIR}:${PYTHONPATH}"

# ══════════════════════════════════════════════════════════════════════════
#  feature_audit: full-data feature exploration (train side)
#  Runs explore_all.py — stream through all features, compute per-fid AUC,
#  save feature_stats.json sidecar for test-side comparison.
# ══════════════════════════════════════════════════════════════════════════

python3 -u "${SCRIPT_DIR}/explore_all.py" "$@"
