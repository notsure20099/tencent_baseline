#!/bin/bash
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
export PYTHONPATH="${SCRIPT_DIR}:${PYTHONPATH}"

# ══════════════════════════════════════════════════════════════════════════
#  feature_audit: full-data feature exploration (train side)
#  Usage:
#    ./run.sh                        → full scan
#    ./run.sh --max_batches 1        → quick 1-batch test
#    ./run.sh --max_batches 10       → 10 batch test
# ══════════════════════════════════════════════════════════════════════════

python3 -u "${SCRIPT_DIR}/explore_all.py" "$@"
