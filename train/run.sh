#!/bin/bash
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
export PYTHONPATH="${SCRIPT_DIR}:${PYTHONPATH}"

# ══════════════════════════════════════════════════════════════════════════
#  feature_audit: random-sampled feature exploration (train side)
#  --sample_ratio 0.1 = 10% random sampling (shuffle + buffer_batches)
#  --max_batches N    = fixed N batches (no shuffle)
# ══════════════════════════════════════════════════════════════════════════

python3 -u "${SCRIPT_DIR}/explore_all.py" --sample_ratio 0.1 "$@"
