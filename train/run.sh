#!/bin/bash
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
export PYTHONPATH="${SCRIPT_DIR}:${PYTHONPATH}"

# ══════════════════════════════════════════════════════════════════════════
#  Probe: sequence position order (position 0 = most recent? or oldest?)
# ══════════════════════════════════════════════════════════════════════════

python3 -u "${SCRIPT_DIR}/probe_order.py"
