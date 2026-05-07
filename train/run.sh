#!/bin/bash
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
export PYTHONPATH="${SCRIPT_DIR}:${PYTHONPATH}"

# ══════════════════════════════════════════════════════════════════════════
#  Exp20 — Exp2 + Item Bridge (item-identity injected into cross-attn Q)
# ══════════════════════════════════════════════════════════════════════════

python3 -u "${SCRIPT_DIR}/train.py" \
    --ns_tokenizer_type group \
    --ns_groups_json "${SCRIPT_DIR}/ns_groups.json" \
    --num_queries 1 \
    --emb_skip_threshold 1000000 \
    --num_workers 8 \
    --label_smoothing 0.05 \
    --warmup_steps 400 \
    --dropout_rate 0.05 \
    --use_item_bridge \
    "$@"