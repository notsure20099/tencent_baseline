#!/bin/bash
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
export PYTHONPATH="${SCRIPT_DIR}:${PYTHONPATH}"

# ══════════════════════════════════════════════════════════════════════════
#  Exp32_capacity_rebalance: num_queries=2 (full + tail-window Q tokens) + learnable domain gates
# ══════════════════════════════════════════════════════════════════════════

python3 -u "${SCRIPT_DIR}/train.py" \
    --ns_tokenizer_type group \
    --ns_groups_json "${SCRIPT_DIR}/ns_groups.json" \
    --num_queries 2 \
    --emb_skip_threshold 1000000 \
    --num_workers 8 \
    --label_smoothing 0.05 \
    --warmup_steps 400 \
    --dropout_rate 0.1 \
    --use_item_bridge \
    --dense_token_groups 4 \
    --dense_aware_qgen \
    --use_time_bias \
    "$@"