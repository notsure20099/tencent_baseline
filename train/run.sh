#!/bin/bash
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
export PYTHONPATH="${SCRIPT_DIR}:${PYTHONPATH}"

# ══════════════════════════════════════════════════════════════════════════
#  Exp38d_GradientFairness: two-stage training
#    Stage 1 (E1-E2): time_bias frozen, ItemGate develops freely
#    Stage 2 (E3+):   time_bias unfrozen, joint training
#  Baseline Exp29. Architecture same as Exp38c (ItemGate + attention pool).
#  d_model=64, T=17 → ffn_only. Full epoch-end monitoring enabled.
# ══════════════════════════════════════════════════════════════════════════

python3 -u "${SCRIPT_DIR}/train.py" \
    --ns_tokenizer_type group \
    --ns_groups_json "${SCRIPT_DIR}/ns_groups.json" \
    --num_queries 1 \
    --emb_skip_threshold 1000000 \
    --num_workers 8 \
    --label_smoothing 0.05 \
    --warmup_steps 400 \
    --dropout_rate 0.1 \
    --use_item_bridge \
    --dense_token_groups 4 \
    --dense_aware_qgen \
    --use_time_bias \
    --use_item_gate \
    "$@"
