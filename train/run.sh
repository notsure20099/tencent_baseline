#!/bin/bash
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
export PYTHONPATH="${SCRIPT_DIR}:${PYTHONPATH}"

# ══════════════════════════════════════════════════════════════════════════
#  Exp44_MultiScaleTime: multi-resolution temporal signals
#  - coarse_time_bias (8 buckets) + fine temporal_bias (65 buckets) in CrossAttention
#  - cls_time_mlp: per-domain time stats injected before classifier
#  ~360 params, no architecture change
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
    "$@"