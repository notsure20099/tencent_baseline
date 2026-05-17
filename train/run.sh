#!/bin/bash
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
export PYTHONPATH="${SCRIPT_DIR}:${PYTHONPATH}"

# ══════════════════════════════════════════════════════════════════════════
#  Exp39: Sequence Time-Delta (dense 4 groups, time_delta only)
#    (1) CrossAttention: per-head time-delta bias (inter-event gaps)
#    (2) Dense: dense_token_groups=4 (Exp27 baseline, proven effective)
#    (3) AMP bfloat16 + set_float32_matmul_precision('high') for speed
#    (4) prefetch_factor=8 for DataLoader throughput
#  Baseline Exp29 (PerHeadTimeBias).
#  Exp38b ns_groups.json (S-tier item + noise compressed user).
#  d_model=64, T=17 → ffn_only.
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
    --use_time_delta \
    "$@"
