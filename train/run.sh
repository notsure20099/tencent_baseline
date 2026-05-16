#!/bin/bash
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
export PYTHONPATH="${SCRIPT_DIR}:${PYTHONPATH}"

# ══════════════════════════════════════════════════════════════════════════
#  Exp38e_5x4: 5 fid × 4 domain per-fid per-domain independent cross
#    No pooling — each S-tier item token independently crosses with
#    each sequence domain via its own gate + cross layers.
#  Baseline Exp29 (PerHeadTimeBias).
#  20 independent interaction paths per block (5 fids × 4 domains).
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
    --use_item_seq_shortcut \
    "$@"
