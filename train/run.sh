#!/bin/bash
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
export PYTHONPATH="${SCRIPT_DIR}:${PYTHONPATH}"

# ══════════════════════════════════════════════════════════════════════════
#  Experiments (3 groups) — uncomment the one to run
# ══════════════════════════════════════════════════════════════════════════

# ---- Experiment 0: Baseline (GroupNSTokenizer) ----
python3 -u "${SCRIPT_DIR}/train.py" \
    --ns_tokenizer_type group \
    --ns_groups_json "${SCRIPT_DIR}/ns_groups.json" \
    --num_queries 1 \
    --emb_skip_threshold 1000000 \
    --num_workers 8 \
    "$@"

# ---- Experiment 1: Label Smoothing 0.05 + LR Warmup+Cosine ----
# python3 -u "${SCRIPT_DIR}/train.py" \
#     --ns_tokenizer_type group \
#     --ns_groups_json "${SCRIPT_DIR}/ns_groups.json" \
#     --num_queries 1 \
#     --emb_skip_threshold 1000000 \
#     --num_workers 8 \
#     --label_smoothing 0.05 \
#     --warmup_steps 400 \
#     "$@"

# ---- Experiment 2: Label Smoothing 0.05 + Warmup + Dropout 0.05 ----
# python3 -u "${SCRIPT_DIR}/train.py" \
#     --ns_tokenizer_type group \
#     --ns_groups_json "${SCRIPT_DIR}/ns_groups.json" \
#     --num_queries 1 \
#     --emb_skip_threshold 1000000 \
#     --num_workers 8 \
#     --label_smoothing 0.05 \
#     --warmup_steps 400 \
#     --dropout_rate 0.05 \
#     "$@"
