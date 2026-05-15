#!/bin/bash
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
export PYTHONPATH="${SCRIPT_DIR}:${PYTHONPATH}"

# ══════════════════════════════════════════════════════════════════════════    
#  Exp37_I2TokenEnhance: split I2 into I2a+I2b (2 independent NS tokens)
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
