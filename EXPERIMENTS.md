# Experiment Log
# =============
# Format:  Date | Branch | Type(train/infer/explore) | Key Metrics | Verdict
# ═══════════════════════════════════════════════════════════════════════════════

═══════════════════════════════════════════════════════════════════════════════
Exp33: RankMixerFull (d_model=76, full mode)
═══════════════════════════════════════════════════════════════════════════════
  Date:        2026-05-14
  Branch:      exp33_rankmixer_full
  Baseline:    Exp29 (Test AUC 0.84727)
  Change:      d_model 64->76 → 76%19=0 & 76%4=0 → RankMixer full mode (no longer ffn_only)
  Extra speed: cudnn.benchmark + foreach clip_grad

  Valid AUC:   0.86394 / 0.86695 / 0.86717 / 0.86722 / 0.86750 / 0.86777 / 0.86757
  Peak:        E6 0.86777 (ALL-TIME HIGH)
  Monitor:
    RankMixer: full T=19 d_model=76 OK
    sep:       1.43 (locked from E4—E7)
    time_bias: norm 8.63→10.02→11.83→12.97 (surge, +50%)
  Verdict:     Best ckpt at E6 (step 45660). Full mode unlocked, but time_bias
               overfit dominates the extra capacity. Must check Test AUC.

═══════════════════════════════════════════════════════════════════════════════
Exp32: Capacity Rebalance (num_queries=2 + domain gates)
═══════════════════════════════════════════════════════════════════════════════
  Date:        2026-05-13
  Branch:      exp32_capacity_rebalance
  Baseline:    Exp29
  Change:      num_queries=2 (Q0: full, Q1: tail 50%) + per-domain learnable gates
  Valid AUC:   0.86343 / 0.86677 / 0.86706 / ... / 0.86747 (E14)
  Test AUC:    0.845305  (-0.00197 vs Exp29)
  Verdict:     FAILED. Q tokens added but RankMixer in ffn_only → extra
               capacity wasted; domain gates with 4 params too weak.
               Also revealed: RankMixer has been in ffn_only since Exp27.

═══════════════════════════════════════════════════════════════════════════════
Exp32 (ours): TimeAwareSeqGate (per-dim multiplicative gate)
═══════════════════════════════════════════════════════════════════════════════
  Date:        2026-05-12
  Branch:      exp32_timeaware_seqgate
  Baseline:    Exp29
  Change:      temporal_bias stats → Linear(4→64) → per-dim sigmoid gate on decoded_q
  Valid AUC:   0.8627 / 0.8658 / 0.8662 / 0.8665
  Verdict:     FAILED. Below Exp29 baseline. Time signal injection
               post-CrossAttention has negative marginal return.

═══════════════════════════════════════════════════════════════════════════════
Exp31a: TimeBias MLP (nonlinear per-head smoothing)
═══════════════════════════════════════════════════════════════════════════════
  Date:        2026-05-11
  Branch:      exp31a_timebias_mlp
  Baseline:    Exp29
  Change:      Embedding(65,4)→MLP→per-head bias (nonlinear smoothing)
  Valid:       ~0.86757
  Test AUC:    0.846817 (-0.00045 vs Exp29)
  Verdict:     MARGINAL. Valid higher but Test lower. MLP smoothing not worth
               8.5K extra params.

═══════════════════════════════════════════════════════════════════════════════
Exp30: ContentAwareTimeBias (dynamic bias conditioned on query content)
═══════════════════════════════════════════════════════════════════════════════
  Date:        2026-05-10
  Branch:      exp30_content_aware_timebias
  Baseline:    Exp29
  Change:      time_bias conditioned on concat(query_context) via MLP
  Test AUC:    0.842565 (-0.00471 vs Exp29)
  Verdict:     FAILED HARD. Content injection into time bias hurts badly.

═══════════════════════════════════════════════════════════════════════════════
Exp29: PerHeadTimeBias (CURRENT BASELINE)
═══════════════════════════════════════════════════════════════════════════════
  Date:        2026-05-09
  Branch:      exp29_perhead_timebias
  Baseline:    Exp28
  Change:      temporal_bias changed from Embedding(65,1) to Embedding(65,4)
               → each head learns independent time preference
  Test AUC:    0.84727  (BEST BASELINE)
  Verdict:     CORE WIN. Per-head temporal bias is the single most impactful
               change in the project history.

═══════════════════════════════════════════════════════════════════════════════
Exp28: TimeBias Baseline
═══════════════════════════════════════════════════════════════════════════════
  Date:        2026-05-08
  Branch:      exp28_timebias_baseline
  Change:      First time-bias in CrossAttention: Embedding(65,1) added to
               attention scores pre-softmax
  Test AUC:    0.84632
  Verdict:     SOLID. Time bias direction proven valid.

═══════════════════════════════════════════════════════════════════════════════
Q4: Content vs Time shuffle ablation (diagnostic)
═══════════════════════════════════════════════════════════════════════════════
  Date:        2026-05-13
  Branch:      exp_q4_content_vs_time
  Purpose:     Shuffle time buckets vs shuffle fid order → measure AUC drop
  Result:      (pending — EVAL_DATA_PATH needs training data with labels)
  Verdict:     AWAITING RESULT

═══════════════════════════════════════════════════════════════════════════════
Q1+Q3: Sequence data exploration (co-occurrence + behavioural patterns)
═══════════════════════════════════════════════════════════════════════════════
  Date:        2026-05-13
  Branch:      exp29_perhead_timebias (train/explore_seq.py)
  Method:      Pure data analysis, no model. 50K samples.
  Q1:          100% fid pairs "significant" but max |diff| only 0.028
               → FM Cross not worth doing.
  Q3:          All sep ≤ 0.11 (len/density/diversity/latest_bucket)
               → behavioural statistics not discriminative.
  Verdict:     Shallow content features have NO differentiating power.
               Deep content interaction (not shallow stats) is the path.

═══════════════════════════════════════════════════════════════════════════════
Diagnosis: Exp32_capacity_rebalance model inspection
═══════════════════════════════════════════════════════════════════════════════
  Date:        2026-05-13
  Branch:      exp32_capacity_rebalance_diagnose
  Method:      Pure weight inspection (no eval)
  Findings:    (pending — infer/model.py version mismatch on AngelML)
  Verdict:     AWAITING RESULT

═══════════════════════════════════════════════════════════════════════════════
Architectural Finding: RankMixer has been in ffn_only mode since Exp27
═══════════════════════════════════════════════════════════════════════════════
  Date:        2026-05-14 (discovered)
  Root Cause:  Exp27 added dense_token_groups=4 → T=19.
               d_model=64, 64%19≠0 → auto-fallback to ffn_only.
               NO experiment since has used RankMixer full mode.
  Impact:      Token mixing (cross-sequence interaction) has been skipped
               across all experiments Exp27—Exp32.
  Fix:         d_model must satisfy d_model%19=0 AND d_model%4=0.
               Smallest: d_model=76 (LCM of 19 and 4).
  Verified in: Exp33 (d_model=76 → full mode OK)

═══════════════════════════════════════════════════════════════════════════════
QUICK REFERENCE: All Test AUCs
═══════════════════════════════════════════════════════════════════════════════
  Exp33          (d_model=76 RankMixer full)    PENDING
  Exp29          (PerHeadTimeBias)              0.84727 ← BEST
  Exp31a         (TimeBias MLP)                 0.846817
  Exp28          (TimeBias baseline)            0.84632
  Exp27          (dense_token_groups=4)         0.846087
  Exp26          (dense_aware_qgen)             0.84579
  Exp32_capacity (num_queries=2 + domain gate)  0.845305
  Exp20          (ItemBridge)                   0.84444
  Exp30          (ContentAwareTimeBias)         0.842565

  Valid ceiling: ~0.8675 across 7+ experiment groups.
  Test range:    0.8425—0.8473 (5x wider than valid → valid is saturated).

═══════════════════════════════════════════════════════════════════════════════
LESSONS LEARNED
═══════════════════════════════════════════════════════════════════════════════
  1. Time is the single most important signal. Per-head time bias (Exp29)
     is the only change with consistent >0.001 Test gain.
  2. Post-CrossAttention time injection (Exp30/31a/32) is negative or zero.
  3. Shallow content statistics (Q1/Q3) have zero differentiating power.
  4. Valid AUC 0.8675 is a hard ceiling — do not use it to judge experiments.
     Must use Test AUC as the ONLY terminal verdict.
  5. RankMixer ffn_only degraded mode was a hidden bottleneck since Exp27.
  6. Content signal path is 4-5x longer than time signal path — model
     naturally gravitates toward time and ignores content.
  7. Every experiment that "added parameters without fixing a bottleneck"
     (Exp30/31a/32) hurt or had zero Test gain.
