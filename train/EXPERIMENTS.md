
===========================================================================
Exp36: AttentionPooling (learnable attention query replaces MeanPool)
===========================================================================
  date:        2026-05-15
  branch:      exp36_attn_pool
  baseline:    Exp29 (Test AUC 0.84727)
  idea:        MeanPool blindly averages all 256 tokens - the last
               remaining content bottleneck.  Replace with learnable
               per-domain attention (4 vectors * 64D = 256 params).
               Each domain learns WHICH token positions matter via
               softmax(token_i * attn_query[domain]).
  change:      MultiSeqQueryGenerator: MeanPool -> Attention Pooling
               + speed: cudnn.benchmark + TF32 + foreach
  params:      +256 (4 domains * 64-dim query each)
  orthogonal:  time_bias untouched, CrossAttn untouched

  valid AUC:   (pending)
  test AUC:    (pending)
  verdict:     (pending)


Order Probe: sequence position  time mapping

  date:        2026-05-15
  branch:      exp35_tapered_posenc (train/probe_order.py)
  method:      30 batches, no model, pure data read
  result:      ALL 4 domains are recentold (position 0 = most recent)
               seq_a: p0=41.0  p511=60.0
               seq_b: p0=41.1  p511=59.7
               seq_c: p0=45.4  p511=62.7
               seq_d: p0=29.8  p511=53.5
  implication: dist_to_end (position-based) maps valid pos500=days-old,
               test pos500=hours-old  position encoding cannot generalize
                all position-based optimisations are DOA for this task


Exp35: TaperedPositionEncoding (learnable position gate in SeqEncoder)

  date:        2026-05-14
  branch:      exp35_tapered_posenc
  baseline:    Exp29 (Test AUC 0.84727)
  change:      4 params (1 per domain)  sigmoid(alpha * dist_to_end)
  valid AUC:   E1=0.86360 E2=0.86621 E3=0.86684 E4=0.86696
               E5=0.86730 E6=0.86735* E7=0.86725
               peak E6, converged at +4 epochs
  alpha final: seq_c=0 / seq_d=+0.057  matches domain ablation
  test AUC:    0.84609 (-0.00118 from Exp29)
  verdict:     FAIL. Position-based encoding cannot generalize when
               train/test have different sequence lengths (different
               absolute-time semantics at the same position index).
  lesson:      content-path optimisation IS valid direction (alpha
               converged to structurally meaningful values), but
               must use time_bucket (absolute time) not position
               (relative) for the encoding basis.
Exp35: TaperedPositionEncoding (learnable position gate in SeqEncoder)

  date:        2026-05-14
  branch:      exp35_tapered_posenc
  baseline:    Exp29 (Test AUC 0.84727)
  idea:        content path is 4-5x longer than time shortcut  model ignores
               content.  Give SeqEncoder explicit position awareness so it
               produces better representations BEFORE time_bias interacts.
  change:      TransformerEncoder learns a per-domain scalar pos_alpha.
               Each token scaled by (1 + sigmoid(alpha * dist_to_tail)).
               Tail tokens get a natural boost; earlier tokens are gated down.
               Pure content-path modification  time_bias untouched.
  params:      +4 (1 scalar per SeqEncoder instance  4 domains)

  valid AUC:   (pending)
  test AUC:    (pending)
  verdict:     (pending)     vs time_bucket鈫扙mbedding鈫抯oftmax锛夛紝妯″瀷澶╃劧鍊惧悜浜庤蛋鏃堕棿鎹峰緞锛屽拷鐣ュ唴瀹广€?  7. 鎵€鏈夈€屽姞浜嗗弬鏁颁絾娌℃湁鎵撻€氱摱棰堛€嶇殑瀹為獙锛圗xp30/31a/32锛夐兘瀵艰嚧浜?Test AUC 涓嬮檷鎴栨寔骞炽€?     鍦ㄦ病鏈夋墦閫氱摱棰堜箣鍓嶏紝澧炲姞鍙傛暟鍙細澧炲姞杩囨嫙鍚堛€?
