
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
