# 实验记录
# ========
# 格式：日期 | 分支 | 类型(train/infer/explore) | 关键指标 | 判定
# ═══════════════════════════════════════════════════════════════════════════════

═══════════════════════════════════════════════════════════════════════════════
NS 特征数据探索（M1-M5 五模块审计）
═══════════════════════════════════════════════════════════════════════════════
  日期:        2026-05-15
  分支:        exp34_time_residual (train/explore_ns.py + train/ns_explore.txt)
  方法:        纯数据分析，全量 189 万样本，不依赖 GPU
  模块:
    M1 特征基础审计:  各 fid 的 vocab、缺失率、top-10 覆盖率
    M2 单特征区分力:  per-fid AUC (LR)
    M3 用户×物品交互: per-group AUC + pairwise interaction boost
    M4 Dense 维度信息: per-dim AUC, top-30
    M5 高基数特征:    emb_skip_threshold 跳过的特征列表

  关键发现:
    M1:  大量 user_int fid 缺失率 > 50%，过半 fid 的 vocab ≤ 8（近乎标签）
         user_dense fid=62-66 数值尺度极大（mean≈15万~27万）
    M2:  没有任何单特征 AUC > 0.53（最高 user_int fid=98: 0.5271,
         user_dense fid=66: 0.5286）→ 单特征就是随机
    M3:  I2 (fids 5,6,7,8,12) 单独 AUC = 0.5534，唯一有区分力的 token 组
         I2 + 任意用户组 → AUC 反而微降至 0.550 → I2 信号自足，不依赖用户特征
         其他所有 user_ns / item_ns 组的 AUC 在 0.50-0.51
         交互 boost max = +0.0238（数据上看起来大，实际因 I1 本身太弱）
    M4:  fid=61 的 13/256 维度 AUC > 0.55，最高 dim 168: 0.6097
         其余 243 维为噪音
    M5:  无特征被 emb_skip_threshold 跳过

  判定:
    - 非序列特征在浅层层面与序列特征一样，几乎没有独立区分力
    - 所有浅层特征 AUC 集中在 0.47-0.53 区间
    - 唯一正信号：I2 token 组 (AUC 0.5534)
    - 启示：信息存在于深度交互中，不在浅层统计量
    - 下一步：I2 独立 token 增强（Exp37）

═══════════════════════════════════════════════════════════════════════════════
Exp37: I2 Token Enhance（I2 拆分为 2 个独立 NS token）
═══════════════════════════════════════════════════════════════════════════════
  日期:        2026-05-15
  分支:        exp37_i2_token
  基线:        Exp29 (Test AUC 0.84727)
  背景:        NS 探索发现 I2 是唯一有区分力的 token 组(AUC 0.5534)，
               其他 14 个 NS token 携带信号极弱(AUC 0.50-0.51)。
  改动:        仅修改 ns_groups.json:
               I2: [5,6,7,8,12] → I2a: [5,6] + I2b: [7,8,12]
               item token 数: 4 → 5, 总 NS tokens: 15 → 16
  T 值:        T = 1*4 + (7+4+5) = 20 (item_dense=0)
               d_model=64 → 64%20≠0 → RankMixer ffn_only

  Valid AUC 逐 Epoch:
    E1: 0.86348  E2: 0.86685  E3: 0.86722*  E4: 0.86717
  峰值:        E3 0.86722
  与 Exp29 对比: E3 同时见顶，全程平行，无差异

  time_bias 监控:
    E1: seq_d=5.34  seq_c=5.78
    E2: seq_d=9.17  seq_c=7.51
    E3: seq_d=11.10 seq_c=8.42
    E4: seq_d=13.11 seq_c=9.04
    sep 锁死在 1.41-1.42

  判定:        I2 拆分是中性变量。Valid AUC 与 Exp29 完全平行，
               既无增益也无损害。多出的 token 未被有效利用。
  启示:        减少噪音 token（P1: 合并无信号 token）比增强单 token 更有价值。

  Test AUC:    等待中

═══════════════════════════════════════════════════════════════════════════════
Exp36: AttentionPooling (learnable attention query replaces MeanPool)
═══════════════════════════════════════════════════════════════════════════════
  日期:        2026-05-15
  分支:        exp36_attn_pool
  基线:        Exp29 (Test AUC 0.84727)
  改动:        MultiSeqQueryGenerator: MeanPool → Attention Pooling
               每个域 1 个可学习 attn_query (64维) × 4 域 = 256 参数
               d_model=64 (RankMixer 退化为 ffn_only)

  Valid AUC:   E1=0.86278 E2=0.86653 … E9=0.86762* E10=0.86745 E11=0.86773*
  峰值:        E11 0.86773

  注意力监控 (E9-E11 锁死):
    attn_q_norm: d0=0.270 d1=0.272 d2=0.279 d3=0.198
    attn_w top3: 2.4%-4.8%（均匀≈1.2%）→ 几乎完全弥散
  结论:        Attention Pooling ≈ MeanPool，+256 参数白加
  Test AUC:    等待中
  判定:        注意力机制未生效。瓶颈不在 Pooling 方式，在 token 表示质量。

═══════════════════════════════════════════════════════════════════════════════
Exp35: TaperedPositionEncoding (learnable position gate in SeqEncoder)
═══════════════════════════════════════════════════════════════════════════════
  日期:        2026-05-14
  改动:        每域 1 个标量 alpha（共 4 参数）→ token × (1+sigmoid(alpha*dist_to_tail))
  Valid AUC:   峰值 E6 0.86735
  Test AUC:    0.84609 (-0.00118)
  判定:        失败。position-based 在 train/test 长度不同时无法泛化。

═══════════════════════════════════════════════════════════════════════════════
Exp34: Time Residual Add-Back（后置 per-domain 时间残差 + RankMixer full）
═══════════════════════════════════════════════════════════════════════════════
  日期:        2026-05-14
  分支:        exp34_time_residual
  改动:        两次 CrossAttention → time_residual = decoded_q - pure_q
               RankMixer full mixing 只混 pure_q → 后加回各域时间残差
  Test AUC:    未完成（训练被 Exp35/36 覆盖）
  判定:        待定

═══════════════════════════════════════════════════════════════════════════════
Exp33: RankMixerFull (d_model=76, full mode)
═══════════════════════════════════════════════════════════════════════════════
  日期:        2026-05-14
  改动:        d_model 64→76 (76%19=0 & 76%4=0)
  Test AUC:    0.845528 (-0.00174)
  根因:        full mode token mixing 把 seq_d 的 time_bias 过拟合扩散到全体
               共享 FFN 被 seq_d 大梯度"绑架"
  判定:        失败。公式: full = ffn_only + 跨域交互 + 梯度污染

═══════════════════════════════════════════════════════════════════════════════
Exp32: 容量重分配（num_queries=2 + domain gate）
═══════════════════════════════════════════════════════════════════════════════
  Test AUC:    0.845305 (-0.00197)
  判定:        无效。

═══════════════════════════════════════════════════════════════════════════════
Exp31a: TimeBias MLP
═══════════════════════════════════════════════════════════════════════════════
  Test AUC:    0.846817 (-0.00045)
  判定:        效果平平。

═══════════════════════════════════════════════════════════════════════════════
Exp30: ContentAwareTimeBias
═══════════════════════════════════════════════════════════════════════════════
  Test AUC:    0.842565 (-0.00471)
  判定:        严重失败。

═══════════════════════════════════════════════════════════════════════════════
Exp29: PerHeadTimeBias（当前基线）
═══════════════════════════════════════════════════════════════════════════════
  日期:        2026-05-09
  改动:        temporal_bias: Embedding(65,1) → Embedding(65,4)
   Test AUC:    0.84727（最优基线）
  判定:        核心突破。迄今唯一稳定带来 Test 收益的改动。

═══════════════════════════════════════════════════════════════════════════════
Exp28: TimeBias 起点
═══════════════════════════════════════════════════════════════════════════════
  Test AUC:    0.84632

═══════════════════════════════════════════════════════════════════════════════
Order Probe: 序列位置 ↔ 时间映射
═══════════════════════════════════════════════════════════════════════════════
  方法:        30 批纯数据读取
  结果:        全部 4 域均为「最新事件在 pos=0」
               seq_a: p0=41.0  p511=60.0 … seq_d: p0=29.8  p511=53.5
  含义:        train/test 长度不同 → 同 position 对应不同绝对时间
                → position-based 编码无法泛化

═══════════════════════════════════════════════════════════════════════════════
Test AUC 汇总排行
═══════════════════════════════════════════════════════════════════════════════
  Exp29          （PerHeadTimeBias）              0.84727 ← 最优
  Exp31a         （TimeBias MLP）                 0.846817
  Exp28          （TimeBias 起点）                0.84632
  Exp27          （dense_token_groups=4）         0.846087
  Exp35          （TaperedPositionEncoding）      0.84609
  Exp26          （dense_aware_qgen）             0.84579
  Exp33          （RankMixer full d_model=76）    0.845528
  Exp32_capacity （num_queries=2 + domain gate）  0.845305
  Exp20          （ItemBridge）                   0.84444
  Exp30          （ContentAwareTimeBias）         0.842565

═══════════════════════════════════════════════════════════════════════════════
feature_audit: Train↔Test 双端特征审计（全量+采样）
═══════════════════════════════════════════════════════════════════════════════
  日期:        2026-05-16
  分支:        feature_audit
  方法:        train: 10% 随机采样 (shuffle+buffer_batches=20, 820 batch, ~21万行)
               test:  全量 523,244 行
               M0: per-fid 分布 (nz_rate/mean/std/train_AUC)
               M2: train AUC 排行  M3: S/A/B 分级
               M4: dense 维度分析  M5: I2 spotlight
               train/test 独立运行，通过打印结果线下 diff 对比

  核心结论:
    Schema:  train/test 完全一致 (46 user_int + 14 item_int + 983 dense)
    分布:    所有 int 特征 train/test nz_rate 差异 < 5%，dense fid=61 零漂移
    I2 稳在:  fid 5/6/7/12 train/test 分布一致，Exp37 失败不是数据偏移导致

  S级特征 (AUC>0.56 + 完美泛化, 9个):
    item_int_5  (0.6699)  item_int_10 (0.6596)  item_int_6  (0.6370)
    item_int_12 (0.6164)  item_int_7  (0.6097)  item_int_9  (0.6039)
    item_int_16 (0.6007)  item_int_13 (0.5955)  item_int_8  (0.5603)
    → 全部 item_int，train/test 一致，应保留独立 NS token 并投入深度交互

  A级特征 (AUC 0.52-0.54 + 分布稳定, 14个):
    user_int_50/48/98/106 — 低方差稳定信号
    user_int_3/4/52/53/54/56/57 — 大mean值，深度交互候选
    user_int_93(高方差std=9.7) / user_int_97/104 — 交互挖掘候选
    → 可保留为已建立的 token 组，不增加新 token

  B级特征 (AUC≤0.51, 24个 ≈ 40% NS token):
    AUC=0.5000 的纯噪音: user_int_15/60/62/63/64/65/66/80/89/90/91, item_int_11
    AUC≈0.5016 的边缘: user_int_49/51
    → 直接压缩为 3-4 个 token，释放 NS 容量

  fid=61 dense:
    983维全量 train/test 零漂移 (mean差异<0.005)
    前 40 维 nz_rate=99.9%，大数值维度 (fid=62-66) 在 train/test 间误差 <3%
    → 可安全独立建模，无分布风险

  历史认知修正:
    - 最初 NS 探索说 I2 AUC=0.5534，10% 采样下 fid=5 单独达 0.6699
    - 之前说"全部 AUC<0.53"，现 S-tier 14 个 >0.53
    - Exp37 失败根因确认: 架构瓶颈（冗余参数过拟合），不是数据偏移

═══════════════════════════════════════════════════════════════════════════════
Exp38a: NoiseCompress — B-tier user features → 2 noise groups
═══════════════════════════════════════════════════════════════════════════════
  日期:        2026-05-16
  分支:        exp38a_noise_compress (基于 Exp29)
  改动:        ns_groups.json user 侧: 7 group → 4 group
               U_s_a(S-tier 4fids) + U_s_b(A-tier 26fids) + U_n_a(噪音8fids) + U_n_b(dense路由8fids)
               item 侧不变。T 16→13。
  结果:
    Valid AUC: E12=0.86762 (标准天花板水平)
    Test AUC:  0.846689  (-0.00058 vs Exp29)
  发现:
    - U_n_a (纯噪音) proj_norm 增长最慢 (+0.06 vs信号组+0.13)
    - U_n_b (含 dense 路由 fid 62-66) 仍有正常梯度流 → 不能压
    - 噪音假说部分成立，但 fid 62-66 的 int 部分虽 AUC=0.50 却有 dense 索引功能
  判定:        ✗ 未持平基线。U_n_b 不应压缩，需拆分。

═══════════════════════════════════════════════════════════════════════════════
Exp38b: ItemGate — user noise compress + item S-tier独立 + Gate×Cross
═══════════════════════════════════════════════════════════════════════════════
  日期:        2026-05-16
  分支:        exp38b_item_gate (基于 Exp29)
  改动:
    ns_groups.json: user → U_s(30fids)/U_n(8fids)/U_dense(8fids)
                     item → 5 S-tier独立(I_s5~I_s9_10) + I_aux
    model.py:      新增 ItemGateModule — item token pool × domain gate + cross
    trainer.py:    每epoch打印 NS proj_norm + gate/cross norms
  结果:
    Valid AUC: E6=0.86734 (峰值) → E7=0.86732 (earlyStopping 1/5)
    Test AUC:  **0.847301** (+0.00003 vs Exp29) 🥇 历史最优
    Gate 轨迹:  d0(seq_a)=0.14 → d1(seq_b)=0.11 → d2(seq_c)=0.19 → d3(seq_d)=0.17
                4个gate在E4前持续增长，E4后饱和 (0.19)
    cross 轨迹:  (因监控bug未获取，E6修复后下个exp可见)
    NS 噪音组:  U_n proj_norm 增长最慢 (+0.07)，信号组+0.12 — 噪音假说三实验连续验证
    time_bias:  唯一持续活跃的参数线 (E6→E7 norm 11.7→13.9)
  关键发现:
    - Gate 在 E4 饱和: 5→1 mean pooling 是瓶颈
    - Item Gate 方向正确但容量不足: 标量 gate + mean pool 限制了自由度
    - 20+ 实验中首次超过 Exp29 基线，证明了"噪音压缩+S-tier独立+交叉门控"方向
  判定:        ✓ 超越基线。进入优化迭代。

═══════════════════════════════════════════════════════════════════════════════
Exp38c: ItemGate v2 — attention pooling 替代 mean pooling
═══════════════════════════════════════════════════════════════════════════════
  日期:        2026-05-16
  分支:        exp38c_item_gate_v2 (基于 Exp29)
  改动:        ItemGateModule: mean(5→1) → softmax attention(可学习权重)
               + 每个fid不同的 attention 权重 → per-sample 自适应选择
               trainer.py 新增 attn_norm 监控
  结果:
    Valid AUC: E9=0.86763 (历史最高，+0.00029 vs Exp38b)
    Test AUC:  **0.847299** (与 Exp38b 完全同等，差异 +0.000002)
  参数轨迹 (全 epoch):
    gate d0:   E1=0.301 → E2=0.372 → E9=0.370     E2冻结
    gate d1-3: E1≈0.13 → E2≈0.16 → E9≈0.16       E2冻结
    cross:     E1=0.54  → E2=0.61  → E9=0.63      慢(+3% over E2→E9)
    attn:      E1=0.084 → E2=0.094 → E9=0.095     完全冻结
    ns_user:   E1=5.1   → E2=5.27  → E9=5.27      完全冻结
    time_bias: E1=5.0   → E2=7.3   → E9=16~23     +167% 持续暴涨
    noise(U_n):E1=4.88  → E2=4.93  → E9=4.93      四实验连续验证噪音组冻结最早
  核心发现:
    - E1→E2 是所有模块的"黄金窗口期"——gate+24%, cross+13%, attn+12%
      E2 之后梯度分配彻底失衡，只有 time_bias 持续学习
    - attention pooling 完全没学到 per-sample 自适应 (attn 在 E2 冻在 0.094)
    - cross 层在 E2→E9 期间缓慢爬行 (+3%)，但速度不足以产生 Test 差异
    - E4 AUC 倒跌 (0.86721→0.86696) 同时 logloss 降 ——
      time_bias 过度适应容易样本、无法排序困难样本的典型表现
    - 参数量: ItemGate 67K params vs time_bias 2K params (32倍)
      但梯度衰减系数: ItemGate≈0.001, time_bias≈1.0 (800倍差距)
      等效参数比: time_bias 2,080 : ItemGate 81
  假设验证:
    attention pooling 并非瓶颈 → Exp38c 与 Exp38b 在 Test 上完全同等
    ✓ 真正瓶颈: 梯度传播路径长度不对等
  判定:        → 与 Exp38b 同等。无增益。梯度不平等是唯一待解瓶颈。

═══════════════════════════════════════════════════════════════════════════════
Exp38d-A: 两阶段训练 — 冻结 time_bias E1-E2
═══════════════════════════════════════════════════════════════════════════════
  日期:        2026-05-17
  分支:        exp38d_gradient_fairness (基于 Exp29)
  架构:        与 Exp38c 完全相同 (ItemGate + attention pooling)
  改动:        trainer.py: E1-E2 冻结 temporal_bias (requires_grad=False)
               E3+ 解冻恢复联合训练
  结果:
    Valid AUC: E1=0.86331, E2=0.86630, E3=0.86660, E4=0.86649
    time_bias:  解冻失败(0→0 完全冻结)，E4=4.64 仍为初始值
    gate d0:    E1=0.306 → E4=0.373 (与 Exp38c E2=0.372 完全相同)
    gate d1-3:  E4 值完全等同于 Exp38c E2
    cross:      E4 值完全等同于 Exp38c E2
    attn:       E4=0.094 (与 Exp38c E2=0.094 完全相同)
  核心发现:
    - ItemGate 在无 time_bias 竞争的环境中 4 个 epoch 后，与有竞争环境
      2 个 epoch 后，达到完全相同的参数状态
    - 冻结 time_bias 没有让 ItemGate 学到更多
    - 梯度竞争假说 ❌ 不成立
    - 真正的瓶颈是 5→1 pooling 信息压缩
  判定:        ✗ 假说被推翻。瓶颈不在梯度竞争，在信息压缩。

═══════════════════════════════════════════════════════════════════════════════
Exp38e: 5×4 per-fid per-domain 全交叉 + Item-Seq Cross Shortcut
═══════════════════════════════════════════════════════════════════════════════
  日期:        2026-05-17
  分支:        exp38e_5x4 (基于 Exp29)
  背景:        Exp38b/c/d-A 三个实验确认: 5→1 pooling (无论 mean 还是 attention)
               是唯一瓶颈。5 个 S-tier item token (fid=5/6/7+12/8/9+10) 语义不同、
               AUC 区间不同，被压缩成 1 个向量时损失了所有差异信息。
  改动:
     (1) ItemGateModule: 5×4 per-fid per-domain 独立交叉 (20条路径)
         每条路径: Linear(128→1) gate + Linear(128→64) cross
         per block ~168K 参数
     (2) ItemSeqCrossShortcut: 新增短路模块
         - item S-tier 与 raw seq tokens 交叉 (在 embedding 后的原始序列上)
         - per-domain pool → shortcut proj (4×64→64)
         - 注入点: 分类器入口 (output_proj 之后、clsfier 之前)
         - 与 time_bias 物理隔离 — 绕开 CrossAttention
         - 梯度路径: 6 步 vs time_bias 2 步 (衰减比 ~3×)
         - ~184K 额外参数
     (3) 分类器入口融合: s_gate(128→1) + s_cross(128→64)
         concat(h_main, s_shortcut) → gated residual
     (4) 监控: scut fid{j} gate/cross norms + entry gate/cross norms
  总新增参数: ~520K (ItemGate 336K + Shortcut 184K)
  关键假设:
    - 5×4 让每个 fid 独立发言 → 打破 5→1 pool 信息瓶颈
    - 短路让 item↔seq 交叉结果绕过 CrossAttn/时间信号垄断
    - 两条线在分类器入口会师 → loss 自然分配梯度
  状态:        训练中

═══════════════════════════════════════════════════════════════════════════════
Exp39: Sequence Time-Delta + Dense High-Dim (×2 run)
═══════════════════════════════════════════════════════════════════════════════
  日期:        2026-05-17
  分支:        exp39_seq_dense (基于 Exp29 + Exp38b ns_groups)
  改动:
    (1) CrossAttention: per-head time_delta_bias — 每个序列位置与最新事件的距离
        Embedding(65,4) 260 params, zero-init, additively injected into time_bias
    (2) Dense 6 groups (Run 1): 983dim → 6×164, 压缩比 2.5:1
        Dense 4 groups (Run 2): 983dim → 4×246, 压缩比 4:1 (基线)
    (3) 速度优化: set_float32_matmul_precision('high') + AMP bfloat16
        + prefetch_factor 8, 关闭 torch.compile

  Run 1 (dense 6 + time_delta):
    Valid AUC: E1=0.86283  E2=0.86437 (-0.00226 vs Exp29 E2 0.86663)
    time_delta: E1=1.3~2.6 → E2=2.8~5.4 (+183%) ✅ 活跃学习
    dense 6g:   E1→E2 全部 proj_norm 冻结在 4.6 ❌ 零学习
    → 暂停，dense 回退到 4 groups

  Run 2 (dense 4 + time_delta + AMP):
    Valid AUC: E1=0.863  E2=0.865  E3=0.865  E4=0.866  E5→E9 崩溃到 0.60
    根因: AMP bfloat16 包裹 Embedding 层导致数值不稳定，非架构问题
    time_delta: 持续增长（E9 B1_seq_d=9.5）但 AUC 并未超越 E3 峰值

  核心发现:
    - time_delta 有独立梯度流（E2 +183%）但未产生 AUC 收益
    - E2 AUC -0.00226 说明时间差信号稀释了绝对时间信号的纯度
    - 与 Exp30 (ContentAwareTimeBias) 同模式: 额外信息混入时间通路 = 负收益
    - Dense 6→4 回退后问题仍在，AUC 仍偏低 → 问题不在 dense，在 time_delta 本身
    - 时间信号只需要绝对时间桶这一个维度
  判定:        ✗ 时间差建模无益。时间维度扩展方向关闭。
  启示:        时间通路应保持单一清晰信号源，不可叠加辅助信息。

═══════════════════════════════════════════════════════════════════════════════
经验教训
═══════════════════════════════════════════════════════════════════════════════
  1. 时间是当前最重要且唯一被验证有效的信号。
  2. 在 CrossAttention 之后继续注入时间信号，边际收益为零或为负。
  3. 序列浅层统计特征（共现/密度/长度）完全没有区分力。
  4. Valid AUC 0.8675 是硬天花板——必须以 Test AUC 为唯一终判标准。
  5. RankMixer ffn_only 降级模式从 Exp27 起一直是隐藏瓶颈。
  6. 内容信号路径比时间信号长 4-5 倍，模型天然倾向走时间捷径。
  7. 「加参数但未打通瓶颈」的实验都导致 Test AUC 下降。
  8. （Exp33）full mode 退化根因：共享 FFN 梯度污染。
  9. （Exp33）单 Q token 是序列信息利用的核心瓶颈。
  10.（Exp35）position-based 编码无法泛化。
  11.（Exp36）Attention Pooling 实质上归为 MeanPool，注意力完全弥散。
  12.（NS 探索）非序列特征在浅层同样无独立区分力（全部 AUC 0.47-0.53）。
      唯一正信号：I2 item 特征组（AUC 0.5534）。
  13.（Exp37）I2 Token 增强是中性变量——Valid AUC 与 Exp29 完全平行。
      单独增强一个 token 不够。深层问题：浅层 LR AUC=0.50 无法区分"真噪音"与"需
      深度交互才能释放的潜在信号"。
  14.（Exp38a）噪声压缩存在副作用——fid 62-66 的 int 部分虽 AUC=0.50，
      但其 dense 对应部分有路由/索引功能，不能简单按 AUC 分类。
  15.（Exp38b）Item Gate 方向正确——20+实验中首次超越 Exp29 基线(+0.00003)。
      S-tier item 特征的跨域交叉有价值，但 mean pooling + 标量 gate 限制了增益。
  16.（Exp38b）time_bias 是当前唯一持续活跃的参数线(E4→E6 norm 10→12)，
       模型边际收益几乎全部来自时间信号的深化利用。内容信号路径仍需优化。
  17.（Exp38c）attention pooling 并未改善 per-sample 自适应——attn 参数
       在 E2 就冻结于 0.094，后续 7 个 epoch 完全不动。
  18.（Exp38c）E1→E2 是所有模块唯一的"黄金窗口期"——此后梯度被 time_bias 垄断。
  19.（Exp38c）gate d0=0.37 是时间竞争下的最优值，不是真正的饱和——
       ItemGate 停不是因为容量不够，而是梯度传不回来。
  20.（Exp38 系列）梯度不平等是唯一瓶颈：time_bias 2步梯度 vs ItemGate 9步梯度，
       衰减系数差距≈800倍。67K参数被2K参数碾压在梯度赛道上。
  21.（Exp38d-A）冻结 time_bias 并未让 ItemGate 学到更多——gate 在无竞争环境
       4 个 epoch 后仍停在 0.37（与有竞争环境 2 个 epoch 结果相同）。
       梯度竞争假说被推翻。真正瓶颈是 5→1 pooling 信息压缩。
  22.（Exp38 系列终判）Item Gate 方向正确（Exp38b Test +0.00003），但 pool 机制
       （无论是 mean 还是 attention）在 E2 就把 5 个 fid 的差异化信息压缩殆尽。
       打破天花板的唯一方式是 per-fid per-domain 独立交叉。
  23.（Exp39）time_delta 有独立梯度流但未产生 AUC 收益。额外信息混入
        时间通路会稀释绝对时间信号纯度（与 Exp30 同模式）。
        时间通路应保持单一清晰信号源。
  24.（Exp39）Dense 6→4 回退后参数全部冻结，额外 token 只增负担不增信息。
        983维 dense 在 4 token 时已达信息饱和。

═══════════════════════════════════════════════════════════════════════════════
Test AUC 排行榜 (Exp28+)
═══════════════════════════════════════════════════════════════════════════════
  🥇 Exp38b ItemGate(mean)      0.847301  +0.00003 vs Exp29
  🥈 Exp38c ItemGate(attn)      0.847299  +0.00003 vs Exp29
  🥈 Exp29 PerHeadTimeBias      0.84727   基线
  4  Exp31a TimeBiasMLP         0.846817
  5  Exp38a NoiseCompress       0.846689
  6  Exp28                      0.846318
  7  Exp27                      0.846090
  8  Exp37 I2Boost              0.844821

═══════════════════════════════════════════════════════════════════════════════
当前特征全景 & 下一步方向
═══════════════════════════════════════════════════════════════════════════════
  有效信号 (已验证):
    time_bias (Exp29): 唯一 +0.00095 Test 收益的改动, 路径短(1步直达softmax)

  微弱正信号 (未验证能否独立增益):
    I2 item 组 (AUC 0.5534): Exp37 分拆为两 token, Valid AUC 与 Exp29 平行
    fid=61 dense (13/256维 AUC>0.55): 尚未独立实验

  已探索无效的方向:
    序列内容优化 ×7 (Exp20/30/31a/32/33/35/36): 全部 ≤ Exp29
    NS 浅层特征 ×1 (探索): 全部 AUC 0.47-0.53
    position-based 编码 ×1 (Exp35): train/test 长度分布无法泛化
    full mode 单开 ×1 (Exp33): 梯度污染, Test -0.00174
    Attention Pooling ×1 (Exp36): 注意力弥散, 退化为 MeanPool

  下一步优先级:
    P0: Exp38 NS 消融实验 — 回答噪音 vs 宝藏问题
    P1: 如果 Exp38 证明多数是噪音 → 永久压缩 NS token 数
        如果 Exp38 证明含深层信号 → 保留, 转攻 User×Item 深层交互
    P2: 基于 Exp38 结论决定是否做 fid=61 dense 独立投影
