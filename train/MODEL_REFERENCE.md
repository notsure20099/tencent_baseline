# PCVRHyFormer 模型速查手册

> 目的: 让你在任何时候能 30 秒内找到任何组件的定义、作用和数据流。
> 配合 `model.py` 行号索引使用。

---

## 一、模型全景图（一张图看完整个 forward）

```
                          输入层
    ┌──────────────────────┼──────────────────────┐
    │ user_int (B,46)      │ item_int (B,14)      │ seq_a/b/c/d (B,13,256)
    │ user_dense (B,983)   │ item_dense (B,0)     │ time_bucket (B,256)
    └──────┬───────────────┴──────┬───────────────┴──────┬───────────┘
           │                      │                      │
    ┌──────▼──────────────────────▼──────────────┐  ┌────▼────────────┐
    │        GroupNSTokenizer (L1068)            │  │ _embed_seq_domain│
    │  user_int → 7 组 → 7 tokens (B,7,64)      │  │ (L1697)           │
    │  item_int → 4 组 → 4 tokens (B,4,64)      │  │ 13fid→cat→proj→  │
    │  dense → 4 tokens (B,4,64)                │  │ gelu + time_embed │
    │  ns_tokens = cat → (B,15,64)              │  │ → (B,256,64) ×4  │
    └──────────────────┬────────────────────────┘  └────┬─────────────┘
                       │                                │
              ┌────────▼────────────────────────────────▼──────────┐
              │          MultiSeqQueryGenerator (L471)              │
              │   ns_flat(960D) ⊕ MeanPool(seq_i) → FFN → Q_i     │
              │   Q_content(若解耦) vs Q_time (含时间信号)          │
              │   → [Q_d0, Q_d1, Q_d2, Q_d3]  each (B,1,64)       │
              └──────────────────────┬──────────────────────────────┘
                                     │
              ┌──────────────────────▼──────────────────────────────┐
              │           MultiSeqHyFormerBlock ×2 (L913)           │
              │                                                      │
              │  ┌─ ① SeqEncoder (L565/607): self-attn on seq      │
              │  ├─ ② CrossAttention (L252): Q×seq 交叉检索        │
              │  │    ├ ItemBridge (gate×item→Q)                    │
              │  │    └ TimeBias  (time_bucket→softmax偏置)        │
              │  └─ ③ RankMixer ffn_only (L371): per-token FFN     │
              └──────────────────────┬──────────────────────────────┘
                                     │
                          ┌──────────▼──────────┐
                          │  Concat(4Q) → (B,256)│
                          │  output_proj (L1540) │
                          │  gate_fusion(若解耦) │
                          │  classifier (L1553)  │
                          └──────────┬──────────┘
                                     │
                              logit → sigmoid → (B,1)
```

---

## 二、组件速查表

### A. 数据定义

| 组件 | 行号 | 作用 | 输入形状 | 输出形状 |
|------|------|------|----------|----------|
| `ModelInput` | 11 | 训练/推理的统一输入 NamedTuple | — | — |
| `user_int_feats` | — | 46 个用户离散特征 | (B, 46) | — |
| `item_int_feats` | — | 14 个物品离散特征 (fid 5-16) | (B, 14) | — |
| `user_dense_feats` | — | 983 个用户连续特征 (fid 61-91) | (B, 983) | — |
| `seq_data[domain]` | — | 序列数据，13 fid × L 位置 | (B, 13, 256/512) | — |
| `seq_lens[domain]` | — | 每条序列的有效长度 | (B,) | — |
| `seq_time_buckets` | — | 每个位置的时间桶 ID (1-64) | (B, 256/512) | — |

### B. Token 构建层

| 组件 | 行号 | 作用 |
|------|------|------|
| `GroupNSTokenizer` | 1068 | 按 ns_groups.json 分组 → per-group 投影为一个 token |
| `RankMixerNSTokenizer` | 1159 | 所有 fid 的 embedding concat → split → per-chunk 投影 (自由 token 数) |
| `_embed_seq_domain` | 1697 | 序列 token 构建: fid Embedding 拼接 + Linear→d_model + time_embedding |
| `_project_dense` | 1597 | 983 维 dense → 等分 4/6 组 → per-group Linear → (B, K, D) |

### C. 序列处理

| 组件 | 行号 | 作用 |
|------|------|------|
| `SwiGLUEncoder` | 565 | 每个 seq position 独立过 SwiGLU FFN |
| `TransformerEncoder` | 607 | 序列 self-attention: token 之间互相看见 |
| `LongerEncoder` | 679 | 长序列优化: top-K 注意力 + 滑动窗口 |
| `RotaryEmbedding` | 26 | RoPE 位置编码 (当前 `use_rope=False`，未启用) |

### D. 交互层（核心）

| 组件 | 行号 | 作用 |
|------|------|------|
| `MultiSeqQueryGenerator` | 471 | 从 NS token + 序列 mean → FFN 生成 per-domain Q token |
| `CrossAttention` | 252 | Q(token) × K/V(seq): 查询在序列中检索相关信息 |
| `RankMixerBlock` | 371 | token 间混合: full=跨token+FFN, ffn_only=独立FFN |
| `MultiSeqHyFormerBlock` | 913 | 一个 HyFormer 层 = SeqEncoder + CrossAttn + RankMixer |

### E. 输出层

| 组件 | 行号 | 作用 |
|------|------|------|
| `output_proj` | 1540 | Concat(4Q) = 256D → 64D |
| `gate_fusion` (Exp41) | 1547 | Content×Time 双路径 128D → 64D |
| `clsfier` | 1553 | 64D → 64D → SiLU → 64D → 1D → sigmoid |

---

## 三、逐层详解

### 3.1 GroupNSTokenizer (L1068)

**做什么**: 把整数 fid 变成 64 维 token。

```
输入: item_int_feats (B, 14)   ← 14 个 fid 的值
      ns_groups.json:
        I1: [11, 13]
        I2: [5, 6, 7, 8, 12]
        I3: [16, 81, 83, 84, 85]
        I4: [9, 10]

对每个组:
  取组内所有 fid → Embedding(5000, 64) → concat → Linear(S×64→64) → SiLU
  例如 I2: 5 个 fid × 64D = 320D → Linear(320→64) → (B, 64)

输出: (B, 4, 64)  ← 一个组 = 一个 token

关键点:
  - 同一组的 fid 信号被 Linear 压缩到 1 个 token
  - S-tier fid (5,6,7,8,12) 在 I2 组共用一个 Linear
  - I3 含 fid=16 (AUC 0.78) 和 fid=81/83/84/85 (AUC≈0.50) — 信号稀释
```

### 3.2 _embed_seq_domain (L1697)

**做什么**: 把序列数据的 13 个 fid → 256 个 d_model 向量。

```
输入: seq[domain] (B, 13, 256), time_bucket (B, 256)

步骤:
  1. 13 个 fid 各自 Embedding(5000, 64) → (B, 256, 13×64)
  2. Linear(832→64) + gelu → (B, 256, 64)
  3. + time_embedding(bucket_id) → (B, 256, 64)  ← 时间信号注入!

add_time=False (Exp41):
  跳过步骤 3 → 纯 fid 内容, 无时间污染

关键点:
  - 步骤 3 是「时间污染」的唯一源头
  - 256 个 token 每个都加了 time_embedding → MeanPool 后时间不灭
```

### 3.3 MultiSeqQueryGenerator (L471)

**做什么**: 为每个序列域生成 1 个 Q token。

```
输入: ns_tokens (B, 15, 64), seq_tokens (B, 256, 64)

步骤:
  ns_flat = ns_tokens.view(B, 15×64)           # (B, 960)
  seq_pooled = mean(seq_tokens, dim=1)          # (B, 64) ← 256→1 盲平均
  global_info = cat(ns_flat, seq_pooled)        # (B, 1024)
  Q_i = FFN(1024→256→64)                        # (B, 64)

输出: [Q_d0, Q_d1, Q_d2, Q_d3] 每个 (B, 1, 64)

关键点:
  - ns_flat 只有 4/15 = 27% 是 item 信号
  - seq_pooled 中的 64 维 ≈ 54% 来自 time_embedding (全链路实验证实)
  - Q 中 item 信号 ≈ 27%×0.74 + 100%×0.86×... = 被稀释
```

### 3.4 CrossAttention (L252)

**做什么**: Q 在序列中检索相关信息。

```
输入:
  query:  (B, 1, 64)    ← Q token
  key_value: (B, 256, 64) ← 序列 token
  time_buckets: (B, 256)   ← 时间桶 ID

步骤:
  1. ItemBridge (若启用):
     item_pooled = mean(I1,I2,I3,I4)      # 4 个 item token → 1 个 64D
     gate = sigmoid(Linear(item_pooled))   # 可学习门控
     query = query + gate × item_pooled    # 注入物品身份

  2. Multi-head attention + TimeBias:
     score = Q×K^T/√64                    # Q 与 256 个 K 的相关性
     score = score + time_bias[bucket]    # 每个位置独立偏置 (65个参数×4头=260)
     weight = softmax(score)              # 归一化注意力权重
     output = weight @ V                  # 加权聚合

输出: (B, 1, 64)

ItemBridge 的 pool_weights (Exp41b):
  w = softmax(learnable_weight[4]) → item_pooled = Σ w_i × token_i
  让模型学会 I2(0.69-0.71) 和 I3(0.78) 比 I1 更重要的权重分配

TimeBias 为什么强:
  - 只有 260 个参数 (65桶×4头)
  - 梯度路径: time_bucket → Embedding → softmax → loss (2步)
  - 模型学到「最近的事件 = 更高注意力」→ 天然与 label 相关
```

### 3.5 RankMixerBlock (L371)

**做什么**: token 间增强。

```
full 模式:
  x_reshaped = reshape(B, 19, 64) → (B, 64, 19)  # 转置, 让 19 个 token 在最后一维
  x_mixed = x + x_reshaped                         # 简单的跨 token 混合 (零参数)
  x_boosted = x_mixed + FFN(x_mixed)               # 共享 FFN

ffn_only 模式 (Exp29/41 当前):
  x_boosted = x + FFN(x)                           # 每个 token 独立 FFN

降级原因: 64 % 19 ≠ 0 → full 不可用
  T = num_queries×4 + num_ns = 1×4 + 15 = 19
```

### 3.6 MultiSeqHyFormerBlock (L913)

**做什么**: 一个完整的处理块 = 三个子步骤。

```
输入:
  q_tokens:    [Q_d0, Q_d1, Q_d2, Q_d3]  each (B, 1, 64)
  ns_tokens:   (B, 15, 64)
  seq_tokens:  [seq_a, seq_b, seq_c, seq_d]  each (B, 256, 64)

步骤:
  ① SeqEncoder (每个域独立):
     seq_i = TransformerEncoder(seq_i)  # self-attention: 256个token互相交互
     [Exp29 使用 SwiGLUEncoder (L565) — 独立 FFN, 无 self-attn]

  ② CrossAttention (每个域独立, L252):
     Q_i = CrossAttention(Q_i, seq_i, time_buckets_i)
     [ItemBridge: Q_i += gate(item_pooled) × item_pooled]
     [TimeBias:   attention += time_bias[bucket]]

  ③ RankMixer (L371):
     combined = cat(Q_4, ns_15) = (B, 19, 64)
     boosted  = RankMixer(combined)     # ffn_only: 各自独立 FFN
     next_Q = boosted[:4], next_ns = boosted[4:]

输出: 更新后的 Q 和 NS token
```

### 3.7 PCVRHyFormer.forward() (L1920)

**全程数据流**:

```python
# 第 1 步: NS tokens
item_ns = GroupNSTokenizer(item_int)     # (B, 4, 64)
user_ns = GroupNSTokenizer(user_int)     # (B, 7, 64)
dense   = _project_dense(user_dense)     # (B, 4, 64)
ns = cat(user_ns, dense, item_ns)        # (B, 15, 64)

# 第 2 步: 序列 embedding
seq_content = _embed_seq_domain(seq, add_time=False)  # 纯内容
seq_mixed   = _embed_seq_domain(seq, add_time=True)   # 含时间

# 第 3 步: Q 生成 (Exp41 双路径)
ns_content = ns.clone()
ns_content[item部分] *= content_item_scale     # 放大 item 信号
Q_content = QGen_content(ns_content, seq_content)  # 纯内容 Q
Q_time    = QGen(ns, seq_mixed)                    # 含时间 Q

# 第 4 步: Block 堆叠 (×2)
for block in self.blocks:
    # Content 路径: Q_content × seq_content, NO time_bias
    Q_content, ns, seq_content, mask = block(Q_content, ns, seq_content, ...)
    # Time 路径:   Q_time × seq_mixed, WITH time_bias
    Q_time, ns, seq_mixed, mask = block(Q_time, ns, seq_mixed, ...)

# 第 5 步: 融合 + 分类
content_out = output_proj_content(concat(Q_content))  # (B, 64)
time_out    = output_proj(concat(Q_time))              # (B, 64)
fused       = gate_fusion(cat(content_out, time_out)) # (B, 64)
logit       = classifier(fused)                       # (B, 1)
```

---

## 四、关键架构决策理解

### 4.1 为什么用 CrossAttention 而不是 Self-Attention？

```
Self-Attention:  256 个 token 互相 attend → 256×256 注意力矩阵 → O(n²) 算力
CrossAttention:  1 个 Q token attend 256 个 K/V → 1×256 → 线性

HyFormer 的设计哲学:
  每条序列有 256 个事件, 但只需要 1 个 Q 去理解「这条序列与当前预测的关系」
  不是让 256 个事件互相聊天, 而是找一个代表去「审阅」所有事件
```

### 4.2 为什么 time_bias 这么强？

```
参数效率:  260 个参数 → Test AUC +0.00095
ItemGate:  ~180K 参数 → Test AUC +0.00003

效率差:    time_bias 的每参数效率是 ItemGate 的 600 倍

原因: time_bias 回答的是「什么时候」, 这是 label 的最强单一预测因子
     ItemGate 回答的是「什么物品」, 但「什么时候买了什么」已经几乎覆盖了「什么」的信息
```

### 4.3 为什么 Exp41 (时-空解耦) 是最好的设计？

```
问题: seq_token = fid_embedding + time_embedding
      → 256 个 token 每个都含时间 → MeanPool 后时间不灭
      → Q 里时间信息占 54%，内容信息只有 46%

Exp41 解法:
  ① 把时间信号从 Q 生成路径中剥离 (add_time=False)
  ② Content Q 是纯内容的 — 第一次以 clean 形态见到 loss
  ③ Time Q 保留原样 — 已验证的 Exp29 通路不动
  ④ gate_fusion 让模型自主决定内容 vs 时间的信任度

但根本性问题:
  即使 Q 清洁了, item fid 的内容信号 (AUC 0.74) 是否已经
  被时间信号 (AUC 0.86) 在统计上覆盖了?
  → 如果答案是"是", 那内容就是冗余 — 和路径长短无关
  → Exp41 是最终的裁定实验
```

---

## 五、20 次实验归因

| 实验 | 类型 | 改动 | 为什么失败 |
|------|------|------|-----------|
| Exp29 | ✅ | Embedding(65,1)→(65,4) | time_bias 2 步到 loss, 极短极强 |
| Exp30 | ❌ | ContentAwareTimeBias | 内容混入时间通路 → 稀释时间纯度 |
| Exp31a | ❌ | TimeBias MLP | 非线性在 time_bias 上无额外信息 |
| Exp32 | ❌ | num_queries=2 | Q 多了, 但信号没多 |
| Exp33 | ❌ | full mode | seq_d 梯度污染全体 token |
| Exp34 | ⏳ | TimeResidual | 待定 |
| Exp35 | ❌ | PositionEncoding | train/test 长度分布不同 |
| Exp36 | ❌ | AttentionPooling | attn 退化为 mean (弥散) |
| Exp37 | ~ | I2Token 增强 | 中性变量 |
| Exp38a | ❌ | NoiseCompress | fid62-66 的 dense 索引功能 |
| Exp38b | ✅ | ItemGate | +0.00003, 方向正确 |
| Exp38c | ~ | ItemGate attn | 与 Exp38b 同等 |
| Exp38d-A | ❌ | 冻结 time_bias | ItemGate 在无竞争下也不学更多 |
| Exp38e | ❌ | 5×4+短路 | 短路路径也在 E2 冻结 |
| Exp39 | ❌ | time_delta | 时间差稀释绝对时间纯度 |
| Exp40 | 🔍 | 全链路诊断 | 定位 Q 生成是信号断崖 |
| Exp41 | ⏳ | 时-空解耦 | 训练中 |

---

## 六、代码行号速查

```
类/函数                             行号    文件
──────────────────────────────────────────────────
ModelInput                          11      model.py
RotaryEmbedding                     26      model.py
SwiGLU                              100     model.py
RoPEMultiheadAttention              117     model.py
CrossAttention                      252     model.py
  ├ item_gate                       293     model.py
  ├ temporal_bias                   298     model.py
  └ item_pool_weights (Exp41b)      300     model.py
RankMixerBlock                      371     model.py
MultiSeqQueryGenerator              471     model.py
SwiGLUEncoder                       565     model.py
TransformerEncoder                  607     model.py
LongerEncoder                       679     model.py
MultiSeqHyFormerBlock               913     model.py
  ├ seq_encoders                    946     model.py
  ├ cross_attns                     962     model.py
  └ mixer                           974     model.py
GroupNSTokenizer                    1068    model.py
RankMixerNSTokenizer                1159    model.py
PCVRHyFormer                        1287    model.py
  ├ _embed_seq_domain               1697    model.py
  ├ _make_padding_mask              1733    model.py
  ├ _run_multi_seq_blocks           1741    model.py
  ├ _run_multi_seq_blocks_dual      1794    model.py
  ├ _run_single_path                1876    model.py
  ├ forward                         1920    model.py
  ├ predict                         1994    model.py
  ├ get_sparse_params               1684    model.py
  └ get_dense_params                1692    model.py

PCVRHyFormerRankingTrainer          46      trainer.py
  ├ train_epoch                     350     trainer.py
  ├ evaluate                        504     trainer.py
  ├ _train_step                     471     trainer.py
  └ _evaluate_step                  598     trainer.py
```

---

## 七、排查任何问题的方法

```
1. 找出你关心的变量在 forward 中的位置 (用行号速查)
2. 往上游追: "这个变量的每 1 维是从哪个输入贡献的?"
3. 往下游追: "这个变量还会被哪些层加工?"
4. 问: "如果我想让模型更关注 X, 需要改这条链路的哪个环节?"
```

例如你想知道 `time_bias` 的影响链:

```
上游: num_time_buckets=65 → Embedding(65, 4) → (65, 4)
中游: CrossAttention.forward(L306):
      time_bias = temporal_bias(key_time_buckets)  # 查表, (B,256,4)
      attn = RoPEAttention(query, key, time_bias=time_bias)  # softmax 入口
下游: attn → weighted V → decoded_Q → RankMixer → classifier → logit

影响: 改变任何一个 time_bucket 的 Embedding 值
      → 改变 softmax 注意力分布
      → 影响 decoded_Q 的内容
      → 最终改变预测概率
```
