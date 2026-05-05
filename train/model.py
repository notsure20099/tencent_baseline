"""DINSeqPCVR: DIN + User Profile architecture for temporal robustness."""

import logging
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, NamedTuple, Tuple, Optional, Dict


class ModelInput(NamedTuple):
    user_int_feats: torch.Tensor
    item_int_feats: torch.Tensor
    user_dense_feats: torch.Tensor
    item_dense_feats: torch.Tensor
    seq_data: dict
    seq_lens: dict
    seq_time_buckets: dict


# ═══════════════════════════════════════════════════════════════════════════════
# Rotary Position Embedding (RoPE)
# ═══════════════════════════════════════════════════════════════════════════════


class RotaryEmbedding(nn.Module):
    def __init__(self, dim: int, max_seq_len: int = 2048, base: float = 10000.0) -> None:
        super().__init__()
        self.dim = dim
        self.max_seq_len = max_seq_len
        self.base = base
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer('inv_freq', inv_freq, persistent=False)
        self._build_cache(max_seq_len)

    def _build_cache(self, seq_len: int) -> None:
        t = torch.arange(seq_len, dtype=self.inv_freq.dtype, device=self.inv_freq.device)
        freqs = torch.outer(t, self.inv_freq)
        emb = torch.cat([freqs, freqs], dim=-1)
        self.register_buffer('cos_cached', emb.cos().unsqueeze(0), persistent=False)
        self.register_buffer('sin_cached', emb.sin().unsqueeze(0), persistent=False)

    def forward(self, seq_len: int, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
        cos = self.cos_cached[:, :seq_len, :].to(device)
        sin = self.sin_cached[:, :seq_len, :].to(device)
        return cos, sin


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat([-x2, x1], dim=-1)


def apply_rope_to_tensor(
    x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor,
) -> torch.Tensor:
    L = x.shape[2]
    cos_ = cos[:, :L, :].unsqueeze(1)
    sin_ = sin[:, :L, :].unsqueeze(1)
    return x * cos_ + rotate_half(x) * sin_


# ═══════════════════════════════════════════════════════════════════════════════
# Attention building blocks
# ═══════════════════════════════════════════════════════════════════════════════


class RoPEMultiheadAttention(nn.Module):
    def __init__(self, d_model: int, num_heads: int, dropout: float = 0.0,
                 rope_on_q: bool = True) -> None:
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.rope_on_q = rope_on_q
        self.dropout = dropout
        assert d_model % num_heads == 0

        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)
        self.W_g = nn.Linear(d_model, d_model)
        nn.init.zeros_(self.W_g.weight)
        nn.init.constant_(self.W_g.bias, 1.0)

    def forward(
        self,
        query: torch.Tensor, key: torch.Tensor, value: torch.Tensor,
        key_padding_mask: Optional[torch.Tensor] = None,
        attn_mask: Optional[torch.Tensor] = None,
        rope_cos: Optional[torch.Tensor] = None,
        rope_sin: Optional[torch.Tensor] = None,
        q_rope_cos: Optional[torch.Tensor] = None,
        q_rope_sin: Optional[torch.Tensor] = None,
        need_weights: bool = False,
    ) -> tuple:
        B, Lq, _ = query.shape
        Lk = key.shape[1]

        Q = self.W_q(query)
        K = self.W_k(key)
        V = self.W_v(value)

        Q = Q.view(B, Lq, self.num_heads, self.head_dim).transpose(1, 2)
        K = K.view(B, Lk, self.num_heads, self.head_dim).transpose(1, 2)
        V = V.view(B, Lk, self.num_heads, self.head_dim).transpose(1, 2)

        if rope_cos is not None and rope_sin is not None:
            K = apply_rope_to_tensor(K, rope_cos, rope_sin)
            if self.rope_on_q:
                q_cos = q_rope_cos if q_rope_cos is not None else rope_cos
                q_sin = q_rope_sin if q_rope_sin is not None else rope_sin
                Q = apply_rope_to_tensor(Q, q_cos, q_sin)

        sdpa_attn_mask = None
        if key_padding_mask is not None:
            sdpa_attn_mask = ~key_padding_mask.unsqueeze(1).unsqueeze(2)
            sdpa_attn_mask = sdpa_attn_mask.expand(B, self.num_heads, Lq, Lk)

        if attn_mask is not None:
            bool_attn = (attn_mask == 0)
            bool_attn = bool_attn.unsqueeze(0).unsqueeze(0).expand(B, self.num_heads, Lq, Lk)
            sdpa_attn_mask = sdpa_attn_mask & bool_attn if sdpa_attn_mask is not None else bool_attn

        dropout_p = self.dropout if self.training else 0.0
        out = F.scaled_dot_product_attention(
            Q, K, V, attn_mask=sdpa_attn_mask, dropout_p=dropout_p,
        )

        out = torch.nan_to_num(out, nan=0.0)
        out = out.transpose(1, 2).contiguous().view(B, Lq, self.d_model)
        G = self.W_g(query)
        out = out * torch.sigmoid(G)
        out = self.W_o(out)
        return out, None


class CrossAttention(nn.Module):
    def __init__(self, d_model: int, num_heads: int, dropout: float = 0.0,
                 ln_mode: str = 'pre') -> None:
        super().__init__()
        self.ln_mode = ln_mode
        self.attn = RoPEMultiheadAttention(
            d_model=d_model, num_heads=num_heads, dropout=dropout, rope_on_q=False,
        )
        if ln_mode in ['pre', 'post']:
            self.norm_q = nn.LayerNorm(d_model)
            self.norm_kv = nn.LayerNorm(d_model)

    def forward(
        self,
        query: torch.Tensor,
        key_value: torch.Tensor,
        key_padding_mask: Optional[torch.Tensor] = None,
        rope_cos: Optional[torch.Tensor] = None,
        rope_sin: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        residual = query
        if self.ln_mode == 'pre':
            query = self.norm_q(query)
            key_value = self.norm_kv(key_value)
        out, _ = self.attn(
            query=query, key=key_value, value=key_value,
            key_padding_mask=key_padding_mask,
            rope_cos=rope_cos, rope_sin=rope_sin,
        )
        out = residual + out
        if self.ln_mode == 'post':
            out = self.norm_q(out)
        return out


# ═══════════════════════════════════════════════════════════════════════════════
# GroupNSTokenizer (shared with HyFormer baseline)
# ═══════════════════════════════════════════════════════════════════════════════


class GroupNSTokenizer(nn.Module):
    def __init__(self, feature_specs: List[Tuple[int, int, int]],
                 groups: List[List[int]], emb_dim: int, d_model: int,
                 emb_skip_threshold: int = 0) -> None:
        super().__init__()
        self.feature_specs = feature_specs
        self.groups = groups
        self.emb_dim = emb_dim
        self.emb_skip_threshold = emb_skip_threshold

        embs = []
        for vs, offset, length in feature_specs:
            skip = int(vs) <= 0 or (emb_skip_threshold > 0 and int(vs) > emb_skip_threshold)
            if skip:
                embs.append(None)
            else:
                embs.append(nn.Embedding(int(vs) + 1, emb_dim, padding_idx=0))
        self.embs = nn.ModuleList([e for e in embs if e is not None])
        self._emb_index = []
        real_idx = 0
        for e in embs:
            if e is not None:
                self._emb_index.append(real_idx)
                real_idx += 1
            else:
                self._emb_index.append(-1)

        self.group_projs = nn.ModuleList([
            nn.Sequential(
                nn.Linear(len(group) * emb_dim, d_model),
                nn.LayerNorm(d_model),
            )
            for group in groups
        ])

    def forward(self, int_feats: torch.Tensor) -> torch.Tensor:
        tokens = []
        for group, proj in zip(self.groups, self.group_projs):
            fid_embs = []
            for fid_idx in group:
                vs, offset, length = self.feature_specs[fid_idx]
                emb_real_idx = self._emb_index[fid_idx]
                if emb_real_idx == -1:
                    fid_emb = int_feats.new_zeros(int_feats.shape[0], self.emb_dim)
                else:
                    emb_layer = self.embs[emb_real_idx]
                    if length == 1:
                        fid_emb = emb_layer(int_feats[:, offset].long())
                    else:
                        vals = int_feats[:, offset:offset + length].long()
                        emb_all = emb_layer(vals)
                        mask = (vals != 0).float().unsqueeze(-1)
                        count = mask.sum(dim=1).clamp(min=1)
                        fid_emb = (emb_all * mask).sum(dim=1) / count
                fid_embs.append(fid_emb)
            cat_emb = torch.cat(fid_embs, dim=-1)
            tokens.append(F.silu(proj(cat_emb)).unsqueeze(1))
        return torch.cat(tokens, dim=1)


# ═══════════════════════════════════════════════════════════════════════════════
# DINSeqPCVR: DIN + User Profile architecture
# ═══════════════════════════════════════════════════════════════════════════════


class DINSeqPCVR(nn.Module):
    """DIN + User Profile architecture.

    Two independent pathways, fused at the end:

    Pathway A — User Profile (stable):
        User Int → GroupNSTokenizer → pool → user_vec
        User Dense → BatchNorm → Project → + user_vec → user_profile

    Pathway B — Item×Sequence DIN matching:
        Item Int → GroupNSTokenizer → pool → item_query
        For each seq domain:
            item_query ──CrossAttention──→ seq_tokens → match_signal

    Fusion:
        [user_profile ‖ match_a ‖ match_b ‖ match_c ‖ match_d] → MLP → logits
    """

    def __init__(
        self,
        # Data schema
        user_int_feature_specs: List[Tuple[int, int, int]],
        item_int_feature_specs: List[Tuple[int, int, int]],
        user_dense_dim: int,
        item_dense_dim: int,
        seq_vocab_sizes: Dict[str, List[int]],
        # NS grouping config
        user_ns_groups: List[List[int]],
        item_ns_groups: List[List[int]],
        # Model hyperparameters
        d_model: int = 64,
        emb_dim: int = 64,
        num_heads: int = 4,
        dropout_rate: float = 0.01,
        action_num: int = 1,
        num_time_buckets: int = 65,
        use_rope: bool = False,
        rope_base: float = 10000.0,
        emb_skip_threshold: int = 0,
        seq_id_threshold: int = 10000,
        # Other (kept for compatibility)
        num_queries: int = 1,
        num_hyformer_blocks: int = 2,
        seq_encoder_type: str = 'transformer',
        hidden_mult: int = 4,
        seq_top_k: int = 50,
        seq_causal: bool = False,
        rank_mixer_mode: str = 'full',
        ns_tokenizer_type: str = 'group',
        user_ns_tokens: int = 0,
        item_ns_tokens: int = 0,
        use_din: bool = False,
    ) -> None:
        super().__init__()
        del num_queries, num_hyformer_blocks, seq_encoder_type, hidden_mult
        del seq_top_k, seq_causal, rank_mixer_mode, ns_tokenizer_type
        del user_ns_tokens, item_ns_tokens, use_din

        self.d_model = d_model
        self.emb_dim = emb_dim
        self.action_num = action_num
        self.seq_domains = sorted(seq_vocab_sizes.keys())
        self.num_sequences = len(self.seq_domains)
        self.num_time_buckets = num_time_buckets
        self.use_rope = use_rope
        self.emb_skip_threshold = emb_skip_threshold
        self.seq_id_threshold = seq_id_threshold

        # ── Pathway A: User Profile ──
        self.user_ns_tokenizer = GroupNSTokenizer(
            feature_specs=user_int_feature_specs,
            groups=user_ns_groups,
            emb_dim=emb_dim,
            d_model=d_model,
            emb_skip_threshold=emb_skip_threshold,
        )
        num_user_ns = len(user_ns_groups)
        user_ns_flat_dim = num_user_ns * d_model

        self.has_user_dense = user_dense_dim > 0
        if self.has_user_dense:
            self.user_dense_norm = nn.InstanceNorm1d(user_dense_dim, affine=False)
            self.user_dense_proj = nn.Sequential(
                nn.Linear(user_dense_dim, d_model),
                nn.LayerNorm(d_model),
            )
            user_ns_flat_dim += d_model

        self.user_profile = nn.Sequential(
            nn.Linear(user_ns_flat_dim, d_model * 2),
            nn.LayerNorm(d_model * 2),
            nn.SiLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(d_model * 2, d_model),
            nn.LayerNorm(d_model),
        )

        # ── Pathway B: Item Query ──
        self.has_item_dense = item_dense_dim > 0
        self.item_ns_tokenizer = GroupNSTokenizer(
            feature_specs=item_int_feature_specs,
            groups=item_ns_groups,
            emb_dim=emb_dim,
            d_model=d_model,
            emb_skip_threshold=emb_skip_threshold,
        )
        num_item_ns = len(item_ns_groups)
        item_flat_dim = num_item_ns * d_model
        if self.has_item_dense:
            self.item_dense_proj = nn.Sequential(
                nn.Linear(item_dense_dim, d_model),
                nn.LayerNorm(d_model),
            )
            item_flat_dim += d_model

        self.item_query_proj = nn.Sequential(
            nn.Linear(item_flat_dim, d_model),
            nn.LayerNorm(d_model),
            nn.SiLU(),
        )

        # ── Sequence Embedding ──
        self.seq_id_emb_dropout = nn.Dropout(dropout_rate * 2)

        def _make_seq_embs(vocab_sizes):
            embs_raw = []
            for vs in vocab_sizes:
                skip = int(vs) <= 0 or (emb_skip_threshold > 0 and int(vs) > emb_skip_threshold)
                if skip:
                    embs_raw.append(None)
                else:
                    embs_raw.append(nn.Embedding(int(vs) + 1, emb_dim, padding_idx=0))
            module_list = nn.ModuleList([e for e in embs_raw if e is not None])
            index_map = []
            real_idx = 0
            for e in embs_raw:
                if e is not None:
                    index_map.append(real_idx)
                    real_idx += 1
                else:
                    index_map.append(-1)
            is_id = [int(vs) > seq_id_threshold for vs in vocab_sizes]
            return module_list, index_map, is_id

        self._seq_embs = nn.ModuleDict()
        self._seq_emb_index = {}
        self._seq_is_id = {}
        self._seq_vocab_sizes = {}
        self._seq_proj = nn.ModuleDict()

        for domain in self.seq_domains:
            vs = seq_vocab_sizes[domain]
            embs, idx_map, is_id = _make_seq_embs(vs)
            self._seq_embs[domain] = embs
            self._seq_emb_index[domain] = idx_map
            self._seq_is_id[domain] = is_id
            self._seq_vocab_sizes[domain] = vs
            self._seq_proj[domain] = nn.Sequential(
                nn.Linear(len(vs) * emb_dim, d_model),
                nn.LayerNorm(d_model),
            )

        if num_time_buckets > 0:
            self.time_embedding = nn.Embedding(num_time_buckets, d_model, padding_idx=0)

        # ── RoPE ──
        if use_rope:
            head_dim = d_model // num_heads
            self.rotary_emb = RotaryEmbedding(dim=head_dim, base=rope_base)
        else:
            self.rotary_emb = None

        # ── DIN Cross-Attention per domain ──
        self.din_attns = nn.ModuleDict({
            domain: CrossAttention(
                d_model=d_model, num_heads=num_heads, dropout=dropout_rate,
            )
            for domain in self.seq_domains
        })

        # ── Dropout ──
        self.dropout = nn.Dropout(dropout_rate)

        # ── Fusion ──
        fusion_in = d_model + d_model * self.num_sequences
        self.fusion = nn.Sequential(
            nn.Linear(fusion_in, d_model * 2),
            nn.LayerNorm(d_model * 2),
            nn.SiLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(d_model * 2, d_model),
            nn.LayerNorm(d_model),
            nn.SiLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(d_model, action_num),
        )

        self._init_params()
        self._log_filtered()

    def _init_params(self) -> None:
        for domain in self.seq_domains:
            for emb in self._seq_embs[domain]:
                nn.init.xavier_normal_(emb.weight.data)
                emb.weight.data[0, :] = 0

    def _log_filtered(self) -> None:
        if self.emb_skip_threshold > 0:
            for domain in self.seq_domains:
                filtered = sum(1 for idx in self._seq_emb_index[domain] if idx == -1)
                total = len(self._seq_vocab_sizes[domain])
                if filtered > 0:
                    logging.info(
                        f"emb_skip_threshold={self.emb_skip_threshold}: "
                        f"{domain} skipped {filtered}/{total} features")
            for name, tokenizer in [
                ("user_ns", self.user_ns_tokenizer),
                ("item_ns", self.item_ns_tokenizer),
            ]:
                f = sum(1 for idx in tokenizer._emb_index if idx == -1)
                t = len(tokenizer._emb_index)
                if f > 0:
                    logging.info(
                        f"emb_skip_threshold={self.emb_skip_threshold}: "
                        f"{name} skipped {f}/{t} features")

    def reinit_high_cardinality_params(
        self, cardinality_threshold: int = 10000
    ) -> "set[int]":
        reinit_count = 0
        skip_count = 0
        reinit_ptrs = set()

        for emb_list, vocab_sizes, emb_index in [
            (self._seq_embs[d], self._seq_vocab_sizes[d], self._seq_emb_index[d])
            for d in self.seq_domains
        ]:
            for i, vs in enumerate(vocab_sizes):
                real_idx = emb_index[i]
                if real_idx == -1:
                    continue
                emb = emb_list[real_idx]
                if int(vs) > cardinality_threshold:
                    nn.init.xavier_normal_(emb.weight.data)
                    emb.weight.data[0, :] = 0
                    reinit_ptrs.add(emb.weight.data_ptr())
                    reinit_count += 1
                else:
                    skip_count += 1

        for tokenizer in [self.user_ns_tokenizer, self.item_ns_tokenizer]:
            specs = tokenizer.feature_specs
            for i, (vs, offset, length) in enumerate(specs):
                real_idx = tokenizer._emb_index[i]
                if real_idx == -1:
                    continue
                emb = tokenizer.embs[real_idx]
                if int(vs) > cardinality_threshold:
                    nn.init.xavier_normal_(emb.weight.data)
                    emb.weight.data[0, :] = 0
                    reinit_ptrs.add(emb.weight.data_ptr())
                    reinit_count += 1
                else:
                    skip_count += 1

        if self.num_time_buckets > 0:
            skip_count += 1

        logging.info(
            f"Re-initialized {reinit_count} high-cardinality Embeddings "
            f"(vocab>{cardinality_threshold}), kept {skip_count}")
        return reinit_ptrs

    def get_sparse_params(self) -> List[nn.Parameter]:
        sparse_params = set()
        for module in self.modules():
            if isinstance(module, nn.Embedding):
                sparse_params.add(module.weight.data_ptr())
        return [p for p in self.parameters() if p.data_ptr() in sparse_params]

    def get_dense_params(self) -> List[nn.Parameter]:
        sparse_ptrs = {p.data_ptr() for p in self.get_sparse_params()}
        return [p for p in self.parameters() if p.data_ptr() not in sparse_ptrs]

    def _embed_seq_domain(
        self,
        data: torch.Tensor,
        embs: nn.ModuleList,
        proj: nn.Sequential,
        is_id: List[bool],
        emb_index: List[int],
        time_bucket_ids: torch.Tensor,
    ) -> torch.Tensor:
        B, num_feats, L = data.shape

        all_emb = []
        for fi in range(num_feats):
            real_idx = emb_index[fi]
            vals = data[:, fi, :].long()
            if real_idx == -1:
                fid_emb = torch.zeros(B, L, self.emb_dim, device=data.device)
            else:
                emb_layer = embs[real_idx]
                fid_emb = emb_layer(vals)
                if is_id[fi]:
                    fid_emb = self.seq_id_emb_dropout(fid_emb)
            all_emb.append(fid_emb)

        token_emb = proj(torch.cat(all_emb, dim=-1))

        if self.num_time_buckets > 0:
            token_emb = token_emb + self.time_embedding(time_bucket_ids)

        return token_emb

    def _make_padding_mask(
        self, seq_lens: torch.Tensor, max_len: int
    ) -> torch.Tensor:
        B = seq_lens.shape[0]
        positions = torch.arange(max_len, device=seq_lens.device).unsqueeze(0)
        return positions >= seq_lens.unsqueeze(1)

    def _build_user_profile(
        self,
        user_int_feats: torch.Tensor,
        user_dense_feats: torch.Tensor,
    ) -> torch.Tensor:
        user_ns = self.user_ns_tokenizer(user_int_feats)
        B = user_ns.shape[0]
        user_flat = user_ns.view(B, -1)

        if self.has_user_dense:
            dense = self.user_dense_norm(user_dense_feats)
            dense = F.silu(self.user_dense_proj(dense))
            user_flat = torch.cat([user_flat, dense], dim=-1)

        user_flat = self.dropout(user_flat)
        return self.user_profile(user_flat)

    def _build_item_query(
        self,
        item_int_feats: torch.Tensor,
        item_dense_feats: torch.Tensor,
    ) -> torch.Tensor:
        item_ns = self.item_ns_tokenizer(item_int_feats)
        B = item_ns.shape[0]
        item_flat = item_ns.view(B, -1)

        if self.has_item_dense:
            dense = F.silu(self.item_dense_proj(item_dense_feats))
            item_flat = torch.cat([item_flat, dense], dim=-1)

        item_flat = self.dropout(item_flat)
        return self.item_query_proj(item_flat)

    def forward(self, inputs: ModelInput) -> torch.Tensor:
        # 1. User Profile
        user_profile = self._build_user_profile(
            inputs.user_int_feats, inputs.user_dense_feats,
        )  # (B, D)

        # 2. Item Query
        item_query = self._build_item_query(
            inputs.item_int_feats, inputs.item_dense_feats,
        )  # (B, D)
        item_query = item_query.unsqueeze(1)  # (B, 1, D)

        # 3. Sequence Embedding + DIN matching
        match_signals = []
        for domain in self.seq_domains:
            seq_tokens = self._embed_seq_domain(
                inputs.seq_data[domain],
                self._seq_embs[domain], self._seq_proj[domain],
                self._seq_is_id[domain], self._seq_emb_index[domain],
                inputs.seq_time_buckets[domain],
            )  # (B, L, D)

            seq_mask = self._make_padding_mask(
                inputs.seq_lens[domain], inputs.seq_data[domain].shape[2],
            )  # (B, L)

            rope_cos = rope_sin = None
            if self.rotary_emb is not None:
                rope_cos, rope_sin = self.rotary_emb(
                    seq_tokens.shape[1], seq_tokens.device)

            din_out = self.din_attns[domain](
                item_query, seq_tokens,
                key_padding_mask=seq_mask,
                rope_cos=rope_cos, rope_sin=rope_sin,
            )  # (B, 1, D)
            match_signals.append(din_out.squeeze(1))  # (B, D)

        # 4. Fusion
        fused = torch.cat([user_profile] + match_signals, dim=-1)
        logits = self.fusion(fused)
        return logits

    def predict(self, inputs: ModelInput) -> Tuple[torch.Tensor, torch.Tensor]:
        """Inference without dropout."""
        user_profile = self._build_user_profile(
            inputs.user_int_feats, inputs.user_dense_feats)

        item_query = self._build_item_query(
            inputs.item_int_feats, inputs.item_dense_feats)
        item_query = item_query.unsqueeze(1)

        match_signals = []
        for domain in self.seq_domains:
            seq_tokens = self._embed_seq_domain(
                inputs.seq_data[domain],
                self._seq_embs[domain], self._seq_proj[domain],
                self._seq_is_id[domain], self._seq_emb_index[domain],
                inputs.seq_time_buckets[domain],
            )
            seq_mask = self._make_padding_mask(
                inputs.seq_lens[domain], inputs.seq_data[domain].shape[2])

            rope_cos = rope_sin = None
            if self.rotary_emb is not None:
                rope_cos, rope_sin = self.rotary_emb(
                    seq_tokens.shape[1], seq_tokens.device)

            din_out = self.din_attns[domain](
                item_query, seq_tokens,
                key_padding_mask=seq_mask,
                rope_cos=rope_cos, rope_sin=rope_sin,
            )
            match_signals.append(din_out.squeeze(1))

        fused = torch.cat([user_profile] + match_signals, dim=-1)
        logits = self.fusion(fused)
        return logits, user_profile
