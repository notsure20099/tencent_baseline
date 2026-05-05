"""PCVR Data-Only Diagnostics — No model, pure feature analysis.

Loads raw Parquet data directly via PyArrow (no PyTorch, no model),
samples a configurable fraction of Row Groups, then prints a diagnostic
report covering seven dimensions: label distribution, sequence truncation,
int-feature discrimination, dense-feature magnitude gap, feature sparsity,
user-level label noise, and timestamp coverage.

Usage:
    python diagnose_data.py
    python diagnose_data.py --sample 0.05

Requires: pyarrow, numpy (already present in the training environment).
"""

import os
import sys
import json
import math
import time
import argparse
from datetime import datetime
from collections import defaultdict
from typing import Any, Dict, List, Tuple

import numpy as np
import pyarrow.parquet as pq


# ═══════════════════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════════════════

def fm(n: int) -> str:
    return f"{n:,}"


def fmf(n: float) -> str:
    return f"{n:,.4f}"


def fmbig(n: float) -> str:
    if abs(n) < 1e-6 and n != 0:
        return f"{n:.4e}"
    return f"{n:,.4f}"


def load_schema(path: str) -> Dict[str, Any]:
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)


def get_parquet_files(data_dir: str) -> List[str]:
    if os.path.isfile(data_dir) and data_dir.endswith('.parquet'):
        return [data_dir]
    files = sorted([os.path.join(data_dir, f)
                    for f in os.listdir(data_dir) if f.endswith('.parquet')])
    if not files:
        raise FileNotFoundError(f'No .parquet files found in {data_dir}')
    return files


def _h(title: str) -> str:
    return f"\n{'='*68}\n  {title}\n{'='*68}"


# ═══════════════════════════════════════════════════════════════════════════════
# Column extraction (parquet → flat numpy)
# ═══════════════════════════════════════════════════════════════════════════════

def extract_int_col(batch, col_name: str, dim: int) -> np.ndarray:
    col = batch.column(col_name)
    if dim == 1:
        arr = col.fill_null(0).to_numpy(zero_copy_only=False).astype(np.int64)
        arr[arr <= 0] = 0
        return arr
    offsets = col.offsets.to_numpy()
    vals = col.values.to_numpy()
    B = len(offsets) - 1
    out = np.zeros((B, dim), dtype=np.int64)
    for i in range(B):
        s, e = int(offsets[i]), int(offsets[i + 1])
        if e <= s:
            continue
        ul = min(e - s, dim)
        out[i, :ul] = vals[s:s + ul]
    out[out <= 0] = 0
    return out.ravel()


def extract_dense_col(batch, col_name: str, dim: int) -> np.ndarray:
    col = batch.column(col_name)
    offsets = col.offsets.to_numpy()
    vals = col.values.to_numpy().astype(np.float64)
    B = len(offsets) - 1
    out = np.zeros((B, dim), dtype=np.float64)
    for i in range(B):
        s, e = int(offsets[i]), int(offsets[i + 1])
        if e <= s:
            continue
        ul = min(e - s, dim)
        out[i, :ul] = vals[s:s + ul]
    return out.ravel()


def extract_label(batch) -> np.ndarray:
    if 'label_type' not in batch.schema.names:
        return np.zeros(batch.num_rows, dtype=np.int64)
    labels = (batch.column('label_type').fill_null(0)
              .to_numpy(zero_copy_only=False).astype(np.int64) == 2).astype(np.int64)
    return labels


def extract_user_ids(batch) -> List[str]:
    return batch.column('user_id').to_pylist()


def extract_seq_lengths(batch, first_col: str) -> np.ndarray:
    offsets = batch.column(first_col).offsets.to_numpy()
    return np.diff(offsets).astype(np.int64)


# ═══════════════════════════════════════════════════════════════════════════════
# Batch iterator
# ═══════════════════════════════════════════════════════════════════════════════

def iter_batches(files: List[str], max_rgs: int):
    rg_count = 0
    for fpath in files:
        if rg_count >= max_rgs:
            break
        pf = pq.ParquetFile(fpath)
        for rg_idx in range(pf.metadata.num_row_groups):
            if rg_count >= max_rgs:
                break
            for batch in pf.iter_batches(batch_size=65536, row_groups=[rg_idx]):
                yield batch
            rg_count += 1


# ═══════════════════════════════════════════════════════════════════════════════
# JS Divergence for two discrete distributions
# ═══════════════════════════════════════════════════════════════════════════════

def js_divergence(p: np.ndarray, q: np.ndarray) -> float:
    eps = 1e-10
    p = np.clip(p, eps, 1 - eps)
    q = np.clip(q, eps, 1 - eps)
    m = 0.5 * (p + q)
    return 0.5 * np.sum(p * np.log(p / m)) + 0.5 * np.sum(q * np.log(q / m))


# ═══════════════════════════════════════════════════════════════════════════════
# Main diagnostics
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description='PCVR Data Diagnostics')
    parser.add_argument('--sample', type=float, default=1.0)
    parser.add_argument('--data_dir', type=str, default=None)
    args = parser.parse_args()

    data_dir = args.data_dir or os.environ.get('TRAIN_DATA_PATH', '')
    if not data_dir:
        print("ERROR: TRAIN_DATA_PATH not set")
        sys.exit(1)

    sp = os.path.join(data_dir, 'schema.json')
    if not os.path.exists(sp):
        print(f"ERROR: schema.json not found at {sp}")
        sys.exit(1)

    t0 = time.time()
    schema = load_schema(sp)
    files = get_parquet_files(data_dir)

    # ── file metadata ──
    total_rgs = 0
    total_rows_meta = 0
    rg_sizes = []
    for f in files:
        pf = pq.ParquetFile(f)
        total_rgs += pf.metadata.num_row_groups
        total_rows_meta += pf.metadata.num_rows
        for i in range(pf.metadata.num_row_groups):
            rg_sizes.append(pf.metadata.row_group(i).num_rows)

    n_sample_rgs = max(1, int(total_rgs * args.sample))
    sample_rows = sum(rg_sizes[:n_sample_rgs])
    columns = pq.ParquetFile(files[0]).schema_arrow.names

    seq_cfg = schema.get('seq', {})
    user_int_cfg = schema['user_int']
    user_dense_cfg = schema['user_dense']

    # Flatten user_int feature list: fid -> (vocab_size, dim)
    ui_fid_info = {}
    for fid, vs, dim in user_int_cfg:
        ui_fid_info[fid] = (vs, dim)

    ud_fid_info = {}
    for fid, dim in user_dense_cfg:
        ud_fid_info[fid] = dim

    # ════════════════════════════════════════════════════════════════════
    # Collect label-conditional statistics
    # ════════════════════════════════════════════════════════════════════

    total = 0
    pos_count = 0
    label_noise_users = defaultdict(set)       # user_id -> set of labels

    # per-seq-domain: list of (length, is_truncated)
    seq_len_samples: Dict[str, List[int]] = {d: [] for d in seq_cfg}
    seq_trunc_counts: Dict[str, int] = {d: 0 for d in seq_cfg}

    # ── user_int non-zero ratio arrays ──
    dense_feat_stats: Dict[int, Tuple[float, float, int, float, float, int, float, float]] = {}
    for fid, dim in user_dense_cfg:
        dense_feat_stats[fid] = (0.0, 0.0, 0, 0.0, 0.0, 0, float('inf'), float('-inf'))

    # user_int sparsity per sample (overall)
    int_nz_ratio_pos = []
    int_nz_ratio_neg = []

    # Also collect dense feature binned AUC: for each dense fid, collect (value, label) pairs
    dense_val_label: Dict[int, List[Tuple[float, int]]] = {fid: [] for fid, _ in user_dense_cfg}

    seq_max_lens: Dict[str, int] = {d: 256 for d in seq_cfg}

    sys.stdout.write(f"[DIAG] Sampling {sample_rows:,} rows / {n_sample_rgs} RGs ...\n")
    sys.stdout.flush()

    for batch in iter_batches(files, n_sample_rgs):
        labels = extract_label(batch)
        B = len(labels)
        total += B
        pos_count += int(labels.sum())

        batch_pos = labels == 1
        batch_neg = labels == 0
        n_pos = batch_pos.sum()
        n_neg = B - n_pos

        # User-level label noise
        if 'user_id' in columns:
            uids = extract_user_ids(batch)
            for i in range(B):
                label_noise_users[uids[i]].add(int(labels[i]))

        # ── Dense features: collect raw values ──
        for fid, dim in user_dense_cfg:
            cn = f'user_dense_feats_{fid}'
            if cn not in columns:
                continue
            arr = extract_dense_col(batch, cn, dim)  # (B*dim,)
            arr_b = arr.reshape(B, dim)
            # Per-sample mean
            sample_vals = arr
            if dim > 1:
                nz_count = (arr_b != 0).sum(axis=1).clip(min=1)
                sample_vals = arr_b.sum(axis=1) / nz_count

            # Collect for binned AUC
            for i in range(B):
                dense_val_label[fid].append((float(sample_vals[i]), int(labels[i])))

            # Stats by label
            pos_vals = sample_vals[batch_pos]
            neg_vals = sample_vals[batch_neg]
            sp_s, ss2_s, cn_s, sp_d, ss2_d, cn_d, vmin, vmax = dense_feat_stats[fid]
            if len(pos_vals) > 0:
                sp_s += pos_vals.sum()
                ss2_s += (pos_vals ** 2).sum()
                cn_s += len(pos_vals)
            if len(neg_vals) > 0:
                sp_d += neg_vals.sum()
                ss2_d += (neg_vals ** 2).sum()
                cn_d += len(neg_vals)
            vmin = min(vmin, float(sample_vals.min())) if len(sample_vals) > 0 else vmin
            vmax = max(vmax, float(sample_vals.max())) if len(sample_vals) > 0 else vmax
            dense_feat_stats[fid] = (sp_s, ss2_s, cn_s, sp_d, ss2_d, cn_d, vmin, vmax)

        # ── Int feature sparsity per sample ──
        user_int_col_first = f'user_int_feats_{user_int_cfg[0][0]}'
        if user_int_col_first in columns:
            total_int_dim = sum(d for _, _, d in user_int_cfg)
            int_arr = np.zeros((B, total_int_dim), dtype=np.int64)
            offset = 0
            for fid, vs, dim in user_int_cfg:
                cn = f'user_int_feats_{fid}'
                if cn not in columns:
                    continue
                if dim == 1:
                    col = batch.column(cn)
                    arr = col.fill_null(0).to_numpy(zero_copy_only=False).astype(np.int64)
                    arr[arr <= 0] = 0
                    int_arr[:, offset] = arr
                else:
                    col = batch.column(cn)
                    offsets = col.offsets.to_numpy()
                    vals = col.values.to_numpy()
                    for i in range(B):
                        s, e = int(offsets[i]), int(offsets[i + 1])
                        if e <= s: continue
                        ul = min(e - s, dim)
                        tmp = vals[s:s + ul].copy()
                        tmp[tmp <= 0] = 0
                        int_arr[i, offset:offset + ul] = tmp
                offset += dim
            nz_ratio = (int_arr != 0).sum(axis=1) / max(1, total_int_dim)
            int_nz_ratio_pos.extend(nz_ratio[batch_pos].tolist())
            int_nz_ratio_neg.extend(nz_ratio[batch_neg].tolist())

        # ── Sequence lengths ──
        for domain in sorted(seq_cfg.keys()):
            cfg_d = seq_cfg[domain]
            fids = cfg_d['features']
            first_col = f"{cfg_d['prefix']}_{fids[0][0]}"
            if first_col not in columns:
                continue
            lengths = extract_seq_lengths(batch, first_col)
            for slen in lengths:
                slen_i = int(slen)
                is_trunc = slen_i > seq_max_lens.get(domain, 256)
                if len(seq_len_samples[domain]) < 50000:
                    seq_len_samples[domain].append(slen_i)
                if is_trunc:
                    seq_trunc_counts[domain] = seq_trunc_counts.get(domain, 0) + 1
                seq_trunc_counts[f'{domain}_total'] = seq_trunc_counts.get(f'{domain}_total', 0) + 1

    # ════════════════════════════════════════════════════════════════════
    # Print report
    # ════════════════════════════════════════════════════════════════════

    lines: List[str] = []
    lines.append("=" * 68)
    lines.append("  PCVR DATA-ONLY DIAGNOSTICS REPORT (no model required)")
    lines.append(f"  Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append("=" * 68)

    # ── 0. Overview ──
    pos_rate = pos_count / max(1, total)
    lines.append(_h("0. Dataset Overview"))
    lines.append(f"  Total rows sampled : {fm(total)}")
    lines.append(f"  Positive (label=1) : {fm(pos_count)}  ({pos_rate:.4%})")
    lines.append(f"  Negative (label=0) : {fm(total - pos_count)}  ({1 - pos_rate:.4%})")
    if pos_count:
        lines.append(f"  Pos:Neg ratio      : 1:{ (total - pos_count) / pos_count:.1f}")

    # ── 1. Sequence truncation analysis ──
    lines.append(_h("1. Sequence Truncation Analysis"))
    lines.append(f"  {'Domain':<8} {'Trunc%':>8}  {'Mean':>8}  "
                 f"{'P50':>8}  {'P90':>8}  {'P95':>8}  {'P99':>8}  {'Max':>8}  {'>10':>8}")
    lines.append(f"  {'-'*64}")
    for domain in sorted(seq_cfg.keys()):
        samples = seq_len_samples.get(domain, [])
        if not samples:
            lines.append(f"  {domain:<8} {'N/A'}")
            continue
        sa = np.array(sorted(samples))
        trunc_total = seq_trunc_counts.get(f'{domain}_total', 1)
        trunc_rate = seq_trunc_counts.get(domain, 0) / max(1, trunc_total)
        over10 = (sa > 10).mean() * 100 if len(sa) > 0 else 0
        lines.append(
            f"  {domain:<8} {trunc_rate:>7.1%}  {sa.mean():>8.1f}  "
            f"{np.percentile(sa, 50):>8.0f}  {np.percentile(sa, 90):>8.0f}  "
            f"{np.percentile(sa, 95):>8.0f}  {np.percentile(sa, 99):>8.0f}  "
            f"{sa.max():>8.0f}  {over10:>7.1f}%"
        )

    # ── 2. Dense feature magnitude gap (pos vs neg) ──
    lines.append(_h("2. Dense Feature Magnitude Gap (mean value pos vs neg)"))
    lines.append(f"  {'FID':>6} {'dim':>5}  {'mean_pos':>12}  {'mean_neg':>12}  "
                 f"{'Δabs':>10}  {'Δrel%':>8}  {'min':>10}  {'max':>12}  {'nz%':>6}")
    lines.append(f"  {'-'*73}")
    for fid, dim in user_dense_cfg:
        sp_s, ss2_s, cn_s, sp_d, ss2_d, cn_d, vmin, vmax = dense_feat_stats[fid]
        mean_p = sp_s / max(1, cn_s)
        mean_n = sp_d / max(1, cn_d)
        delta_abs = abs(mean_p - mean_n)
        max_abs = max(abs(mean_p), abs(mean_n), 1e-10)
        delta_rel = delta_abs / max_abs * 100
        # Count non-zero in dense_val_label for this fid
        dl = dense_val_label.get(fid, [])
        nz_pct = sum(1 for v, _ in dl if v != 0) / max(1, len(dl)) * 100 if dl else 0
        if cn_s + cn_d == 0:
            lines.append(f"  {fid:>6} {dim:>5}  {'NO DATA'}")
        else:
            lines.append(f"  {fid:>6} {dim:>5}  {fmbig(mean_p):>12}  {fmbig(mean_n):>12}  "
                         f"{fmbig(delta_abs):>10}  {delta_rel:>7.1f}%  "
                         f"{fmbig(vmin):>10}  {fmbig(vmax):>12}  {nz_pct:>5.1f}%")

    # ── 3. Dense feature binned AUC ──
    lines.append(_h("3. Dense Feature Per-Feature AUC (raw value → label)"))
    lines.append(f"  {'FID':>6}  {'AUC':>8}  {'HighMean':>12}  {'LowMean':>12}  "
                 f"{'HighPos%':>8}")
    lines.append(f"  {'-'*50}")
    for fid, dim in user_dense_cfg:
        dl = dense_val_label.get(fid, [])
        if len(dl) < 100:
            continue
        vals = np.array([v for v, _ in dl])
        labs = np.array([l for _, l in dl])
        nz_mask = vals != 0
        if nz_mask.sum() < 50:
            continue
        try:
            from sklearn.metrics import roc_auc_score
            if len(np.unique(labs[nz_mask])) < 2:
                continue
            auc_f = float(roc_auc_score(labs[nz_mask], vals[nz_mask]))
        except Exception:
            continue
        # High vs low split
        median = np.median(vals[nz_mask])
        hi_mask = nz_mask & (vals >= median)
        lo_mask = nz_mask & (vals < median)
        hi_pos = labs[hi_mask].mean() if hi_mask.sum() > 0 else 0
        lo_pos = labs[lo_mask].mean() if lo_mask.sum() > 0 else 0
        hi_mean = vals[hi_mask].mean() if hi_mask.sum() > 0 else 0
        lo_mean = vals[lo_mask].mean() if lo_mask.sum() > 0 else 0
        # Marker for large dynamic range
        flag = " ⚠ BIG" if (vals[nz_mask].max() / max(abs(vals[nz_mask].min()), 1e-6) > 1000) else ""
        lines.append(f"  {fid:>6}  {auc_f:>8.4f}  {fmbig(hi_mean):>12}  {fmbig(lo_mean):>12}  "
                     f"{hi_pos:>7.1%}{flag}")

    # ── 4. Feature sparsity ──
    lines.append(_h("4. Feature Sparsity by Label"))
    pos_sp_mean = np.mean(int_nz_ratio_pos) if int_nz_ratio_pos else 0
    neg_sp_mean = np.mean(int_nz_ratio_neg) if int_nz_ratio_neg else 0
    lines.append(f"  User-int non-zero ratio (mean):")
    lines.append(f"    Positive samples: {pos_sp_mean:.4f}  (n={len(int_nz_ratio_pos):,})")
    lines.append(f"    Negative samples: {neg_sp_mean:.4f}  (n={len(int_nz_ratio_neg):,})")
    lines.append(f"    Δ = {pos_sp_mean - neg_sp_mean:+.4f}")
    if pos_sp_mean > 0 and neg_sp_mean > 0:
        gap_pct = abs(pos_sp_mean - neg_sp_mean) / max(pos_sp_mean, neg_sp_mean) * 100
        lines.append(f"    Relative gap: {gap_pct:.1f}%")

    # ── 5. User-level label noise ──
    lines.append(_h("5. User-Level Label Consistency"))
    multi_sample = {u: labs for u, labs in label_noise_users.items() if len(labs) >= 2}
    consistent = sum(1 for labs in multi_sample.values() if len(labs) == 1)
    inconsistent = len(multi_sample) - consistent
    if multi_sample:
        lines.append(f"  Users with >=2 samples: {len(multi_sample):,}")
        lines.append(f"  Consistent labels:      {consistent:,} "
                     f"({consistent / len(multi_sample) * 100:.1f}%)")
        lines.append(f"  Inconsistent labels:    {inconsistent:,} "
                     f"({inconsistent / len(multi_sample) * 100:.1f}%)")
        if inconsistent > len(multi_sample) * 0.15:
            lines.append(f"  ⚠ WARNING: >15% users have conflicting labels — potential noise issue")
    else:
        lines.append("  No users with >=2 samples found (or user_id column missing)")

    # ── 6. OOB / high-cardinality features ──
    lines.append(_h("6. High-Cardinality Feature Risk"))
    high_vocab = []
    for fid, vs, dim in user_int_cfg:
        if vs > 100000:
            high_vocab.append((fid, vs, dim))
    for fid, vs, dim in high_vocab:
        lines.append(f"  user_int_feats_{fid}: vocab={fm(vs)}, dim={dim}")
    for domain in sorted(seq_cfg.keys()):
        for fid, vs in seq_cfg[domain]['features']:
            if vs > 1000:
                lines.append(f"  seq_{domain}_fid{fid}: vocab={fm(vs)}")
    if not high_vocab:
        lines.append("  No features with vocab > 100K")

    # ── 7. Sequence truncation deep dive ──
    lines.append(_h("7. Sequence Truncation Deep-Dive"))
    for domain in sorted(seq_cfg.keys()):
        samples = seq_len_samples.get(domain, [])
        if not samples:
            continue
        sa = np.array(samples)
        max_len = seq_max_lens.get(domain, 256)
        edges = [0, max_len * 0.25, max_len * 0.5, max_len * 0.75,
                 max_len, max_len * 1.5, 999999]
        lines.append(f"  [{domain}] max_len={max_len}")
        lines.append(f"  {'Bucket':>16}  {'Count':>10}  {'Ratio':>8}")
        for i in range(len(edges) - 1):
            lo, hi = edges[i], edges[i + 1]
            mask = (sa >= lo) & (sa < hi)
            cnt = mask.sum()
            if cnt == 0:
                continue
            label = f"[{lo:.0f}, {hi:.0f})"
            if hi >= 99999:
                label = f"[{lo:.0f}, ∞)"
            lines.append(f"  {label:>16}  {cnt:>10,}  {cnt/len(sa):>7.1%}")

    # ── Done ──
    elapsed = time.time() - t0
    lines.append(_h("Done"))
    lines.append(f"  Elapsed  : {elapsed:.1f}s")
    lines.append(f"  Processed: {fm(total)} samples across {n_sample_rgs} RGs")
    lines.append(f"  Lines    : {len(lines)}")

    for line in lines:
        print(line)


if __name__ == '__main__':
    main()
