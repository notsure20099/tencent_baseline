"""
Feature Exploration — full data, streaming, single pass.

Runs on TRAIN data. Computes per-feature AUC, distribution stats,
and saves a feature_stats.json sidecar for test-side comparison.

Activated by: EXPLORE_MODE=true in run.sh (reads TRAIN_DATA_PATH env var).
"""

import os
import sys
import json
import time
import logging
import numpy as np
from typing import Dict, List, Tuple
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from dataset import PCVRParquetDataset

logging.basicConfig(level=logging.INFO, format="%(message)s")
log = logging.getLogger("explore_all")

# ═══════════════════════════════════════════════════════════════
# Streaming accumulators
# ═══════════════════════════════════════════════════════════════

class BinnedAUC:
    """Streaming approximate AUC via histogram binning."""
    __slots__ = ("n_bins", "exact", "vocab_size", "pos", "neg", "total_pos", "total_neg")

    def __init__(self, vocab_size: int, max_bins: int = 100):
        self.vocab_size = max(vocab_size, 1)
        if self.vocab_size <= max_bins:
            self.n_bins = self.vocab_size
            self.exact = True
        else:
            self.n_bins = max_bins
            self.exact = False
        self.pos = np.zeros(self.n_bins, dtype=np.int64)
        self.neg = np.zeros(self.n_bins, dtype=np.int64)
        self.total_pos = 0
        self.total_neg = 0

    def update(self, values: np.ndarray, labels: np.ndarray):
        if self.exact:
            bins = np.clip(values, 0, self.n_bins - 1)
        else:
            bins = np.clip((values.astype(np.float64) * self.n_bins / self.vocab_size).astype(np.int64),
                           0, self.n_bins - 1)
        pos_mask = labels == 1
        neg_mask = labels == 0
        self.pos += np.bincount(bins[pos_mask], minlength=self.n_bins)
        self.neg += np.bincount(bins[neg_mask], minlength=self.n_bins)
        self.total_pos += int(pos_mask.sum())
        self.total_neg += int(neg_mask.sum())

    def auc(self) -> float:
        if self.total_pos == 0 or self.total_neg == 0:
            return 0.5
        rates = np.divide(self.pos, self.pos + self.neg,
                          out=np.zeros(self.n_bins), where=(self.pos + self.neg) > 0)
        order = np.argsort(rates)
        cum_neg = 0
        auc_sum = 0.0
        for b in order:
            p = int(self.pos[b])
            n = int(self.neg[b])
            auc_sum += p * cum_neg + p * n * 0.5
            cum_neg += n
        return auc_sum / (self.total_pos * self.total_neg)


class RunningStats:
    """Welford single-pass mean/variance."""
    __slots__ = ("n", "mean", "M2")

    def __init__(self):
        self.n = 0
        self.mean = 0.0
        self.M2 = 0.0

    def update_many(self, values: np.ndarray):
        for x in values.flat:
            self.n += 1
            delta = x - self.mean
            self.mean += delta / self.n
            self.M2 += delta * (x - self.mean)

    @property
    def std(self) -> float:
        return float(np.sqrt(self.M2 / self.n)) if self.n > 1 else 0.0


class DistStats:
    """Streaming per-feature distribution accumulator."""
    __slots__ = ("non_zero", "total", "running")

    def __init__(self):
        self.non_zero = 0
        self.total = 0
        self.running = RunningStats()

    def update(self, values: np.ndarray):
        self.total += values.size
        self.non_zero += int((values != 0).sum())
        self.running.update_many(values)

    @property
    def nz_rate(self) -> float:
        return self.non_zero / self.total if self.total > 0 else 0.0

    @property
    def mean(self) -> float:
        return self.running.mean

    @property
    def std(self) -> float:
        return self.running.std


# ═══════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════

def main():
    t0 = time.time()

    data_dir = os.environ.get("TRAIN_DATA_PATH", "")
    if not data_dir:
        log.error("TRAIN_DATA_PATH env var is not set. Abort.")
        return

    schema_path = os.path.join(data_dir, "schema.json")
    if not os.path.exists(schema_path):
        log.error(f"schema.json not found at {schema_path}")
        return

    sml_str = os.environ.get("SEQ_MAX_LENS", "seq_a:256,seq_b:256,seq_c:512,seq_d:512")
    seq_max_lens = {}
    for pair in sml_str.split(","):
        k, v = pair.split(":")
        seq_max_lens[k.strip()] = int(v.strip())

    output_json = os.environ.get("EXPLORE_OUTPUT_JSON",
                                  os.path.join(os.path.dirname(data_dir), "feature_stats.json"))

    batch_size = 256
    num_workers = int(os.environ.get("EXPLORE_NUM_WORKERS", "4"))

    log.info("Loading dataset (full, streaming)...")
    dataset = PCVRParquetDataset(
        parquet_path=data_dir,
        schema_path=schema_path,
        batch_size=batch_size,
        seq_max_lens=seq_max_lens,
        shuffle=False,
        buffer_batches=0,
        is_training=True,
    )
    loader = DataLoader(dataset, batch_size=None, num_workers=num_workers)
    total_rows = dataset.num_rows

    # ── Extract feature metadata ──
    user_int_entries = dataset.user_int_schema.entries        # [(fid, offset, length), ...]
    item_int_entries = dataset.item_int_schema.entries
    user_dense_entries = dataset.user_dense_schema.entries     # [(fid, offset, length), ...]

    user_int_vocab = dataset.user_int_vocab_sizes              # flat, per-position
    item_int_vocab = dataset.item_int_vocab_sizes
    user_dense_dim = dataset.user_dense_schema.total_dim

    n_user_int = len(user_int_entries)
    n_item_int = len(item_int_entries)
    n_dense = user_dense_dim

    log.info("Data: %d rows, %d user_int fids, %d item_int fids, %d dense dims",
             total_rows, n_user_int, n_item_int, n_dense)

    # ── Accumulators ──
    int_aucs: Dict[str, BinnedAUC] = {}    # key: "user_int_{fid}" or "item_int_{fid}"
    int_dists: Dict[str, DistStats] = {}
    dense_dists: List[DistStats] = [DistStats() for _ in range(n_dense)]

    for fid, offset, length in user_int_entries:
        vs = max(user_int_vocab[offset:offset + length])
        key = f"user_int_{fid}"
        int_aucs[key] = BinnedAUC(vs)
        int_dists[key] = DistStats()

    for fid, offset, length in item_int_entries:
        vs = max(item_int_vocab[offset:offset + length])
        key = f"item_int_{fid}"
        int_aucs[key] = BinnedAUC(vs)
        int_dists[key] = DistStats()

    # ── Scan ──
    log.info("Scanning batches...")
    batch_count = 0
    for batch in loader:
        labels = batch["label"].numpy().astype(np.int64)
        user_int = batch["user_int_feats"].numpy()
        item_int = batch["item_int_feats"].numpy()
        user_dense = batch["user_dense_feats"].numpy()

        for fid, offset, length in user_int_entries:
            key = f"user_int_{fid}"
            vals = user_int[:, offset:offset + length].astype(np.int64)
            if length == 1:
                vals = vals.ravel()
                int_aucs[key].update(vals, labels)
                int_dists[key].update(vals)
            else:
                int_dists[key].update(vals)
                for l in range(length):
                    auc_key = f"{key}_d{l}"
                    if auc_key not in int_aucs:
                        int_aucs[auc_key] = BinnedAUC(int_aucs[key].vocab_size)
                    int_aucs[auc_key].update(vals[:, l], labels)

        for fid, offset, length in item_int_entries:
            key = f"item_int_{fid}"
            vals = item_int[:, offset:offset + length].astype(np.int64)
            if length == 1:
                vals = vals.ravel()
                int_aucs[key].update(vals, labels)
                int_dists[key].update(vals)
            else:
                int_dists[key].update(vals)
                for l in range(length):
                    auc_key = f"{key}_d{l}"
                    if auc_key not in int_aucs:
                        int_aucs[auc_key] = BinnedAUC(int_aucs[key].vocab_size)
                    int_aucs[auc_key].update(vals[:, l], labels)

        for d in range(n_dense):
            dense_dists[d].update(user_dense[:, d])

        batch_count += 1
        if batch_count % 500 == 0:
            log.info("  %d batches, %d rows...", batch_count, batch_count * batch_size)

    scan_time = time.time() - t0
    log.info("Scan done: %d batches in %.1fs", batch_count, scan_time)

    # ── Compute AUCs ──
    auc_results = []
    for key, auc_acc in int_aucs.items():
        auc_val = auc_acc.auc()
        if auc_val > 0.4:
            auc_results.append((key, auc_val))
    auc_results.sort(key=lambda x: -x[1])

    # ── Build feature summary ──
    int_summary = {}
    for key, dist in int_dists.items():
        if key in int_aucs:
            auc_val = int_aucs[key].auc()
        else:
            auc_val = max((int_aucs.get(f"{key}_d{l}", BinnedAUC(1)).auc()
                           for l in range(10) if f"{key}_d{l}" in int_aucs), default=0.5)
        parts = key.split("_", 2)
        ftype = f"{parts[0]}_{parts[1]}"
        fid = parts[2] if len(parts) > 2 else "?"
        int_summary[key] = {
            "fid": fid, "type": ftype,
            "auc": round(auc_val, 4),
            "nz_rate": round(dist.nz_rate, 4),
            "mean": round(dist.mean, 2), "std": round(dist.std, 2),
        }

    dense_summary = []
    for d in range(n_dense):
        dist = dense_dists[d]
        if dist.total > 0 and dist.nz_rate > 0.001:
            dense_summary.append({
                "dim": d, "nz_rate": round(dist.nz_rate, 4),
                "mean": round(dist.mean, 4), "std": round(dist.std, 4),
            })

    # ── Save JSON sidecar ──
    sidecar = {
        "source": "train",
        "total_rows": total_rows,
        "n_user_int": n_user_int,
        "n_item_int": n_item_int,
        "n_dense_dims": n_dense,
        "user_int_fids": [fid for fid, _, _ in user_int_entries],
        "item_int_fids": [fid for fid, _, _ in item_int_entries],
        "user_dense_fids": [fid for fid, _, _ in user_dense_entries],
        "int_features": int_summary,
        "dense_dims": dense_summary,
    }
    try:
        with open(output_json, "w") as f:
            json.dump(sidecar, f, indent=2)
        log.info("Saved feature stats to %s", output_json)
    except Exception as e:
        log.warning("Failed to save JSON: %s", e)

    # ═══════════════════════════════════════════════════════════
    # REPORT
    # ═══════════════════════════════════════════════════════════

    print("=" * 72)
    print("FEATURE EXPLORATION REPORT — TRAIN DATA")
    print("=" * 72)
    print(f"Rows: {total_rows:,}  |  user_int fids: {n_user_int}  |  item_int fids: {n_item_int}  |  dense dims: {n_dense}")
    print(f"Scan time: {scan_time:.1f}s")
    print()

    # M1: Schema overview
    print("─ M1: INT FEATURE SCHEMA ─")
    print(f"{'FID':>5} {'Type':>12} {'Vocab':>8} {'Dim':>5}")
    print("-" * 35)
    for fid, offset, length in user_int_entries:
        vs = max(user_int_vocab[offset:offset + length])
        print(f"{fid:>5} {'user_int':>12} {vs:>8} {length:>5}")
    for fid, offset, length in item_int_entries:
        vs = max(item_int_vocab[offset:offset + length])
        print(f"{fid:>5} {'item_int':>12} {vs:>8} {length:>5}")
    print(f"\nDense features: {user_dense_dim} dims from {len(user_dense_entries)} fids")
    fid_list = [str(fid) for fid, _, _ in user_dense_entries]
    print(f"  fids: {', '.join(fid_list)}")
    print()

    # M2: Int feature AUC ranking
    print("─ M2: INT FEATURE AUC (single feature -> label, streaming binned) ─")
    n_show = min(len(auc_results), 60)
    print(f"{'Rank':>4} {'Key':>24} {'AUC':>8} {'NZ%':>7} {'Mean':>10} {'Std':>10}")
    print("-" * 70)
    for i, (key, auc_val) in enumerate(auc_results[:n_show]):
        info = int_summary.get(key, {})
        print(f"{i+1:>4} {key:>24} {auc_val:>8.4f} {info.get('nz_rate', 0):>7.1%} "
              f"{info.get('mean', 0):>10.1f} {info.get('std', 0):>10.1f}")
    if len(auc_results) > n_show:
        print(f"  ... ({len(auc_results) - n_show} more with AUC ≤ {auc_results[n_show-1][1]:.4f})")
    print()

    # M3: Feature classification
    print("─ M3: FEATURE CLASSIFICATION ─")
    strong = [(k, v) for k, v in auc_results if v > 0.53]
    medium = [(k, v) for k, v in auc_results if 0.51 < v <= 0.53]
    weak = [(k, v) for k, v in auc_results if v <= 0.51]
    print(f"S-tier (AUC > 0.53):  {len(strong)} features")
    for k, v in strong:
        info = int_summary.get(k, {})
        print(f"  {k:>24s}  AUC={v:.4f}  nz={info.get('nz_rate', 0):.1%}")
    if not strong:
        print("  (none)")
    print(f"A-tier (AUC 0.51-0.53): {len(medium)} features — low signal, may need depth")
    for k, v in medium[:10]:
        info = int_summary.get(k, {})
        print(f"  {k:>24s}  AUC={v:.4f}  nz={info.get('nz_rate', 0):.1%}")
    if len(medium) > 10:
        print(f"  ... and {len(medium) - 10} more")
    print(f"B-tier (AUC ≤ 0.51): {len(weak)} features — noise candidates")
    print()

    # M4: Dense dimension stats (top by non-zero rate)
    print("─ M4: DENSE DIM STATS (top 30 by non-zero rate) ─")
    dense_sorted = sorted(dense_summary, key=lambda x: -x["nz_rate"])[:30]
    print(f"{'Rank':>4} {'Dim':>6} {'NZ%':>8} {'Mean':>14} {'Std':>14}")
    print("-" * 55)
    for i, d in enumerate(dense_sorted):
        print(f"{i+1:>4} {d['dim']:>6} {d['nz_rate']:>7.1%} {d['mean']:>14.6f} {d['std']:>14.6f}")
    print()

    # M5: I2 group spotlight
    print("─ M5: I2 ITEM GROUP SPOTLIGHT (fids 5,6,7,8,12) ─")
    i2_fids = ["5", "6", "7", "8", "12"]
    for key in int_summary:
        fid = str(int_summary[key]["fid"])
        if fid in i2_fids and "item_int" in key:
            info = int_summary[key]
            print(f"  fid={fid}: AUC={info['auc']:.4f}  nz={info['nz_rate']:.1%}  "
                  f"mean={info['mean']:.1f}  std={info['std']:.1f}")
    print()

    # Summary
    print("─ SUMMARY ─")
    print(f"Total features with AUC > 0.5: {sum(1 for _, v in auc_results if v > 0.5)}/{len(auc_results)}")
    print(f"Max single-feature AUC: {auc_results[0][1]:.4f} ({auc_results[0][0]})" if auc_results else "N/A")
    print(f"Median AUC: {auc_results[len(auc_results)//2][1]:.4f}" if auc_results else "N/A")
    print(f"Output JSON: {output_json}")
    print("=" * 72)

    total_time = time.time() - t0
    log.info("Total time: %.1fs", total_time)


if __name__ == "__main__":
    main()
