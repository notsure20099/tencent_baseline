"""
Feature Exploration — full data, streaming, single pass.

Runs on TRAIN data. Computes per-feature AUC and distribution stats.
All results are printed (no file I/O).
"""

import os
import sys
import time
import logging
import numpy as np
from typing import Dict, List
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from dataset import PCVRParquetDataset

logging.basicConfig(level=logging.INFO, format="%(message)s")
log = logging.getLogger("explore_all")


class BinnedAUC:
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


def main():
    t0 = time.time()
    data_dir = os.environ.get("TRAIN_DATA_PATH", "")
    if not data_dir:
        log.error("TRAIN_DATA_PATH env var is not set. Abort.")
        return
    schema_path = os.path.join(data_dir, "schema.json")
    if not os.path.exists(schema_path):
        log.error("schema.json not found at %s", schema_path)
        return

    sml_str = os.environ.get("SEQ_MAX_LENS", "seq_a:256,seq_b:256,seq_c:512,seq_d:512")
    seq_max_lens = {}
    for pair in sml_str.split(","):
        k, v = pair.split(":")
        seq_max_lens[k.strip()] = int(v.strip())

    batch_size = 256
    num_workers = int(os.environ.get("EXPLORE_NUM_WORKERS", "4"))

    log.info("Loading dataset (full, streaming)...")
    dataset = PCVRParquetDataset(
        parquet_path=data_dir, schema_path=schema_path,
        batch_size=batch_size, seq_max_lens=seq_max_lens,
        shuffle=False, buffer_batches=0, is_training=True,
    )
    loader = DataLoader(dataset, batch_size=None, num_workers=num_workers)

    user_int_entries = dataset.user_int_schema.entries
    item_int_entries = dataset.item_int_schema.entries
    user_dense_entries = dataset.user_dense_schema.entries
    user_int_vocab = dataset.user_int_vocab_sizes
    item_int_vocab = dataset.item_int_vocab_sizes
    user_dense_dim = dataset.user_dense_schema.total_dim

    n_user = len(user_int_entries)
    n_item = len(item_int_entries)
    n_dense = user_dense_dim
    total_rows = dataset.num_rows

    log.info("Data: %d rows, %d user_int, %d item_int, %d dense dims",
             total_rows, n_user, n_item, n_dense)

    int_aucs: Dict[str, BinnedAUC] = {}
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
            int_dists[key].update(vals.reshape(-1))

        for fid, offset, length in item_int_entries:
            key = f"item_int_{fid}"
            vals = item_int[:, offset:offset + length].astype(np.int64)
            if length == 1:
                vals = vals.ravel()
                int_aucs[key].update(vals, labels)
            int_dists[key].update(vals.reshape(-1))

        for d in range(n_dense):
            dense_dists[d].update(user_dense[:, d])

        batch_count += 1
        if batch_count % 500 == 0:
            log.info("  %d batches, %d rows...", batch_count, batch_count * batch_size)

    scan_time = time.time() - t0
    log.info("Scan done: %d batches in %.1fs", batch_count, scan_time)

    # ═══════════════════════════════════════════════════════════
    REPORT
    # ═══════════════════════════════════════════════════════════

    auc_results = []
    for key, auc_acc in int_aucs.items():
        val = auc_acc.auc()
        if val > 0.4:
            auc_results.append((key, val))
    auc_results.sort(key=lambda x: -x[1])

    # Build summary
    int_summary = {}
    for key, dist in int_dists.items():
        auc_val = int_aucs[key].auc() if key in int_aucs else 0.5
        parts = key.split("_", 2)
        fid = parts[2] if len(parts) > 2 else "?"
        int_summary[key] = {
            "fid": fid, "type": f"{parts[0]}_{parts[1]}",
            "auc": round(auc_val, 4),
            "nz_rate": round(dist.nz_rate, 4),
            "mean": round(dist.mean, 2), "std": round(dist.std, 2),
        }

    dense_summary = []
    for d in range(n_dense):
        dist = dense_dists[d]
        if dist.total > 0:
            dense_summary.append({
                "dim": d,
                "nz_rate": round(dist.nz_rate, 4),
                "mean": round(dist.mean, 6),
                "std": round(dist.std, 6),
            })

    print("=" * 72)
    print("FEATURE EXPLORATION REPORT — TRAIN DATA")
    print("=" * 72)
    print(f"Rows: {total_rows:,}  user_int: {n_user}  item_int: {n_item}  dense: {n_dense}")
    print(f"Scan time: {scan_time:.1f}s")
    print()

    # M0: Raw data dump — all features with full stats (for offline comparison)
    print("─ M0: RAW FEATURE RECORD (copy to compare with test output) ─")
    print("FEATURE_TYPE FID AUC NZ_RATE MEAN STD")
    for key in sorted(int_summary.keys()):
        s = int_summary[key]
        print(f"FEAT {s['type']} {s['fid']} {s['auc']:.4f} {s['nz_rate']:.4f} {s['mean']:.2f} {s['std']:.2f}")
    for d in dense_summary:
        # find which fid this dim belongs to
        fid = "?"
        offset = 0
        for _fid, _, length in user_dense_entries:
            if offset <= d["dim"] < offset + length:
                fid = str(_fid)
                break
            offset += length
        print(f"DENSE user_dense {fid} d{d['dim']} _ {d['nz_rate']:.4f} {d['mean']:.6f} {d['std']:.6f}")
    print()

    # M1: Schema
    print("─ M1: INT FEATURE SCHEMA ─")
    print(f"{'FID':>5} {'Type':>12} {'Vocab':>8} {'Dim':>5}")
    print("-" * 35)
    for fid, offset, length in user_int_entries:
        vs = max(user_int_vocab[offset:offset + length])
        print(f"{fid:>5} {'user_int':>12} {vs:>8} {length:>5}")
    for fid, offset, length in item_int_entries:
        vs = max(item_int_vocab[offset:offset + length])
        print(f"{fid:>5} {'item_int':>12} {vs:>8} {length:>5}")
    print(f"Dense: {n_dense} dims, fids: {', '.join(str(f) for f, _, _ in user_dense_entries)}")
    print()

    # M2: AUC ranking
    print("─ M2: INT FEATURE AUC ─")
    n_show = min(len(auc_results), 70)
    print(f"{'Rank':>4} {'Key':>24} {'AUC':>8} {'NZ%':>7} {'Mean':>10} {'Std':>10}")
    print("-" * 70)
    for i, (key, auc_val) in enumerate(auc_results[:n_show]):
        info = int_summary.get(key, {})
        print(f"{i+1:>4} {key:>24} {auc_val:>8.4f} {info.get('nz_rate', 0):>7.1%} "
              f"{info.get('mean', 0):>10.1f} {info.get('std', 0):>10.1f}")
    print()

    # M3: Classification
    print("─ M3: FEATURE CLASSIFICATION ─")
    strong = [(k, v) for k, v in auc_results if v > 0.53]
    medium = [(k, v) for k, v in auc_results if 0.51 < v <= 0.53]
    weak = [(k, v) for k, v in auc_results if v <= 0.51]
    print(f"S (AUC>0.53): {len(strong)}")
    for k, v in strong:
        print(f"  {k} AUC={v:.4f} nz={int_summary.get(k, {}).get('nz_rate', 0):.1%}")
    if not strong:
        print("  (none)")
    print(f"A (AUC 0.51-0.53): {len(medium)}")
    for k, v in medium[:10]:
        print(f"  {k} AUC={v:.4f}")
    if len(medium) > 10:
        print(f"  ... and {len(medium) - 10} more")
    print(f"B (AUC<=0.51): {len(weak)}")
    print()

    # M4: Dense dims
    print("─ M4: DENSE DIM STATS (top 40 by nz_rate) ─")
    dense_sorted = sorted(dense_summary, key=lambda x: -x["nz_rate"])[:40]
    print(f"{'Rank':>4} {'Dim':>6} {'NZ%':>8} {'Mean':>14} {'Std':>14}")
    print("-" * 55)
    for i, d in enumerate(dense_sorted):
        print(f"{i+1:>4} {d['dim']:>6} {d['nz_rate']:>7.1%} {d['mean']:>14.6f} {d['std']:>14.6f}")
    # Also print dims by mean value (catch large-value outliers)
    print(f"\nTop 20 dims by |mean|:")
    dense_by_mean = sorted(dense_summary, key=lambda x: -abs(x["mean"]))[:20]
    for i, d in enumerate(dense_by_mean):
        print(f"  dim={d['dim']:>6}  mean={d['mean']:>14.6f}  std={d['std']:>14.6f}  nz={d['nz_rate']:.1%}")
    print()

    # M5: I2 spotlight
    print("─ M5: I2 SPOTLIGHT (item fids 5,6,7,8,12) ─")
    for fid_s in ["5", "6", "7", "8", "12"]:
        key = f"item_int_{fid_s}"
        s = int_summary.get(key, {})
        print(f"  fid={fid_s}: AUC={s.get('auc', '?'):>7s} nz={s.get('nz_rate', 0):.1%} "
              f"mean={s.get('mean', 0):.1f} std={s.get('std', 0):.1f}")
    print()

    # Summary
    print("─ SUMMARY ─")
    print(f"Features with AUC>0.5: {sum(1 for _,v in auc_results if v>0.5)}/{len(auc_results)}")
    if auc_results:
        print(f"Max AUC: {auc_results[0][1]:.4f} ({auc_results[0][0]})")
        print(f"Median AUC: {auc_results[len(auc_results)//2][1]:.4f}")
    print("=" * 72)

    total_time = time.time() - t0
    log.info("Total time: %.1fs", total_time)


if __name__ == "__main__":
    main()
