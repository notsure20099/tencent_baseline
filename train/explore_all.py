"""
Feature Exploration — full data, streaming, single pass.

Runs on TRAIN data. Computes per-feature AUC and distribution stats.
All results are printed (no file I/O).
"""

import argparse
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
    parser = argparse.ArgumentParser(description="Train-side feature exploration")
    parser.add_argument("--data_dir", type=str, default=os.environ.get("TRAIN_DATA_PATH", ""),
                        help="Training data directory")
    parser.add_argument("--max_batches", type=int, default=0,
                        help="Limit batches (0=full). Use 1 for quick test.")
    args, _ = parser.parse_known_args()

    if not args.data_dir:
        log.error("--data_dir not set and TRAIN_DATA_PATH env var is not set. Abort.")
        return

    t0 = time.time()
    data_dir = args.data_dir
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
        if args.max_batches > 0 and batch_count >= args.max_batches:
            break

    scan_time = time.time() - t0
    log.info("Scan done: %d batches in %.1fs", batch_count, scan_time)

    # ═══════════════════════════════════════════════════════════
    # REPORT
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

    # ── Helper ──
    def _pack(items, per_line=5):
        for i in range(0, len(items), per_line):
            yield "  ".join(str(x) for x in items[i:i + per_line])

    print("=" * 72)
    print("FEATURE EXPLORATION REPORT — TRAIN DATA")
    print("=" * 72)
    print(f"Rows: {total_rows:,}  user_int: {n_user}  item_int: {n_item}  dense: {n_dense}  scan: {scan_time:.1f}s")
    print()

    print("─ M0: INT FEATURE DISTRIBUTION + AUC (5 per line, copy to diff) ─")
    print("COL: FEATURE_TYPE FID AUC NZ_RATE MEAN STD | ...")
    feats = [f"{s['type']} {s['fid']} {s['auc']:.4f} {s['nz_rate']:.4f} {s['mean']:.2f} {s['std']:.2f}"
             for key in sorted(int_summary.keys()) for s in [int_summary[key]]]
    for line in _pack(feats, 5):
        print(line)
    print()

    print("─ M1: SCHEMA SUMMARY ─")
    nz_above_50 = sum(1 for s in int_summary.values() if s['nz_rate'] > 0.5)
    nz_above_10 = sum(1 for s in int_summary.values() if s['nz_rate'] > 0.1)
    nz_below_1 = sum(1 for s in int_summary.values() if s['nz_rate'] < 0.01)
    print(f"  user_int: {n_user} fids  item_int: {n_item} fids  nz>50%: {nz_above_50}  nz>10%: {nz_above_10}  nz<1%: {nz_below_1}")
    print(f"  dense: {n_dense} dims from fids {', '.join(str(f) for f, _, _ in user_dense_entries)}")
    print()

    print("─ M2: INT FEATURE AUC (key auc|nz%, 5 per line) ─")
    auc_items = [f"{k} {v:.4f}|{int_summary.get(k, {}).get('nz_rate', 0):.1%}"
                 for k, v in auc_results]
    for line in _pack(auc_items, 5):
        print(line)
    print()

    print("─ M3: FEATURE CLASSIFICATION ─")
    strong = [(k, v) for k, v in auc_results if v > 0.53]
    medium = [(k, v) for k, v in auc_results if 0.51 < v <= 0.53]
    print(f"S (AUC>0.53): {len(strong)} {', '.join(k for k,_ in strong) if strong else '(none)'}")
    print(f"A (AUC 0.51-0.53): {len(medium)} {', '.join(k for k,_ in medium) if medium else '(none)'}")
    print(f"B (AUC<=0.51): {len(weak)}")
    print()

    print("─ M4: DENSE DIM STATS ─")
    dense_sorted = sorted(dense_summary, key=lambda x: -x["nz_rate"])[:40]
    items = [f"d{d['dim']}:{d['nz_rate']:.1%}|{d['mean']:.4f}|{d['std']:.4f}"
             for d in dense_sorted]
    print(f"Top 40 by nz_rate: {'  '.join(items[:20])}")
    if len(items) > 20:
        print(f"  {'  '.join(items[20:])}")
    dense_by_mean = sorted(dense_summary, key=lambda x: -abs(x["mean"]))[:20]
    items2 = [f"d{d['dim']}:{d['mean']:.6f}|{d['nz_rate']:.1%}" for d in dense_by_mean]
    print(f"Top 20 by |mean|: {'  '.join(items2)}")
    print()

    print("─ M5: I2 SPOTLIGHT (item fids 5,6,7,8,12) ─")
    i2_parts = []
    for fid_s in ["5", "6", "7", "8", "12"]:
        s = int_summary.get(f"item_int_{fid_s}", {})
        auc_v = s.get('auc', 0)
        i2_parts.append(f"{fid_s}:AUC={auc_v:.4f}|nz={s.get('nz_rate', 0):.1%}|m={s.get('mean', 0):.1f}|s={s.get('std', 0):.1f}")
    print(f"  {'  '.join(i2_parts)}")
    print()

    print("─ SUMMARY ─")
    if auc_results:
        print(f"AUC>0.5: {sum(1 for _,v in auc_results if v>0.5)}/{len(auc_results)}  Max: {auc_results[0][1]:.4f}({auc_results[0][0]})  Median: {auc_results[len(auc_results)//2][1]:.4f}")
    print("=" * 72)

    total_time = time.time() - t0
    log.info("Total time: %.1fs", total_time)


if __name__ == "__main__":
    main()
