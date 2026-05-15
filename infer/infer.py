"""feature_audit: test-side standalone feature distribution analysis.

No model loading, no inference, no predictions. Pure data exploration:
streams through test parquet and prints per-feature distribution stats.

Environment variables:
    EVAL_DATA_PATH     Test data directory (*.parquet + schema.json).
"""

import os
import json
import logging
from typing import Dict


logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
)

_FALLBACK_SEQ_MAX_LENS = 'seq_a:256,seq_b:256,seq_c:512,seq_d:512'
_FALLBACK_BATCH_SIZE = 256


def _parse_seq_max_lens(sml_str: str) -> Dict[str, int]:
    seq_max_lens = {}
    for pair in sml_str.split(','):
        k, v = pair.split(':')
        seq_max_lens[k.strip()] = int(v.strip())
    return seq_max_lens


def main():
    import numpy as np
    from torch.utils.data import DataLoader
    from dataset import PCVRParquetDataset

    data_dir = os.environ.get('EVAL_DATA_PATH')
    if not data_dir:
        logging.error("EVAL_DATA_PATH must be set.")
        return

    schema_path = os.path.join(data_dir, 'schema.json')
    logging.info("Using schema: %s", schema_path)

    # Read seq_max_lens from train_config if available (next to schema)
    train_config_path = os.path.join(data_dir, 'train_config.json')
    sml_str = _FALLBACK_SEQ_MAX_LENS
    batch_size = _FALLBACK_BATCH_SIZE
    if os.path.exists(train_config_path):
        try:
            with open(train_config_path) as f:
                tc = json.load(f)
            sml_str = tc.get('seq_max_lens', sml_str)
            batch_size = int(tc.get('batch_size', batch_size))
        except Exception:
            pass

    seq_max_lens = _parse_seq_max_lens(sml_str)
    logging.info("seq_max_lens: %s, batch_size: %d", seq_max_lens, batch_size)

    # Load test dataset (no labels)
    dataset = PCVRParquetDataset(
        parquet_path=data_dir,
        schema_path=schema_path,
        batch_size=batch_size,
        seq_max_lens=seq_max_lens,
        shuffle=False,
        buffer_batches=0,
        is_training=False,
    )
    loader = DataLoader(dataset, batch_size=None,
                        num_workers=min(4, os.cpu_count() or 1))

    user_int_entries = dataset.user_int_schema.entries
    item_int_entries = dataset.item_int_schema.entries
    user_dense_entries = dataset.user_dense_schema.entries
    n_dense = dataset.user_dense_schema.total_dim
    test_rows = dataset.num_rows
    logging.info("Test data: %d rows, %d user_int, %d item_int, %d dense dims",
                 test_rows, len(user_int_entries), len(item_int_entries), n_dense)

    # Accumulators
    class DistAcc:
        __slots__ = ("non_zero", "total", "sum_v", "sum_v2")
        def __init__(self):
            self.non_zero = 0
            self.total = 0
            self.sum_v = 0.0
            self.sum_v2 = 0.0
        def update(self, vals):
            v = vals.astype(np.float64)
            self.total += v.size
            self.non_zero += int((vals != 0).sum())
            s = v.sum()
            self.sum_v += float(s)
            self.sum_v2 += float((v * v).sum())
        @property
        def nz_rate(self):
            return self.non_zero / self.total if self.total > 0 else 0.0
        @property
        def mean(self):
            return self.sum_v / self.total if self.total > 0 else 0.0
        @property
        def std(self):
            if self.total < 2:
                return 0.0
            var = (self.sum_v2 - self.sum_v * self.sum_v / self.total) / (self.total - 1)
            return float(np.sqrt(max(var, 0.0)))

    int_dists = {}
    dense_dists = [DistAcc() for _ in range(n_dense)]
    for fid, offset, length in user_int_entries:
        int_dists[f"user_int_{fid}"] = DistAcc()
    for fid, offset, length in item_int_entries:
        int_dists[f"item_int_{fid}"] = DistAcc()

    # Scan
    logging.info("Scanning test data...")
    batch_count = 0
    for batch in loader:
        user_int = batch["user_int_feats"].numpy()
        item_int = batch["item_int_feats"].numpy()
        user_dense = batch["user_dense_feats"].numpy()
        for fid, offset, length in user_int_entries:
            int_dists[f"user_int_{fid}"].update(user_int[:, offset:offset + length])
        for fid, offset, length in item_int_entries:
            int_dists[f"item_int_{fid}"].update(item_int[:, offset:offset + length])
        for d in range(n_dense):
            dense_dists[d].update(user_dense[:, d])
        batch_count += 1
        if batch_count % 500 == 0:
            logging.info("  %d batches...", batch_count)
    logging.info("Scan done: %d batches", batch_count)

    # ══════════════════════════════════════════════════════════════
    REPORT
    # ══════════════════════════════════════════════════════════════
    print("=" * 72)
    print("FEATURE DISTRIBUTION REPORT — TEST DATA")
    print("=" * 72)
    print(f"Rows: {test_rows:,}  user_int: {len(user_int_entries)}  item_int: {len(item_int_entries)}  dense: {n_dense}")
    print()

    # M0: Raw data dump — same format as train M0 for offline diff
    print("─ M0: RAW FEATURE RECORD (copy to diff with train M0 output) ─")
    print("FEATURE_TYPE FID AUC NZ_RATE MEAN STD")
    for key in sorted(int_dists.keys()):
        d = int_dists[key]
        parts = key.split("_", 2)
        ftype = f"{parts[0]}_{parts[1]}"
        fid = parts[2] if len(parts) > 2 else "?"
        print(f"FEAT {ftype} {fid} _ {d.nz_rate:.4f} {d.mean:.2f} {d.std:.2f}")
    offset = 0
    for _fid, _, length in user_dense_entries:
        for l in range(length):
            d_idx = offset + l
            if d_idx < len(dense_dists):
                dd = dense_dists[d_idx]
                if dd.total > 0:
                    print(f"DENSE user_dense {_fid} d{d_idx} _ {dd.nz_rate:.4f} {dd.mean:.6f} {dd.std:.6f}")
        offset += length
    print()

    # M1: Schema overview
    print("─ M1: SCHEMA OVERVIEW ─")
    print(f"{'FID':>5} {'Type':>12} {'Dim':>5} {'NZ%':>7} {'Mean':>14} {'Std':>14}")
    print("-" * 65)
    for fid, offset, length in user_int_entries:
        d = int_dists[f"user_int_{fid}"]
        print(f"{fid:>5} {'user_int':>12} {length:>5} {d.nz_rate:>7.1%} {d.mean:>14.2f} {d.std:>14.2f}")
    for fid, offset, length in item_int_entries:
        d = int_dists[f"item_int_{fid}"]
        print(f"{fid:>5} {'item_int':>12} {length:>5} {d.nz_rate:>7.1%} {d.mean:>14.2f} {d.std:>14.2f}")
    print(f"Dense: {n_dense} dims, fids: {', '.join(str(f) for f, _, _ in user_dense_entries)}")
    print()

    # M2: I2 item group focus
    print("─ M2: I2 GROUP (item fids 5,6,7,8,12) ─")
    for fid_s in ["5", "6", "7", "8", "12"]:
        key = f"item_int_{fid_s}"
        d = int_dists.get(key)
        if d is not None:
            print(f"  fid={fid_s}: nz={d.nz_rate:.1%} mean={d.mean:.1f} std={d.std:.1f}")
        else:
            print(f"  fid={fid_s}: NOT IN TEST SCHEMA")
    print()

    # M3: Dense dimension stats
    if n_dense > 0:
        print("─ M3: DENSE DIM STATS (top 40 by non-zero rate) ─")
        dense_info = []
        for d in range(n_dense):
            dd = dense_dists[d]
            if dd.total > 0:
                fid = "?"
                off = 0
                for _fid, _, length in user_dense_entries:
                    if off <= d < off + length:
                        fid = str(_fid)
                        break
                    off += length
                dense_info.append((d, fid, dd.nz_rate, dd.mean, dd.std))
        dense_sorted = sorted(dense_info, key=lambda x: -x[2])[:40]
        print(f"{'Rank':>4} {'Dim':>6} {'FID':>5} {'NZ%':>8} {'Mean':>14} {'Std':>14}")
        print("-" * 60)
        for i, (d, fid, nz, mean, std) in enumerate(dense_sorted):
            print(f"{i+1:>4} {d:>6} {fid:>5} {nz:>7.1%} {mean:>14.6f} {std:>14.6f}")
        # Top by |mean|
        print(f"\nTop 20 dims by |mean|:")
        dense_by_mean = sorted(dense_info, key=lambda x: -abs(x[3]))[:20]
        for d, fid, nz, mean, std in dense_by_mean:
            print(f"  dim={d:>6} fid={fid:>4} mean={mean:>14.6f} std={std:>14.6f} nz={nz:.1%}")
        print()

    # Summary
    print("─ SUMMARY ─")
    zero_nz = sum(1 for d in int_dists.values() if d.nz_rate < 0.001)
    print(f"Features with nz_rate < 0.1%: {zero_nz}/{len(int_dists)}")
    print(f"I2 fids present: {sum(1 for fid in ['5','6','7','8','12'] if f'item_int_{fid}' in int_dists)}/5")
    print("=" * 72)


if __name__ == "__main__":
    main()
