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

    model_dir = os.environ.get('MODEL_OUTPUT_PATH')
    data_dir = os.environ.get('EVAL_DATA_PATH')
    if not data_dir:
        logging.error("EVAL_DATA_PATH must be set.")
        return

    schema_path = os.path.join(model_dir, 'schema.json') if model_dir else None
    if not schema_path or not os.path.exists(schema_path):
        schema_path = os.path.join(data_dir, 'schema.json')
    logging.info("Using schema: %s", schema_path)

    # Read seq_max_lens from train_config if available
    train_config_path = os.path.join(model_dir, 'train_config.json') if model_dir else None
    if not train_config_path or not os.path.exists(train_config_path):
        train_config_path = os.path.join(data_dir, 'train_config.json')
    if not os.path.exists(train_config_path):
        train_config_path = None
    sml_str = _FALLBACK_SEQ_MAX_LENS
    batch_size = _FALLBACK_BATCH_SIZE
    if train_config_path and os.path.exists(train_config_path):
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
    # REPORT
    # ══════════════════════════════════════════════════════════════
    print("=" * 72)
    print("FEATURE DISTRIBUTION REPORT — TEST DATA")
    print("=" * 72)
    print(f"Rows: {test_rows:,}  user_int: {len(user_int_entries)}  item_int: {len(item_int_entries)}  dense: {n_dense}")
    print()

    # ── Helper ──
    def _pack(items, per_line=5):
        for i in range(0, len(items), per_line):
            yield "  ".join(str(x) for x in items[i:i + per_line])

    # M0: Int feature distribution records (for offline diff with train M0)
    print("─ M0: INT FEATURE DISTRIBUTION (5 per line, copy to diff) ─")
    print("COL: FEATURE_TYPE FID NZ_RATE MEAN STD | ...")
    feats = []
    for key in sorted(int_dists.keys()):
        d = int_dists[key]
        parts = key.split("_", 2)
        ftype = f"{parts[0]}_{parts[1]}"
        fid = parts[2] if len(parts) > 2 else "?"
        feats.append(f"{ftype} {fid} {d.nz_rate:.4f} {d.mean:.2f} {d.std:.2f}")
    for line in _pack(feats, 5):
        print(line)
    print()

    # M1: Schema summary
    print("─ M1: SCHEMA SUMMARY ─")
    nz_above_50 = sum(1 for d in int_dists.values() if d.nz_rate > 0.5)
    nz_above_10 = sum(1 for d in int_dists.values() if d.nz_rate > 0.1)
    nz_below_1 = sum(1 for d in int_dists.values() if d.nz_rate < 0.01)
    print(f"  user_int: {len(user_int_entries)} fids  item_int: {len(item_int_entries)} fids  nz>50%: {nz_above_50}  nz>10%: {nz_above_10}  nz<1%: {nz_below_1}")
    print(f"  dense: {n_dense} dims from fids {', '.join(str(f) for f, _, _ in user_dense_entries)}")

    # M2: I2 item group focus
    print("─ M2: I2 GROUP (item fids 5,6,7,8,12) ─")
    i2_parts = []
    for fid_s in ["5", "6", "7", "8", "12"]:
        key = f"item_int_{fid_s}"
        d = int_dists.get(key)
        if d is not None:
            i2_parts.append(f"{fid_s}:nz={d.nz_rate:.1%}|m={d.mean:.1f}|s={d.std:.1f}")
        else:
            i2_parts.append(f"{fid_s}:MISSING")
    print(f"  {'  '.join(i2_parts)}")
    print()

    # M3: Dense dimension stats (packed)
    if n_dense > 0:
        print("─ M3: DENSE DIM STATS ─")
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
        items = [f"d{d}({fid}){nz:.1%}|{mean:.4f}" for d, fid, nz, mean, _ in dense_sorted]
        print(f"Top 40 by nz: {'  '.join(items[:20])}")
        if len(items) > 20:
            print(f"  {'  '.join(items[20:])}")
        dense_by_mean = sorted(dense_info, key=lambda x: -abs(x[3]))[:20]
        items2 = [f"d{d}({fid}){mean:.6f}|{nz:.1%}" for d, fid, nz, mean, _ in dense_by_mean]
        print(f"Top 20 by |mean|: {'  '.join(items2)}")
        print()

    # Summary
    print("─ SUMMARY ─")
    zero_nz = sum(1 for d in int_dists.values() if d.nz_rate < 0.001)
    print(f"Features nz<0.1%: {zero_nz}/{len(int_dists)}  I2 fids present: {sum(1 for fid in ['5','6','7','8','12'] if f'item_int_{fid}' in int_dists)}/5")
    print("=" * 72)


if __name__ == "__main__":
    main()
