"""feature_audit: train↔test feature distribution comparison.

No model loading, no inference. Pure data exploration: reads train-side
``feature_stats.json``, streams through test parquet, and prints a
schema diff + distribution drift report.

Environment variables:
    MODEL_OUTPUT_PATH  Checkpoint directory (contains schema.json + feature_stats.json).
    EVAL_DATA_PATH     Test data directory (*.parquet + schema.json).
"""

import os
import json
import logging
from typing import Any, Dict


logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
)

_FALLBACK_SEQ_MAX_LENS = 'seq_a:256,seq_b:256,seq_c:512,seq_d:512'
_FALLBACK_BATCH_SIZE = 256


def _parse_seq_max_lens(sml_str: str) -> Dict[str, int]:
    seq_max_lens: Dict[str, int] = {}
    for pair in sml_str.split(','):
        k, v = pair.split(':')
        seq_max_lens[k.strip()] = int(v.strip())
    return seq_max_lens


def load_train_config(model_dir: str) -> Dict[str, Any]:
    train_config_path = os.path.join(model_dir, 'train_config.json')
    if os.path.exists(train_config_path):
        with open(train_config_path, 'r') as f:
            cfg = json.load(f)
        logging.info(f"Loaded train_config from {train_config_path}")
        return cfg
    logging.info(f"train_config.json not found in {model_dir}, using fallback defaults")
    return {}


def _run_data_diagnose(model_dir, data_dir, schema_path, seq_max_lens, batch_size):
    """DIAGNOSE_DATA: train↔test feature distribution comparison (label-free).

    Loads train-side ``feature_stats.json`` from the checkpoint directory,
    computes per-feature distributions on the test set, and prints a
    comparison report (schema diff, distribution drift, risk assessment).
    """
    import numpy as np
    from torch.utils.data import DataLoader
    from dataset import PCVRParquetDataset

    logging.info("=" * 60)
    logging.info("DIAGNOSE_DATA: TRAIN↔TEST FEATURE DISTRIBUTION COMPARISON")
    logging.info("=" * 60)

    # 1. Load train stats
    train_json = os.path.join(model_dir, "feature_stats.json")
    if not os.path.exists(train_json):
        train_json = os.path.join(os.path.dirname(data_dir), "feature_stats.json")
    if not os.path.exists(train_json):
        logging.error("feature_stats.json not found in model_dir or parent of data_dir. "
                       "Run explore_all.py on train first.")
        return
    with open(train_json) as f:
        train_stats = json.load(f)
    logging.info("Loaded train stats: %d rows, %d user_int fids, %d item_int fids",
                 train_stats.get("total_rows", 0),
                 train_stats.get("n_user_int", 0),
                 train_stats.get("n_item_int", 0))

    # 2. Load test dataset (no labels)
    test_dataset = PCVRParquetDataset(
        parquet_path=data_dir,
        schema_path=schema_path,
        batch_size=batch_size,
        seq_max_lens=seq_max_lens,
        shuffle=False,
        buffer_batches=0,
        is_training=False,
    )
    test_loader = DataLoader(test_dataset, batch_size=None,
                             num_workers=min(4, os.cpu_count() or 1))
    test_rows = test_dataset.num_rows
    logging.info("Test data: %d rows", test_rows)

    # 3. Prepare test-side accumulators matching train's feature layout
    user_int_entries = test_dataset.user_int_schema.entries
    item_int_entries = test_dataset.item_int_schema.entries
    n_dense = test_dataset.user_dense_schema.total_dim

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

    test_int_dists = {}
    test_dense_dists = [DistAcc() for _ in range(n_dense)]

    for fid, offset, length in user_int_entries:
        test_int_dists[f"user_int_{fid}"] = DistAcc()
    for fid, offset, length in item_int_entries:
        test_int_dists[f"item_int_{fid}"] = DistAcc()

    # 4. Scan test data
    logging.info("Scanning test data...")
    batch_count = 0
    for batch in test_loader:
        user_int = batch["user_int_feats"].numpy()
        item_int = batch["item_int_feats"].numpy()
        user_dense = batch["user_dense_feats"].numpy()

        for fid, offset, length in user_int_entries:
            vals = user_int[:, offset:offset + length]
            test_int_dists[f"user_int_{fid}"].update(vals)
        for fid, offset, length in item_int_entries:
            vals = item_int[:, offset:offset + length]
            test_int_dists[f"item_int_{fid}"].update(vals)
        for d in range(n_dense):
            test_dense_dists[d].update(user_dense[:, d])

        batch_count += 1
        if batch_count % 500 == 0:
            logging.info("  %d batches...", batch_count)

    logging.info("Scan done: %d batches", batch_count)

    # 5. Compare with train stats
    train_int = train_stats.get("int_features", {})

    # Build drift report for int features
    drift_report = []
    for key, tdist in test_int_dists.items():
        if key not in train_int:
            drift_report.append((key, -1, -1, -1, "MISSING_IN_TRAIN"))
            continue
        ti = train_int[key]
        t_nz = ti.get("nz_rate", 0)
        t_mean = ti.get("mean", 0)
        tt_nz = tdist.nz_rate
        tt_mean = tdist.mean
        nz_drift = abs(tt_nz - t_nz)
        mean_norm = max(abs(t_mean), 1e-6)
        mean_drift = abs(tt_mean - t_mean) / mean_norm if mean_norm > 0 else 0
        if nz_drift > 0.1 or mean_drift > 0.5:
            status = "HIGH_DRIFT"
        elif nz_drift > 0.03 or mean_drift > 0.1:
            status = "MEDIUM_DRIFT"
        else:
            status = "OK"
        drift_report.append((key, nz_drift, mean_drift, ti.get("auc", 0), status))

    drift_report.sort(key=lambda x: -(x[1] + x[2]))

    # Schema diff
    train_fids_user = set(str(fid) for fid in train_stats.get("user_int_fids", []))
    train_fids_item = set(str(fid) for fid in train_stats.get("item_int_fids", []))
    test_fids_user = set(str(fid) for fid, _, _ in user_int_entries)
    test_fids_item = set(str(fid) for fid, _, _ in item_int_entries)

    missing_in_test_u = train_fids_user - test_fids_user
    missing_in_test_i = train_fids_item - test_fids_item
    missing_in_train_u = test_fids_user - train_fids_user
    missing_in_train_i = test_fids_item - train_fids_item

    # ══════════════════════════════════════════════════════════════
    # REPORT
    # ══════════════════════════════════════════════════════════════
    print("=" * 72)
    print("FEATURE EXPLORATION REPORT — TEST (vs TRAIN)")
    print("=" * 72)
    print(f"Train rows: {train_stats.get('total_rows', '?'):,}  |  Test rows: {test_rows:,}")
    print()

    # M0: Schema diff
    print("─ M0: SCHEMA DIFF ─")
    if missing_in_test_u or missing_in_test_i:
        print(f"  MISSING in test (train-only user_int): {sorted(missing_in_test_u)}")
        print(f"  MISSING in test (train-only item_int): {sorted(missing_in_test_i)}")
    if missing_in_train_u or missing_in_train_i:
        print(f"  NEW in test (test-only user_int): {sorted(missing_in_train_u)}")
        print(f"  NEW in test (test-only item_int): {sorted(missing_in_train_i)}")
    if not (missing_in_test_u or missing_in_test_i or missing_in_train_u or missing_in_train_i):
        print("  Schema identical — all fids present in both train and test.")
    print()

    # M1: Distribution drift
    print("─ M1: DISTRIBUTION DRIFT (int features, top 50 by drift) ─")
    n_show = min(len(drift_report), 50)
    print(f"{'Rank':>4} {'Feature':>24s} {'NZ_drift':>9s} {'Mean_drift':>11s} {'Train_AUC':>10s} {'Status':>14s}")
    print("-" * 78)
    for i, (key, nz_d, mean_d, auc, status) in enumerate(drift_report[:n_show]):
        if nz_d < 0:
            print(f"{i+1:>4} {key:>24s} {'N/A':>9s} {'N/A':>11s} {'N/A':>10s} {status:>14s}")
        else:
            print(f"{i+1:>4} {key:>24s} {nz_d:>9.4f} {mean_d:>11.4f} {auc:>10.4f} {status:>14s}")
    print()

    # M2: I2 feature focus
    print("─ M2: I2 GROUP (fids 5,6,7,8,12) TRAIN↔TEST ─")
    for fid_s in ["5", "6", "7", "8", "12"]:
        key = f"item_int_{fid_s}"
        ti = train_int.get(key, {})
        td = test_int_dists.get(key)
        if td is not None:
            t_nz = ti.get("nz_rate", 0) if ti else 0
            print(f"  fid={fid_s}: train nz={t_nz:.1%} auc={ti.get('auc', '?'):>7s}"
                  f"  test nz={td.nz_rate:.1%} mean={td.mean:.1f} std={td.std:.1f}")
        else:
            print(f"  fid={fid_s}: MISSING in test schema")
    print()

    # M3: Dense dim drift (top 20)
    if n_dense > 0:
        print("─ M3: DENSE DIM DRIFT (top 20 by non-zero rate diff) ─")
        train_dense = train_stats.get("dense_dims", [])
        train_dense_map = {d["dim"]: d for d in train_dense}
        dense_drift = []
        for d in range(n_dense):
            td = test_dense_dists[d]
            if td.total == 0:
                continue
            td_info = train_dense_map.get(d, {})
            t_nz = td_info.get("nz_rate", 0)
            drift = abs(td.nz_rate - t_nz)
            dense_drift.append((d, drift, td.nz_rate, t_nz, td.mean, td.std))
        dense_drift.sort(key=lambda x: -x[1])
        print(f"{'Rank':>4} {'Dim':>6} {'Drift':>8} {'Test_NZ%':>9} {'Train_NZ%':>10} {'Test_Mean':>14} {'Test_Std':>14}")
        print("-" * 70)
        for i, (d, dr, t_nz, tr_nz, t_mean, t_std) in enumerate(dense_drift[:20]):
            print(f"{i+1:>4} {d:>6} {dr:>8.4f} {t_nz:>8.1%} {tr_nz:>9.1%} {t_mean:>14.6f} {t_std:>14.6f}")
        print()

    # M4: Risk assessment
    print("─ M4: RISK ASSESSMENT ─")
    high = [(k, s) for k, _, _, _, s in drift_report if s == "HIGH_DRIFT"]
    medium = [(k, s) for k, _, _, _, s in drift_report if s == "MEDIUM_DRIFT"]
    missing = [k for k, _, _, _, s in drift_report if s == "MISSING_IN_TRAIN"]

    if missing:
        print(f"  ⚫ MISSING_IN_TRAIN ({len(missing)}): these features exist in test but NOT in train stats.")
        print(f"     Model didn't train on them → they contribute ZERO signal.")
        for k in missing:
            print(f"     {k}")
    if high:
        print(f"  🔴 HIGH DRIFT ({len(high)}): distribution shifted significantly. These features may hurt generalization.")
        for k, _ in high[:10]:
            ti = train_int.get(k, {})
            td = test_int_dists.get(k)
            t_auc = ti.get("auc", "?")
            if td:
                print(f"     {k:>24s}  train_nz={ti.get('nz_rate', 0):.1%} test_nz={td.nz_rate:.1%} train_auc={t_auc}")
    if medium:
        print(f"  🟡 MEDIUM DRIFT ({len(medium)}): moderate shift. Monitor but likely OK.")
    ok_count = len(drift_report) - len(high) - len(medium) - len(missing)
    print(f"  🟢 OK ({ok_count}): distributions consistent between train and test.")
    print()

    # M5: Feature grade
    print("─ M5: FEATURE GRADE (considering AUC + drift) ─")
    graded_s = []
    graded_a = []
    graded_b = []
    graded_drop = []
    for key, nz_d, mean_d, auc, status in drift_report:
        if status == "MISSING_IN_TRAIN":
            graded_drop.append(key)
        elif status == "HIGH_DRIFT":
            graded_drop.append(key)
        elif auc > 0.53 and status != "MEDIUM_DRIFT":
            graded_s.append(key)
        elif auc > 0.51 or status == "MEDIUM_DRIFT":
            graded_a.append(key)
        else:
            graded_b.append(key)

    print(f"  S-tier (AUC>0.53 + stable): {len(graded_s)} — safe to invest depth")
    for k in graded_s:
        ti = train_int.get(k, {})
        print(f"    {k:>24s}  AUC={ti.get('auc', '?')}")
    if not graded_s:
        print("    (none)")

    print(f"  A-tier (AUC~0.51 or moderate drift): {len(graded_a)} — use with caution")
    for k in graded_a[:8]:
        ti = train_int.get(k, {})
        print(f"    {k:>24s}  AUC={ti.get('auc', '?')}")
    if len(graded_a) > 8:
        print(f"    ... and {len(graded_a) - 8} more")

    print(f"  B-tier (AUC<0.51, stable): {len(graded_b)} — noise, safe to compress")
    print(f"  DROP (missing/high drift): {len(graded_drop)} — unreliable, consider removing")
    for k in graded_drop[:5]:
        print(f"    {k}")
    if len(graded_drop) > 5:
        print(f"    ... and {len(graded_drop) - 5} more")
    print()

    print("─ SUMMARY ─")
    print(f"  Total features compared: {len(drift_report) - len(missing)}")
    print(f"  Consistent (OK):         {ok_count}")
    print(f"  High drift:              {len(high)}")
    print(f"  Train AUC source:        {train_json}")
    print(f"  → Run explore_all.py on train first to generate feature_stats.json")
    print("=" * 72)


def main() -> None:
    """feature_audit: train↔test feature distribution comparison.

    No model loading, no inference. Pure data exploration: reads train-side
    ``feature_stats.json``, streams through test parquet, and prints a
    schema diff + distribution drift report.
    """
    model_dir = os.environ.get('MODEL_OUTPUT_PATH')
    data_dir = os.environ.get('EVAL_DATA_PATH')
    if not model_dir or not data_dir:
        logging.error("MODEL_OUTPUT_PATH and EVAL_DATA_PATH must be set.")
        return

    schema_path = os.path.join(model_dir, 'schema.json')
    if not os.path.exists(schema_path):
        schema_path = os.path.join(data_dir, 'schema.json')
    logging.info(f"Using schema: {schema_path}")

    train_config = load_train_config(model_dir)

    sml_str = train_config.get('seq_max_lens', _FALLBACK_SEQ_MAX_LENS)
    seq_max_lens = _parse_seq_max_lens(sml_str)
    logging.info(f"seq_max_lens: {seq_max_lens}")

    batch_size = int(os.environ.get('EVAL_BATCH_SIZE',
                     train_config.get('batch_size', _FALLBACK_BATCH_SIZE)))

    _run_data_diagnose(model_dir, data_dir, schema_path, seq_max_lens, batch_size)


if __name__ == "__main__":
    main()
