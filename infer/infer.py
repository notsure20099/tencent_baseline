"""PCVRHyFormer inference script (uploaded by the contestant into the
evaluation container).

Model construction mirrors ``train.py``: we rebuild the model from
``schema.json`` + ``ns_groups.json`` + ``train_config.json``. All model
hyperparameters are resolved first from the ckpt directory's
``train_config.json`` (written by ``trainer.py`` when saving a checkpoint),
falling back to ``_FALLBACK_MODEL_CFG`` below (which must stay consistent
with the CLI defaults in ``train.py``).

Only the Parquet data format is supported.

Environment variables:
    MODEL_OUTPUT_PATH  Checkpoint directory (points at the ``global_step``
                       sub-directory containing ``model.pt`` / ``train_config.json``).
    EVAL_DATA_PATH     Test data directory (*.parquet + schema.json).
    EVAL_RESULT_PATH   Directory for the generated ``predictions.json``.
"""

import os
import json
import logging
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from dataset import FeatureSchema, PCVRParquetDataset, NUM_TIME_BUCKETS
from model import PCVRHyFormer, ModelInput


logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
)


def _log_gpu_memory(tag: str = '') -> None:
    """Log CUDA memory usage and per-process GPU utilization.

    Emits ``nvidia-smi`` output (when available) and PyTorch's allocator
    summary so that OOM diagnostics are self-contained in the log.
    """
    if not torch.cuda.is_available():
        logging.info(f"[GPU:{tag}] CUDA not available")
        return

    free_bytes, total_bytes = torch.cuda.mem_get_info()
    allocated = torch.cuda.memory_allocated()
    reserved = torch.cuda.memory_reserved()
    used_pct = 100.0 * (total_bytes - free_bytes) / total_bytes

    logging.info(
        f"[GPU:{tag}] total={total_bytes / 1024**3:.2f} GiB  "
        f"free={free_bytes / 1024**3:.2f} GiB  "
        f"used={used_pct:.1f}%  "
        f"allocated={allocated / 1024**3:.2f} GiB  "
        f"reserved={reserved / 1024**3:.2f} GiB"
    )

    try:
        import subprocess
        result = subprocess.run(
            ['nvidia-smi', '--query-compute-apps=pid,used_memory',
             '--format=csv,noheader,nounits'],
            capture_output=True, text=True, timeout=5,
        )
        if result.returncode == 0 and result.stdout.strip():
            logging.info(f"[GPU:{tag}] nvidia-smi processes:\n{result.stdout.strip()}")
    except Exception:
        pass


# Fallback values used only when ``train_config.json`` is missing from the
# ckpt directory.
#
# These MUST match the argparse defaults in ``train.py``; otherwise once the
# fallback path is actually taken the built model will shape-mismatch the
# saved state_dict.
#
# Special note on ``num_time_buckets``: this value is strictly determined by
# ``dataset.BUCKET_BOUNDARIES`` and is NOT an independent hyperparameter.
# When the feature is enabled we therefore use the constant exposed by the
# dataset module; ``0`` means disabled.
_FALLBACK_MODEL_CFG = {
    'd_model': 64,
    'emb_dim': 64,
    'num_queries': 1,
    'num_hyformer_blocks': 2,
    'num_heads': 4,
    'seq_encoder_type': 'transformer',
    'hidden_mult': 4,
    'dropout_rate': 0.01,
    'seq_top_k': 50,
    'seq_causal': False,
    'action_num': 1,
    'num_time_buckets': NUM_TIME_BUCKETS,
    'rank_mixer_mode': 'full',
    'use_rope': False,
    'rope_base': 10000.0,
    'emb_skip_threshold': 0,
    'seq_id_threshold': 10000,
    'ns_tokenizer_type': 'rankmixer',
    'user_ns_tokens': 0,
    'item_ns_tokens': 0,
    'use_item_bridge': False,
    'dense_token_groups': 1,
    'dense_aware_qgen': False,
    'use_time_bias': False,
}

_FALLBACK_SEQ_MAX_LENS = 'seq_a:256,seq_b:256,seq_c:512,seq_d:512'
_FALLBACK_BATCH_SIZE = 256
_FALLBACK_NUM_WORKERS = 16


# Hyperparameter keys used to build the model. Everything else in
# ``train_config.json`` is ignored when constructing ``PCVRHyFormer``.
_MODEL_CFG_KEYS = list(_FALLBACK_MODEL_CFG.keys())


def build_feature_specs(
    schema: FeatureSchema,
    per_position_vocab_sizes: List[int],
) -> List[Tuple[int, int, int]]:
    """Build ``feature_specs = [(vocab_size, offset, length), ...]`` in the
    order of ``schema.entries``.
    """
    specs: List[Tuple[int, int, int]] = []
    for fid, offset, length in schema.entries:
        vs = max(per_position_vocab_sizes[offset:offset + length])
        specs.append((vs, offset, length))
    return specs


def _parse_seq_max_lens(sml_str: str) -> Dict[str, int]:
    """Parse a string like ``'seq_a:256,seq_b:256,...'`` into a dict."""
    seq_max_lens: Dict[str, int] = {}
    for pair in sml_str.split(','):
        k, v = pair.split(':')
        seq_max_lens[k.strip()] = int(v.strip())
    return seq_max_lens


def load_train_config(model_dir: str) -> Dict[str, Any]:
    """Load ``train_config.json`` from the ckpt directory.

    Returns an empty dict (which triggers fallback resolution) if the file is
    not present.
    """
    train_config_path = os.path.join(model_dir, 'train_config.json')
    if os.path.exists(train_config_path):
        with open(train_config_path, 'r') as f:
            cfg = json.load(f)
        logging.info(f"Loaded train_config from {train_config_path}")
        return cfg
    logging.warning(
        f"train_config.json not found in {model_dir}, "
        f"falling back to hardcoded defaults. "
        f"Shape mismatch may occur if training used non-default hyperparameters.")
    return {}


def resolve_model_cfg(train_config: Dict[str, Any]) -> Dict[str, Any]:
    """Extract model hyperparameters from ``train_config``; missing keys fall
    back to ``_FALLBACK_MODEL_CFG``.

    Special handling for ``num_time_buckets``: it is not exposed on the CLI
    as an independent hyperparameter; the bucket count is uniquely determined
    by the length of ``dataset.BUCKET_BOUNDARIES``. Resolution order:

      1) ``train_config`` contains ``num_time_buckets`` directly (legacy ckpt)
         -> use that value;
      2) ``train_config`` contains ``use_time_buckets`` (new-style training)
         -> derive as ``NUM_TIME_BUCKETS`` or ``0``;
      3) neither is present -> fall back to ``_FALLBACK_MODEL_CFG[...]``.
    """
    cfg: Dict[str, Any] = {}
    for key in _MODEL_CFG_KEYS:
        if key == 'num_time_buckets':
            if 'num_time_buckets' in train_config:
                cfg[key] = train_config['num_time_buckets']
            elif 'use_time_buckets' in train_config:
                cfg[key] = NUM_TIME_BUCKETS if train_config['use_time_buckets'] else 0
            else:
                cfg[key] = _FALLBACK_MODEL_CFG[key]
                logging.warning(
                    f"train_config missing both 'num_time_buckets' and 'use_time_buckets', "
                    f"using fallback = {cfg[key]}")
            continue

        if key in train_config:
            cfg[key] = train_config[key]
        else:
            cfg[key] = _FALLBACK_MODEL_CFG[key]
            logging.warning(
                f"train_config missing '{key}', using fallback = {cfg[key]}")
    return cfg


def build_model(
    dataset: PCVRParquetDataset,
    model_cfg: Dict[str, Any],
    ns_groups_json: Optional[str] = None,
    device: str = 'cpu',
) -> PCVRHyFormer:
    """Construct a ``PCVRHyFormer`` from the dataset schema, an NS-groups JSON,
    and a resolved ``model_cfg`` dict.

    Args:
        dataset: a ``PCVRParquetDataset`` providing the feature schema.
        model_cfg: resolved model hyperparameters, typically the output of
            ``resolve_model_cfg``.
        ns_groups_json: path to the NS-groups JSON file, or ``None`` / empty
            string to disable it (each feature becomes its own singleton group).
        device: torch device.
    """
    # NS grouping. The JSON schema uses *fid* (feature id) values; convert
    # them to positional indices into ``user_int_schema.entries`` /
    # ``item_int_schema.entries`` so ``GroupNSTokenizer`` /
    # ``RankMixerNSTokenizer`` can index ``feature_specs`` directly. This is
    # the same conversion ``train.py`` performs when loading the JSON; doing
    # it here keeps infer.py symmetric with training.
    user_ns_groups: List[List[int]]
    item_ns_groups: List[List[int]]
    if ns_groups_json and os.path.exists(ns_groups_json):
        logging.info(f"Loading NS groups from {ns_groups_json}")
        with open(ns_groups_json, 'r') as f:
            ns_groups_cfg = json.load(f)
        user_fid_to_idx = {
            fid: i for i, (fid, _, _) in enumerate(dataset.user_int_schema.entries)
        }
        item_fid_to_idx = {
            fid: i for i, (fid, _, _) in enumerate(dataset.item_int_schema.entries)
        }
        try:
            user_ns_groups = [
                [user_fid_to_idx[f] for f in fids]
                for fids in ns_groups_cfg['user_ns_groups'].values()
            ]
            item_ns_groups = [
                [item_fid_to_idx[f] for f in fids]
                for fids in ns_groups_cfg['item_ns_groups'].values()
            ]
        except KeyError as exc:
            raise KeyError(
                f"NS-groups JSON references fid {exc.args[0]} which is not "
                f"present in the checkpoint's schema.json. The ns_groups.json "
                f"and schema.json must come from the same training run."
            ) from exc
    else:
        logging.info("No NS groups JSON found, using default: each feature as one group")
        user_ns_groups = [[i] for i in range(len(dataset.user_int_schema.entries))]
        item_ns_groups = [[i] for i in range(len(dataset.item_int_schema.entries))]

    # Feature specs.
    user_int_feature_specs = build_feature_specs(
        dataset.user_int_schema, dataset.user_int_vocab_sizes)
    item_int_feature_specs = build_feature_specs(
        dataset.item_int_schema, dataset.item_int_vocab_sizes)

    logging.info(f"Building PCVRHyFormer with cfg: {model_cfg}")
    model = PCVRHyFormer(
        user_int_feature_specs=user_int_feature_specs,
        item_int_feature_specs=item_int_feature_specs,
        user_dense_dim=dataset.user_dense_schema.total_dim,
        item_dense_dim=dataset.item_dense_schema.total_dim,
        seq_vocab_sizes=dataset.seq_domain_vocab_sizes,
        user_ns_groups=user_ns_groups,
        item_ns_groups=item_ns_groups,
        **model_cfg,
    ).to(device)

    return model


def load_model_state_strict(
    model: nn.Module,
    ckpt_path: str,
    device: str,
) -> None:
    """Strictly load ``state_dict``; any missing/unexpected key fails fast
    with a diagnostic message.
    """
    state_dict = torch.load(ckpt_path, map_location=device)
    if any(k.startswith('_orig_mod.') for k in state_dict.keys()):
        state_dict = {k.removeprefix('_orig_mod.'): v for k, v in state_dict.items()}
    try:
        model.load_state_dict(state_dict, strict=True)
    except RuntimeError as e:
        logging.error(
            "Failed to load state_dict in strict mode. This usually means the "
            "model constructed by build_model does NOT match the checkpoint. "
            "Check that train_config.json in the ckpt dir is present and matches "
            "the training hyperparameters.")
        raise e


def get_ckpt_path() -> Optional[str]:
    """Locate the first ``*.pt`` file inside the directory pointed at by
    ``$MODEL_OUTPUT_PATH``. Returns ``None`` if no checkpoint is found.
    """
    ckpt_path = os.environ.get("MODEL_OUTPUT_PATH")
    if not ckpt_path:
        return None
    for item in os.listdir(ckpt_path):
        if item.endswith(".pt"):
            return os.path.join(ckpt_path, item)
    return None


def _manual_auc(labels: List[int], probs: List[float]) -> float:
    """Manual AUC computation — no sklearn dependency."""
    pairs = sorted(zip(probs, labels), key=lambda x: x[0])
    pos_count = sum(labels)
    neg_count = len(labels) - pos_count
    if pos_count == 0 or neg_count == 0:
        return float("nan")
    auc = 0.0
    neg_below = 0
    for prob, lbl in pairs:
        if lbl == 1:
            auc += neg_below
        else:
            neg_below += 1
    return auc / (pos_count * neg_count)


def _run_q4_diagnose(model, dataset, device):
    """Q4: Content vs Time awareness — shuffle ablation."""
    import numpy as np
    logging.info("=" * 60)
    logging.info("Q4: CONTENT vs TIME AWARENESS (shuffle ablation)")
    logging.info("=" * 60)

    # need labels — verify dataset has them
    test_loader = DataLoader(dataset, batch_size=None, num_workers=min(4, os.cpu_count() or 1),
                             prefetch_factor=2, pin_memory=torch.cuda.is_available())

    def _collect_auc(loader, shuffle_mode="none", max_batches=0):
        probs_all, labels_all = [], []
        with torch.no_grad():
            for bi, batch in enumerate(loader):
                if max_batches and bi >= max_batches:
                    break
                # move to device
                db = {}
                for k, v in batch.items():
                    if isinstance(v, torch.Tensor):
                        db[k] = v.to(device, non_blocking=True)
                    else:
                        db[k] = v

                label = db.get("label")
                if label is None:
                    return float("nan")
                if hasattr(label, "numpy"):
                    label = label.cpu().numpy()
                elif isinstance(label, list):
                    label = np.array(label)
                else:
                    label = np.zeros(len(db[list(db.keys())[0]]))

                # apply shuffle
                if shuffle_mode == "time":
                    for d in db.get("_seq_domains", []):
                        tb_key = f"{d}_time_bucket"
                        if tb_key in db:
                            tb = db[tb_key]
                            idx = torch.randperm(tb.shape[0], device=device)
                            db[tb_key] = tb[idx]
                elif shuffle_mode == "fid":
                    for d in db.get("_seq_domains", []):
                        seq = db[d]
                        B, nf, L = seq.shape
                        for bb in range(B):
                            perm = torch.randperm(L, device=device)
                            db[d][bb] = seq[bb, :, perm]

                mi = _batch_to_model_input(db, device)
                logits, _ = model.predict(mi)
                p = torch.sigmoid(logits.squeeze(-1)).cpu().numpy()
                probs_all.extend(p.tolist())
                labels_all.extend(label.tolist())

        return _manual_auc(labels_all, probs_all)

    # Sampling: read Q4_SAMPLE_BATCHES env var (default 300 batches)
    n_max = int(os.environ.get('Q4_SAMPLE_BATCHES', '300'))
    logging.info(f"Q4 sampling: max {n_max} batches per pass")

    logging.info("Computing baseline AUC...")
    auc_base = _collect_auc(test_loader, "none", max_batches=n_max)
    logging.info("Computing shuffle-time AUC...")
    auc_time = _collect_auc(test_loader, "time", max_batches=n_max)
    logging.info("Computing shuffle-fid AUC...")
    auc_fid = _collect_auc(test_loader, "fid", max_batches=n_max)

    time_drop = auc_base - auc_time
    fid_drop = auc_base - auc_fid

    logging.info("=" * 60)
    logging.info("Q4 RESULTS")
    logging.info(f"  baseline           AUC = {auc_base:.6f}")
    logging.info(f"  shuffle time       AUC = {auc_time:.6f}  (drop = {time_drop:+.6f})")
    logging.info(f"  shuffle fid order  AUC = {auc_fid:.6f}  (drop = {fid_drop:+.6f})")
    logging.info("  --- interpretation ---")
    if time_drop > 0.001 and fid_drop < 0.0003:
        logging.info("  model relies HEAVILY on time, barely on content")
        logging.info("  -> content-interaction optimisations may have room")
    elif fid_drop > 0.001 and time_drop < 0.0003:
        logging.info("  model relies on content, time is weak")
        logging.info("  -> time bias direction fully exploited; content mining is path")
    elif time_drop > 0.001 and fid_drop > 0.001:
        logging.info("  model uses both time and content — balanced")
        logging.info("  -> either direction valid, pick cheaper one")
    else:
        logging.info("  both drops small — model may rely on non-seq features")
    logging.info("=" * 60)


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
                       "Run EXPLORE_MODE=true on train first.")
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
    user_dense_entries = test_dataset.user_dense_schema.entries
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
    print(f"  → Run EXPLORE_MODE=true on train first to generate feature_stats.json")
    print("=" * 72)


def _batch_to_model_input(
    batch: Dict[str, Any],
    device: str,
) -> ModelInput:
    """Convert a batch dict to ``ModelInput``, handling dynamic seq domains."""
    device_batch: Dict[str, Any] = {}
    for k, v in batch.items():
        if isinstance(v, torch.Tensor):
            device_batch[k] = v.to(device, non_blocking=True)
        else:
            device_batch[k] = v

    seq_domains = device_batch['_seq_domains']
    seq_data: Dict[str, torch.Tensor] = {}
    seq_lens: Dict[str, torch.Tensor] = {}
    seq_time_buckets: Dict[str, torch.Tensor] = {}
    for domain in seq_domains:
        seq_data[domain] = device_batch[domain]
        seq_lens[domain] = device_batch[f'{domain}_len']
        B, _, L = device_batch[domain].shape
        seq_time_buckets[domain] = device_batch.get(
            f'{domain}_time_bucket',
            torch.zeros(B, L, dtype=torch.long, device=device))

    return ModelInput(
        user_int_feats=device_batch['user_int_feats'],
        item_int_feats=device_batch['item_int_feats'],
        user_dense_feats=device_batch['user_dense_feats'],
        item_dense_feats=device_batch['item_dense_feats'],
        seq_data=seq_data,
        seq_lens=seq_lens,
        seq_time_buckets=seq_time_buckets,
    )


def main() -> None:
    # ---- Read environment variables ----
    model_dir = os.environ.get('MODEL_OUTPUT_PATH')
    data_dir = os.environ.get('EVAL_DATA_PATH')
    result_dir = os.environ.get('EVAL_RESULT_PATH')

    os.makedirs(result_dir, exist_ok=True)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # ---- Schema: prefer the one from model_dir (to exactly match training);
    #      fall back to the one in data_dir if missing. ----
    schema_path = os.path.join(model_dir, 'schema.json')
    if not os.path.exists(schema_path):
        schema_path = os.path.join(data_dir, 'schema.json')
    logging.info(f"Using schema: {schema_path}")

    # ---- Load train_config.json (single source of truth for all hyperparams) ----
    train_config = load_train_config(model_dir)

    # ---- Parse seq_max_lens ----
    sml_str = train_config.get('seq_max_lens', _FALLBACK_SEQ_MAX_LENS)
    seq_max_lens = _parse_seq_max_lens(sml_str)
    logging.info(f"seq_max_lens: {seq_max_lens}")

    # ---- Data loading: reuse batch_size / num_workers from training config ----
    # Allow overriding batch_size via EVAL_BATCH_SIZE env var (useful when
    # GPU memory is tight, e.g. shared GPU with other processes).
    eval_batch_size = os.environ.get('EVAL_BATCH_SIZE')
    if eval_batch_size is not None:
        batch_size = int(eval_batch_size)
        logging.info(f"Using EVAL_BATCH_SIZE={batch_size} (overriding training config)")
    else:
        batch_size = int(train_config.get('batch_size', _FALLBACK_BATCH_SIZE))
    num_workers = int(train_config.get('num_workers', _FALLBACK_NUM_WORKERS))

    test_dataset = PCVRParquetDataset(
        parquet_path=data_dir,
        schema_path=schema_path,
        batch_size=batch_size,
        seq_max_lens=seq_max_lens,
        shuffle=False,
        buffer_batches=0,
        is_training=False,
    )
    total_test_samples = test_dataset.num_rows
    logging.info(f"Total test samples: {total_test_samples}")

    # ---- Build model: every structural hyperparameter is resolved from train_config ----
    model_cfg = resolve_model_cfg(train_config)

    # ns_groups_json also comes from training config (e.g. run.sh may have
    # passed an empty string to disable it). When trainer.py has copied the
    # JSON into the ckpt dir, train_config records just the basename, so try
    # resolving against ``model_dir`` first before honoring the raw (possibly
    # absolute) path as a fallback.
    ns_groups_json = train_config.get('ns_groups_json', None)
    if ns_groups_json:
        local_candidate = os.path.join(model_dir, os.path.basename(ns_groups_json))
        if os.path.exists(local_candidate):
            ns_groups_json = local_candidate

    _log_gpu_memory('before_build')
    model = build_model(
        test_dataset,
        model_cfg=model_cfg,
        ns_groups_json=ns_groups_json,
        device=device,
    )
    _log_gpu_memory('after_build')

    # ---- Strictly load weights ----
    ckpt_path = get_ckpt_path()
    if ckpt_path is None:
        raise FileNotFoundError(
            f"No *.pt file found under MODEL_OUTPUT_PATH={model_dir!r}. "
            f"The directory contains: {os.listdir(model_dir) if model_dir and os.path.isdir(model_dir) else 'N/A'}. "
            "This typically means the training job wrote only the sidecar "
            "files (schema.json / train_config.json) for this step but did "
            "not persist model.pt — a symptom of a race between "
            "_remove_old_best_dirs and EarlyStopping.save_checkpoint."
        )
    logging.info(f"Loading checkpoint from {ckpt_path}")
    load_model_state_strict(model, ckpt_path, device)
    model.eval()
    logging.info("Model loaded successfully")

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        logging.info("Cleared CUDA cache before inference")

    _log_gpu_memory('before_inference')

    # ── Data diagnostic: train↔test feature distribution comparison ──
    if os.environ.get('DIAGNOSE_DATA', '').lower() in ('true', '1', 'yes'):
        _run_data_diagnose(model_dir, data_dir, schema_path, seq_max_lens, batch_size)
        return

    # ── Q4 diagnostic: Content vs Time awareness ──
    if os.environ.get('Q4_DIAGNOSE', '').lower() in ('true', '1', 'yes'):
        # Q4 needs labels → use training data path (TRAIN_DATA_PATH env var
        # or fallback: strip the task-id suffix from EVAL_DATA_PATH)
        q4_data_dir = os.environ.get('TRAIN_DATA_PATH', '')
        if not q4_data_dir:
            # fallback: try replacing the task dir suffix in EVAL_DATA_PATH
            # e.g. .../84887/test_data → .../train_data
            q4_data_dir = data_dir.replace('/test/', '/train/').replace('/eval/', '/train/')
        if not q4_data_dir or not os.path.isdir(q4_data_dir):
            q4_data_dir = data_dir
            logging.warning("Q4: TRAIN_DATA_PATH not set, using EVAL_DATA_PATH (labels may be all-zero!)")

        q4_dataset = PCVRParquetDataset(
            parquet_path=q4_data_dir,
            schema_path=schema_path,
            batch_size=batch_size,
            seq_max_lens=seq_max_lens,
            shuffle=True,
            buffer_batches=20,
            is_training=True,  # ← read real labels
        )
        logging.info(f"Q4: using data from {q4_data_dir} ({q4_dataset.num_rows} rows, is_training=True)")
        _run_q4_diagnose(model, q4_dataset, device)
        return

    test_loader = DataLoader(
        test_dataset,
        batch_size=None,
        num_workers=num_workers,
        prefetch_factor=2,
        pin_memory=torch.cuda.is_available(),
    )

    all_probs = []
    all_user_ids = []
    logging.info("Starting inference...")

    try:
        with torch.no_grad():
            for batch_idx, batch in enumerate(test_loader):
                model_input = _batch_to_model_input(batch, device)
                user_ids = batch.get('user_id', [])

                logits, _ = model.predict(model_input)
                logits = logits.squeeze(-1)
                probs = torch.sigmoid(logits).cpu().numpy()
                all_probs.extend(probs.tolist())
                all_user_ids.extend(user_ids)

                if (batch_idx + 1) % 100 == 0:
                    logging.info(f"  Processed {(batch_idx + 1) * batch_size} samples")
    except RuntimeError as e:
        _log_gpu_memory(f'OOM_at_batch_{batch_idx}')
        raise e

    logging.info(f"Inference complete: {len(all_probs)} predictions")

    predictions = {
        "predictions": dict(zip(all_user_ids, all_probs)),
    }

    # ---- Save predictions.json ----
    output_path = os.path.join(result_dir, 'predictions.json')
    with open(output_path, 'w') as f:
        json.dump(predictions, f)
    logging.info(f"Saved {len(all_probs)} predictions to {output_path}")


if __name__ == "__main__":
    main()
