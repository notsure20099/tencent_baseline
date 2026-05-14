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

import numpy as np
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


# ═══════════════════════════════════════════════════════════════════════════════
# Model & Data Diagnostic Functions (infer stage)
# ═══════════════════════════════════════════════════════════════════════════════


def _run_model_diagnose(model: nn.Module) -> None:
    """打印模型结构与权重诊断信息。"""
    logging.info("=" * 60)
    logging.info("[Diagnose] 模型结构诊断开始")

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logging.info(f"[Diagnose] 总参数量: {total_params:,}  可训练: {trainable_params:,}")

    emb_count = 0
    emb_total_vocab = 0
    for m in model.modules():
        if isinstance(m, nn.Embedding):
            emb_count += 1
            emb_total_vocab += m.num_embeddings
    logging.info(f"[Diagnose] Embedding 表数量: {emb_count}  总词表大小: {emb_total_vocab:,}")

    if hasattr(model, 'blocks') and len(model.blocks) > 0:
        for block_idx, block in enumerate(model.blocks):
            mixer = block.mixer
            rm_mode = mixer.mode if hasattr(mixer, 'mode') else 'unknown'
            T_val = mixer.T if hasattr(mixer, 'T') else 0
            D_val = mixer.D if hasattr(mixer, 'D') else 0
            full_check = 'OK' if (rm_mode == 'full' and T_val > 0 and D_val % T_val == 0) else 'DEGRADED'
            logging.info(
                f"[Diagnose] Block{block_idx} RankMixer mode={rm_mode} T={T_val} d_model={D_val} "
                f"{D_val}%{T_val}={D_val % T_val if T_val else '?'} ({full_check})")

            if hasattr(block, 'cross_attns'):
                for ca_idx, ca in enumerate(block.cross_attns):
                    if hasattr(ca, 'use_time_bias') and ca.use_time_bias:
                        tb = ca.temporal_bias.weight.detach().cpu()
                        tb_norm = float(tb.norm())
                        tb_mean = float(tb[1:].mean())
                        domain_map = {0: 'seq_a', 1: 'seq_b', 2: 'seq_c', 3: 'seq_d'}
                        domain = domain_map.get(ca_idx, f'ca_{ca_idx}')
                        logging.info(
                            f"[Diagnose] Block{block_idx} time_bias {domain} "
                            f"norm={tb_norm:.3f} mean={tb_mean:+.3f}")

    cfg_items = {
        'd_model': model.d_model, 'emb_dim': model.emb_dim,
        'num_queries': model.num_queries, 'num_sequences': model.num_sequences,
        'num_ns': model.num_ns, 'rank_mixer_mode': model.rank_mixer_mode,
        'use_time_bias': model.use_time_bias, 'use_rope': model.use_rope,
        'ns_tokenizer_type': model.ns_tokenizer_type, 'seq_domains': model.seq_domains,
    }
    for k, v in cfg_items.items():
        logging.info(f"[Diagnose] cfg.{k} = {v}")

    logging.info("[Diagnose] 模型结构诊断结束")
    logging.info("=" * 60)


def _run_test_seq_explore(
    test_dataset: PCVRParquetDataset,
    model: nn.Module,
    device: str,
    num_workers: int,
    sample_batches: int = 20,
) -> None:
    """对测试集前 sample_batches 批进行序列特征探索（独立 DataLoader）。"""
    logging.info("=" * 60)
    logging.info(f"[Diagnose] 测试集序列探索开始 (采样 {sample_batches} 批)")

    explore_loader = DataLoader(
        test_dataset, batch_size=None, num_workers=num_workers,
        prefetch_factor=2 if num_workers > 0 else None, pin_memory=False,
    )

    seq_domains = model.seq_domains
    all_lens = {d: [] for d in seq_domains}
    all_tb = {d: [] for d in seq_domains}
    all_density = {d: [] for d in seq_domains}

    batch_count = 0
    with torch.no_grad():
        for batch in explore_loader:
            if batch_count >= sample_batches:
                break
            for domain in seq_domains:
                seq_len_tensor = batch.get(f'{domain}_len')
                seq_data_tensor = batch.get(domain)
                seq_tb_tensor = batch.get(f'{domain}_time_bucket')
                if seq_len_tensor is not None:
                    all_lens[domain].append(seq_len_tensor.numpy().astype(np.int32))
                if seq_data_tensor is not None:
                    B, S, L = seq_data_tensor.shape
                    non_zero = (seq_data_tensor.numpy() != 0).astype(np.float32).sum(axis=(1, 2))
                    all_density[domain].append((non_zero / (S * L)).astype(np.float32))
                if seq_tb_tensor is not None:
                    all_tb[domain].append(seq_tb_tensor.numpy().astype(np.int32))
            batch_count += 1

    logging.info(f"[Diagnose] 实际采样批次: {batch_count}")
    for domain in seq_domains:
        if all_lens[domain]:
            lens = np.concatenate(all_lens[domain])
            logging.info(
                f"[Diagnose] {domain} 序列长度: min={lens.min():.0f} max={lens.max():.0f} "
                f"mean={lens.mean():.1f} median={np.median(lens):.1f} std={lens.std():.1f} "
                f"样本数={len(lens)}")
        if all_density[domain]:
            dens = np.concatenate(all_density[domain])
            logging.info(
                f"[Diagnose] {domain} fid非零密度: min={dens.min():.4f} max={dens.max():.4f} "
                f"mean={dens.mean():.4f} std={dens.std():.4f}")
        if all_tb[domain]:
            tbs = np.concatenate(all_tb[domain])
            non_zero = tbs[tbs > 0]
            if len(non_zero) > 0:
                pcts = [10, 25, 50, 75, 90]
                pct_vals = np.percentile(non_zero, pcts)
                pct_str = " ".join(f"P{p}={pct_vals[i]:.0f}" for i, p in enumerate(pcts))
                logging.info(
                    f"[Diagnose] {domain} time_bucket(非零): "
                    f"比例={len(non_zero)/len(tbs.ravel()):.2%} "
                    f"min={non_zero.min()} max={non_zero.max()} mean={non_zero.mean():.1f} {pct_str}")
            else:
                logging.info(f"[Diagnose] {domain} time_bucket: 全部为0")

    if len(seq_domains) >= 2:
        logging.info("[Diagnose] 跨域序列长度相关系数矩阵:")
        lens_matrix = [np.concatenate(all_lens[d]) for d in seq_domains if all_lens[d]]
        if len(lens_matrix) >= 2:
            corr = np.corrcoef(lens_matrix)
            header = "        " + " ".join(f"{d:>8s}" for d in seq_domains)
            logging.info(header)
            for i, d in enumerate(seq_domains):
                row = " ".join(f"{corr[i][j]:8.3f}" for j in range(len(seq_domains)))
                logging.info(f"{d:>8s} {row}")

    logging.info("[Diagnose] 测试集序列探索结束")
    logging.info("=" * 60)


def _run_pred_dist_diagnose(all_probs: list) -> None:
    """打印预测概率分布诊断。"""
    logging.info("=" * 60)
    logging.info("[Diagnose] 预测概率分布诊断")

    probs_np = np.array(all_probs, dtype=np.float32)
    pcts = [1, 5, 10, 25, 50, 75, 90, 95, 99]
    pct_vals = np.percentile(probs_np, pcts)
    pct_str = " ".join(f"P{p}={pct_vals[i]:.4f}" for i, p in enumerate(pcts))
    logging.info(
        f"[Diagnose] 预测概率: N={len(probs_np)} "
        f"min={probs_np.min():.4f} max={probs_np.max():.4f} "
        f"mean={probs_np.mean():.4f} std={probs_np.std():.4f}")
    logging.info(f"[Diagnose] 百分位: {pct_str}")

    bins = [0, 0.01, 0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 0.9, 1.0]
    hist, _ = np.histogram(probs_np, bins=bins)
    total = len(probs_np)
    bin_desc = []
    for i in range(len(bins) - 1):
        if hist[i] > 0:
            bin_desc.append(f"[{bins[i]:.2f},{bins[i+1]:.2f})={hist[i]/total:.2%}")
    logging.info(f"[Diagnose] 概率分布: {' | '.join(bin_desc)}")

    logging.info("[Diagnose] 预测概率分布诊断结束")
    logging.info("=" * 60)


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
    'd_model': 76,
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

    # ── 模型诊断 ──
    _run_model_diagnose(model)

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        logging.info("Cleared CUDA cache before inference")

    _log_gpu_memory('before_inference')

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

    # ── 测试集序列探索 (独立 DataLoader，不影响主推理) ──
    _run_test_seq_explore(test_dataset, model, device, num_workers)

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

    # ── 预测概率分布诊断 ──
    _run_pred_dist_diagnose(all_probs)

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
