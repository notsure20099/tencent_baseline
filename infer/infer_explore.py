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
import math
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from dataset import FeatureSchema, PCVRParquetDataset, BUCKET_BOUNDARIES, NUM_TIME_BUCKETS
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


def _bucket_label(idx: int) -> str:
    if idx == 0:
        return "pad"
    edge = BUCKET_BOUNDARIES[idx - 1]
    if edge < 60:
        return f"{edge}s"
    if edge < 3600:
        return f"{edge // 60}m"
    if edge < 86400:
        return f"{edge // 3600}h"
    if edge < 604800:
        return f"{edge // 86400}d"
    if edge < 2592000:
        return f"{edge // 604800}w"
    return f"{edge // 2592000}mon"


def _run_diagnose(
    model: PCVRHyFormer,
    test_dataset: PCVRParquetDataset,
    device: str,
) -> None:
    num_blocks = len(model.blocks)
    num_sequences = model.blocks[0].num_sequences
    num_heads = model.blocks[0].cross_attns[0].attn.num_heads

    print("=" * 72)
    print("  DIAGNOSE MODE: Time Bias Analysis")
    print(f"  Model: {num_blocks} HyFormer blocks, {num_sequences} sequences, {num_heads} heads")
    print(f"  Time buckets: {NUM_TIME_BUCKETS} (0=pad, 1..{NUM_TIME_BUCKETS - 1}=event)")
    print("=" * 72)

    # Collect only Block 0, Sequence 0 (representative layer)
    ca = model.blocks[0].cross_attns[0]
    if not ca.use_time_bias:
        print("WARNING: use_time_bias=False, no temporal_bias to analyze. Aborting.")
        return
    w0 = ca.temporal_bias.weight.detach().cpu().numpy()  # (65, H)

    # ── 1. Per-head stats + top5 ──
    print("── 1. Per-Head Time Bias Stats (Block0 CrossAttn0) ──")
    for h in range(num_heads):
        vals = w0[1:, h]
        top5 = np.argsort(w0[:, h])[::-1][:5]
        labels = ",".join(f"{_bucket_label(i)}:{w0[i, h]:+.2f}" for i in top5)
        print(f"  Head {h}: mean={vals.mean():+.3f} std={vals.std():.3f} "
              f"range=[{vals.min():+.3f},{vals.max():+.3f}] top5={{{labels}}}")

    # ── 2. Head overlap ──
    top5_sets = [set(np.argsort(w0[1:, h])[::-1][:5]) for h in range(num_heads)]
    print("── 2. Head Preference Overlap (Jaccard on top-5 buckets) ──")
    overlap_m = []
    for h1 in range(num_heads):
        row = []
        for h2 in range(num_heads):
            j = 1.0 if h1 == h2 else len(top5_sets[h1] & top5_sets[h2]) / max(1, len(top5_sets[h1] | top5_sets[h2]))
            row.append(f"{j:.2f}")
        print(f"  H{h1}:  " + "  ".join(row))
        overlap_m.append(row)
    vals_o = [float(overlap_m[h1][h2]) for h1 in range(num_heads) for h2 in range(h1 + 1, num_heads)]
    avg_overlap = float(np.mean(vals_o)) if vals_o else 1.0

    # ── 3. Per-bucket head disagreement ──
    per_bucket_std = w0[1:, :].std(axis=1)  # (64,)
    mean_disagreement = float(per_bucket_std.mean())
    top5_disagree = np.argsort(per_bucket_std)[::-1][:5]
    disagree_labels = ",".join(f"{_bucket_label(i + 1)}:{per_bucket_std[i]:.3f}" for i in top5_disagree)
    print(f"── 3. Head Disagreement ──")
    print(f"  mean={mean_disagreement:.4f}  top5_max_std=[{disagree_labels}]")

    # ── 4. Curve shape ──
    n_step = 0
    print("── 4. Curve Shape ──")
    for h in range(num_heads):
        vals = w0[1:, h]
        diffs = np.abs(np.diff(vals))
        n_jumps = int((diffs > 2 * float(diffs.mean())).sum())
        second_diff = np.diff(vals, n=2)
        smooth_L2 = float(np.sum(second_diff ** 2))
        if n_jumps >= 3:
            shape = "step"
            n_step += 1
        elif smooth_L2 < 0.5:
            shape = "smooth"
        else:
            shape = "mixed"
        print(f"  Head {h}: |Δ|_avg={diffs.mean():.3f} jumps={n_jumps} smooth_L2={smooth_L2:.4f} → {shape}")

    # ── 5. Cross-block stability ──
    print("── 5. Cross-Block Stability ──")
    if num_blocks > 1 and model.blocks[1].cross_attns[0].use_time_bias:
        w1 = model.blocks[1].cross_attns[0].temporal_bias.weight.detach().cpu().numpy()
        for h in range(num_heads):
            corr = float(np.corrcoef(w0[1:, h], w1[1:, h])[0, 1])
            print(f"  Head {h}: corr(Block0,Block1)={corr:+.3f}")
        f0 = w0[1:, :].flatten()
        f1 = w1[1:, :].flatten()
        cos = float(np.dot(f0, f1) / (np.linalg.norm(f0) * np.linalg.norm(f1)))
        print(f"  cosine(Block0,Block1)={cos:+.4f}")
    else:
        print("  only 1 block → skip")

    # ── 6. RECOMMENDATION (most important — at end) ──
    print("=" * 72)
    print("  >>> RECOMMENDATION <<<")
    print(f"  signals: disagreement={mean_disagreement:.3f} overlap={avg_overlap:.2f} step_heads={n_step}/{num_heads}")
    print("=" * 72)

    # Exp31a
    if n_step >= 2:
        print("  [+] Exp31a (TimeBias MLP): jumps detected, MLP can smooth → TOP PICK")
    elif n_step == 0:
        print("  [~] Exp31a (TimeBias MLP): already smooth, MLP may add little")
    else:
        print("  [~] Exp31a (TimeBias MLP): mixed, mild potential")

    # Exp31c
    if mean_disagreement > 0.15 and avg_overlap < 0.3:
        print("  [+] Exp31c (multi-grain): heads differentiated, multi-grain can amplify")
    elif mean_disagreement < 0.05 or avg_overlap > 0.6:
        print("  [-] Exp31c (multi-grain): heads too similar, skip")
    else:
        print("  [~] Exp31c (multi-grain): moderate, keep as candidate")

    # Exp31d
    if mean_disagreement > 0.2:
        print("  [+] Exp31d (SelfAttn+TB): large divergence, worth extending")
    elif mean_disagreement < 0.05:
        print("  [-] Exp31d (SelfAttn+TB): heads identical, skip")
    else:
        print("  [~] Exp31d (SelfAttn+TB): lower priority than deepening PerHead")

    # ── 7. OPTIONAL Ablation (DIAGNOSE_ABLATE=true) ──
    run_ablate = os.environ.get('DIAGNOSE_ABLATE', '').lower() in ('true', '1', 'yes')
    if run_ablate:
        print("── 7. Ablation: Time Bias ON vs OFF ──")
        try:
            from sklearn.metrics import roc_auc_score
        except ImportError:
            print("  SKIP: sklearn not available")
            return

        nw = min(4, os.cpu_count() or 1)
        test_loader = DataLoader(
            test_dataset, batch_size=None, num_workers=nw,
            prefetch_factor=2, pin_memory=torch.cuda.is_available(),
        )

        def _collect_auc(m: PCVRHyFormer) -> float:
            probs_all, labels_all = [], []
            with torch.no_grad():
                for batch in test_loader:
                    mi = _batch_to_model_input(batch, device)
                    logits, _ = m.predict(mi)
                    p = torch.sigmoid(logits.squeeze(-1)).cpu().numpy()
                    lbl = batch.get('label')
                    if hasattr(lbl, 'numpy'):
                        lbl = lbl.numpy()
                    elif isinstance(lbl, list):
                        lbl = np.array(lbl)
                    elif lbl is None:
                        lbl = np.zeros(len(p))
                    probs_all.extend(p.tolist())
                    labels_all.extend(lbl.tolist())
            try:
                return float(roc_auc_score(labels_all, probs_all))
            except ValueError:
                return float('nan')

        auc_on = _collect_auc(model)

        saved = {}
        for b_idx, block in enumerate(model.blocks):
            for s_idx, ca in enumerate(block.cross_attns):
                if ca.use_time_bias:
                    saved[(b_idx, s_idx)] = ca.temporal_bias.weight.data.clone()
                    ca.temporal_bias.weight.data.zero_()

        auc_off = _collect_auc(model)
        for (b_idx, s_idx), w in saved.items():
            model.blocks[b_idx].cross_attns[s_idx].temporal_bias.weight.data.copy_(w)

        delta = auc_on - auc_off
        print(f"  AUC(ON) ={auc_on:.5f}  AUC(OFF)={auc_off:.5f}  Δ={delta:+.5f}")
        if delta > 0.003:
            print("  → time bias is significant, continue deepening")
        elif delta > 0.001:
            print("  → has effect, prefer stacking with orthogonal changes")
        else:
            print("  → little impact, pivot to orthogonal directions")

        # ── 8. Q4: Content vs Time awareness ──
        print("")
        print("── 8. Q4: Content vs Time awareness (shuffle ablation) ──")

        def _batch_shuffle_time(batch, dev):
             batch = {k: v.to(dev, non_blocking=True) if isinstance(v, torch.Tensor) else v
                      for k, v in batch.items()}
             for d in batch.get('_seq_domains', []):
                 tb_key = f'{d}_time_bucket'
                 if tb_key in batch:
                     tb = batch[tb_key]
                     idx = torch.randperm(tb.shape[0], device=dev)
                     batch[tb_key] = tb[idx]  # shuffle time across batch
             return _batch_to_model_input(batch, dev)

         def _batch_shuffle_content(batch, dev):
             batch = {k: v.to(dev, non_blocking=True) if isinstance(v, torch.Tensor) else v
                      for k, v in batch.items()}
             for d in batch.get('_seq_domains', []):
                 seq = batch[d]  # (B, n_feats, L)
                 B2, nf, L2 = seq.shape
                 for bb in range(B2):
                     perm = torch.randperm(L2, device=dev)
                     batch[d][bb] = seq[bb, :, perm]  # shuffle fid order within event
             return _batch_to_model_input(batch, dev)

        def _collect_auc_custom(m, loader, batch_fn):
            probs_all, labels_all = [], []
            with torch.no_grad():
                for batch in loader:
                    mi = batch_fn(batch, device)
                    logits, _ = m.predict(mi)
                    p = torch.sigmoid(logits.squeeze(-1)).cpu().numpy()
                    lbl = batch.get('label')
                    if hasattr(lbl, 'numpy'):
                        lbl = lbl.numpy()
                    elif isinstance(lbl, list):
                        lbl = np.array(lbl)
                    elif lbl is None:
                        lbl = np.zeros(len(p))
                    probs_all.extend(p.tolist())
                    labels_all.extend(lbl.tolist())
            try:
                return float(roc_auc_score(labels_all, probs_all))
            except ValueError:
                return float('nan')

        auc_time_shuffle = _collect_auc_custom(model, test_loader, _batch_shuffle_time)
        auc_fid_shuffle = _collect_auc_custom(model, test_loader, _batch_shuffle_content)

        time_drop = auc_on - auc_time_shuffle
        fid_drop = auc_on - auc_fid_shuffle
        print(f"  baseline          AUC = {auc_on:.6f}")
        print(f"  shuffle time      AUC = {auc_time_shuffle:.6f}  (drop = {time_drop:+.6f})")
        print(f"  shuffle fid order AUC = {auc_fid_shuffle:.6f}  (drop = {fid_drop:+.6f})")
        print("  --- interpretation ---")
        if time_drop > 0.001 and fid_drop < 0.0003:
            print("  model relies HEAVILY on time, barely on content")
            print("  → content interaction optimisations (FM Cross etc) may have room")
        elif fid_drop > 0.001 and time_drop < 0.0003:
            print("  model relies on content, time is weak")
            print("  → time bias direction fully exploited; content mining is the path")
        elif time_drop > 0.001 and fid_drop > 0.001:
            print("  model uses both — balanced")
            print("  → either direction valid, pick the cheaper one")
        else:
            print("  both drops small — model may be underfitting or relying on non-seq features")
            print("  → shift focus to non-sequence dimensions")

    print("=" * 72)
    print("  END. Set DIAGNOSE_MODE=false to run normal inference.")
    print("=" * 72)


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

    # ── Diagnostic mode ──
    if os.environ.get('DIAGNOSE_MODE', '').lower() in ('true', '1', 'yes'):
        _run_diagnose(model, test_dataset, device)
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
