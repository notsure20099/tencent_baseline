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


def _run_capacity_diagnose(model):
    """Diagnose Exp32_capacity model — pure weight inspection, no data."""
    import numpy as np
    num_blocks = len(model.blocks)
    num_seqs = model.blocks[0].num_sequences
    num_heads = model.blocks[0].cross_attns[0].attn.num_heads
    num_queries = model.blocks[0].num_queries

    log = logging.getLogger("diagnose")
    sep = "-" * 62

    def _p(s):
        logging.info(s)

    _p("=" * 62)
    _p("Exp32_capacity_rebalance — Model Diagnosis")
    _p(f"blocks={num_blocks}  sequences={num_seqs}  heads={num_heads}  queries={num_queries}")
    _p(sep)

    # ── 1. Domain Gates ──
    _p("1. DOMAIN GATES (per-block)")
    for bi, block in enumerate(model.blocks):
        gates = block.domain_gates.detach().cpu().numpy()  # (S,)
        labels = ",".join(f"{g:+.3f}" for g in gates)
        _p(f"  Block{bi}: [{labels}]  sum={gates.sum():.3f}  range=[{gates.min():+.3f},{gates.max():+.3f}]")
        # which domain is strongest/weakest
        top = int(np.argmax(gates))
        bot = int(np.argmin(gates))
        _p(f"    strongest=seq_{top}  weakest=seq_{bot}  ratio={gates[top]/max(abs(gates[bot]),1e-6):.1f}x")
    _p("")

    # ── 2. Q Token FFN divergence (Q0 full vs Q1 tail) ──
    if hasattr(model, 'query_generator'):
        qg = model.query_generator
        if hasattr(qg, 'query_ffns_per_seq') and num_queries >= 2:
            _p("2. Q TOKEN FFN DIVERGENCE (Q0-full vs Q1-tail)")
            for si in range(num_seqs):
                w0 = qg.query_ffns_per_seq[si][0][0].weight  # first Linear of Q0
                w1 = qg.query_ffns_per_seq[si][1][0].weight  # first Linear of Q1
                cos = float(torch.cosine_similarity(w0.flatten(), w1.flatten(), dim=0))
                l2 = float((w0 - w1).norm())
                _p(f"  seq_{si}: cosine(Q0,Q1)={cos:+.4f}  ||W0-W1||={l2:.3f}")
            _p("")
    else:
        _p("2. Q TOKEN DIVERGENCE: query_generator not found, skipping")
        _p("")

    # ── 3. Per-Head Time Bias ──
    _p("3. PER-HEAD TIME BIAS (Block0 CrossAttn0)")
    ca = model.blocks[0].cross_attns[0]
    if ca.use_time_bias:
        w0 = ca.temporal_bias.weight.detach().cpu().numpy()  # (65, H)
        for h in range(num_heads):
            vals = w0[1:, h]  # exclude padding bucket
            top3 = np.argsort(vals)[::-1][:3]
            top_labels = ",".join(f"b{ti+1}:{vals[ti]:+.2f}" for ti in top3)
            _p(f"  H{h}: mean={vals.mean():+.3f} std={vals.std():.3f} [{vals.min():+.2f},{vals.max():+.2f}] top3={{{top_labels}}}")

        # overlap
        top5_sets = [set(np.argsort(w0[1:, h])[::-1][:5]) for h in range(num_heads)]
        overlaps = []
        for h1 in range(num_heads):
            for h2 in range(h1 + 1, num_heads):
                j = len(top5_sets[h1] & top5_sets[h2]) / max(1, len(top5_sets[h1] | top5_sets[h2]))
                overlaps.append(j)
        avg_ov = float(np.mean(overlaps)) if overlaps else 1.0
        _p(f"  head overlap (Jaccard top5): {avg_ov:.2f}  (0=fully distinct, 1=identical)")

        # curve shape
        n_step = 0
        for h in range(num_heads):
            vals = w0[1:, h]
            diffs = np.abs(np.diff(vals))
            n_jumps = int((diffs > 2 * float(diffs.mean())).sum())
            if n_jumps >= 3:
                n_step += 1
        _p(f"  step heads: {n_step}/{num_heads}  (>2 jumps = step)")
        _p("")
    else:
        _p("3. PER-HEAD TIME BIAS: disabled, skipping")
        _p("")

    # ── 4. Cross-Block Time Bias Stability ──
    if num_blocks > 1:
        _p("4. CROSS-BLOCK TIME BIAS STABILITY")
        for s_idx in range(num_seqs):
            ca0 = model.blocks[0].cross_attns[s_idx]
            ca1 = model.blocks[1].cross_attns[s_idx]
            if not ca0.use_time_bias or not ca1.use_time_bias:
                _p(f"  seq_{s_idx}: time_bias disabled in one block, skip")
                continue
            w0 = ca0.temporal_bias.weight.detach().cpu().numpy()
            w1 = ca1.temporal_bias.weight.detach().cpu().numpy()
            f0, f1 = w0[1:, :].flatten(), w1[1:, :].flatten()
            cos = float(np.dot(f0, f1) / (np.linalg.norm(f0) * np.linalg.norm(f1)))
            corrs = [float(np.corrcoef(w0[1:, h], w1[1:, h])[0, 1]) for h in range(num_heads)]
            _p(f"  seq_{s_idx}: cosine={cos:+.4f}  per-head corr={[f'{c:+.3f}' for c in corrs]}")
        _p("")

    # ── 5. QUERY GENERATOR weight norms ──
    if hasattr(model, 'query_generator'):
        qg = model.query_generator
        _p("5. QUERY GENERATOR WEIGHT NORMS")
        for si in range(num_seqs):
            norms = []
            for q in range(num_queries):
                w = qg.query_ffns_per_seq[si][q][0].weight
                norms.append(float(w.norm()))
            _p(f"  seq_{si}: " + "  ".join(f"Q{q}:{n:.3f}" for q, n in enumerate(norms)))
        _p("")

    # ── Summary ──
    _p(sep)
    _p("SUMMARY & INTERPRETATION")
    _p(sep)

    issues = []

    # domain gate check
    for bi, block in enumerate(model.blocks):
        gates = block.domain_gates.detach().cpu().numpy()
        if abs(float(gates.sum()) - 4.0) > 1.0:
            issues.append(f"domain gates in Block{bi} sum={gates.sum():.2f} (far from 4.0)")
        r = gates.max() / max(abs(gates.min()), 1e-6)
        if r < 1.5:
            issues.append(f"domain gates in Block{bi} nearly uniform (ratio={r:.1f}x)")

    # Q token divergence check
    if hasattr(model, 'query_generator') and num_queries >= 2:
        qg = model.query_generator
        cos_vals = []
        for si in range(num_seqs):
            w0 = qg.query_ffns_per_seq[si][0][0].weight
            w1 = qg.query_ffns_per_seq[si][1][0].weight
            cos_vals.append(float(torch.cosine_similarity(w0.flatten(), w1.flatten(), dim=0)))
        avg_cos = sum(cos_vals)/len(cos_vals)
        if avg_cos > 0.95:
            issues.append(f"Q0-Q1 nearly identical (avg cosine={avg_cos:.4f}) — extra Q wasted")
        elif avg_cos < 0.5:
            _p(f"  [+] Q0-Q1 well differentiated (avg cosine={avg_cos:.4f})")
        else:
            _p(f"  [~] Q0-Q1 moderately distinct (avg cosine={avg_cos:.4f})")

    # time bias check
    if model.blocks[0].cross_attns[0].use_time_bias:
        w0 = model.blocks[0].cross_attns[0].temporal_bias.weight.detach().cpu().numpy()
        top5_sets = [set(np.argsort(w0[1:, h])[::-1][:5]) for h in range(num_heads)]
        ov = []
        for h1 in range(num_heads):
            for h2 in range(h1+1, num_heads):
                ov.append(len(top5_sets[h1]&top5_sets[h2])/max(1,len(top5_sets[h1]|top5_sets[h2])))
        avg_ov = float(np.mean(ov)) if ov else 1.0
        if avg_ov < 0.15:
            _p(f"  [+] heads highly differentiated (overlap={avg_ov:.2f})")
        elif avg_ov < 0.4:
            _p(f"  [~] heads moderately distinct (overlap={avg_ov:.2f})")
        else:
            issues.append(f"heads too similar (overlap={avg_ov:.2f}) — multi-head wasted")

    # cross-block
    if num_blocks > 1:
        cos_vals = []
        for s_idx in range(num_seqs):
            ca0 = model.blocks[0].cross_attns[s_idx]
            ca1 = model.blocks[1].cross_attns[s_idx]
            if ca0.use_time_bias and ca1.use_time_bias:
                w0 = ca0.temporal_bias.weight.detach().cpu().numpy()[1:,:].flatten()
                w1 = ca1.temporal_bias.weight.detach().cpu().numpy()[1:,:].flatten()
                cos_vals.append(float(np.dot(w0,w1)/(np.linalg.norm(w0)*np.linalg.norm(w1))+1e-9))
        if cos_vals:
            avg_cb_cos = sum(cos_vals)/len(cos_vals)
            if avg_cb_cos > 0.8:
                issues.append(f"cross-block time bias too similar ({avg_cb_cos:.2f}) — 2nd block may be redundant")
            else:
                _p(f"  [+] cross-block time bias differentiated (cos={avg_cb_cos:.3f})")

    if issues:
        _p("  Issues found:")
        for iss in issues:
            _p(f"    [!] {iss}")
    else:
        _p("  No major issues detected. Model structure looks healthy.")
    _p("=" * 62)


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

    # ── Diagnostic mode: inspect model weights (no eval) ──
    if os.environ.get('CAPACITY_DIAGNOSE', '').lower() in ('true', '1', 'yes'):
        _run_capacity_diagnose(model)
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
