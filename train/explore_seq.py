"""Sequence Feature Exploration — Full Data, Incremental, Single Pass.

Reuses ``train.py`` data loading (``TRAIN_DATA_PATH`` env var).
Runs Q1 (fid co-occurrence) and Q3 (behavioural patterns) on the FULL valid
set in ONE pass, using *only* incremental/numpy accumulators (no giant Python
lists).  All conclusions are collected and printed together at the end
(well within 1000 lines).
"""

import os
import sys
import time
import logging
import numpy as np
from collections import defaultdict
from torch.utils.data import DataLoader

from dataset import get_pcvr_data

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(message)s")
log = logging.getLogger("explore_seq")

# ═══════════════════════════════════════════════════════════════════════════════
# Q3 incremental stats helper  (Welford's algorithm — O(1) memory per metric)
# ═══════════════════════════════════════════════════════════════════════════════

class _RunningStats:
    """Welford single-pass mean / variance.  Memory: 3 floats."""
    __slots__ = ("n", "mean", "M2")

    def __init__(self):
        self.n = 0
        self.mean = 0.0
        self.M2 = 0.0

    def update(self, x: float):
        self.n += 1
        delta = x - self.mean
        self.mean += delta / self.n
        self.M2 += delta * (x - self.mean)

    @property
    def std(self) -> float:
        if self.n < 2:
            return 0.0
        return float(np.sqrt(self.M2 / self.n))


_BEHAVIOUR_METRICS = ["len", "density", "diversity"]


def main():
    t_start = time.time()

    # ── data loading ──
    data_dir = os.environ.get("TRAIN_DATA_PATH", "")
    if not data_dir:
        log.error("TRAIN_DATA_PATH env var is not set. Abort.")
        return

    schema_path = os.path.join(data_dir, "schema.json")
    log.info("Loading data from %s", data_dir)

    train_loader, valid_loader, dataset = get_pcvr_data(
        data_dir=data_dir,
        schema_path=schema_path,
        batch_size=256,
        valid_ratio=0.1,
        train_ratio=1.0,
        num_workers=0,
        buffer_batches=20,
    )
    domains = dataset.seq_domains
    sideinfo_fids = dataset.sideinfo_fids

    # Pre-build fid name lists for final report
    nf_per_domain = {d: len(sideinfo_fids[d]) for d in domains}
    total_samples = dataset.num_rows
    log.info("Dataset: %d rows, %d domains: %s", total_samples, len(domains), ",".join(domains))

    # ═══════════════════════════════════════════════════════════════════
    # Q1 accumulators  (numpy — tiny, ~420 ints)
    # ═══════════════════════════════════════════════════════════════════
    q1_cooc_pos = {}
    q1_cooc_neg = {}
    q1_total_event_slots_pos = {d: 0 for d in domains}
    q1_total_event_slots_neg = {d: 0 for d in domains}
    for d in domains:
        nf = nf_per_domain[d]
        q1_cooc_pos[d] = np.zeros((nf, nf), dtype=np.int64)
        q1_cooc_neg[d] = np.zeros((nf, nf), dtype=np.int64)

    # ═══════════════════════════════════════════════════════════════════
    # Q3 accumulators  (RunningStats + small lists for latest_bucket)
    # ═══════════════════════════════════════════════════════════════════
    q3_stats = {
        d: {
            tag: {m: _RunningStats() for m in _BEHAVIOUR_METRICS}
            for tag in ("pos", "neg")
        }
        for d in domains
    }
    q3_latest_bucket = {
        d: {"pos": [], "neg": []} for d in domains
    }

    # ═══════════════════════════════════════════════════════════════════
    # Single pass over shuffled data
    # ═══════════════════════════════════════════════════════════════════
    loader = DataLoader(dataset, batch_size=None, num_workers=0)
    total_processed = 0
    log.info("Running single-pass Q1+Q3 exploration...")

    for bi, batch in enumerate(loader):
        labels = batch["label"].numpy()
        B = labels.shape[0]

        for d in domains:
            seq = batch[d].numpy()  # (B, n_feats, L)
            _, nf, L = seq.shape
            tb = batch.get(f"{d}_time_bucket")
            tb_np = tb.numpy() if tb is not None else None

            # ── Q1: vectorised co-occurrence ──
            active = (seq > 0)  # (B, nf, L)
            for b in range(B):
                lbl = int(labels[b])
                tag = "pos" if lbl == 1 else "neg"
                if tag == "pos":
                    q1_total_event_slots_pos[d] += L
                else:
                    q1_total_event_slots_neg[d] += L
                for i in range(nf):
                    for j in range(i + 1, nf):
                        cooc = int((active[b, i, :] & active[b, j, :]).sum())
                        if cooc:
                            if tag == "pos":
                                q1_cooc_pos[d][i, j] += cooc
                            else:
                                q1_cooc_neg[d][i, j] += cooc

            # ── Q3: behavioural stats ──
            for b in range(B):
                lbl = int(labels[b])
                tag = "pos" if lbl == 1 else "neg"
                # length
                lens = (seq[b, :, :] > 0).sum(axis=1).astype(int)
                q3_stats[d][tag]["len"].update(float(lens.mean()))
                # density
                density = float(lens.sum()) / max(L * nf, 1)
                q3_stats[d][tag]["density"].update(density)
                # diversity
                unique_fids = len(set(int(v) for v in seq[b].ravel() if v > 0))
                q3_stats[d][tag]["diversity"].update(float(unique_fids))
                # latest_bucket
                if tb_np is not None:
                    active_b = tb_np[b][tb_np[b] > 0]
                    if len(active_b):
                        q3_latest_bucket[d][tag].append(float(active_b.min()))
                    else:
                        q3_latest_bucket[d][tag].append(float(L))  # no event → sentinel L

        total_processed += B
        if (bi + 1) % 500 == 0:
            pct = 100.0 * total_processed / max(total_samples, 1)
            elapsed = time.time() - t_start
            log.info("  progress: %d batches  %d samples (%.1f%%)  elapsed=%.0fs",
                     bi + 1, total_processed, pct, elapsed)

    elapsed_total = time.time() - t_start
    log.info("Data pass complete: %d samples in %.0fs (%.0f samples/s)",
             total_processed, elapsed_total, total_processed / max(elapsed_total, 1))

    # ═══════════════════════════════════════════════════════════════════
    # Build final report
    # ═══════════════════════════════════════════════════════════════════
    lines = []
    sep = "-" * 72

    def _emit(s: str):
        lines.append(s)

    _emit("=" * 72)
    _emit("SEQUENCE FEATURE EXPLORATION REPORT")
    _emit("=" * 72)
    _emit(f"Samples: {total_processed}  Domains: {len(domains)}  Time: {elapsed_total:.0f}s")
    _emit(sep)

    # ── Q1 ──
    _emit("")
    _emit("Q1: EVENT FID-PAIR CO-OCCURRENCE")
    _emit("    Are fid-pairs within the same event discriminative?")
    _emit(sep)

    for d in domains:
        nf = nf_per_domain[d]
        total_pairs = nf * (nf - 1) // 2
        tp = max(q1_total_event_slots_pos[d], 1)
        tn = max(q1_total_event_slots_neg[d], 1)

        sig_pairs = []
        for i in range(nf):
            for j in range(i + 1, nf):
                vp = q1_cooc_pos[d][i, j]
                vn = q1_cooc_neg[d][i, j]
                pr = vp / tp
                nr = vn / tn
                diff = abs(pr - nr)
                if diff > 1e-6:
                    sig_pairs.append((i, j, pr, nr, diff))

        sig_pairs.sort(key=lambda x: -x[4])
        n_sig = len(sig_pairs)
        ratio = 100.0 * n_sig / max(total_pairs, 1)

        _emit(f"  Domain {d}  ({nf} fids, {total_pairs} pairs)")
        _emit(f"    significant: {n_sig}/{total_pairs} ({ratio:.1f}%)")
        if sig_pairs:
            _emit("    top-10:")
            for i, j, pr, nr, diff in sig_pairs[:10]:
                fid_i = sideinfo_fids[d][i]
                fid_j = sideinfo_fids[d][j]
                _emit(f"      fid_{fid_i} x fid_{fid_j}   pos={pr:.6f}  neg={nr:.6f}  |diff|={diff:.6f}")

        if ratio > 30:
            _emit("    >>> RICH combinational signal  -> FM Cross STRONGLY recommended")
        elif ratio > 5:
            _emit("    >>> MODERATE signal              -> FM Cross worth trying")
        else:
            _emit("    >>> SPARSE signal               -> FM Cross may add little")
        _emit("")

    # ── Q3 ──
    _emit(sep)
    _emit("Q3: SEQ-LEVEL BEHAVIOURAL PATTERNS")
    _emit("    Do pos/neg samples differ in seq-level statistics?")
    _emit(sep)

    for d in domains:
        _emit(f"  Domain {d} ({nf_per_domain[d]} fids)")
        for metric in _BEHAVIOUR_METRICS:
            ps = q3_stats[d]["pos"][metric]
            ns = q3_stats[d]["neg"][metric]
            diff = ps.mean - ns.mean
            sep = abs(diff) / max((ps.std + ns.std) / 2, 1e-9)
            tag = "*** STRONG ***" if sep > 0.5 else ("** moderate **" if sep > 0.2 else "~ weak")
            _emit(f"    {metric:>14s}  pos={ps.mean:7.3f}+-{ps.std:6.3f}  "
                  f"neg={ns.mean:7.3f}+-{ns.std:6.3f}  diff={diff:+7.3f}  sep={sep:5.2f}  {tag}")

        # latest_bucket
        pv = q3_latest_bucket[d]["pos"]
        nv = q3_latest_bucket[d]["neg"]
        if pv and nv:
            # Strip sentinel values (L)
            pv_arr = np.array([v for v in pv if v < 1e5], dtype=np.float32)
            nv_arr = np.array([v for v in nv if v < 1e5], dtype=np.float32)
            if len(pv_arr) > 0 and len(nv_arr) > 0:
                pm, ps = pv_arr.mean(), pv_arr.std()
                nm, ns = nv_arr.mean(), nv_arr.std()
                diff = pm - nm
                sep = abs(diff) / max((ps + ns) / 2, 1e-9)
                tag = "*** STRONG ***" if sep > 0.5 else ("** moderate **" if sep > 0.2 else "~ weak")
                _emit(f"    {'latest_bucket':>14s}  pos={pm:7.3f}+-{ps:6.3f}  "
                      f"neg={nm:7.3f}+-{ns:6.3f}  diff={diff:+7.3f}  sep={sep:5.2f}  {tag}")
        _emit("")

    # ── Summary ──
    _emit("=" * 72)
    _emit("  SUMMARY & NEXT-STEP MAPPING")
    _emit(sep)

    # Q1 summary
    rich_domains = []
    moderate_domains = []
    for d in domains:
        nf = nf_per_domain[d]
        tp = max(q1_total_event_slots_pos[d], 1)
        tn = max(q1_total_event_slots_neg[d], 1)
        sig_count = 0
        for i in range(nf):
            for j in range(i + 1, nf):
                pr = q1_cooc_pos[d][i, j] / tp
                nr = q1_cooc_neg[d][i, j] / tn
                if abs(pr - nr) > 1e-6:
                    sig_count += 1
        ratio = 100.0 * sig_count / max(nf * (nf - 1) // 2, 1)
        if ratio > 30:
            rich_domains.append(d)
        elif ratio > 5:
            moderate_domains.append(d)

    if rich_domains:
        _emit(f"Q1: RICH co-occurrence in {','.join(rich_domains)} -> FM Cross STRONGLY recommended")
    if moderate_domains:
        _emit(f"Q1: MODERATE co-occurrence in {','.join(moderate_domains)} -> FM Cross worth trying")
    if not rich_domains and not moderate_domains:
        _emit("Q1: SPARSE co-occurrence across all domains -> skip FM Cross")

    # Q3 summary
    strong_metrics = []
    moderate_metrics = []
    for d in domains:
        for metric in _BEHAVIOUR_METRICS:
            ps = q3_stats[d]["pos"][metric]
            ns = q3_stats[d]["neg"][metric]
            sep = abs(ps.mean - ns.mean) / max((ps.std + ns.std) / 2, 1e-9)
            if sep > 0.5:
                strong_metrics.append(f"{d}/{metric}")
            elif sep > 0.2:
                moderate_metrics.append(f"{d}/{metric}")

    if strong_metrics:
        _emit(f"Q3: STRONG separation in {','.join(strong_metrics)} -> inject as features")
    if moderate_metrics:
        _emit(f"Q3: MODERATE separation in {','.join(moderate_metrics)} -> consider as features")
    if not strong_metrics and not moderate_metrics:
        _emit("Q3: weak separation across all metrics -> skip behavioural-feature injection")

    _emit(sep)
    _emit(f"Exploration complete.  Total time: {time.time() - t_start:.0f}s")
    _emit("=" * 72)

    # write out
    report = "\n".join(lines)
    sys.stdout.write(report + "\n")

    # also write to file
    out_path = os.environ.get("EXPLORE_OUTPUT", os.path.join(os.getcwd(), "explore_report.txt"))
    with open(out_path, "w") as f:
        f.write(report + "\n")
    log.info("Report saved to %s", out_path)


if __name__ == "__main__":
    main()
