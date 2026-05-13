"""Sequence Feature Exploration — Pure Data Analysis (no model needed).

Reuses the exact same data loading as train.py (TRAIN_DATA_PATH env var).
Runs Q1 (fid co-occurrence) and Q3 (behavioural patterns) on valid set.
"""

import os
import logging
import numpy as np
from collections import defaultdict
from torch.utils.data import DataLoader

from dataset import get_pcvr_data

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(message)s")
log = logging.getLogger("explore_seq")


def main():
    # ── data loading: EXACT same pattern as train.py ──
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
        buffer_batches=0,
    )
    log.info("Dataset: %d rows, %d domains: %s", dataset.num_rows, len(dataset.seq_domains),
             ",".join(dataset.seq_domains))

    # We use the valid DataLoader for analysis
    loader = DataLoader(dataset, batch_size=None, num_workers=0)
    domains = dataset.seq_domains
    sideinfo_fids = dataset.sideinfo_fids
    max_batches = 200

    # ═══════════════════════════════════════════════════════════════
    # Q1: Event fid-pair co-occurrence
    # ═══════════════════════════════════════════════════════════════
    log.info("=" * 60)
    log.info("Q1: Event fid-pair co-occurrence")
    log.info("=" * 60)

    pair_counts = {}
    for d in domains:
        nf = len(sideinfo_fids[d])
        pair_counts[d] = {"pos": defaultdict(int), "neg": defaultdict(int), "total_pos": 0, "total_neg": 0}

    total = 0
    for bi, batch in enumerate(loader):
        if bi >= max_batches:
            break
        labels = batch["label"].numpy()
        for d in domains:
            seq = batch[d].numpy()
            B, nf, L = seq.shape
            for b in range(B):
                lbl = int(labels[b])
                tag = "pos" if lbl == 1 else "neg"
                pair_counts[d][f"total_{tag}"] += L
                for slot in range(L):
                    active = [i for i in range(nf) if seq[b, i, slot] > 0]
                    for ai in range(len(active)):
                        for aj in range(ai + 1, len(active)):
                            pair_counts[d][tag][(active[ai], active[aj])] += 1
        total += B
        if (bi + 1) % 50 == 0:
            log.info("  Q1 %d batches (%d samples)", bi + 1, total)

    for d in domains:
        pfids = sideinfo_fids[d]
        nf = len(pfids)
        total_pairs = nf * (nf - 1) // 2
        pos = pair_counts[d]["pos"]
        neg = pair_counts[d]["neg"]
        tp = max(pair_counts[d]["total_pos"], 1)
        tn = max(pair_counts[d]["total_neg"], 1)

        significant = []
        for i in range(nf):
            for j in range(i + 1, nf):
                pr = pos.get((i, j), 0) / tp
                nr = neg.get((i, j), 0) / tn
                diff = abs(pr - nr)
                if diff > 1e-4:
                    significant.append((i, j, pr, nr, diff))

        significant.sort(key=lambda x: -x[4])
        n_sig = len(significant)
        n_sig_ratio = n_sig / max(total_pairs, 1)

        log.info("  --- %s: %d/%d significant (%.1f%%) ---", d, n_sig, total_pairs, 100 * n_sig_ratio)
        for i, j, pr, nr, diff in significant[:10]:
            log.info("    fid_%d x fid_%d  pos=%.4f  neg=%.4f  |diff|=%.4f", pfids[i], pfids[j], pr, nr, diff)
        if n_sig_ratio > 0.3:
            log.info("    -> RICH combinational signal -> FM Cross recommended")
        elif n_sig_ratio > 0.05:
            log.info("    -> MODERATE -> FM Cross worth trying")
        else:
            log.info("    -> SPARSE -> FM Cross may add little")

    # ═══════════════════════════════════════════════════════════════
    # Q3: Seq-level behavioural patterns
    # ═══════════════════════════════════════════════════════════════
    log.info("=" * 60)
    log.info("Q3: Seq-level behavioural patterns")
    log.info("=" * 60)

    stats = {d: {"pos": defaultdict(list), "neg": defaultdict(list)} for d in domains}

    total = 0
    for bi, batch in enumerate(loader):
        if bi >= max_batches:
            break
        labels = batch["label"].numpy()
        for d in domains:
            seq = batch[d].numpy()
            tb = batch.get(f"{d}_time_bucket")
            if tb is not None:
                tb = tb.numpy()
            B, nf, L = seq.shape
            for b in range(B):
                lbl = int(labels[b])
                tag = "pos" if lbl == 1 else "neg"
                lens = [(seq[b, i, :] > 0).sum() for i in range(nf)]
                stats[d][tag]["len"].append(int(np.mean(lens)))
                density = (seq[b] > 0).sum() / max(L * nf, 1)
                stats[d][tag]["density"].append(float(density))
                unique_fids = len(set(seq[b].ravel().tolist()) - {0})
                stats[d][tag]["diversity"].append(unique_fids)
                if tb is not None:
                    active_buckets = tb[b][tb[b] > 0]
                    if len(active_buckets):
                        stats[d][tag]["latest_bucket"].append(float(active_buckets.min()))
        total += B
        if (bi + 1) % 50 == 0:
            log.info("  Q3 %d batches (%d samples)", bi + 1, total)

    for d in domains:
        log.info("  --- %s ---", d)
        for metric in ["len", "density", "diversity", "latest_bucket"]:
            pv = stats[d]["pos"].get(metric, [])
            nv = stats[d]["neg"].get(metric, [])
            if not pv or not nv:
                continue
            pm, ps = np.mean(pv), np.std(pv)
            nm, ns = np.mean(nv), np.std(nv)
            diff = pm - nm
            sep = abs(diff) / max((ps + ns) / 2, 1e-9)
            log.info("    %-16s  pos=%.3f+-%.3f  neg=%.3f+-%.3f  diff=%+.3f  sep=%.2f",
                     metric, pm, ps, nm, ns, diff, sep)
            if sep > 0.5:
                log.info("      ^ STRONG separation - consider as feature")
            elif sep > 0.2:
                log.info("      ^ MODERATE separation")

    log.info("=" * 60)
    log.info("DONE. Conclusions:")
    log.info("  Q1 rich co-occurrence -> FM Cross (event-internal fid interaction)")
    log.info("  Q3 strong stat sep    -> inject behavioural statistics as features")
    log.info("=" * 60)


if __name__ == "__main__":
    main()
