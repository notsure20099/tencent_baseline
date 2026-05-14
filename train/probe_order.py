"""Sequence Position Order Probe — is position 0 the most recent or oldest event?

Samples 30 batches, prints per-position avg time_bucket.  No labels, no model.
"""

import os, sys, logging
import numpy as np
from collections import defaultdict
from torch.utils.data import DataLoader
from dataset import get_pcvr_data

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(message)s")
log = logging.getLogger("probe")

BATCHES = 30

def main():
    data_dir = os.environ.get("TRAIN_DATA_PATH", "")
    if not data_dir:
        log.error("TRAIN_DATA_PATH is not set"); return
    schema_path = os.path.join(data_dir, "schema.json")

    _, _, dataset = get_pcvr_data(
        data_dir=data_dir, schema_path=schema_path,
        batch_size=256, valid_ratio=0.0, train_ratio=1.0,
        num_workers=0, buffer_batches=0, shuffle_train=False,
        seq_max_lens={"seq_a": 512, "seq_b": 512, "seq_c": 512, "seq_d": 512},
    )
    domains = dataset.seq_domains
    log.info("Domains: %s  Samples: %d", ",".join(domains), dataset.num_rows)

    # accumulators: per-domain, per-position → sum of buckets, count
    acc = {d: {"sum": defaultdict(float), "cnt": defaultdict(int)} for d in domains}

    loader = DataLoader(dataset, batch_size=None, num_workers=0)
    processed = 0
    for bi, batch in enumerate(loader):
        if bi >= BATCHES:
            break
        for d in domains:
            tb = batch.get(f"{d}_time_bucket")
            if tb is None:
                continue
            tb_np = tb.numpy()  # (B, L)
            B, L = tb_np.shape
            for pos in range(L):
                col = tb_np[:, pos]
                valid = col[col > 0]
                if len(valid):
                    acc[d]["sum"][pos] += float(valid.sum())
                    acc[d]["cnt"][pos] += len(valid)
        processed += B
        log.info("  batch %d/%d  %d samples", bi + 1, BATCHES, processed)

    # ── Report ──
    print("\n" + "=" * 72)
    print("  SEQUENCE ORDER PROBE — avg time_bucket per position")
    print("  (small bucket ≈ recent, large bucket ≈ old)")
    print("  (if bucket increases with position → position 0 = most recent)")
    print("=" * 72)

    for d in domains:
        print(f"\n  Domain {d}:")
        pos_list = sorted(acc[d]["cnt"].keys())
        if not pos_list:
            print("    (no data)")
            continue
        # Print first 10, last 10, and every 50th in between
        key_pos = set()
        for p in pos_list[:10]:
            key_pos.add(p)
        for p in pos_list[-10:]:
            key_pos.add(p)
        for p in range(50, max(pos_list), 50):
            if p in acc[d]["cnt"]:
                key_pos.add(p)
        key_pos = sorted(key_pos)

        vals = []
        for p in key_pos:
            s = acc[d]["sum"][p]
            c = acc[d]["cnt"][p]
            avg = s / c if c else 0.0
            vals.append(f"p{p}={avg:.1f}")
        print(f"    {'  '.join(vals)}")
        # trend
        all_vals = [acc[d]["sum"].get(p, 0) / max(acc[d]["cnt"].get(p, 1), 1)
                    for p in pos_list]
        first_m = float(np.mean(all_vals[:5])) if len(all_vals) >= 5 else 0
        last_m  = float(np.mean(all_vals[-5:])) if len(all_vals) >= 5 else 0
        trend = "recent→old" if last_m > first_m + 0.5 else ("old→recent" if first_m > last_m + 0.5 else "flat")
        print(f"    first5_avg={first_m:.1f}  last5_avg={last_m:.1f}  → {trend}")

    print("\n" + "=" * 72)
    print("  INTERPRETATION")
    print("  recent→old → position 0 = most recent  → alpha boosts OLD events")
    print("  old→recent → position 0 = oldest       → alpha boosts RECENT events")
    print("=" * 72)


if __name__ == "__main__":
    main()
