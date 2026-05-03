"""
PCVR 训练集数据探索工具 (AngelML 平台专用 · 流式版本)

- 通过 TRAIN_DATA_PATH 环境变量读取数据路径
- 逐 batch 流式统计，不缓存全量数据，避免 OOM
- 全部计算完成后一次性输出，控制行数在平台限制内

用法:
    python explore_data.py
    python explore_data.py --sample 0.1
"""

import os
import sys
import json
import math
import time
import argparse
from datetime import datetime
from collections import defaultdict
from typing import Any, Dict, List

import numpy as np
import pyarrow.parquet as pq


def load_schema(path: str) -> Dict[str, Any]:
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)


def get_parquet_files(data_dir: str) -> List[str]:
    if os.path.isfile(data_dir) and data_dir.endswith('.parquet'):
        return [data_dir]
    files = sorted([os.path.join(data_dir, f)
                    for f in os.listdir(data_dir) if f.endswith('.parquet')])
    if not files:
        raise FileNotFoundError(f'No .parquet files found in {data_dir}')
    return files


def fm(n: int) -> str:
    return f"{n:,}"


# ═══════════════════════════ 流式统计累加器 ═══════════════════════════


class IntFeatureStats:
    def __init__(self, fid: int, vocab_size: int, dim: int):
        self.fid = fid
        self.vocab_size = vocab_size
        self.dim = dim
        self.total_slots = 0
        self.nonzero_count = 0
        self.max_val = -(2 ** 63)
        self.oob_count = 0
        self._mean = 0.0
        self._n = 0

    def update(self, arr: np.ndarray, oob_arr: np.ndarray):
        self.total_slots += len(arr)
        if self.vocab_size > 0:
            self.oob_count += int(oob_arr.sum())
        nonzero = arr[arr > 0]
        self.nonzero_count += len(nonzero)
        if len(nonzero) > 0:
            mx = int(nonzero.max())
            if mx > self.max_val:
                self.max_val = mx
        for v in nonzero:
            self._n += 1
            self._mean += (float(v) - self._mean) / self._n

    @property
    def zero_rate(self) -> float:
        return 1.0 - self.nonzero_count / self.total_slots if self.total_slots > 0 else 1.0

    @property
    def mean_nonzero(self) -> float:
        return self._mean if self._n > 0 else 0.0

    def line(self) -> str:
        mx = fm(self.max_val) if self.max_val > -(2 ** 63) else "N/A"
        return (f"  {self.fid:>6} {fm(self.vocab_size):>10} {self.dim:>4}"
                f"  {self.zero_rate:>8.4f}  {self.mean_nonzero:>10.1f}"
                f"  {mx:>10}  {fm(self.oob_count):>8}")


class DenseFeatureStats:
    def __init__(self, fid: int, dim: int):
        self.fid = fid
        self.dim = dim
        self.count = 0
        self.min_val = float('inf')
        self.max_val = float('-inf')
        self.zero_count = 0
        self._mean = 0.0
        self._m2 = 0.0
        self._n = 0

    def update(self, arr: np.ndarray):
        self.count += len(arr)
        self.zero_count += int((arr == 0).sum())
        for v in arr:
            vf = float(v)
            if vf < self.min_val: self.min_val = vf
            if vf > self.max_val: self.max_val = vf
            self._n += 1
            delta = vf - self._mean
            self._mean += delta / self._n
            self._m2 += delta * (vf - self._mean)

    @property
    def std_val(self) -> float:
        return math.sqrt(self._m2 / self._n) if self._n > 1 else 0.0

    @property
    def zero_rate(self) -> float:
        return self.zero_count / self.count if self.count > 0 else 1.0

    def line(self) -> str:
        return (f"  {self.fid:>6} {self.dim:>5}"
                f"  {self._mean:>12.6f}  {self.std_val:>12.6f}"
                f"  {self.min_val:>12.6f}  {self.max_val:>12.6f}"
                f"  {self.zero_rate:>9.4f}")


# ═══════════════════════════ 列提取 ═══════════════════════════


def extract_int_col(batch, col_name: str, dim: int) -> np.ndarray:
    col = batch.column(col_name)
    if dim == 1:
        arr = col.fill_null(0).to_numpy(zero_copy_only=False).astype(np.int64)
        arr[arr <= 0] = 0
        return arr
    offsets = col.offsets.to_numpy()
    vals = col.values.to_numpy()
    B = len(offsets) - 1
    out = np.zeros((B, dim), dtype=np.int64)
    for i in range(B):
        s, e = int(offsets[i]), int(offsets[i + 1])
        if e <= s: continue
        ul = min(e - s, dim)
        out[i, :ul] = vals[s:s + ul]
    out[out <= 0] = 0
    return out.ravel()


def extract_dense_col(batch, col_name: str, dim: int) -> np.ndarray:
    col = batch.column(col_name)
    offsets = col.offsets.to_numpy()
    vals = col.values.to_numpy().astype(np.float64)
    B = len(offsets) - 1
    out = np.zeros((B, dim), dtype=np.float64)
    for i in range(B):
        s, e = int(offsets[i]), int(offsets[i + 1])
        if e <= s: continue
        ul = min(e - s, dim)
        out[i, :ul] = vals[s:s + ul]
    return out.ravel()


# ═══════════════════════════ 流式数据迭代 ═══════════════════════════


def iter_batches(files: List[str], sample_rgs: int):
    rg_count = 0
    for f in files:
        if rg_count >= sample_rgs:
            break
        pf = pq.ParquetFile(f)
        for rg_idx in range(pf.metadata.num_row_groups):
            if rg_count >= sample_rgs:
                break
            for batch in pf.iter_batches(batch_size=65536, row_groups=[rg_idx]):
                yield batch
            rg_count += 1


# ═══════════════════════════ 统计计算 (无打印) ═══════════════════════════


def compute_label_dist(files, sample_rgs, columns) -> dict:
    total = 0
    pos = 0
    has = 'label_type' in columns
    for batch in iter_batches(files, sample_rgs):
        total += batch.num_rows
        if has:
            labels = (batch.column('label_type').fill_null(0)
                      .to_numpy(zero_copy_only=False).astype(np.int64) == 2).astype(np.int64)
            pos += int(labels.sum())
    return {'total': total, 'pos': pos}


def compute_int_features(feature_list, col_prefix, columns, files, sample_rgs) -> Dict[int, IntFeatureStats]:
    stats = {}
    for fid, vs, dim in feature_list:
        if f'{col_prefix}_{fid}' in columns:
            stats[fid] = IntFeatureStats(fid, vs, dim)
    for batch in iter_batches(files, sample_rgs):
        for fid, vs, dim in feature_list:
            st = stats.get(fid)
            if st is None: continue
            arr = extract_int_col(batch, f'{col_prefix}_{fid}', dim)
            oob = (arr >= vs) if vs > 0 else np.zeros(len(arr), dtype=bool)
            st.update(arr, oob)
    return stats


def compute_dense_features(feature_list, col_prefix, columns, files, sample_rgs) -> Dict[int, DenseFeatureStats]:
    if not feature_list:
        return {}
    stats = {}
    for fid, dim in feature_list:
        if f'{col_prefix}_{fid}' in columns:
            stats[fid] = DenseFeatureStats(fid, dim)
    for batch in iter_batches(files, sample_rgs):
        for fid, dim in feature_list:
            st = stats.get(fid)
            if st is None: continue
            st.update(extract_dense_col(batch, f'{col_prefix}_{fid}', dim))
    return stats


def compute_seq_features(seq_cfg, columns, files, sample_rgs) -> dict:
    result = {}
    for domain, cfg in sorted(seq_cfg.items()):
        prefix = cfg['prefix']
        feats = cfg['features']
        seq_count = 0
        seq_sum = 0.0
        seq_min = float('inf')
        seq_max = float('-inf')
        sample = []
        max_sample = 10000
        oob_by = defaultdict(int)
        total_by = defaultdict(int)
        first_col = f'{prefix}_{feats[0][0]}'

        for batch in iter_batches(files, sample_rgs):
            if first_col in columns:
                offsets = batch.column(first_col).offsets.to_numpy()
                for i in range(len(offsets) - 1):
                    sl = int(offsets[i + 1]) - int(offsets[i])
                    seq_count += 1; seq_sum += sl
                    if sl < seq_min: seq_min = sl
                    if sl > seq_max: seq_max = sl
                    if len(sample) < max_sample: sample.append(sl)
            for fid, vs in feats:
                cn = f'{prefix}_{fid}'
                if cn not in columns or vs <= 0: continue
                vals = batch.column(cn).values.to_numpy()
                oob_by[fid] += int((vals >= vs).sum())
                total_by[fid] += len(vals)

        result[domain] = {
            'prefix': prefix, 'ts_fid': cfg['ts_fid'], 'features': feats,
            'seq_count': seq_count, 'seq_sum': seq_sum,
            'seq_min': seq_min, 'seq_max': seq_max,
            'seq_sample': sample, 'oob_by': dict(oob_by), 'total_by': dict(total_by),
        }
    return result


def compute_timestamp(files, sample_rgs, columns) -> dict:
    if 'timestamp' not in columns:
        return {'found': False}
    t_min = float('inf')
    t_max = float('-inf')
    total = 0
    for batch in iter_batches(files, sample_rgs):
        ts = batch.column('timestamp').to_numpy().astype(np.int64)
        total += len(ts)
        bmin, bmax = int(ts.min()), int(ts.max())
        if bmin < t_min: t_min = bmin
        if bmax > t_max: t_max = bmax
    return {'found': True, 'total': total, 'min': int(t_min), 'max': int(t_max)}


# ═══════════════════════════ 输出 (一次性打印) ═══════════════════════════


def _h(title: str) -> str:
    return f"\n{'='*70}\n  {title}\n{'='*70}"


def render_overview(schema, files, sample_ratio, sample_rgs, sample_rows,
                    total_rgs, total_rows, columns, data_dir) -> List[str]:
    ui = schema['user_int']; ii = schema['item_int']
    ud = schema['user_dense']; id_ = schema.get('item_dense', [])
    seq = schema.get('seq', {})
    lines = [_h("0. Dataset Overview")]
    lines.append(f"  Data dir  : {data_dir}")
    lines.append(f"  Files     : {len(files)}  |  RowGroups: {total_rgs}  |  "
                 f"Total rows: {fm(total_rows)}")
    lines.append(f"  Sampled   : {fm(sample_rows)} rows ({sample_rgs} RGs, ratio={sample_ratio})")
    lines.append(f"  Columns   : {len(columns)}")
    lines.append(f"  user_int  : {len(ui):>3} feats  (dim={sum(d for _,_,d in ui)})")
    lines.append(f"  item_int  : {len(ii):>3} feats  (dim={sum(d for _,_,d in ii)})")
    lines.append(f"  user_dense: {len(ud):>3} feats  (dim={sum(d for _,d in ud)})")
    lines.append(f"  item_dense: {len(id_):>3} feats  (dim={sum(d for _,d in id_)})")
    lines.append(f"  seq domains: {len(seq)}  ({', '.join(sorted(seq.keys()))})")
    return lines


def render_label(label_res: dict) -> List[str]:
    t, pos = label_res['total'], label_res['pos']
    neg = t - pos
    lines = [_h("1. Label Distribution")]
    lines.append(f"  Total   : {fm(t)}")
    lines.append(f"  Pos (1) : {fm(pos)}  ({pos/t:.4%})" if t else "  Pos (1) : 0")
    lines.append(f"  Neg (0) : {fm(neg)}  ({neg/t:.4%})" if t else "  Neg (0) : 0")
    if pos:
        lines.append(f"  Ratio   : 1:{neg/pos:.1f}")
    return lines


def render_int(stats: Dict[int, IntFeatureStats], feature_list, col_prefix) -> List[str]:
    lines = [_h(f"2. {col_prefix} Features")]
    lines.append(f"  {'fid':>6} {'vocab':>10} {'dim':>4}"
                 f"  {'zero%':>8}  {'mean(nz)':>10}  {'max':>10}  {'OOB':>8}")
    for fid, vs, dim in feature_list:
        st = stats.get(fid)
        if st:
            lines.append(st.line())
        else:
            lines.append(f"  {fid:>6} {fm(vs):>10} {dim:>4}  {'MISSING':>8}")
    return lines


def render_dense(stats: Dict[int, DenseFeatureStats], feature_list, col_prefix) -> List[str]:
    if not feature_list:
        return []
    lines = [_h(f"3. {col_prefix} Features")]
    lines.append(f"  {'fid':>6} {'dim':>5}"
                 f"  {'mean':>12}  {'std':>12}  {'min':>12}  {'max':>12}  {'zero%':>9}")
    for fid, dim in feature_list:
        st = stats.get(fid)
        if st:
            lines.append(st.line())
        else:
            lines.append(f"  {fid:>6} {dim:>5}  {'MISSING':>12}")
    return lines


def render_seq(seq_res: dict, columns) -> List[str]:
    lines = [_h("4. Sequence Features")]
    for domain, r in seq_res.items():
        feats = r['features']
        lines.append(f"\n  Domain: {domain}  prefix={r['prefix']}  ts_fid={r['ts_fid']}"
                     f"  feats={len(feats)}")
        sc = r['seq_count']
        if sc > 0:
            m = r['seq_sum'] / sc
            lines.append(f"  Sequences: {fm(sc)}  "
                         f"mean={m:.1f}  min={r['seq_min']:.0f}  max={r['seq_max']:.0f}")
            if r['seq_sample']:
                sa = np.array(sorted(r['seq_sample']))
                lines.append(f"  Percentiles(sampled): "
                             f"p50={np.percentile(sa,50):.0f}  "
                             f"p90={np.percentile(sa,90):.0f}  "
                             f"p95={np.percentile(sa,95):.0f}  "
                             f"p99={np.percentile(sa,99):.0f}")
        else:
            lines.append(f"  [No sequence data]")
        lines.append(f"  {'fid':>6} {'vocab':>10} {'OOB':>8} {'OOB%':>10}")
        for fid, vs in feats:
            cn = f'{r["prefix"]}_{fid}'
            oob = r['oob_by'].get(fid, 0)
            tot = r['total_by'].get(fid, 1)
            st = "OK" if cn in columns else "MISS"
            lines.append(f"  {fid:>6} {fm(vs):>10} {fm(oob):>8}"
                         f" {oob/tot:>9.4f}  {st}" if tot else f"  {fid:>6} {fm(vs):>10} {'--':>8} ----  {st}")
    return lines


def render_timestamp(ts_res: dict) -> List[str]:
    lines = [_h("5. Timestamp Statistics")]
    if not ts_res['found']:
        lines.append("  Column 'timestamp' not found.")
        return lines
    range_d = (ts_res['max'] - ts_res['min']) / 86400
    dmin = datetime.fromtimestamp(ts_res['min'])
    dmax = datetime.fromtimestamp(ts_res['max'])
    lines.append(f"  Count : {fm(ts_res['total'])}")
    lines.append(f"  Min   : {ts_res['min']}  ({dmin})")
    lines.append(f"  Max   : {ts_res['max']}  ({dmax})")
    lines.append(f"  Range : {range_d:.1f} days")
    return lines


# ═══════════════════════════ 主流程 ═══════════════════════════


def main():
    parser = argparse.ArgumentParser(description='PCVR Data Explorer')
    parser.add_argument('--sample', type=float, default=1.0)
    parser.add_argument('--data_dir', type=str, default=None)
    args = parser.parse_args()

    data_dir = args.data_dir or os.environ.get('TRAIN_DATA_PATH')
    if not data_dir:
        print("ERROR: TRAIN_DATA_PATH not set"); sys.exit(1)
    sp = os.path.join(data_dir, 'schema.json')
    if not os.path.exists(sp):
        print(f"ERROR: schema.json not found"); sys.exit(1)

    t0 = time.time()
    schema = load_schema(sp)
    files = get_parquet_files(data_dir)

    # 元信息
    total_rgs = 0; total_rows_meta = 0
    rg_sizes = []
    for f in files:
        pf = pq.ParquetFile(f)
        total_rgs += pf.metadata.num_row_groups
        total_rows_meta += pf.metadata.num_rows
        for i in range(pf.metadata.num_row_groups):
            rg_sizes.append(pf.metadata.row_group(i).num_rows)

    n_sample_rgs = max(1, int(total_rgs * args.sample))
    sample_rows = sum(rg_sizes[:n_sample_rgs])
    columns = pq.ParquetFile(files[0]).schema_arrow.names

    print(f"[EXPLORE] Sampling {fm(sample_rows)} rows / {n_sample_rgs} RGs ...")
    sys.stdout.flush()

    # ═══════ 计算（静默）═══════
    label_res = compute_label_dist(files, n_sample_rgs, columns)
    user_int_stats = compute_int_features(schema['user_int'], 'user_int_feats', columns, files, n_sample_rgs)
    item_int_stats = compute_int_features(schema['item_int'], 'item_int_feats', columns, files, n_sample_rgs)
    user_dense_stats = compute_dense_features(schema['user_dense'], 'user_dense_feats', columns, files, n_sample_rgs)
    item_dense_stats = compute_dense_features(
        schema.get('item_dense', []), 'item_dense_feats', columns, files, n_sample_rgs)
    seq_res = compute_seq_features(schema.get('seq', {}), columns, files, n_sample_rgs)
    ts_res = compute_timestamp(files, n_sample_rgs, columns)

    # ═══════ 一次性输出 ═══════
    all_lines = []
    all_lines.append("=" * 70)
    all_lines.append(f"  PCVR Data Explorer Report")
    all_lines.append(f"  Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    all_lines.append("=" * 70)

    all_lines += render_overview(schema, files, args.sample, n_sample_rgs,
                                 sample_rows, total_rgs, total_rows_meta, columns, data_dir)
    all_lines += render_label(label_res)
    all_lines += render_int(user_int_stats, schema['user_int'], 'user_int_feats')
    all_lines += render_int(item_int_stats, schema['item_int'], 'item_int_feats')
    all_lines += render_dense(user_dense_stats, schema['user_dense'], 'user_dense_feats')
    all_lines += render_dense(item_dense_stats, schema.get('item_dense', []), 'item_dense_feats')
    all_lines += render_seq(seq_res, columns)
    all_lines += render_timestamp(ts_res)

    elapsed = time.time() - t0
    all_lines.append(_h("Done"))
    all_lines.append(f"  Elapsed  : {elapsed:.1f}s")
    all_lines.append(f"  Processed: {fm(sample_rows)} rows / {n_sample_rgs} RGs")
    all_lines.append(f"  Lines    : {len(all_lines)}")

    for line in all_lines:
        print(line)

    print(f"\n[EXPLORE] Total output lines: {len(all_lines)}")


if __name__ == '__main__':
    main()
