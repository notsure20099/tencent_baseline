"""
解析 explore_data.txt 中的探索结果，生成结构化 JSON 文件。
用法: python parse_report.py
"""
import os
import re
import json

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
INPUT_FILE = os.path.join(SCRIPT_DIR, 'explore_data.txt')
OUTPUT_FILE = os.path.join(SCRIPT_DIR, 'explore_report.json')


def parse_overview(lines, i):
    result = {}
    while i < len(lines) and not lines[i].startswith('==='):
        line = lines[i].strip()
        if not line:
            i += 1
            continue
        # Handle multi-part line: "Files : 1000 | RowGroups: 1044 | Total rows: 2,099,956"
        if '|' in line:
            parts = [p.strip() for p in line.split('|')]
            for part in parts:
                if ':' in part:
                    k, _, v = part.partition(':')
                    k = k.strip(); v = v.strip()
                    if 'Files' in k:
                        m = re.match(r'(\d+)', v)
                        if m: result['files'] = int(m.group(1))
                    elif 'RowGroups' in k or 'Row Groups' in k:
                        m = re.match(r'(\d+)', v)
                        if m: result['row_groups'] = int(m.group(1))
                    elif 'Total rows' in k:
                        result['total_rows'] = int(v.replace(',', ''))
            i += 1
            continue

        if ':' not in line:
            i += 1; continue

        key, _, val = line.partition(':')
        key = key.strip(); val = val.strip()

        if key == 'Data dir':
            result['data_dir'] = val
        elif key == 'Sampled':
            m = re.match(r'([\d,]+) rows \((\d+) RGs', val)
            if m:
                result['sampled_rows'] = int(m.group(1).replace(',', ''))
                result['sampled_rgs'] = int(m.group(2))
            m_ratio = re.search(r'ratio=([\d.]+)', val)
            if m_ratio: result['sample_ratio'] = float(m_ratio.group(1))
        elif key == 'Columns':
            result['columns'] = int(val)
        elif key == 'user_int':
            m = re.match(r'(\d+) feats\s+\(dim=(\d+)\)', val)
            if m: result['user_int'] = {'count': int(m.group(1)), 'total_dim': int(m.group(2))}
        elif key == 'item_int':
            m = re.match(r'(\d+) feats\s+\(dim=(\d+)\)', val)
            if m: result['item_int'] = {'count': int(m.group(1)), 'total_dim': int(m.group(2))}
        elif key == 'user_dense':
            m = re.match(r'(\d+) feats\s+\(dim=(\d+)\)', val)
            if m: result['user_dense'] = {'count': int(m.group(1)), 'total_dim': int(m.group(2))}
        elif key == 'item_dense':
            m = re.match(r'(\d+) feats\s+\(dim=(\d+)\)', val)
            if m: result['item_dense'] = {'count': int(m.group(1)), 'total_dim': int(m.group(2))}
        elif key == 'seq domains':
            m = re.match(r'(\d+)\s+\((.+)\)', val)
            if m: result['seq_domains'] = {'count': int(m.group(1)), 'names': [s.strip() for s in m.group(2).split(',')]}
        i += 1
    return result, i


def parse_label(lines, i):
    result = {}
    while i < len(lines) and not lines[i].startswith('==='):
        line = lines[i].strip()
        if ':' not in line:
            i += 1
            continue
        key, _, val = line.partition(':')
        key = key.strip()
        val = val.strip()
        if key == 'Total':
            result['total'] = int(val.replace(',', ''))
        elif key.startswith('Pos'):
            m = re.match(r'([\d,]+)\s+\(([\d.]+)%\)', val)
            if m:
                result['positive'] = int(m.group(1).replace(',', ''))
                result['positive_pct'] = float(m.group(2))
        elif key.startswith('Neg'):
            m = re.match(r'([\d,]+)\s+\(([\d.]+)%\)', val)
            if m:
                result['negative'] = int(m.group(1).replace(',', ''))
                result['negative_pct'] = float(m.group(2))
        elif key == 'Ratio':
            result['ratio'] = val
        i += 1
    return result, i


def parse_int_features(lines, i):
    """Parse list of int feature rows until next '===' section."""
    features = []
    while i < len(lines) and not lines[i].startswith('==='):
        line = lines[i].strip()
        if not line or line.startswith('fid'):
            i += 1
            continue
        parts = line.split()
        if len(parts) < 7:
            i += 1
            continue
        try:
            fid = int(parts[0])
            vocab = int(parts[1].replace(',', ''))
            dim = int(parts[2])
            if parts[3] == 'MISSING':
                features.append({'fid': fid, 'vocab': vocab, 'dim': dim, 'status': 'MISSING'})
            else:
                features.append({
                    'fid': fid,
                    'vocab': vocab,
                    'dim': dim,
                    'zero_rate': float(parts[3]),
                    'mean_nonzero': float(parts[4]),
                    'max': int(parts[5].replace(',', '')) if parts[5] != 'N/A' else None,
                    'oob_count': int(parts[6].replace(',', '')),
                })
        except (ValueError, IndexError):
            pass
        i += 1
    return features, i


def parse_dense_features(lines, i):
    features = []
    while i < len(lines) and not lines[i].startswith('===') and 'Sequence Features' not in lines[i]:
        line = lines[i].strip()
        if not line or line.startswith('fid'):
            i += 1
            continue
        parts = line.split()
        if len(parts) < 7:
            i += 1
            continue
        try:
            fid = int(parts[0])
            dim = int(parts[1])
            if parts[2] == 'MISSING':
                features.append({'fid': fid, 'dim': dim, 'status': 'MISSING'})
            else:
                features.append({
                    'fid': fid,
                    'dim': dim,
                    'mean': float(parts[2]),
                    'std': float(parts[3]),
                    'min': float(parts[4]),
                    'max': float(parts[5]),
                    'zero_rate': float(parts[6]),
                })
        except (ValueError, IndexError):
            pass
        i += 1
    return features, i


def parse_sequences(lines, i):
    seqs = []
    while i < len(lines):
        line = lines[i].strip()
        # Stop at next top-level section separator
        if (line.startswith('===') and len(line) >= 10 and all(c == '=' for c in line)) or \
           'Timestamp Statistics' in line or 'Done' in line:
            while i < len(lines) and not lines[i].strip().startswith('==='):
                i += 1
            break
        if line.startswith('Domain:'):
            seq = _parse_one_seq(lines, i)
            if seq:
                seqs.append(seq)
            i += 1
            while i < len(lines):
                li = lines[i].strip()
                if li.startswith('Domain:') or (li.startswith('===') and len(li) >= 10 and all(c == '=' for c in li)):
                    break
                i += 1
            continue
        i += 1
    return seqs, i


def _parse_one_seq(lines, start):
    domain_line = lines[start].strip()
    m = re.match(r'Domain: (\w+)\s+prefix=(\S+)\s+ts_fid=(\d+)\s+feats=(\d+)', domain_line)
    seq = {}
    if m:
        seq['domain'] = m.group(1)
        seq['prefix'] = m.group(2)
        seq['ts_fid'] = int(m.group(3))
        seq['feature_count'] = int(m.group(4))

    for j in range(start + 1, min(start + 10, len(lines))):
        l = lines[j].strip()
        if l.startswith('Sequences:'):
            n = re.search(r'Sequences:\s+([\d,]+)\s+mean=([\d.]+)\s+min=([\d.]+)\s+max=([\d.]+)', l)
            if n:
                seq['sequence_count'] = int(n.group(1).replace(',', ''))
                seq['seq_len_mean'] = float(n.group(2))
                seq['seq_len_min'] = float(n.group(3))
                seq['seq_len_max'] = float(n.group(4))
        if l.startswith('Percentiles'):
            n = re.search(r'p50=([\d.]+)\s+p90=([\d.]+)\s+p95=([\d.]+)\s+p99=([\d.]+)', l)
            if n:
                seq['seq_len_p50'] = float(n.group(1))
                seq['seq_len_p90'] = float(n.group(2))
                seq['seq_len_p95'] = float(n.group(3))
                seq['seq_len_p99'] = float(n.group(4))
        if l.startswith('fid'):
            break

    # Parse feature rows
    features = []
    for j in range(start + 1, len(lines)):
        l = lines[j].strip()
        if l.startswith('Domain:') or (l.startswith('===') and len(l) >= 10 and all(c == '=' for c in l)):
            break
        if l.startswith('fid') or not l or 'Sequences:' in l or 'Percentiles' in l or 'No sequence' in l:
            continue
        parts = l.split()
        if len(parts) >= 5 and parts[0].isdigit():
            fid = int(parts[0])
            vocab = int(parts[1].replace(',', ''))
            oob = int(parts[2].replace(',', ''))
            oob_pct = float(parts[3])
            status = parts[4] if len(parts) >= 5 else 'OK'
            features.append({
                'fid': fid,
                'vocab': vocab,
                'oob_count': oob,
                'oob_rate': oob_pct,
                'status': status,
            })
    seq['features'] = features
    return seq


def parse_timestamp(lines, i):
    result = {}
    while i < len(lines) and not lines[i].startswith('===') and 'Done' not in lines[i]:
        line = lines[i].strip()
        if ':' not in line:
            i += 1
            continue
        key, _, val = line.partition(':')
        key = key.strip()
        val = val.strip()
        if key == 'Count':
            result['count'] = int(val.replace(',', ''))
        elif key == 'Min':
            m = re.match(r'(\d+)\s+\((.+)\)', val)
            if m:
                result['min_ts'] = int(m.group(1))
                result['min_datetime'] = m.group(2)
        elif key == 'Max':
            m = re.match(r'(\d+)\s+\((.+)\)', val)
            if m:
                result['max_ts'] = int(m.group(1))
                result['max_datetime'] = m.group(2)
        elif key == 'Range':
            m = re.match(r'([\d.]+) days', val)
            if m:
                result['range_days'] = float(m.group(1))
        i += 1
    return result, i


def parse_done(lines, i):
    result = {}
    while i < len(lines):
        line = lines[i].strip()
        if ':' in line:
            key, _, val = line.partition(':')
            key = key.strip()
            val = val.strip()
            if key == 'Elapsed':
                m = re.match(r'([\d.]+)s', val)
                if m:
                    result['elapsed_seconds'] = float(m.group(1))
            elif key == 'Processed':
                m = re.match(r'([\d,]+) rows / (\d+) RGs', val)
                if m:
                    result['processed_rows'] = int(m.group(1).replace(',', ''))
                    result['processed_rgs'] = int(m.group(2))
            elif key == 'Lines':
                result['output_lines'] = int(val)
        i += 1
    return result, i


def main():
    with open(INPUT_FILE, 'r', encoding='utf-8') as f:
        content = f.read()

    lines = content.split('\n')

    # Find start of report
    i = 0
    while i < len(lines) and 'PCVR Data Explorer Report' not in lines[i]:
        i += 1
    # skip header lines
    while i < len(lines) and not lines[i].startswith('==') and '0. Dataset Overview' not in lines[i]:
        i += 1

    report = {}

    # Parse sections by detecting section headers
    section_map = {
        '0. Dataset Overview': ('overview', parse_overview),
        '1. Label Distribution': ('label', parse_label),
        '2. user_int_feats': ('user_int_features', parse_int_features),
        '2. item_int_feats': ('item_int_features', parse_int_features),
        '3. user_dense_feats': ('user_dense_features', parse_dense_features),
        '3. item_dense_feats': ('item_dense_features', parse_dense_features),
    }

    while i < len(lines):
        line = lines[i].strip()
        matched = False
        for section_title, (key, parser) in section_map.items():
            if section_title in line:
                j = i + 2
                parsed, new_i = parser(lines, j)
                report[key] = parsed
                i = new_i
                matched = True
                break

        if not matched:
            if '4. Sequence Features' in line:
                j = i + 2
                seqs, new_i = parse_sequences(lines, j)
                report['sequence_features'] = seqs
                i = new_i
            elif '5. Timestamp Statistics' in line:
                j = i + 2
                ts, new_i = parse_timestamp(lines, j)
                report['timestamp'] = ts
                i = new_i
            elif line == 'Done':
                done, _ = parse_done(lines, i + 2)
                report['run_info'] = done
                break
            else:
                i += 1

    # Also extract time and info line
    for line in lines:
        if 'Time:' in line and 'PCVR' not in line:
            m = re.search(r'Time:\s+(.+)', line)
            if m:
                report['report_time'] = m.group(1).strip()
                break

    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    print(f"Structured report saved to: {OUTPUT_FILE}")
    print(f"Keys: {list(report.keys())}")


if __name__ == '__main__':
    main()
