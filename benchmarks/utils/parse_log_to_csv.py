#!/usr/bin/env python3
"""Parse e2e_runtime or e2e_accuracy logs into CSV.

Detects log type from filename:
  e2e_runtime_*.log  -> per-run timing samples (uses MEDIAN samples line)
  e2e_accuracy_*.log -> per-seed final OOB accuracy (last 'Train tree N/N accuracy:' per run)

Usage: parse_log_to_csv.py <log_file> <out_csv>

Exit codes:
  0 - CSV written successfully
  1 - parse failure (no data) or IO error; caller should preserve the log
  2 - bad arguments
"""
import csv
import math
import os
import re
import sys

CMD_RE = re.compile(r'^\./bazel-bin/examples/train_oblique_forest (.+?)\s*$')
RUN_RE = re.compile(r'^----- Run (\d+)/(\d+)(?: \(seed=(\d+)\))? -----$')
ACC_RE = re.compile(r'Train tree \d+/\d+ accuracy:([\d.]+)')
# STDDEV segment is optional so logs predating it still parse.
MEDIAN_RUN_RE = re.compile(
    r'^MEDIAN of \d+/\d+ runs: ([\d.eE+\-]+) s'
    r'(?:\s+STDDEV: (?:[\d.eE+\-]+|N/A) s)?'
    r'\s+\(samples: ([0-9. eE+\-]+)\)$'
)


def sample_stddev(samples):
    """Sample stddev (Bessel-corrected). Empty string if n<2."""
    xs = [float(s) for s in samples]
    n = len(xs)
    if n < 2:
        return ''
    mean = sum(xs) / n
    var = sum((x - mean) ** 2 for x in xs) / (n - 1)
    return f"{math.sqrt(var):.6f}"


def parse_cmd(args):
    m_csv = re.search(r'--train_csv "([^"]+)"', args)
    m_trunk = re.search(r'--input_mode trunk.*?--rows (\d+)(?:.*?--cols (\d+))?', args)
    if m_csv:
        path = m_csv.group(1)
        parts = path.split('/')
        if 'cc18_binary_csv' in parts:
            idx = parts.index('cc18_binary_csv')
            dataset = parts[idx + 1]
        else:
            dataset = re.sub(r'\.csv$', '', parts[-1])
    elif m_trunk:
        rows = m_trunk.group(1)
        cols = m_trunk.group(2)
        dataset = f"trunk_{rows}_x_{cols}" if cols else f"trunk_{rows}rows"
    else:
        dataset = "unknown"

    m_split = re.search(r'--feature_split_type "([^"]+)"', args)
    split_type = m_split.group(1).replace(' ', '_') if m_split else "Oblique"
    m_method = re.search(r'--numerical_split_type "([^"]+)"', args)
    method = m_method.group(1).replace(' ', '_') if m_method else "unknown"
    m_thresh = re.search(r'--dynamic_split_threshold=(\d+)', args)
    thresh = m_thresh.group(1) if m_thresh else None
    gpu = '--use_gpu=true' in args

    algo = f"{split_type}_{method}"
    if thresh:
        algo += f"_thresh{thresh}"
    if gpu:
        algo += "_GPU"
    return dataset, algo


def parse_runtime(log_path):
    rows = []
    cmd = None
    with open(log_path) as f:
        for line in f:
            line = line.rstrip('\n')
            m = CMD_RE.match(line)
            if m:
                cmd = m.group(1)
                continue
            m = MEDIAN_RUN_RE.match(line)
            if m and cmd is not None:
                median = m.group(1)
                samples = m.group(2).split()
                dataset, algo = parse_cmd(cmd)
                rows.append((dataset, algo, median, samples))
                cmd = None
    return rows


def parse_accuracy(log_path):
    rows = []
    state = {'cmd': None, 'seeds': {}, 'cur_seed': None}

    def finish_cmd():
        cmd = state['cmd']
        seeds = state['seeds']
        if cmd is None:
            return
        if seeds:
            dataset, algo = parse_cmd(cmd)
            samples = [
                ('' if seeds[s] is None else f"{seeds[s]}")
                for s in sorted(seeds.keys())
            ]
            rows.append((dataset, algo, samples))
        state['cmd'] = None
        state['seeds'] = {}
        state['cur_seed'] = None

    with open(log_path) as f:
        for line in f:
            line = line.rstrip('\n')
            m_cmd = CMD_RE.match(line)
            if m_cmd:
                finish_cmd()
                state['cmd'] = m_cmd.group(1)
                continue
            m_run = RUN_RE.match(line)
            if m_run:
                seed_s = m_run.group(3)
                if seed_s is not None:
                    state['cur_seed'] = int(seed_s)
                else:
                    state['cur_seed'] = int(m_run.group(1))
                state['seeds'].setdefault(state['cur_seed'], None)
                continue
            m_acc = ACC_RE.search(line)
            if m_acc and state['cur_seed'] is not None:
                state['seeds'][state['cur_seed']] = float(m_acc.group(1))
    finish_cmd()
    return rows


def main():
    if len(sys.argv) != 3:
        print("Usage: parse_log_to_csv.py <log_file> <out_csv>", file=sys.stderr)
        return 2
    log_path, out_path = sys.argv[1], sys.argv[2]
    if not os.path.exists(log_path):
        print(f"ERROR: log file not found: {log_path}", file=sys.stderr)
        return 1

    basename = os.path.basename(log_path)
    if 'accuracy' in basename:
        rows = parse_accuracy(log_path)
        col_prefix = 'seed'
        is_runtime = False
    elif 'runtime' in basename:
        rows = parse_runtime(log_path)
        col_prefix = 'run'
        is_runtime = True
    else:
        print(f"ERROR: cannot determine log type from filename '{basename}'", file=sys.stderr)
        return 1

    if not rows:
        print(f"ERROR: no data rows parsed from {log_path}", file=sys.stderr)
        return 1

    if is_runtime:
        # Runtime CSV reports median + sample stddev (n-1); raw samples stay
        # in the log (which is itself deleted on success unless something fails).
        header = ['dataset', 'algorithm', 'median_s', 'stddev_s']
    else:
        max_samples = max(len(r[2]) for r in rows)
        header = ['dataset', 'algorithm'] + [f'{col_prefix}_{i}' for i in range(1, max_samples + 1)]
    with open(out_path, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(header)
        if is_runtime:
            for dataset, algo, median, samples in rows:
                w.writerow([dataset, algo, median, sample_stddev(samples)])
        else:
            for dataset, algo, samples in rows:
                w.writerow([dataset, algo] + samples + [''] * (max_samples - len(samples)))
    print(f"Wrote {len(rows)} rows to {out_path}")
    return 0


if __name__ == '__main__':
    sys.exit(main())
