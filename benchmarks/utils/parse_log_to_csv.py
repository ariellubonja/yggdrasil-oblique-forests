#!/usr/bin/env python3
"""Parse e2e_runtime or e2e_accuracy logs into CSV.

Log type is taken from an explicit 3rd argument when given, otherwise it is
inferred from the filename (substring 'runtime' or 'accuracy'):
  *runtime*.log  -> per-run timing samples (uses MEDIAN samples line)
  *accuracy*.log -> per-seed final accuracy (last 'Train tree N/N ... accuracy:'
                    per run; RF OOB accuracy or GBT train-accuracy)

Usage: parse_log_to_csv.py <log_file> <out_csv> [runtime|accuracy]

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
# Random Forest logs   "Train tree N/N accuracy:0.83 logloss:..."
# Gradient Boosted Trees log "Train tree N/N train-loss:... train-accuracy:0.73 [total:...]"
# One regex captures the accuracy value in either format: the `\b` before
# `accuracy` also matches the GBT `train-accuracy:` variant (since `-` is a
# non-word char, there is a word boundary between it and `accuracy`). The
# accuracy parser keeps the LAST match per seed, i.e. the final-model accuracy
# in both cases.
# CAVEAT: GBT's number is TRAIN-set accuracy (the harness builds GBT with
# validation_set_ratio=0), whereas RF's is OOB accuracy. The two are NOT
# directly comparable -- GBT train-accuracy is optimistic.
ACC_RE = re.compile(r'Train tree \d+/\d+.*?\baccuracy:([\d.]+)')
# STDDEV segment is optional so logs predating it still parse.
MEDIAN_RUN_RE = re.compile(
    r'^MEDIAN of \d+/\d+ runs: ([\d.eE+\-]+) s'
    r'(?:\s+STDDEV: (?:[\d.eE+\-]+|N/A) s)?'
    r'\s+\(samples: ([0-9. eE+\-]+)\)$'
)
# A dataset whose every run failed: the median_s cell carries the failure label
# (OOM = OOM-killed, ERROR = any other crash) so it is visible in the CSV.
MEDIAN_FAIL_RE = re.compile(r'^MEDIAN of \d+/\d+ runs: (OOM|ERROR)$')


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
                continue
            m = MEDIAN_FAIL_RE.match(line)
            if m and cmd is not None:
                # Failure label goes in the median_s cell; no samples => empty stddev.
                dataset, algo = parse_cmd(cmd)
                rows.append((dataset, algo, m.group(1), []))
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
    if len(sys.argv) not in (3, 4):
        print("Usage: parse_log_to_csv.py <log_file> <out_csv> [runtime|accuracy]", file=sys.stderr)
        return 2
    log_path, out_path = sys.argv[1], sys.argv[2]
    if not os.path.exists(log_path):
        print(f"ERROR: log file not found: {log_path}", file=sys.stderr)
        return 1

    # Log type comes from an explicit 3rd argument when given; otherwise it is
    # inferred from the filename (substring 'runtime' or 'accuracy').
    basename = os.path.basename(log_path)
    log_type = sys.argv[3].lower() if len(sys.argv) > 3 else None
    if log_type is None:
        if 'accuracy' in basename:
            log_type = 'accuracy'
        elif 'runtime' in basename:
            log_type = 'runtime'

    if log_type == 'accuracy':
        rows = parse_accuracy(log_path)
        col_prefix = 'seed'
        is_runtime = False
    elif log_type == 'runtime':
        rows = parse_runtime(log_path)
        col_prefix = 'run'
        is_runtime = True
    else:
        print(f"ERROR: cannot determine log type for '{basename}'; "
              f"pass 'runtime' or 'accuracy' as the 3rd argument", file=sys.stderr)
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
