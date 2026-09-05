#!/usr/bin/env python3
"""Parse e2e_runtime or e2e_accuracy logs into CSV.

Log type is taken from an explicit 3rd argument when given, otherwise it is
inferred from the filename (substring 'runtime' or 'accuracy'):
  *runtime*.log  -> per-run timing samples (uses MEDIAN samples line)
  *accuracy*.log -> per-fold held-out accuracy (preferred 'test-accuracy:' line
                    from --test_csv; falls back to the last 'Train tree N/N ...
                    accuracy:' = RF OOB / GBT train-accuracy when no test fold).
                    When the log also carries 'test-auc:' / 'test-logloss:'
                    (newer binaries), side CSVs '<out>_auc.csv' /
                    '<out>_logloss.csv' are written with the same shape.

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
# The sample-axis marker carries either a model seed (legacy seed sweep) or a CV
# fold index (accuracy.sh 10-fold mode); group 3 captures whichever is present
# and becomes the per-sample key.
RUN_RE = re.compile(r'^----- Run (\d+)/(\d+)(?: \((?:seed|fold)=(\d+)\))? -----$')
# Random Forest logs   "Train tree N/N accuracy:0.83 logloss:..."
# Gradient Boosted Trees log "Train tree N/N train-loss:... train-accuracy:0.73 [total:...]"
# One regex captures the accuracy value in either format: the `\b` before
# `accuracy` also matches the GBT `train-accuracy:` variant (since `-` is a
# non-word char, there is a word boundary between it and `accuracy`). The
# accuracy parser keeps the LAST match per sample, i.e. the final-model accuracy
# in both cases.
# CAVEAT: this is only the FALLBACK and is NOT comparable across ensembles --
# GBT's number is TRAIN-set accuracy (the harness builds GBT with
# validation_set_ratio=0), optimistic, whereas RF's is OOB accuracy. Prefer the
# held-out 'test-accuracy:' line (TEST_ACC_RE below), which the accuracy parser
# uses whenever it is present.
ACC_RE = re.compile(r'Train tree \d+/\d+.*?\baccuracy:([\d.]+)')
# Held-out test-set accuracy emitted by train_oblique_forest's --test_csv path
# ("... test-accuracy:0.94"). Disjoint from ACC_RE (no "Train tree N/N" prefix).
# When present for a sample it is PREFERRED over the OOB/train-accuracy value.
TEST_ACC_RE = re.compile(r'\btest-accuracy:([\d.]+)')
# Optional extra held-out metrics on the same line (newer binaries only).
# AUC is logged as -1 when no ROC was computed; treat that as missing.
TEST_AUC_RE = re.compile(r'\btest-auc:(-?[\d.eE+\-]+)')
TEST_LOGLOSS_RE = re.compile(r'\btest-logloss:([\d.eE+\-]+)')
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
    """Sample stddev (Bessel-corrected) over non-empty samples. '' if n<2."""
    xs = [float(s) for s in samples if s != '']
    n = len(xs)
    if n < 2:
        return ''
    mean = sum(xs) / n
    var = sum((x - mean) ** 2 for x in xs) / (n - 1)
    return f"{math.sqrt(var):.6f}"


def sample_mean(samples):
    """Mean over non-empty samples. '' if there are none."""
    xs = [float(s) for s in samples if s != '']
    if not xs:
        return ''
    return f"{sum(xs) / len(xs):.6f}"


# The vectorized experiments rebuild the binary with an AVX --config, which is
# invisible on the command line. accuracy.sh emits this banner once, right
# before the vectorized runs, so the parser latches on it and tags every
# subsequent run's method as "Vectorized_<method>".
VEC_RE = re.compile(r'USING INSTRUCTION SET:\s*\w+')


def parse_cmd(args, vectorized=False):
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

    # Binary defaults: feature_split_type=Oblique, ensemble_method=Bagging.
    # Flags may be written `--flag value`, `--flag "quoted value"` or
    # `--flag=value` (EXTRA_TRAIN_ARGS is free-form); accept all three.
    m_split = re.search(r'--feature_split_type[= ]"?([^"=][^"]*?)"?(?=\s+--|\s*$)', args)
    is_oblique = 'Oblique' in (m_split.group(1) if m_split else "Oblique")
    m_ens = re.search(r'--ensemble_method[= ]"?(\w+)"?', args)
    is_boosting = (m_ens.group(1) if m_ens else "Bagging") == "Boosting"

    # Family label = obliqueness x ensemble:
    #   Oblique  + Bagging  -> SPORF     Oblique  + Boosting -> SPO-GBT
    #   non-obl. + Bagging  -> RF        non-obl. + Boosting -> GBT
    if is_oblique:
        family = "SPO-GBT" if is_boosting else "SPORF"
    else:
        family = "GBT" if is_boosting else "RF"

    m_method = re.search(r'--numerical_split_type[= ]"?([^"=][^"]*?)"?(?=\s+--|\s*$)', args)
    method = m_method.group(1).replace(' ', '_') if m_method else "unknown"
    if vectorized:
        method = f"Vectorized_{method}"
    m_thresh = re.search(r'--dynamic_split_threshold=(\d+)', args)
    thresh = m_thresh.group(1) if m_thresh else None
    gpu = '--use_gpu=true' in args

    algo = f"{family}_{method}"
    if thresh:
        algo += f"_thresh{thresh}"
    if gpu:
        algo += "_GPU"
    return dataset, algo


def parse_runtime(log_path):
    rows = []
    cmd = None
    vectorized = False
    with open(log_path) as f:
        for line in f:
            line = line.rstrip('\n')
            if VEC_RE.search(line):
                vectorized = True
                continue
            m = CMD_RE.match(line)
            if m:
                cmd = m.group(1)
                continue
            m = MEDIAN_RUN_RE.match(line)
            if m and cmd is not None:
                median = m.group(1)
                samples = m.group(2).split()
                dataset, algo = parse_cmd(cmd, vectorized)
                rows.append((dataset, algo, median, samples))
                cmd = None
                continue
            m = MEDIAN_FAIL_RE.match(line)
            if m and cmd is not None:
                # Failure label goes in the median_s cell; no samples => empty stddev.
                dataset, algo = parse_cmd(cmd, vectorized)
                rows.append((dataset, algo, m.group(1), []))
                cmd = None
    return rows


def parse_accuracy(log_path):
    rows = []
    # 'seeds' holds the OOB/train-accuracy per sample (seed or CV fold); 'test'
    # holds the held-out test-set accuracy per sample (preferred when present).
    # 'auc'/'logloss' hold the optional extra held-out metrics (no fallback:
    # they exist only for samples with a test fold and a new-enough binary).
    state = {'cmd': None, 'seeds': {}, 'test': {}, 'auc': {}, 'logloss': {},
             'cur_seed': None, 'vectorized': False}

    def finish_cmd():
        cmd = state['cmd']
        seeds = state['seeds']
        test = state['test']
        if cmd is None:
            return
        if seeds:
            dataset, algo = parse_cmd(cmd, state['vectorized'])
            samples = []
            auc_samples = []
            logloss_samples = []
            for s in sorted(seeds.keys()):
                val = test[s] if s in test else seeds[s]
                samples.append('' if val is None else f"{val}")
                auc = state['auc'].get(s)
                auc_samples.append('' if auc is None else f"{auc}")
                ll = state['logloss'].get(s)
                logloss_samples.append('' if ll is None else f"{ll}")
            rows.append((dataset, algo, samples, auc_samples, logloss_samples))
        state['cmd'] = None
        state['seeds'] = {}
        state['test'] = {}
        state['auc'] = {}
        state['logloss'] = {}
        state['cur_seed'] = None

    with open(log_path) as f:
        for line in f:
            line = line.rstrip('\n')
            if VEC_RE.search(line):
                # Finalize the pending (still non-vectorized) command BEFORE
                # latching, so only the runs that follow this banner are tagged.
                finish_cmd()
                state['vectorized'] = True
                continue
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
            m_test = TEST_ACC_RE.search(line)
            if m_test and state['cur_seed'] is not None:
                state['test'][state['cur_seed']] = float(m_test.group(1))
                m_auc = TEST_AUC_RE.search(line)
                if m_auc and float(m_auc.group(1)) >= 0:
                    state['auc'][state['cur_seed']] = float(m_auc.group(1))
                m_ll = TEST_LOGLOSS_RE.search(line)
                if m_ll:
                    state['logloss'][state['cur_seed']] = float(m_ll.group(1))
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
        col_prefix = 'fold'
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
        with open(out_path, 'w', newline='') as f:
            w = csv.writer(f)
            w.writerow(header)
            for dataset, algo, median, samples in rows:
                w.writerow([dataset, algo, median, sample_stddev(samples)])
        print(f"Wrote {len(rows)} rows to {out_path}")
        return 0

    # Accuracy mode: the main CSV keeps its historical shape (held-out
    # accuracy per fold). The optional test-auc / test-logloss metrics go to
    # side CSVs '<out>_auc.csv' / '<out>_logloss.csv' with the identical
    # shape, written only when the log carried at least one such value.
    max_samples = max(len(r[2]) for r in rows)
    header = (['dataset', 'algorithm']
              + [f'{col_prefix}_{i}' for i in range(1, max_samples + 1)]
              + ['avg', 'std'])

    def write_metric_csv(path, sample_idx):
        with open(path, 'w', newline='') as f:
            w = csv.writer(f)
            w.writerow(header)
            for r in rows:
                dataset, algo, samples = r[0], r[1], r[sample_idx]
                padded = samples + [''] * (max_samples - len(samples))
                w.writerow([dataset, algo] + padded
                           + [sample_mean(samples), sample_stddev(samples)])

    write_metric_csv(out_path, 2)
    print(f"Wrote {len(rows)} rows to {out_path}")
    base, ext = os.path.splitext(out_path)
    for name, sample_idx in (('auc', 3), ('logloss', 4)):
        if any(v != '' for r in rows for v in r[sample_idx]):
            side_path = f"{base}_{name}{ext}"
            write_metric_csv(side_path, sample_idx)
            print(f"Wrote {len(rows)} rows to {side_path}")
    return 0


if __name__ == '__main__':
    sys.exit(main())
