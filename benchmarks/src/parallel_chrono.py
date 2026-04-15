#!/usr/bin/env python3
"""Run YDF with parallel-chrono, write per-tree-depth CSV (thread-pivoted)."""

from __future__ import annotations
import argparse, csv, os, re, subprocess, sys, time
from pathlib import Path
from pathlib import Path
import subprocess, sys, logging

import pandas as pd
import utils.utils as utils
import shlex

log = logging.getLogger(__name__)


def get_args():
    # Get base parser as parent
    parent_parser = utils.get_base_parser()
    
    # Create this script's parser with the base as parent
    p = argparse.ArgumentParser(parents=[parent_parser])
    
    # Add script-specific arguments
    p.add_argument("--histogram_num_bins", type=int, default=64)
    p.add_argument("--rows", type=int, default=4096)
    p.add_argument("--cols", type=int, default=4096)
    p.add_argument("--save_log", action="store_true")
    p.add_argument("--skip_build", help="Skip building target. Use whatever's in .bazel-bin", action="store_true")
    p.add_argument("--disable_ecores", action="store_true",
                   help="Disable Intel E-cores for stable measurements (default: leave E-cores on)")
    p.add_argument("--gpu_mode", choices=["per_depth", "per_node"], default="per_depth",
                   help="GPU batching mode: per_depth (BFS, one kernel per depth level) "
                        "or per_node (DFS, one kernel per node). Only relevant when --use_gpu=true")

    return p.parse_args()


# “Classic” (no histogram)
TIMING_RX_SORT = re.compile(
    r"thread\s+(\d+)\s+tree\s+(\d+)\s+depth\s+(\d+)\s+"
    r"nodes\s+(\d+)\s+samples\s+(\d+)\s+"
    r"SampleProj\s+([0-9.eE+-]+)s\s+"          #  6
    r"ProjEval\s+([0-9.eE+-]+)s\s+"            #  7
    r"EvalProj\s+([0-9.eE+-]+)s\s+"            #  8
    r"kSortFillExampleBucketSet\s+([0-9.eE+-]+)s\s+"   #  9
    r"kSortScanSplits\s+([0-9.eE+-]+)s\s+"             # 10
    r"kSortInitBuckets\s+([0-9.eE+-]+)s\s+"             # 11
    r"kSortFillBuckets\s+([0-9.eE+-]+)s\s+"             # 12
    r"kSortFinalizeBuckets\s+([0-9.eE+-]+)s\s+"         # 13
    r"kSortFeatures\s+([0-9.eE+-]+)s\s+"                # 14
    r"kSortLabels\s+([0-9.eE+-]+)s"                     # 15
    r"(?:\s+GpuInit\s+([0-9.eE+-]+)s)?"
    r"(?:\s+GpuCsrFlatten\s+([0-9.eE+-]+)s)?"
    r"(?:\s+GpuKernel\s+([0-9.eE+-]+)s)?"
    r"(?:\s+GpuUnpack\s+([0-9.eE+-]+)s)?"
    r"(?:\s+GpuMutex\s+([0-9.eE+-]+)s)?"
    r"(?:\s+GpuSampleBatch\s+([0-9.eE+-]+)s)?"
    r"(?:\s+GpuApplyProj\s+([0-9.eE+-]+)s)?"
)

# Extended with the 4 histogram phases
TIMING_RX_HISTO = re.compile(
    r"thread\s+(\d+)\s+tree\s+(\d+)\s+depth\s+(\d+)\s+"
    r"nodes\s+(\d+)\s+samples\s+(\d+)\s+"
    r"SampleProj\s+([0-9.eE+-]+)s\s+"
    r"ProjEval\s+([0-9.eE+-]+)s\s+"
    r"EvalProj\s+([0-9.eE+-]+)s\s+"
    r"kFindSplitHistogram\s+([0-9.eE+-]+)s\s+"
    r"kChecksHistogram\s+([0-9.eE+-]+)s\s+"
    r"kFindMinMaxHistogram\s+([0-9.eE+-]+)s\s+"
    r"kGenHistogramBins\s+([0-9.eE+-]+)s\s+"
    r"kHistogramSetNumClasses\s+([0-9.eE+-]+)s\s+"
    r"kAssignSamplesToHistogram\s+([0-9.eE+-]+)s\s+"
    r"kUpdateDistributionsHistogram\s+([0-9.eE+-]+)s\s+"
    r"kComputeEntropy\s+([0-9.eE+-]+)s\s+"
    r"kSelectBestThresholdHistogram\s+([0-9.eE+-]+)s"
    r"(?:\s+GpuInit\s+([0-9.eE+-]+)s)?"
    r"(?:\s+GpuCsrFlatten\s+([0-9.eE+-]+)s)?"
    r"(?:\s+GpuKernel\s+([0-9.eE+-]+)s)?"
    r"(?:\s+GpuUnpack\s+([0-9.eE+-]+)s)?"
    r"(?:\s+GpuMutex\s+([0-9.eE+-]+)s)?"
    r"(?:\s+GpuSampleBatch\s+([0-9.eE+-]+)s)?"
    r"(?:\s+GpuApplyProj\s+([0-9.eE+-]+)s)?"
)


def parse_parallel_chrono(raw_log: str) -> pd.DataFrame:
    histo_mode = "kSelectBestThresholdHistogram" in raw_log
    rx = TIMING_RX_HISTO if histo_mode else TIMING_RX_SORT

    rows = []
    for m in rx.finditer(raw_log):
        g = m.groups()

        # Helper for optional GPU fields (None → 0.0)
        def opt_float(val):
            return float(val) if val is not None else 0.0

        if histo_mode:
            (tid, tree, depth, nodes, samples,
             sp, pe, ep,
             fsh, chk, fmm, ghb, hsnc, ast, udh, ce, sbt,
             gpu_init, gpu_csr, gpu_kernel, gpu_unpack, gpu_mutex, gpu_sample,
             gpu_apply) = g

            rows.append(dict(
                thread                       = int(tid),
                tree                         = int(tree),
                depth                        = int(depth),
                nodes                        = int(nodes),
                samples                      = int(samples),
                SampleProjection             = float(sp),
                ProjectionEvaluate           = float(pe),
                EvaluateProjection           = float(ep),
                FindSplitHistogram           = float(fsh),
                ChecksHistogram              = float(chk),
                FindMinMaxHistogram          = float(fmm),
                GenHistogramBins             = float(ghb),
                HistogramSetNumClasses       = float(hsnc),
                AssignSamplesToHist          = float(ast),
                UpdateDistributionsHistogram = float(udh),
                ComputeEntropy               = float(ce),
                SelectBestThresholdHistogram = float(sbt),
                GpuInit                      = opt_float(gpu_init),
                GpuCsrFlatten                = opt_float(gpu_csr),
                GpuKernel                    = opt_float(gpu_kernel),
                GpuUnpack                    = opt_float(gpu_unpack),
                GpuMutex                     = opt_float(gpu_mutex),
                GpuSampleBatch               = opt_float(gpu_sample),
                GpuApplyProj                 = opt_float(gpu_apply),
            ))
        else:
            (tid, tree, depth, nodes, samples,
             sp, pe, ep,
             fill_example, scan_splits,
             init_buckets, fill_buckets, finalize_buckets,
             features, labels,
             gpu_init, gpu_csr, gpu_kernel, gpu_unpack, gpu_mutex, gpu_sample,
             gpu_apply) = g

            rows.append(dict(
                thread                       = int(tid),
                tree                         = int(tree),
                depth                        = int(depth),
                nodes                        = int(nodes),
                samples                      = int(samples),
                SampleProjection             = float(sp),
                ProjectionEvaluate           = float(pe),
                EvaluateProjection           = float(ep),
                SortFillExampleBucketSet     = float(fill_example),
                SortScanSplits               = float(scan_splits),
                SortInitBuckets              = float(init_buckets),
                SortFillBuckets              = float(fill_buckets),
                SortFinalizeBuckets          = float(finalize_buckets),
                SortFeatures                 = float(features),
                SortLabels                   = float(labels),
                GpuInit                      = opt_float(gpu_init),
                GpuCsrFlatten                = opt_float(gpu_csr),
                GpuKernel                    = opt_float(gpu_kernel),
                GpuUnpack                    = opt_float(gpu_unpack),
                GpuMutex                     = opt_float(gpu_mutex),
                GpuSampleBatch               = opt_float(gpu_sample),
                GpuApplyProj                 = opt_float(gpu_apply),
            ))

    if not rows:
        raise ValueError("no parallel-chrono lines found in log")

    df = pd.DataFrame(rows)

    # ──────────────────────────────────────────────────────────────────────
    # Build one block per thread (unchanged logic)
    # ──────────────────────────────────────────────────────────────────────
    blocks = []
    for tid, g in df.groupby("thread", sort=True):
        g = g.sort_values(["tree", "depth"]).reset_index(drop=True)

        # Friendlier column names
        g = g.rename(columns={
            "samples": "Active Samples",
            "ProjectionEvaluate": "ApplyProjection"
        })

        g = g.drop(columns=["thread"])

        # Header rows
        thread_header = pd.DataFrame(
            [[f"Thread {tid}"] + [""] * (len(g.columns) - 1)],
            columns=g.columns)
        col_names = pd.DataFrame([g.columns.tolist()], columns=g.columns)

        blocks.append(pd.concat([thread_header, col_names, g], ignore_index=True))

    # Side-by-side layout with a blank separator
    max_len = max(len(b) for b in blocks)
    gap = pd.DataFrame({"": [""] * max_len})

    padded = []
    for i, blk in enumerate(blocks):
        padded.append(blk.reindex(range(max_len)).fillna(""))
        if i + 1 < len(blocks):
            padded.append(gap)

    return pd.concat(padded, axis=1)

def write_csv(left: pd.DataFrame, cmds: list[tuple[str, str]], path: str):
    """
    left : timing table (threads × depths) produced by parse_parallel_chrono
    cmds : list of (description, command-line) tuples
    """
    right = pd.DataFrame(cmds, columns=["Description", "Command"])

    # One blank column + spacer (same as before)
    n = max(len(left), len(right) + 1)     # +1 for our header row
    gap = pd.DataFrame({"": [""] * n, "  ": [""] * n})

    # Add header row for the command section
    cmds_with_headers = pd.concat(
        [
            pd.DataFrame([["", "", "Description", "Command"]],
                         columns=["", "  ", "Description", "Command"]),
            right
        ],
        ignore_index=True
    )

    (left.reindex(range(n)).fillna("")
         .pipe(lambda l: pd.concat([l, gap,
                                    cmds_with_headers.reindex(range(n)).fillna("")],
                                   axis=1))
     ).to_csv(path, index=False, header=False, quoting=csv.QUOTE_MINIMAL)

if __name__ == "__main__":
    utils.setup_signal_handlers()
    a = get_args()

    # The way this helper script itself was called
    helper_invocation = "python3 " + " ".join(shlex.quote(arg)
                                          for arg in sys.argv)

    if (not a.skip_build):
        if not utils.build_binary(a, chrono_mode=True):
            print("❌ build failed", file=sys.stderr)
            sys.exit(1)

    gpu_mode_label = ""
    if a.use_gpu:
        gpu_mode_label = f"GPU {a.gpu_mode} | "
        print(f"GPU mode: {a.gpu_mode}")
    exp = f"{gpu_mode_label}{a.sample_projection_mode} projections | {a.feature_split_type} | {a.numerical_split_type} | {a.experiment_name}"
    
    cmd = ["./bazel-bin/examples/train_oblique_forest",
           f"--num_trees={a.num_trees}",
           f"--feature_split_type={a.feature_split_type}",
           "--compute_oob_performances=false",
           f"--histogram_num_bins={a.histogram_num_bins}"]
    
    if a.max_num_projections is not None:
        cmd.append(f"--max_num_projections={a.max_num_projections}")
    if a.projection_density_factor is not None:
        cmd.append(f"--projection_density_factor={a.projection_density_factor}")
    if a.num_threads is not None:
        cmd.append(f"--num_threads={a.num_threads}")
    if a.tree_depth is not None:
        cmd.append(f"--tree_depth={a.tree_depth}")

    cmd.append(f"--use_gpu={'true' if a.use_gpu else 'false'}")

    if (a.numerical_split_type == "Dynamic Random Histogramming" or a.numerical_split_type == "Dynamic Equal Width Histogramming"):
        cmd.append("--numerical_split_type=Exact")
    elif a.numerical_split_type == "Vectorized Random":
        cmd.append("--numerical_split_type=Random")
    else:
        cmd.append(f"--numerical_split_type={a.numerical_split_type}")
 

    # Use CSV filename (without extension) if using CSV input, otherwise use matrix dimensions
    if a.input_mode == "csv":
        if a.rows != 4096 or a.cols != 4096:
            raise ValueError("Input Mode specified to be CSV, however synthetic --rows or --cols was also specified")

        csv_filename = Path(a.train_csv).stem  # Gets filename without extension
        dataset_name = csv_filename

        cmd += ["--input_mode=csv",
        f"--train_csv={a.train_csv}",
        f"--label_col={a.label_col}"]
    elif a.input_mode == "uniform" or a.input_mode == "trunk":
        dataset_name = f"{a.input_mode}_{a.rows}_x_{a.cols}"
        cmd += [f"--input_mode={a.input_mode}", f"--rows={a.rows}", f"--cols={a.cols}"]

    if a.use_gpu:
        device_name = utils.get_gpu_name() or "Unknown_GPU"
    else:
        device_name = utils.get_cpu_model_proc()
    out_dir = Path("benchmarks/results/per_function_timing") / device_name / exp / dataset_name
    out_dir.mkdir(parents=True, exist_ok=True)   

    binary_cmd_str = " ".join(cmd)
    print("\nRunning binary with command:\n", binary_cmd_str, "\n\n")

    try:
        if a.disable_ecores:
            utils.configure_cpu_for_benchmarks(True)
            print("E-cores: DISABLED")
        else:
            print("E-cores: ON (default — use --disable_ecores to disable)")
        t0 = time.perf_counter()
        proc = subprocess.run(
                cmd,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                check=False)
        log = proc.stdout

        if a.save_log:
            # ------------------------------------------------------------------
            #  Save the raw log (without ANSI colour codes) next to the CSV file
            # ------------------------------------------------------------------
            ansi_rx = re.compile(r'\x1B\[[0-?]*[ -/]*[@-~]')
            log_plain = ansi_rx.sub("", log)

            # Timestamped file name to avoid accidental overwrites
            ts = time.strftime("%Y%m%d-%H%M%S")
            log_fp = out_dir / f"{a.feature_split_type}-{a.numerical_split_type}-{a.num_threads}t-{ts}.log"

            try:
                log_fp.write_text(log_plain, encoding="utf-8")
                print("Raw log saved to", log_fp)
            except Exception as err:
                print(f"⚠️  Could not write log file: {err}", file=sys.stderr)

        if proc.returncode < 0:
            print(f"binary died with signal {-proc.returncode}")

        dt = time.perf_counter() - t0
        print(f"\n⏱  Binary subprocess ran for {dt:.4f} s\n")
        log_plain = re.sub(r'\x1B\[[0-?]*[ -/]*[@-~]', '', log)

        print(log_plain[:1000])

        table = parse_parallel_chrono(log_plain)

        # ------------------------------------------------------------------
        #  >>>  NEW: build file name from arguments instead of wall-time
        # ------------------------------------------------------------------
        d = -1 if a.tree_depth is None else a.tree_depth
        fname  = f"{d}Depth-{a.num_threads}Threads.csv"
        out_fp = out_dir / fname

        cmd_lines = [
            ("Helper invocation", helper_invocation),
            ("Bazel build", utils.last_build_cmd if utils.last_build_cmd else "(build skipped)"),
            ("Binary command", binary_cmd_str),
        ]

        write_csv(table, cmd_lines, out_fp)

        print("CSV written to", out_fp)

    except Exception as e:
        print("❌", e, file=sys.stderr)
    finally:
        utils.cleanup_and_exit()
