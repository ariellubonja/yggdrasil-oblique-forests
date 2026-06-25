#!/usr/bin/env python3
"""Run YDF with parallel-chrono, write per-tree-depth CSV (thread-pivoted)."""

from __future__ import annotations
import argparse, csv, os, re, subprocess, sys, time
from pathlib import Path
from pathlib import Path
import subprocess, sys, logging

import pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import benchmarks.utils.utils as utils  # noqa: E402
import shlex

log = logging.getLogger(__name__)


def get_args():
    # Get base parser as parent
    parent_parser = utils.get_base_parser()
    
    # Create this script's parser with the base as parent
    p = argparse.ArgumentParser(parents=[parent_parser])
    
    # Add script-specific arguments
    p.add_argument("--histogram_num_bins", type=int, default=64)
    p.add_argument("--rows", type=int, default=3000000)
    p.add_argument("--cols", type=int, default=4096)
    p.add_argument("--save_log", action="store_true")
    p.add_argument("--chrono_level", type=int, choices=[1, 2], default=1,
                   help="CHRONO profiling tier (see .bazelrc / parallel_chrono.h): "
                        "1=coarse (default; only the top-level scopes — "
                        "TreeTrain, NodeTrain, SampleProjection, EvaluateProj, "
                        "ProjEval, BfsNodeLoop — for lower measurement overhead), "
                        "2=fine (coarse scopes plus all sub-scopes).")
    p.add_argument("--skip_build", help="Skip building target. Use whatever's in .bazel-bin", action="store_true")
    p.add_argument("--disable_ecores", action=argparse.BooleanOptionalAction, default=True,
                   help="Disable Intel E-cores for stable measurements (default: disabled; pass --no-disable_ecores to keep them on)")
    p.add_argument("--gpu_mode", choices=["per_depth", "per_node"], default="per_depth",
                   help="GPU batching mode: per_depth (BFS, one kernel per depth level) "
                        "or per_node (DFS, one kernel per node). Only relevant when --use_gpu=true")

    # Bazel config passthrough: any unknown bare `--<name>` flag (no `=value`)
    # is forwarded as `--config=<name>` to the bazel build. Lets new .bazelrc
    # configs be exercised without touching this script — e.g. `--bfs_only`
    # → `--config=bfs_only`. Anything else (typos with values, positional
    # garbage, valid flags spelled wrong) still errors as usual.
    args, unknown = p.parse_known_args()
    for tok in unknown:
        if tok.startswith("--") and "=" not in tok and len(tok) > 2:
            args.bazel_config.append(tok[2:])
        else:
            p.error(f"unrecognized argument: {tok}")
    return args


# Tail of the per-depth LOG line: flat per-stage GPU timings (measured via
# cudaEvent_t bridging in oblique_gpu_kernels.cu.cc). Every capture group is
# optional so lines from CPU-only or older GPU builds still parse.
_GPU_TAIL_RX = (
    r"(?:\s+GpuInit\s+([0-9.eE+-]+)s)?"
    r"(?:\s+GpuCsrFlatten\s+([0-9.eE+-]+)s)?"
    r"(?:\s+GpuUnpack\s+([0-9.eE+-]+)s)?"
    r"(?:\s+GpuMutex\s+([0-9.eE+-]+)s)?"
    r"(?:\s+GpuSampleBatch\s+([0-9.eE+-]+)s)?"
    r"(?:\s+ApplyColumnADD\s+([0-9.eE+-]+)s)?"
    r"(?:\s+ApplyColumnADDMultiNode\s+([0-9.eE+-]+)s)?"
    r"(?:\s+RandomHistogram\s+([0-9.eE+-]+)s)?"
    r"(?:\s+SplitHistogram\s+([0-9.eE+-]+)s)?"
    r"(?:\s+SortIndices\s+([0-9.eE+-]+)s)?"
    r"(?:\s+ExactSplit\s+([0-9.eE+-]+)s)?"
    r"(?:\s+GpuOther\s+([0-9.eE+-]+)s)?"
    # ApplyProjectionsSymmetricDepthwiseAP sub-phases. Optional — only
    # emitted when the binary is built with -DSYMMETRIC_DEPTHWISE_AP.
    r"(?:\s+SymBuildBag\s+([0-9.eE+-]+)s)?"
    r"(?:\s+SymSortBag\s+([0-9.eE+-]+)s)?"
    r"(?:\s+SymSweep\s+([0-9.eE+-]+)s)?"
    # ApplyProjectionsDepthwise1Pass sub-phases. Optional — only emitted
    # when the binary is built with -DDEPTHWISE_1_PASS.
    r"(?:\s+Dw1PreSize\s+([0-9.eE+-]+)s)?"
    r"(?:\s+Dw1Sweep\s+([0-9.eE+-]+)s)?"
    # Sub-phases of Dw1Sweep. Optional — only emitted under -DDEPTHWISE_1_PASS.
    # ColWalk (gather/FMA) vs Bucket+Scatter (counting-sort) splits the 93%.
    r"(?:\s+Dw1SweepCtor\s+([0-9.eE+-]+)s)?"
    r"(?:\s+Dw1SweepBucket\s+([0-9.eE+-]+)s)?"
    r"(?:\s+Dw1SweepScatter\s+([0-9.eE+-]+)s)?"
    r"(?:\s+Dw1SweepColWalk\s+([0-9.eE+-]+)s)?"
    r"(?:\s+Dw1SweepBig\s+([0-9.eE+-]+)s)?"
    r"(?:\s+Dw1SweepGeneric\s+([0-9.eE+-]+)s)?"
    # Per-node bookkeeping scopes inside NodeTrain / FindBestConditionSparseOblique.
    # Always emitted (no #ifdef). Close the BfsNodeLoop residual gap.
    r"(?:\s+NodeTrain\s+([0-9.eE+-]+)s)?"
    r"(?:\s+FindBestCondition\s+([0-9.eE+-]+)s)?"
    r"(?:\s+ObliqueSplitSearch\s+([0-9.eE+-]+)s)?"
    # Sub-scopes of ObliqueSplitSearch closing the FBC residual gap.
    r"(?:\s+FindObliqueSetup\s+([0-9.eE+-]+)s)?"
    r"(?:\s+EvaluateProj\s+([0-9.eE+-]+)s)?"
    # Sub-scope of EvaluateProj — BuildCountLogCountTable(N_node).
    r"(?:\s+EntropyTableSetup\s+([0-9.eE+-]+)s)?"
    # Sub-scope of EvaluateProj — FindSplitLabelClassificationFeatureNumericalCart.
    r"(?:\s+CartPath\s+([0-9.eE+-]+)s)?"
    # Sub-scope of CartPath — feature_filler lambda (EffectiveStrategy etc.).
    r"(?:\s+CartSetup\s+([0-9.eE+-]+)s)?"
    # Sub-scope of EvaluateProj — whole histogram split-finder body (symmetric
    # to CartPath; includes the reverse cumulative sweep and local destructors).
    r"(?:\s+HistoPath\s+([0-9.eE+-]+)s)?"
    r"(?:\s+AxisAlignedSplitSearch\s+([0-9.eE+-]+)s)?"
    r"(?:\s+SampleProjection\s+([0-9.eE+-]+)s)?"
    r"(?:\s+SplitExamplesInPlace\s+([0-9.eE+-]+)s)?"
    r"(?:\s+SetLeafValue\s+([0-9.eE+-]+)s)?"
    # BFS-only scheduler scope. BfsNodeLoop only fires under -DBFS_ONLY.
    # Zero on DFS builds.
    r"(?:\s+BfsNodeLoop\s+([0-9.eE+-]+)s)?"
    # Top-level per-tree scope. Only depth=0 of each tree is non-zero —
    # the entire tree-train accumulates there. Works as the DFS-build
    # analogue of BfsNodeLoop for CHRONO-coverage comparison.
    r"(?:\s+TreeTrain\s+([0-9.eE+-]+)s)?"
)

# Thread identifier printed by random_forest.cc is `std::this_thread::get_id()`
# streamed via operator<<. That representation is implementation-defined:
# libstdc++ (Linux benchmark box) prints a decimal number, libc++ (macOS)
# prints a hex pointer like `0x16da57000`. Accept either so the parser works
# on both — the value is only used as a per-thread grouping key.
_THREAD_ID_RX = r"(0x[0-9a-fA-F]+|\d+)"

# "Classic" (Exact / sort-based CPU split)
TIMING_RX_SORT = re.compile(
    r"thread\s+" + _THREAD_ID_RX + r"\s+tree\s+(\d+)\s+depth\s+(\d+)\s+"
    r"nodes\s+(\d+)\s+samples\s+(\d+)\s+"
    r"ProjEval\s+([0-9.eE+-]+)s\s+"            #  7
    r"kGetCandidateAttributes\s+([0-9.eE+-]+)s\s+"      #  7a
    r"kAxisAlignedColumnFetch\s+([0-9.eE+-]+)s\s+"      #  7c
    r"kSortFillExampleBucketSet\s+([0-9.eE+-]+)s\s+"   #  9
    r"kSortScanSplits\s+([0-9.eE+-]+)s\s+"             # 10
    r"kSortInitBuckets\s+([0-9.eE+-]+)s\s+"             # 11
    r"kSortFillBuckets\s+([0-9.eE+-]+)s\s+"             # 12
    r"kSortFinalizeBuckets\s+([0-9.eE+-]+)s\s+"         # 13
    r"kSortFeatures\s+([0-9.eE+-]+)s\s+"                # 14
    r"kSortLabels\s+([0-9.eE+-]+)s\s+"                  # 15
    r"kScanPresorted\s+([0-9.eE+-]+)s"                  # 16
    + _GPU_TAIL_RX
)

# Extended with the histogram-based CPU split phases. The 8 Exact / sort-path
# fields are optional: a pure-histogram run leaves them at zero, but
# DYNAMIC_RANDOM_HISTOGRAM / DYNAMIC_EQUAL_WIDTH_HISTOGRAM downgrade per-node
# to EXACT below dynamic_split_threshold (oblique.cc:194-203), so this branch
# can carry non-zero Sort-path values on Dynamic runs.
TIMING_RX_HISTO = re.compile(
    r"thread\s+" + _THREAD_ID_RX + r"\s+tree\s+(\d+)\s+depth\s+(\d+)\s+"
    r"nodes\s+(\d+)\s+samples\s+(\d+)\s+"
    r"ProjEval\s+([0-9.eE+-]+)s\s+"
    r"kGetCandidateAttributes\s+([0-9.eE+-]+)s\s+"
    r"kAxisAlignedColumnFetch\s+([0-9.eE+-]+)s\s+"
    r"kHistogramSetup\s+([0-9.eE+-]+)s\s+"
    r"kMinMaxNumerical\s+([0-9.eE+-]+)s\s+"
    r"kAssignSamplesToHistogram\s+([0-9.eE+-]+)s\s+"
    r"kSelectBestThresholdHistogram\s+([0-9.eE+-]+)s"
    # Dynamic-fallback Exact-path scopes. Optional — zero on pure-histogram
    # builds, non-zero when DYNAMIC_* downgrades small nodes to EXACT.
    r"(?:\s+kSortFillExampleBucketSet\s+([0-9.eE+-]+)s)?"
    r"(?:\s+kSortScanSplits\s+([0-9.eE+-]+)s)?"
    r"(?:\s+kSortInitBuckets\s+([0-9.eE+-]+)s)?"
    r"(?:\s+kSortFillBuckets\s+([0-9.eE+-]+)s)?"
    r"(?:\s+kSortFinalizeBuckets\s+([0-9.eE+-]+)s)?"
    r"(?:\s+kSortFeatures\s+([0-9.eE+-]+)s)?"
    r"(?:\s+kSortLabels\s+([0-9.eE+-]+)s)?"
    r"(?:\s+kScanPresorted\s+([0-9.eE+-]+)s)?"
    + _GPU_TAIL_RX
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
             pe, get_cand, aa_col_fetch,
             setup, minmax, ast, sbt,
             fill_example, scan_splits,
             init_buckets, fill_buckets, finalize_buckets,
             features, labels, scan_presorted,
             gpu_init, gpu_csr, gpu_unpack, gpu_mutex, gpu_sample,
             gpu_apply_cad, gpu_apply_cad_mn, gpu_random_hist,
             gpu_split_hist, gpu_sort_idx, gpu_exact_split, gpu_other,
             sym_build, sym_sort, sym_sweep,
             dw1_presize, dw1_sweep,
             dw1_sweep_ctor, dw1_sweep_bucket, dw1_sweep_scatter,
             dw1_sweep_colwalk, dw1_sweep_big, dw1_sweep_generic,
             node_train, find_best_condition, oblique_split_search,
             find_oblique_setup, evaluate_proj, entropy_table_setup,
             cart_path, cart_setup, histo_path,
             axis_aligned_split_search,
             sample_proj, split_in_place, set_leaf_value,
             bfs_node_loop, tree_train) = g

            rows.append(dict(
                thread                       = int(tid, 0),  # 0x… (libc++) or decimal (libstdc++)
                tree                         = int(tree),
                depth                        = int(depth),
                nodes                        = int(nodes),
                samples                      = int(samples),
                ProjectionEvaluate           = float(pe),
                GetCandidateAttributes       = float(get_cand),
                AxisAlignedColumnFetch       = float(aa_col_fetch),
                HistogramSetup               = float(setup),
                MinMaxNumerical              = float(minmax),
                AssignSamplesToHist          = float(ast),
                SelectBestThresholdHistogram = float(sbt),
                # Exact / sort-path fields. Non-zero only when a DYNAMIC_*
                # histogram run falls back to EXACT for small nodes.
                SortFillExampleBucketSet     = opt_float(fill_example),
                SortScanSplits               = opt_float(scan_splits),
                SortInitBuckets              = opt_float(init_buckets),
                SortFillBuckets              = opt_float(fill_buckets),
                SortFinalizeBuckets          = opt_float(finalize_buckets),
                SortFeatures                 = opt_float(features),
                SortLabels                   = opt_float(labels),
                ScanPresorted                = opt_float(scan_presorted),
                GpuInit                      = opt_float(gpu_init),
                GpuCsrFlatten                = opt_float(gpu_csr),
                GpuUnpack                    = opt_float(gpu_unpack),
                GpuMutex                     = opt_float(gpu_mutex),
                GpuSampleBatch               = opt_float(gpu_sample),
                ApplyColumnADD               = opt_float(gpu_apply_cad),
                ApplyColumnADDMultiNode      = opt_float(gpu_apply_cad_mn),
                RandomHistogram              = opt_float(gpu_random_hist),
                SplitHistogram               = opt_float(gpu_split_hist),
                SortIndices                  = opt_float(gpu_sort_idx),
                ExactSplit                   = opt_float(gpu_exact_split),
                GpuOther                     = opt_float(gpu_other),
                SymBuildBag                  = opt_float(sym_build),
                SymSortBag                   = opt_float(sym_sort),
                SymSweep                     = opt_float(sym_sweep),
                Dw1PreSize                   = opt_float(dw1_presize),
                Dw1Sweep                     = opt_float(dw1_sweep),
                Dw1SweepCtor                 = opt_float(dw1_sweep_ctor),
                Dw1SweepBucket               = opt_float(dw1_sweep_bucket),
                Dw1SweepScatter              = opt_float(dw1_sweep_scatter),
                Dw1SweepColWalk              = opt_float(dw1_sweep_colwalk),
                Dw1SweepBig                  = opt_float(dw1_sweep_big),
                Dw1SweepGeneric              = opt_float(dw1_sweep_generic),
                NodeTrain                    = opt_float(node_train),
                FindBestCondition            = opt_float(find_best_condition),
                ObliqueSplitSearch           = opt_float(oblique_split_search),
                FindObliqueSetup             = opt_float(find_oblique_setup),
                EvaluateProj                 = opt_float(evaluate_proj),
                EntropyTableSetup            = opt_float(entropy_table_setup),
                CartPath                     = opt_float(cart_path),
                CartSetup                    = opt_float(cart_setup),
                HistoPath                    = opt_float(histo_path),
                AxisAlignedSplitSearch       = opt_float(axis_aligned_split_search),
                SampleProjection             = opt_float(sample_proj),
                SplitExamplesInPlace         = opt_float(split_in_place),
                SetLeafValue                 = opt_float(set_leaf_value),
                BfsNodeLoop                  = opt_float(bfs_node_loop),
                TreeTrain                    = opt_float(tree_train),
            ))
        else:
            (tid, tree, depth, nodes, samples,
             pe, get_cand, aa_col_fetch,
             fill_example, scan_splits,
             init_buckets, fill_buckets, finalize_buckets,
             features, labels, scan_presorted,
             gpu_init, gpu_csr, gpu_unpack, gpu_mutex, gpu_sample,
             gpu_apply_cad, gpu_apply_cad_mn, gpu_random_hist,
             gpu_split_hist, gpu_sort_idx, gpu_exact_split, gpu_other,
             sym_build, sym_sort, sym_sweep,
             dw1_presize, dw1_sweep,
             dw1_sweep_ctor, dw1_sweep_bucket, dw1_sweep_scatter,
             dw1_sweep_colwalk, dw1_sweep_big, dw1_sweep_generic,
             node_train, find_best_condition, oblique_split_search,
             find_oblique_setup, evaluate_proj, entropy_table_setup,
             cart_path, cart_setup, histo_path,
             axis_aligned_split_search,
             sample_proj, split_in_place, set_leaf_value,
             bfs_node_loop, tree_train) = g

            rows.append(dict(
                thread                       = int(tid, 0),  # 0x… (libc++) or decimal (libstdc++)
                tree                         = int(tree),
                depth                        = int(depth),
                nodes                        = int(nodes),
                samples                      = int(samples),
                ProjectionEvaluate           = float(pe),
                GetCandidateAttributes       = float(get_cand),
                AxisAlignedColumnFetch       = float(aa_col_fetch),
                SortFillExampleBucketSet     = float(fill_example),
                SortScanSplits               = float(scan_splits),
                SortInitBuckets              = float(init_buckets),
                SortFillBuckets              = float(fill_buckets),
                SortFinalizeBuckets          = float(finalize_buckets),
                SortFeatures                 = float(features),
                SortLabels                   = float(labels),
                ScanPresorted                = float(scan_presorted),
                GpuInit                      = opt_float(gpu_init),
                GpuCsrFlatten                = opt_float(gpu_csr),
                GpuUnpack                    = opt_float(gpu_unpack),
                GpuMutex                     = opt_float(gpu_mutex),
                GpuSampleBatch               = opt_float(gpu_sample),
                ApplyColumnADD               = opt_float(gpu_apply_cad),
                ApplyColumnADDMultiNode      = opt_float(gpu_apply_cad_mn),
                RandomHistogram              = opt_float(gpu_random_hist),
                SplitHistogram               = opt_float(gpu_split_hist),
                SortIndices                  = opt_float(gpu_sort_idx),
                ExactSplit                   = opt_float(gpu_exact_split),
                GpuOther                     = opt_float(gpu_other),
                SymBuildBag                  = opt_float(sym_build),
                SymSortBag                   = opt_float(sym_sort),
                SymSweep                     = opt_float(sym_sweep),
                Dw1PreSize                   = opt_float(dw1_presize),
                Dw1Sweep                     = opt_float(dw1_sweep),
                Dw1SweepCtor                 = opt_float(dw1_sweep_ctor),
                Dw1SweepBucket               = opt_float(dw1_sweep_bucket),
                Dw1SweepScatter              = opt_float(dw1_sweep_scatter),
                Dw1SweepColWalk              = opt_float(dw1_sweep_colwalk),
                Dw1SweepBig                  = opt_float(dw1_sweep_big),
                Dw1SweepGeneric              = opt_float(dw1_sweep_generic),
                NodeTrain                    = opt_float(node_train),
                FindBestCondition            = opt_float(find_best_condition),
                ObliqueSplitSearch           = opt_float(oblique_split_search),
                FindObliqueSetup             = opt_float(find_oblique_setup),
                EvaluateProj                 = opt_float(evaluate_proj),
                EntropyTableSetup            = opt_float(entropy_table_setup),
                CartPath                     = opt_float(cart_path),
                CartSetup                    = opt_float(cart_setup),
                HistoPath                    = opt_float(histo_path),
                AxisAlignedSplitSearch       = opt_float(axis_aligned_split_search),
                SampleProjection             = opt_float(sample_proj),
                SplitExamplesInPlace         = opt_float(split_in_place),
                SetLeafValue                 = opt_float(set_leaf_value),
                BfsNodeLoop                  = opt_float(bfs_node_loop),
                TreeTrain                    = opt_float(tree_train),
            ))

    if not rows:
        raise ValueError("no parallel-chrono lines found in log")

    df = pd.DataFrame(rows)

    # Session-level GpuInit is emitted once by random_forest.cc after the
    # per-tree loop. It's not per-depth (fires before any TreeScope), so
    # we park its value in the otherwise-unused tree=0, depth=0
    # placeholder row — the one with nodes=0 and all timings zero.
    m_session = re.search(r"session\s+GpuInit\s+([0-9.eE+-]+)s", raw_log)
    if m_session:
        placeholder = (df["tree"] == 0) & (df["depth"] == 0)
        df.loc[placeholder, "GpuInit"] = float(m_session.group(1))

    # ──────────────────────────────────────────────────────────────────────
    # Build one block per thread (unchanged logic)
    # ──────────────────────────────────────────────────────────────────────
    blocks = []
    for tid, g in df.groupby("thread", sort=True):
        g = g.sort_values(["tree", "depth"]).reset_index(drop=True)

        # Friendlier column names + dashes to show scope nesting. GPU
        # columns are flat (one column per helper stage) — no dashes. CPU
        # split-finder columns retain their nested dashes.
        g = g.rename(columns={
            "samples": "Active Samples",
            "ProjectionEvaluate": "ApplyProjection",
            # Inside EvaluateProjection → FindSplitHistogram (CPU histogram)
            "HistogramSetup":               "--HistogramSetup",
            "MinMaxNumerical":              "---MinMaxNumerical",
            "AssignSamplesToHist":          "--AssignSamplesToHist",
            "SelectBestThresholdHistogram": "--SelectBestThresholdHistogram",
            # Inside EvaluateProjection (CPU Exact/Sort splitter)
            "SortFillExampleBucketSet":     "-SortFillExampleBucketSet",
            "SortInitBuckets":              "--SortInitBuckets",
            "SortFillBuckets":              "--SortFillBuckets",
            "SortFinalizeBuckets":          "--SortFinalizeBuckets",
            "SortFeatures":                 "--SortFeatures",
            "SortLabels":                   "--SortLabels",
            "SortScanSplits":               "-SortScanSplits",
            "ScanPresorted":                "-ScanPresorted",
            # ApplyProjectionsSymmetricDepthwiseAP sub-phases (nest under
            # ApplyProjection — they sum to it).
            "SymBuildBag":                  "-SymBuildBag",
            "SymSortBag":                   "-SymSortBag",
            "SymSweep":                     "-SymSweep",
            # ApplyProjectionsDepthwise1Pass sub-phases (also sum to
            # ApplyProjection, mutually exclusive with Sym* by build mode).
            "Dw1PreSize":                   "-Dw1PreSize",
            "Dw1Sweep":                     "-Dw1Sweep",
            # Sub-phases of Dw1Sweep (nest under it — they sum to Dw1Sweep
            # modulo the evaluator ctor and per-task loop glue).
            "Dw1SweepCtor":                 "--Dw1SweepCtor",
            "Dw1SweepBucket":               "--Dw1SweepBucket",
            "Dw1SweepScatter":              "--Dw1SweepScatter",
            "Dw1SweepColWalk":              "--Dw1SweepColWalk",
            "Dw1SweepBig":                  "--Dw1SweepBig",
            "Dw1SweepGeneric":              "--Dw1SweepGeneric",
            # BFS-only scheduler scope (top-level, not nested under
            # ApplyProjection). BfsNodeLoop = per-node NodeTrain
            # dispatch in the BFS_ONLY fallback path.
            "BfsNodeLoop":                  "BfsNodeLoop",
            # Per-node bookkeeping scopes inside NodeTrain /
            # FindBestConditionSparseObliqueTemplate. Close the
            # BfsNodeLoop − Σ(splitter scopes) gap.
            "NodeTrain":                    "NodeTrain",
            "FindBestCondition":            "-FindBestCondition",
            "ObliqueSplitSearch":           "--ObliqueSplitSearch",
            "FindObliqueSetup":             "---FindObliqueSetup",
            "EvaluateProj":                 "---EvaluateProj",
            "EntropyTableSetup":            "----EntropyTableSetup",
            "CartPath":                     "----CartPath",
            "CartSetup":                    "-----CartSetup",
            "HistoPath":                    "----HistoPath",
            "AxisAlignedSplitSearch":        "--AxisAlignedSplitSearch",
            "SampleProjection":             "SampleProjection",
            "SplitExamplesInPlace":         "SplitExamplesInPlace",
            "SetLeafValue":                 "SetLeafValue",
            # Top-level per-tree scope. Non-zero only at depth=0 of each
            # tree. The DFS analogue of BFS top-level scopes.
            "TreeTrain":                    "TreeTrain",
        })

        g = g.drop(columns=["thread"])

        # Reorder columns. ProjEval first, then the flat per-stage GPU
        # kernel columns + GpuOther residual, then the CPU split-finder
        # subtree. Columns missing for the current mode are filtered out
        # below.
        # Call-stack order from outermost-in. CART and Histogram children
        # of EvaluateProj are kept in adjacent blocks so each path can be
        # read top-to-bottom. Columns missing for the current mode are
        # zero-dropped below.
        desired_order = [
            "tree", "depth", "nodes", "Active Samples",
            # GPU kernels (run before CPU work in GPU mode).
            "GpuSampleBatch",
            "GpuInit",
            "GpuMutex",
            "GpuCsrFlatten",
            "ApplyColumnADD",
            "ApplyColumnADDMultiNode",
            "RandomHistogram",
            "SplitHistogram",
            "SortIndices",
            "ExactSplit",
            "GpuOther",
            "GpuUnpack",
            # Top-level ground-truth scope.
            "TreeTrain",
            # BFS-only scheduler scope (top-level under TreeTrain).
            "BfsNodeLoop",
            # Per-node body.
            "NodeTrain",
            "-FindBestCondition",
            "---FindObliqueSetup",
            "--ObliqueSplitSearch",
            "SampleProjection",
            "--AxisAlignedSplitSearch",
            "ApplyProjection",
            # Sub-phases of ApplyProjection.
            "-SymBuildBag",
            "-SymSortBag",
            "-SymSweep",
            "-Dw1PreSize",
            "-Dw1Sweep",
            # Sub-phases of Dw1Sweep.
            "--Dw1SweepCtor",
            "--Dw1SweepBucket",
            "--Dw1SweepScatter",
            "--Dw1SweepColWalk",
            "--Dw1SweepBig",
            "--Dw1SweepGeneric",
            "---EvaluateProj",
            # --- CART path inside EvaluateProj ---
            "----CartPath",
            "-----CartSetup",
            "--SortInitBuckets",
            "--SortFillBuckets",
            "--SortFeatures",
            "-SortScanSplits",
            # --- Histogram path inside EvaluateProj ---
            "----HistoPath",
            "----EntropyTableSetup",
            "--HistogramSetup",
            "--AssignSamplesToHist",
            "--SelectBestThresholdHistogram",
            # Other splitter-related scopes (rare in current modes).
            "EvaluateProjection",
            "-SortFillExampleBucketSet",
            "--SortFinalizeBuckets",
            "--SortLabels",
            "-ScanPresorted",
            "GetCandidateAttributes",
            "AxisAlignedColumnFetch",
            # Per-node finish.
            "SplitExamplesInPlace",
            "SetLeafValue",
        ]
        ordered = [c for c in desired_order if c in g.columns]
        remaining = [c for c in g.columns if c not in desired_order]
        g = g[ordered + remaining]

        # Columns intentionally excluded from the CSV (still chrono'd in
        # C++). MinMaxNumerical is nested inside --HistogramSetup, so no
        # coverage is lost.
        excluded = ["---MinMaxNumerical"]
        g = g.drop(columns=[c for c in excluded if c in g.columns])

        # Drop timing columns that are all-zero for this run (e.g. GPU
        # columns in a CPU run, Histogram columns in an Exact run). Keep
        # preamble columns even if zero.
        preamble = {"tree", "depth", "nodes", "Active Samples"}
        zero_cols = [c for c in g.columns
                     if c not in preamble and (g[c] == 0.0).all()]
        g = g.drop(columns=zero_cols)

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
    exp = f"{gpu_mode_label}{a.feature_split_type} | {a.numerical_split_type}"
    
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

    # C++ --use_gpu flag is commented out until hybrid CPU-GPU offloading is
    # added; GPU vs CPU is chosen at compile time via --config=oblique_gpu.
    # cmd.append(f"--use_gpu={'true' if a.use_gpu else 'false'}")

    if (a.numerical_split_type == "Dynamic Random Histogramming" or a.numerical_split_type == "Dynamic Equal Width Histogramming"):
        cmd.append("--numerical_split_type=Exact")
    elif a.numerical_split_type == "Vectorized Random":
        cmd.append("--numerical_split_type=Random")
    else:
        cmd.append(f"--numerical_split_type={a.numerical_split_type}")
 

    # Use CSV filename (without extension) if using CSV input, otherwise use matrix dimensions
    if a.input_mode == "csv":
        csv_filename = Path(a.train_csv).stem  # Gets filename without extension
        dataset_name = csv_filename

        cmd += ["--input_mode=csv",
        f"--train_csv={a.train_csv}",
        f"--label_col={a.label_col}"]
    elif a.input_mode == "uniform" or a.input_mode == "trunk":
        dataset_name = f"{a.input_mode}_{a.rows}_x_{a.cols}"
        cmd += [f"--input_mode={a.input_mode}", f"--rows={a.rows}", f"--cols={a.cols}"]
    # Alternate dataset layouts apply to both synthetic and CSV inputs
    # (Dynamic_Row_Col_Major mirrors CSV numerical columns since 2026-06-11).
    if a.input_mode in ("csv", "uniform", "trunk") and \
            getattr(a, "dataset_layout", "column") != "column":
        cmd.append(f"--dataset_layout={a.dataset_layout}")

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
            print("E-cores: DISABLED (default — use --no-disable_ecores to keep on)")
        else:
            print("E-cores: ON")
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
        log_plain = re.sub(r'\x1B\[[0-?]*[ -/]*[@-~]', '', log)

        # Prefer the training-only wall time logged by random_forest.cc
        # ("Training block took: X s"). The subprocess wall time also
        # includes dataset generation, learner config, GPU init, etc., so
        # it overstates the training cost. Fall back to subprocess wall
        # time if the marker is missing (e.g. binary crashed before the
        # log line, or chrono build dropped the marker).
        m_train = re.search(
            r"random_forest\.cc Training block took:\s*([0-9.eE+-]+)\s*s",
            log_plain)
        if m_train:
            train_block = float(m_train.group(1))
            print(f"\n⏱  Training block took {train_block:.4f} s"
                  f"   (subprocess wall: {dt:.4f} s)\n")
        else:
            train_block = None
            print(f"\n⏱  Binary subprocess ran for {dt:.4f} s"
                  f"   (Training-block marker not found in log)\n")

        # ── CHRONO coverage vs Training-block wall ────────────────────
        # Sum top-level per-tree-train chrono scopes across all
        # (tree, depth, thread) entries in the log. With BFS_ONLY,
        # BfsNodeLoop covers the entire GrowTreeLocalBFS body
        # (≈ all of one tree's training; the frontier pop-loop is
        # un-chrono'd, <0.1 s for 3M rows).
        #
        # When N trees train concurrently on T threads, each tree's
        # internal time is single-thread, so:
        #     Σ scope ≈ train_block × min(N_trees, T) for load-balanced
        # case. For num_threads=1, Σ scope ≈ train_block exactly.
        def _sum_scope(name: str) -> float:
            return sum(float(v) for v in re.findall(
                rf"\s{name}\s+([0-9.eE+-]+)s", log_plain))

        if train_block is not None:
            bfs_total = _sum_scope("BfsNodeLoop")
            n_threads = max(1, int(getattr(a, "num_threads", 1)))
            try:
                n_trees = int(getattr(a, "num_trees", 1))
            except Exception:
                n_trees = 1
            cpu_budget = train_block * min(n_trees, n_threads)

            print("── CHRONO coverage ─────────────────────────────────")
            print(f"  Training block (wall):       {train_block:>10.4f} s")
            print(f"  CPU budget (wall × min(trees, threads) = "
                  f"{train_block:.3f} × min({n_trees},{n_threads})): "
                  f"{cpu_budget:>10.4f} s")
            tree_train = _sum_scope("TreeTrain")
            if bfs_total > 0:
                pct = (bfs_total / cpu_budget * 100.0) if cpu_budget > 0 else 0.0
                print(f"  Σ BfsNodeLoop:               {bfs_total:>10.4f} s  "
                      f"({pct:5.1f}% of CPU budget)")
                gap = cpu_budget - bfs_total
                print(f"  Unaccounted (CPU budget − Σ BFS): "
                      f"{gap:>10.4f} s  ({gap/cpu_budget*100:5.1f}%)")
            elif tree_train > 0:
                # DFS-build path: kTreeTrain is the top-level per-tree
                # scope and wraps the entire tree training. Sum across
                # trees ≈ training_block × min(trees, threads).
                pct = (tree_train / cpu_budget * 100.0) if cpu_budget > 0 else 0.0
                print(f"  Σ TreeTrain (DFS top-level): {tree_train:>10.4f} s  "
                      f"({pct:5.1f}% of CPU budget)")
                gap = cpu_budget - tree_train
                print(f"  Unaccounted (CPU budget − Σ TreeTrain): "
                      f"{gap:>10.4f} s  ({gap/cpu_budget*100:5.1f}%)")
            else:
                # Pre-TreeTrain-emit binary (rebuild needed). Fall back to
                # ApplyProjection as a partial coverage proxy.
                proj = _sum_scope("ProjEval")
                print(f"  Σ ProjEval (ApplyProjection): {proj:>10.4f} s   "
                      f"(no BFS / TreeTrain scope in log — rebuild)")
            print("────────────────────────────────────────────────────\n")

        print(log_plain[:1000])

        t_parse = time.perf_counter()
        table = parse_parallel_chrono(log_plain)
        parse_dt = time.perf_counter() - t_parse
        print(f"\n⏱  Result parsing took {parse_dt:.4f} s\n")

        # experiment_name sets the filename; fall back to Depth/Threads label.
        if a.experiment_name:
            fname = f"{a.experiment_name}.csv"
        else:
            d = -1 if a.tree_depth is None else a.tree_depth
            fname = f"{d}Depth-{a.num_threads}Threads.csv"
        out_fp = out_dir / fname

        cmd_lines = [
            ("Helper invocation", helper_invocation),
            ("Bazel build", utils.last_build_cmd if utils.last_build_cmd else "(build skipped)"),
            ("Binary command", binary_cmd_str),
            ("Training time (s)",
             f"{train_block:.4f}" if train_block is not None else "(marker not found)"),
        ]

        write_csv(table, cmd_lines, out_fp)

        print("CSV written to", out_fp)

    except Exception as e:
        print("❌", e, file=sys.stderr)
    finally:
        utils.cleanup_and_exit()
