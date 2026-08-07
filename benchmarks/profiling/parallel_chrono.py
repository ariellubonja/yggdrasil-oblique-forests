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
    p.add_argument("--histogram_num_bins", type=int, default=None,
                   help="Histogram bin count. Default follows "
                        "--vectorized (avx2 -> 64, avx512 -> 256).")
    p.add_argument("--rows", type=int, default=3000000)
    p.add_argument("--cols", type=int, default=4096)
    p.add_argument("--save_log", action="store_true")
    p.add_argument("--chrono_level", choices=["coarse", "ap", "ep", "both"],
                   default="coarse",
                   help="CHRONO profiling selector (see .bazelrc / "
                        "parallel_chrono.h): 'coarse' (default) = only the "
                        "top-level scopes (TreeTrain, NodeTrain, "
                        "SampleProjection, EvaluateProj, ProjEval, BfsNodeLoop) "
                        "plus node-bookkeeping / split-manager / GBT scopes; "
                        "'ap' = coarse + inner scopes of "
                        "ProjectionEvaluator::Evaluate (symmetric / "
                        "depthwise_1pass); 'ep' = coarse + inner scopes of "
                        "EvaluateProjection (histogram / Cart split search); "
                        "'both' = both fine axes at once.")
    p.add_argument("--skip_build", help="Skip building target. Use whatever's in .bazel-bin", action="store_true")
    p.add_argument("--disable_ecores", action=argparse.BooleanOptionalAction, default=True,
                   help="Disable Intel E-cores for stable measurements (default: disabled; pass --no-disable_ecores to keep them on)")
    p.add_argument("--gpu_mode", choices=["per_depth", "per_node"], default="per_depth",
                   help="GPU batching mode: per_depth (BFS, one kernel per depth level) "
                        "or per_node (DFS, one kernel per node). Only relevant when --use_gpu=true")
    p.add_argument("--ensemble_method", choices=["Bagging", "Boosting"], default="Bagging",
                   help="Ensemble method: 'Bagging' (Random Forest) or 'Boosting' (Gradient Boosted Trees/MART).")
    p.add_argument("--shrinkage", type=float, default=0.1,
                   help="Learning rate for boosting (only used when --ensemble_method=Boosting).")

    # Bazel config passthrough: an unknown bare `--<name>` (no `=value`) becomes
    # `--config=<name>`, so new .bazelrc configs need no change here. Anything
    # else still errors as usual.
    args, unknown = p.parse_known_args()
    for tok in unknown:
        if tok.startswith("--") and "=" not in tok and len(tok) > 2:
            args.bazel_config.append(tok[2:])
        else:
            p.error(f"unrecognized argument: {tok}")
    apply_vectorized(args)
    return args


# --vectorized -> the two things that actually select the SIMD histogram-binning
# kernel in HistogramBinner::Init (training.cc): the bin count picks the kernel
# shape (64 -> AVX2 8x8, 256 -> AVX-512 16x16; anything else falls through to
# scalar std::upper_bound) and the build macro decides whether the SIMD code is
# compiled in at all. Without this the flag was parsed and dropped.
_VEC_BINS = {"avx2": 64, "avx512": 256, "None": 64}
_VEC_LABEL = {"avx2": "", "avx512": " | AVX512", "None": " | Scalar"}
_NO_SIMD_CONFIG = "disable_std_upper_bound_vectorization"


def apply_vectorized(args):
    """Resolve --vectorized into histogram_num_bins + the bazel config."""
    iso = args.vectorized
    want_bins = _VEC_BINS[iso]

    if args.histogram_num_bins is None:
        args.histogram_num_bins = want_bins
    elif iso != "None" and args.histogram_num_bins != want_bins:
        print(f"⚠️  --vectorized={iso} selects the {want_bins}-bin kernel, but "
              f"--histogram_num_bins={args.histogram_num_bins} was given: "
              "HistogramBinner will use scalar std::upper_bound.",
              file=sys.stderr)

    if iso == "None":
        if _NO_SIMD_CONFIG not in args.bazel_config:
            args.bazel_config.append(_NO_SIMD_CONFIG)
        if args.skip_build:
            print(f"⚠️  --vectorized=None needs a rebuild with "
                  f"--config={_NO_SIMD_CONFIG}; --skip_build reuses whatever "
                  "bazel-bin already has (likely still vectorized).",
                  file=sys.stderr)
    return args


def chrono_level_name(chrono_level: str) -> str:
    names = {
        "coarse": "COARSE",
        "ap": "FINE_AP",
        "ep": "FINE_EP",
        "both": "FINE",
    }
    if chrono_level in names:
        return names[chrono_level]
    raise ValueError(f"Unsupported chrono_level: {chrono_level}")


# Log token (per-depth LOG line) -> internal column name. Only tokens spelled
# differently need an entry; the rest pass through, so a new C++ CHRONO scope is
# captured automatically rather than silently dropped.
_LOG_TOKEN_TO_COL = {
    "ProjEval":                      "ProjectionEvaluate",
    "kGetCandidateAttributes":       "GetCandidateAttributes",
    "kGetCandidateAttributesAssign": "GetCandidateAttributesAssign",
    "kGetCandidateAttributesShuffle": "GetCandidateAttributesShuffle",
    "kGetCandidateAttributesNumToTest": "GetCandidateAttributesNumToTest",
    "kColumnWithCast":               "ColumnWithCast",
    "kHistogramSetup":               "HistogramSetup",
    "kMinMaxNumerical":              "MinMaxNumerical",
    "kAssignSamplesToHistogram":     "AssignSamplesToHist",
    "kSelectBestThresholdHistogram": "SelectBestThresholdHistogram",
    "kSortFillExampleBucketSet":     "SortFillExampleBucketSet",
    "kSortScanSplits":               "SortScanSplits",
    "kSortInitBuckets":              "SortInitBuckets",
    "kSortFillBuckets":              "SortFillBuckets",
    "kSortFinalizeBuckets":          "SortFinalizeBuckets",
    "kSortFeatures":                 "SortFeatures",
    "kSortLabels":                   "SortLabels",
    "kScanPresorted":                "ScanPresorted",
}

# GBT session-level scopes, dumped once as "GBT chrono (ms): startup=.. .." in
# `name=value` ms form, which the per-depth parser cannot read. Captured
# separately in parse_parallel_chrono and converted ms->s.
_GBT_TOKEN_TO_COL = {
    "startup":            "GbtStartup",
    "preprocess":         "GbtPreprocess",
    "update_gradients":   "GbtUpdateGradients",
    "sample_examples":    "GbtSampleExamples",
    "train_tree":         "GbtTrainTree",
    "update_predictions": "GbtUpdatePredictions",
    "validation_eval":    "GbtValidationEval",
    "finalize":           "GbtFinalize",
}

# One "<name>=<milliseconds>" pair on the "GBT chrono (ms):" line.
_GBT_PAIR_RX = re.compile(r"(\w+)=([0-9.eE+-]+)")

# Every timing column the pipeline knows about, seeded to 0.0 per row so a
# thread/depth that never emitted a token still carries it. Captured tokens
# missing here are still added, surfacing as trailing CSV columns.
_TIMING_COLS = (
    "ProjectionEvaluate", "GetCandidateAttributes",
    "GetCandidateAttributesAssign", "GetCandidateAttributesShuffle", "GetCandidateAttributesNumToTest",
    "ColumnWithCast",
    "HistogramSetup", "MinMaxNumerical", "AssignSamplesToHist",
    "SelectBestThresholdHistogram",
    "SortFillExampleBucketSet", "SortScanSplits", "SortInitBuckets",
    "SortFillBuckets", "SortFinalizeBuckets", "SortFeatures", "SortLabels",
    "ScanPresorted",
    "GpuInit", "GpuCsrFlatten", "GpuUnpack", "GpuMutex", "GpuSampleBatch",
    "ApplyColumnADD", "ApplyColumnADDMultiNode", "RandomHistogram",
    "SplitHistogram", "SortIndices", "ExactSplit", "GpuOther",
    "SymBuildBag", "SymSortBag", "SymSweep",
    "SymSweepColSetup", "SymSweepMainLoop",
    "Dw1PreSize", "Dw1Sweep",
    "Dw1SweepSetup",
    "Dw1SweepColWalk", "Dw1ColWalkGroupByNode", "Dw1ColWalkBagScatter",
    "Dw1SweepBig", "Dw1SweepGeneric", "Dw1SharedBag",
    "NodeTrain", "FindBestCondition", "ObliqueSplitSearch",
    "FindObliqueSetup", "EvaluateProj", "EntropyTableSetup", "CartPath",
    "CartSetup", "HistoPath", "AxisAlignedSplitSearch", "SampleProjection",
    "SplitExamplesInPlace", "SetLeafValue", "BfsNodeLoop", "DepthTrain",
    "TreeTrain",
    # GBT session-level scopes (overlaid from the "GBT chrono (ms):" line onto
    # one placeholder row; see parse_parallel_chrono). Absent -> stay 0.0 ->
    # zero-dropped, so Bagging/RF runs are unaffected.
    "GbtStartup", "GbtPreprocess", "GbtUpdateGradients", "GbtSampleExamples",
    "GbtTrainTree", "GbtUpdatePredictions", "GbtValidationEval", "GbtFinalize",
)

# One "<Token> <seconds>s" timing pair on a per-depth LOG line.
_TIMING_PAIR_RX = re.compile(r"([A-Za-z_]\w*)\s+(-?[0-9.eE+-]+)s")

# std::this_thread::get_id() streams implementation-defined: decimal under
# libstdc++ (Linux), a hex pointer under libc++ (macOS). Accept either — it is
# only a grouping key.
_THREAD_ID_RX = r"(0x[0-9a-fA-F]+|\d+)"

# Per-depth LOG line (one per thread x tree x depth): capture the fixed prefix
# only, scanning the trailing pairs with _TIMING_PAIR_RX. Order-independent, so
# an unknown scope no longer truncates the line as the old positional regex did.
_PER_DEPTH_RX = re.compile(
    r"thread\s+" + _THREAD_ID_RX +
    r"\s+tree\s+(\d+)\s+depth\s+(\d+)\s+nodes\s+(\d+)\s+samples\s+(\d+)"
    r"([^\n]*)"
)


def parse_parallel_chrono(raw_log: str) -> pd.DataFrame:
    rows = []
    for m in _PER_DEPTH_RX.finditer(raw_log):
        tid, tree, depth, nodes, samples, rest = m.groups()

        # Seed every known column to 0.0, then overlay this line's pairs. Unknown
        # tokens map to themselves, so nothing is dropped.
        row = dict.fromkeys(_TIMING_COLS, 0.0)
        row["thread"]  = int(tid, 0)   # 0x.. (libc++) or decimal (libstdc++)
        row["tree"]    = int(tree)
        row["depth"]   = int(depth)
        row["nodes"]   = int(nodes)
        row["samples"] = int(samples)
        for tok, val in _TIMING_PAIR_RX.findall(rest):
            row[_LOG_TOKEN_TO_COL.get(tok, tok)] = float(val)
        rows.append(row)

    if not rows:
        raise ValueError("no parallel-chrono lines found in log")

    df = pd.DataFrame(rows)

    # Session-level GpuInit is emitted once after the per-tree loop and is not
    # per-depth, so park it in the unused tree=0/depth=0 placeholder row.
    m_session = re.search(r"session\s+GpuInit\s+([0-9.eE+-]+)s", raw_log)
    if m_session:
        placeholder = (df["tree"] == 0) & (df["depth"] == 0)
        df.loc[placeholder, "GpuInit"] = float(m_session.group(1))

    # GBT session scopes arrive on one "GBT chrono (ms): ..." line and are global,
    # so — like GpuInit — park them in the first tree=0/depth=0 slot; one row only,
    # so a cross-thread sum can't double-count. ms -> s to match the rest.
    m_gbt = re.search(r"GBT chrono \(ms\):(.*)", raw_log)
    if m_gbt:
        ph_idx = df.index[(df["tree"] == 0) & (df["depth"] == 0)]
        if len(ph_idx):
            target = ph_idx[0]
            for tok, val in _GBT_PAIR_RX.findall(m_gbt.group(1)):
                col = _GBT_TOKEN_TO_COL.get(tok)
                if col is None:
                    continue
                if col not in df.columns:
                    df[col] = 0.0
                df.at[target, col] = float(val) / 1000.0  # ms -> s

    # ──────────────────────────────────────────────────────────────────────
    # Build one block per thread (unchanged logic)
    # ──────────────────────────────────────────────────────────────────────
    blocks = []
    for tid, g in df.groupby("thread", sort=True):
        g = g.sort_values(["tree", "depth"]).reset_index(drop=True)

        # Symmetric optimized is special: its shared SampleProjection and
        # bag-wide ApplyProjection run in GrowTreeLocalBFS before NodeTrain, so
        # detect it from the Sym* scopes and render them as TreeTrain children.
        symmetric_optimized = any(
            c in g.columns and (g[c] != 0.0).any()
            for c in ("SymBuildBag", "SymSortBag", "SymSweep"))

        # A column's dash prefix = its depth in the CHRONO hierarchy. GPU stages
        # stay flat: they replace ApplyProjection rather than nest under it.
        # symmetric_optimized lifts SampleProjection/AP to 1 and Sym* to 2.
        #   0  TreeTrain
        #   1  ├─ DepthTrain (BFS builds: the whole depth, ⊇ everything below)
        #   1  ├─ NodeTrain (and the BFS-only BfsNodeLoop scheduler scope)
        #   2  │  ├─ SetLeafValue / FindBestCondition / SplitExamplesInPlace
        #   3  │  │  ├─ ObliqueSplitSearch / AxisAlignedSplitSearch
        #   4  │  │  │  ├─ FindObliqueSetup / SampleProjection /
        #   4  │  │  │  │  ApplyProjection / EvaluateProj
        #   5  │  │  │  │  ├─ (ApplyProjection) Sym*/Dw1* ; (EvaluateProj) Cart/Histo
        #   6  │  │  │  │  │  └─ Dw1Sweep* ; Cart/Histo split-finder leaves
        #   7  │  │  │  │  │     └─ MinMaxNumerical (under HistogramSetup)
        g = g.rename(columns={
            "samples": "Active Samples",
            # depth 0 — top-level per-tree scope (non-zero only at depth=0 of
            # each tree). The DFS analogue of the BFS top-level scope.
            "TreeTrain":                    "TreeTrain",
            # depth 1 — under TreeTrain. DepthTrain wraps a whole BFS depth
            # (fused Apply included), so it is a parent of the two below;
            # rendered as their sibling, same as BfsNodeLoop already is.
            "DepthTrain":                   "-DepthTrain",
            "BfsNodeLoop":                  "-BfsNodeLoop",
            "NodeTrain":                    "-NodeTrain",
            # depth 2 — under NodeTrain.
            "SetLeafValue":                 "--SetLeafValue",
            "FindBestCondition":            "--FindBestCondition",
            "SplitExamplesInPlace":         "--SplitExamplesInPlace",
            # depth 3 — under FindBestCondition.
            "ObliqueSplitSearch":           "---ObliqueSplitSearch",
            "AxisAlignedSplitSearch":       "---AxisAlignedSplitSearch",
            # depth 4 — under ObliqueSplitSearch (per-node setup + per-K loop).
            "FindObliqueSetup":             "----FindObliqueSetup",
            "SampleProjection": (
                "-SampleProjection" if symmetric_optimized
                else "----SampleProjection"),
            "ProjectionEvaluate": (
                "-ApplyProjection" if symmetric_optimized
                else "----ApplyProjection"),
            "EvaluateProj":                 "----EvaluateProj",
            # Axis-aligned candidate-selection/search tail under FindBestCondition.
            "GetCandidateAttributes": (
                "---GetCandidateAttributes" if symmetric_optimized
                else "----GetCandidateAttributes"),
            "GetCandidateAttributesAssign": (
                "----GetCandidateAttributesAssign" if symmetric_optimized
                else "-----GetCandidateAttributesAssign"),
            "GetCandidateAttributesShuffle": (
                "----GetCandidateAttributesShuffle" if symmetric_optimized
                else "-----GetCandidateAttributesShuffle"),
            "GetCandidateAttributesNumToTest": (
                "----GetCandidateAttributesNumToTest" if symmetric_optimized
                else "-----GetCandidateAttributesNumToTest"),
            "ColumnWithCast":               "----ColumnWithCast",
            # ApplyProjection sub-phases. In symmetric depthwise AP they are
            # under the TreeTrain-level ApplyProjection; otherwise they are
            # displayed under the node-local ApplyProjection scope.
            "SymBuildBag": (
                "--SymBuildBag" if symmetric_optimized else "-----SymBuildBag"),
            "SymSortBag": (
                "--SymSortBag" if symmetric_optimized else "-----SymSortBag"),
            "SymSweep": (
                "--SymSweep" if symmetric_optimized else "-----SymSweep"),
            # One level under SymSweep: per-k column setup + the bag pass.
            "SymSweepColSetup": (
                "---SymSweepColSetup" if symmetric_optimized
                else "------SymSweepColSetup"),
            "SymSweepMainLoop": (
                "---SymSweepMainLoop" if symmetric_optimized
                else "------SymSweepMainLoop"),
            "Dw1PreSize":                   "-----Dw1PreSize",
            "Dw1Sweep":                     "-----Dw1Sweep",
            # depth 5 — EvaluateProj's two split-finder paths.
            "CartPath":                     "-----CartPath",
            "HistoPath":                    "-----HistoPath",
            # depth 6 — Dw1Sweep sub-phases (sum to Dw1Sweep modulo ctor/glue).
            # DW1 shared-rows: merged per-block bag build + sort (replaces the
            # per-node gather; sits beside the Bucket/Scatter/ColWalk siblings).
            "Dw1SharedBag":                 "------Dw1SharedBag",
            # Column bucketing that precedes the sweep's column walk.
            "Dw1SweepSetup":                "------Dw1SweepSetup",
            "Dw1SweepColWalk":              "------Dw1SweepColWalk",
            # depth 7 — ColWalk sub-loops (group-by-node pass + bag scatter pass).
            "Dw1ColWalkGroupByNode":        "-------Dw1ColWalkGroupByNode",
            "Dw1ColWalkBagScatter":         "-------Dw1ColWalkBagScatter",
            "Dw1SweepBig":                  "------Dw1SweepBig",
            "Dw1SweepGeneric":              "------Dw1SweepGeneric",
            # depth 6 — CartPath leaves (CPU Exact/Sort splitter).
            "CartSetup":                    "------CartSetup",
            "SortFillExampleBucketSet":     "------SortFillExampleBucketSet",
            "SortInitBuckets":              "------SortInitBuckets",
            "SortFillBuckets":              "------SortFillBuckets",
            "SortFinalizeBuckets":          "------SortFinalizeBuckets",
            "SortFeatures":                 "------SortFeatures",
            "SortLabels":                   "------SortLabels",
            "SortScanSplits":               "------SortScanSplits",
            "ScanPresorted":                "------ScanPresorted",
            # depth 6 — HistoPath leaves (CPU histogram splitter).
            "EntropyTableSetup":            "------EntropyTableSetup",
            "HistogramSetup":               "------HistogramSetup",
            "AssignSamplesToHist":          "------AssignSamplesToHist",
            "SelectBestThresholdHistogram": "------SelectBestThresholdHistogram",
            # depth 7 — under HistogramSetup (excluded from the CSV below).
            "MinMaxNumerical":              "-------MinMaxNumerical",
        })

        g = g.drop(columns=["thread"])

        # Columns in pre-order over the CHRONO hierarchy, so left-to-right walks
        # the call tree. GPU kernels and the Sym*/Dw1* phases sit under
        # ApplyProjection, being alternative implementations of it.
        desired_order = [
            "tree", "depth", "nodes", "Active Samples",

            # GBT session scopes: flat, wrapping the whole boosting run above any
            # TreeTrain, non-zero only in the placeholder row. GbtTrainTree wraps
            # every decision_tree::Train call (⊇ Σ TreeTrain); RF zero-drops all.
            "GbtStartup",
            "GbtPreprocess",
            "GbtUpdateGradients",
            "GbtSampleExamples",
            "GbtTrainTree",
            "GbtUpdatePredictions",
            "GbtValidationEval",
            "GbtFinalize",

            # depth 0 — top-level per-tree scope.
            "TreeTrain",
            # Symmetric depthwise AP: these run once per depth cohort before
            # per-node NodeTrain, so they are TreeTrain children.
            "-SampleProjection",
            "-ApplyProjection",
            "--SymBuildBag",
            "--SymSortBag",
            "--SymSweep",
            "---SymSweepColSetup",
            "---SymSweepMainLoop",
            # depth 1 — under TreeTrain.
            "-DepthTrain",           # BFS builds: the depth's full wall-time
            "-BfsNodeLoop",          # BFS-only scheduler scope
            "-NodeTrain",
            # depth 2 — under NodeTrain.
            "--SetLeafValue",
            "--FindBestCondition",
            # depth 3 — under FindBestCondition.
            "---ObliqueSplitSearch",
            # depth 4 — under ObliqueSplitSearch (per-node setup + per-K loop).
            "----FindObliqueSetup",
            "----SampleProjection",
            "----ApplyProjection",   # kProjectionEvaluate — col-major gather/FMA
            #   GPU kernels (flat): ApplyProjection replacement on the GPU path.
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
            #   depth 5 — ApplyProjection sub-phases (Sym*/Dw1*, build-mode
            #   exclusive), then depth 6 Dw1Sweep children.
            "-----SymBuildBag",
            "-----SymSortBag",
            "-----SymSweep",
            "------SymSweepColSetup",
            "------SymSweepMainLoop",
            "-----Dw1PreSize",
            "-----Dw1Sweep",
            "------Dw1SharedBag",
            "------Dw1SweepSetup",
            "------Dw1SweepColWalk",
            "-------Dw1ColWalkGroupByNode",
            "-------Dw1ColWalkBagScatter",
            "------Dw1SweepBig",
            "------Dw1SweepGeneric",
            # depth 4 — EvaluateProj split-finder dispatch.
            "----EvaluateProj",
            #   depth 5 — CART path (EXACT, the DFS col-major default), depth 6 leaves.
            "-----CartPath",
            "------CartSetup",
            "------SortFillExampleBucketSet",
            "------SortInitBuckets",
            "------SortFillBuckets",
            "------SortFinalizeBuckets",
            "------SortFeatures",
            "------SortLabels",
            "------SortScanSplits",
            "------ScanPresorted",
            #   depth 5 — Histogram path (DYNAMIC_* split types), depth 6 leaves.
            "-----HistoPath",
            "------EntropyTableSetup",
            "------HistogramSetup",
            "------AssignSamplesToHist",
            "------SelectBestThresholdHistogram",
            # depth 3 — axis-aligned splitter tail (timing noise under oblique).
            "---AxisAlignedSplitSearch",
            "---GetCandidateAttributes",
            "----GetCandidateAttributesAssign",
            "----GetCandidateAttributesShuffle",
            "----GetCandidateAttributesNumToTest",
            "----GetCandidateAttributes",
            "-----GetCandidateAttributesAssign",
            "-----GetCandidateAttributesShuffle",
            "-----GetCandidateAttributesNumToTest",
            "----ColumnWithCast",
            # depth 2 — per-node finish.
            "--SplitExamplesInPlace",
        ]
        ordered = [c for c in desired_order if c in g.columns]
        remaining = [c for c in g.columns if c not in desired_order]
        g = g[ordered + remaining]

        # Columns intentionally excluded from the CSV (still chrono'd in
        # C++). MinMaxNumerical is nested inside ------HistogramSetup, so no
        # coverage is lost.
        excluded = ["-------MinMaxNumerical"]
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
    ensemble_label = ""
    if a.ensemble_method == "Boosting":
        ensemble_label = "Boosting | "
    # Non-default vectorization gets its own results subtree so a scalar or
    # AVX-512 run cannot clobber the avx2 default it is compared against.
    exp = (f"{gpu_mode_label}{ensemble_label}{a.feature_split_type} | "
           f"{a.numerical_split_type}{_VEC_LABEL[a.vectorized]}")
    
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
    if a.ensemble_method is not None:
        cmd.append(f"--ensemble_method={a.ensemble_method}")
    if a.shrinkage is not None:
        cmd.append(f"--shrinkage={a.shrinkage}")

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
    # Raw timing CSVs (and their .log) go in <dataset_name>/raw/; avg_per_depth.py
    # writes the averages in <dataset_name>/ itself — that split, not a filename
    # suffix, is what distinguishes the two.
    out_dir = (
        Path("benchmarks/results/per_function_timing")
        / chrono_level_name(a.chrono_level)
        / device_name
        / exp
        / dataset_name
        / "raw"
    )
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
            ensemble_suffix = f"-{a.ensemble_method}" if a.ensemble_method == "Boosting" else ""
            log_fp = out_dir / f"{a.feature_split_type}-{a.numerical_split_type}{ensemble_suffix}-{a.num_threads}t-{ts}.log"

            try:
                log_fp.write_text(log_plain, encoding="utf-8")
                print("Raw log saved to", log_fp)
            except Exception as err:
                print(f"⚠️  Could not write log file: {err}", file=sys.stderr)

        if proc.returncode < 0:
            print(f"binary died with signal {-proc.returncode}")

        dt = time.perf_counter() - t0
        log_plain = re.sub(r'\x1B\[[0-?]*[ -/]*[@-~]', '', log)

        # Prefer the learner's "Training block took: X s": subprocess wall time
        # also covers dataset generation, config and GPU init, overstating the
        # training cost. Fall back to it only when the marker is missing.
        m_train = re.search(
            r"(?:random_forest|gradient_boosted_trees)\.cc Training block took:\s*([0-9.eE+-]+)\s*s",
            log_plain)
        m_total = re.search(
            r"train_oblique_forest wall-time - Training \+ Post-Processing:\s*([0-9.eE+-]+)s",
            log_plain)
        train_from_learner_marker = m_train is not None
        if not m_train:
            m_train = m_total
        if m_train:
            train_block = float(m_train.group(1))
            print(f"\n⏱  Training block took {train_block:.4f} s"
                  f"   (subprocess wall: {dt:.4f} s)\n")
        else:
            train_block = None
            print(f"\n⏱  Binary subprocess ran for {dt:.4f} s"
                  f"   (Training-block marker not found in log)\n")

        # ── Phase decomposition: Loading/Init | Training | Post-processing ──
        # Loading/Init = harness pre-train wall + the "Dataset load took" markers
        # from inside TrainWithStatus. Post-processing = harness end wall − that
        # load − the training block. A manual exit() drops the end marker.
        dataset_load = sum(float(x) for x in re.findall(
            r"Dataset load took:\s*([0-9.eE+-]+)\s*s", log_plain))
        m_init = re.search(
            r"train_oblique_forest wall-time - Loading/Init \(pre-train\):\s*([0-9.eE+-]+)s",
            log_plain)
        loading_init = (float(m_init.group(1)) + dataset_load) if m_init else None
        if m_total and train_from_learner_marker:
            post_proc = float(m_total.group(1)) - dataset_load - train_block
        else:
            post_proc = None

        def _fmt_phase(v, missing):
            return f"{v:.4f} s" if v is not None else missing

        print("⏱  Phases: "
              f"Loading/Init {_fmt_phase(loading_init, '(no marker — rebuild binary)')} | "
              f"Training {_fmt_phase(train_block, '(no marker)')} | "
              f"Post-processing {_fmt_phase(post_proc, '(skipped)')}"
              f"   (subprocess wall {dt:.4f} s)\n")

        # ── CHRONO coverage vs Training-block wall ────────────────────
        # Sums the top-level scopes over every (tree, depth, thread) entry. Each
        # tree's time is single-threaded, so Σ scope ≈ train_block ×
        # min(N_trees, T) when load-balanced, and exactly train_block at T=1.
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
            # BFS builds only: covers the fused Apply that runs outside
            # NodeTrain, so it is the honest per-depth total under DW1.
            depth_total = _sum_scope("DepthTrain")
            if depth_total > 0:
                pct = (depth_total / cpu_budget * 100.0) if cpu_budget > 0 else 0.0
                print(f"  Σ DepthTrain (BFS depth loop):{depth_total:>10.4f} s  "
                      f"({pct:5.1f}% of CPU budget)")
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
            ("Machine", f"{utils.get_cpu_model_proc()} (nproc={os.cpu_count()})"),
            ("Machine serial", utils.get_machine_serial()),
            ("Helper invocation", helper_invocation),
            ("Bazel build", utils.last_build_cmd if utils.last_build_cmd else "(build skipped)"),
            ("Binary command", binary_cmd_str),
            ("Vectorization",
             f"{a.vectorized} (histogram_num_bins={a.histogram_num_bins})"),
            ("Loading/init time (s)",
             f"{loading_init:.4f}" if loading_init is not None else "(marker not found)"),
            ("Training time (s)",
             f"{train_block:.4f}" if train_block is not None else "(marker not found)"),
            ("Post-processing time (s)",
             f"{post_proc:.4f}" if post_proc is not None else "(skipped)"),
        ]

        write_csv(table, cmd_lines, out_fp)

        print("CSV written to", out_fp)

    except Exception as e:
        print("❌", e, file=sys.stderr)
    finally:
        utils.cleanup_and_exit()
