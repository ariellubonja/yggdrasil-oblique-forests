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
    p.add_argument("--ensemble_method", choices=["Bagging", "Boosting"], default="Bagging",
                   help="Ensemble method: 'Bagging' (Random Forest) or 'Boosting' (Gradient Boosted Trees/MART).")
    p.add_argument("--shrinkage", type=float, default=0.1,
                   help="Learning rate for boosting (only used when --ensemble_method=Boosting).")

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


def chrono_level_name(chrono_level: int) -> str:
    if chrono_level == 1:
        return "COARSE"
    if chrono_level == 2:
        return "FINE"
    raise ValueError(f"Unsupported chrono_level: {chrono_level}")


# Log token (as printed by random_forest.cc's per-depth LOG line) -> internal
# column name. Only tokens whose printed spelling differs from the internal
# name need an entry; every other token passes through unchanged, so a CHRONO
# scope newly added on the C++ side is captured automatically instead of being
# silently dropped.
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

# GBT session-level scopes. gradient_boosted_trees.cc dumps these once, on a
# single aggregate line ("GBT chrono (ms): startup=.. train_tree=.. ..") in
# MILLISECONDS and in `name=value` form -- neither the per-depth line shape nor
# the "<Token> <seconds>s" pair shape the per-depth parser understands. So they
# are captured separately (see parse_parallel_chrono) and mapped to these
# columns; values are converted ms->s to match every other timing column.
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

# Every timing column the downstream pipeline knows about, seeded to 0.0 on
# each row so a thread/depth that never emitted a given token still carries the
# column (the old fixed-schema behaviour). Tokens captured but absent here
# (e.g. a brand-new scope) are still added per row and surface as trailing
# "unordered" CSV columns -- never dropped.
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
    "Dw1PreSize", "Dw1Sweep",
    "Dw1SweepColWalk", "Dw1ColWalkGroupByNode", "Dw1ColWalkBagScatter",
    "Dw1ColWalkSlabAccum",
    "Dw1SweepBig", "Dw1SweepGeneric", "Dw1SharedBag",
    "NodeTrain", "FindBestCondition", "ObliqueSplitSearch",
    "FindObliqueSetup", "EvaluateProj", "EntropyTableSetup", "CartPath",
    "CartSetup", "HistoPath", "AxisAlignedSplitSearch", "SampleProjection",
    "SplitExamplesInPlace", "SetLeafValue", "BfsNodeLoop", "TreeTrain",
    # GBT session-level scopes (overlaid from the "GBT chrono (ms):" line onto
    # one placeholder row; see parse_parallel_chrono). Absent -> stay 0.0 ->
    # zero-dropped, so Bagging/RF runs are unaffected.
    "GbtStartup", "GbtPreprocess", "GbtUpdateGradients", "GbtSampleExamples",
    "GbtTrainTree", "GbtUpdatePredictions", "GbtValidationEval", "GbtFinalize",
)

# One "<Token> <seconds>s" timing pair on a per-depth LOG line.
_TIMING_PAIR_RX = re.compile(r"([A-Za-z_]\w*)\s+(-?[0-9.eE+-]+)s")

# Thread identifier printed by random_forest.cc is `std::this_thread::get_id()`
# streamed via operator<<. That representation is implementation-defined:
# libstdc++ (Linux benchmark box) prints a decimal number, libc++ (macOS)
# prints a hex pointer like `0x16da57000`. Accept either so the parser works
# on both — the value is only used as a per-thread grouping key.
_THREAD_ID_RX = r"(0x[0-9a-fA-F]+|\d+)"

# Per-depth LOG line emitted by random_forest.cc (one per thread x tree x
# depth). Capture the fixed prefix; the trailing timing pairs are scanned
# separately with _TIMING_PAIR_RX. This is order-independent and forward
# compatible -- a new CHRONO scope no longer truncates the line. (The old
# positional regex stopped at the first unrecognized token, so the
# DW1_SHARED_ROWS `Dw1SharedBag` token hid SampleProjection and the whole
# NodeTrain tail emitted after it.)
_PER_DEPTH_RX = re.compile(
    r"thread\s+" + _THREAD_ID_RX +
    r"\s+tree\s+(\d+)\s+depth\s+(\d+)\s+nodes\s+(\d+)\s+samples\s+(\d+)"
    r"([^\n]*)"
)


def parse_parallel_chrono(raw_log: str) -> pd.DataFrame:
    rows = []
    for m in _PER_DEPTH_RX.finditer(raw_log):
        tid, tree, depth, nodes, samples, rest = m.groups()

        # Seed every known column to 0.0, then overlay whatever timing pairs
        # this line actually carries. Unknown tokens map to themselves, so
        # nothing is ever dropped -- order and presence are both irrelevant.
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

    # Session-level GpuInit is emitted once by random_forest.cc after the
    # per-tree loop. It's not per-depth (fires before any TreeScope), so
    # we park its value in the otherwise-unused tree=0, depth=0
    # placeholder row — the one with nodes=0 and all timings zero.
    m_session = re.search(r"session\s+GpuInit\s+([0-9.eE+-]+)s", raw_log)
    if m_session:
        placeholder = (df["tree"] == 0) & (df["depth"] == 0)
        df.loc[placeholder, "GpuInit"] = float(m_session.group(1))

    # GBT session-level scopes come on a single "GBT chrono (ms): ..." line
    # (gradient_boosted_trees.cc, CHRONO_PROFILE>=2). They are global, not
    # per-thread/depth, so -- like GpuInit -- park them in a single placeholder
    # row (the first tree=0/depth=0 slot). Only one row gets them so a
    # cross-thread column sum is not double-counted. ms -> s to match the rest.
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

        # Friendlier column names + dashes to show scope nesting. GPU
        # columns are flat (one column per helper stage) — no dashes. CPU
        # split-finder columns retain their nested dashes.
        #
        # Symmetric depthwise AP is special: shared SampleProjection and
        # bag-wide ApplyProjection run in GrowTreeLocalBFS before NodeTrain,
        # not inside FindBestConditionSparseObliqueTemplate. Detect it from
        # its Sym* sub-scopes and render those scopes as TreeTrain children.
        symmetric_depthwise = any(
            c in g.columns and (g[c] != 0.0).any()
            for c in ("SymBuildBag", "SymSortBag", "SymSweep"))

        # The dash prefix on each renamed column equals the scope's displayed
        # depth in the CHRONO hierarchy, with TreeTrain at depth 0:
        #   0  TreeTrain
        #   1  ├─ NodeTrain (and the BFS-only BfsNodeLoop scheduler scope)
        #   2  │  ├─ SetLeafValue / FindBestCondition / SplitExamplesInPlace
        #   3  │  │  ├─ ObliqueSplitSearch / AxisAlignedSplitSearch
        #   4  │  │  │  ├─ FindObliqueSetup / SampleProjection /
        #   4  │  │  │  │  ApplyProjection / EvaluateProj
        #   5  │  │  │  │  ├─ (ApplyProjection) Sym*/Dw1* ; (EvaluateProj) Cart/Histo
        #   6  │  │  │  │  │  └─ Dw1Sweep* ; Cart/Histo split-finder leaves
        #   7  │  │  │  │  │     └─ MinMaxNumerical (under HistogramSetup)
        # In symmetric_depthwise, SampleProjection / ApplyProjection move to
        # depth 1 and Sym* moves to depth 2 because they run before NodeTrain.
        # GPU helper-stage columns stay flat (no dashes): they are alternative
        # implementations of ApplyProjection, not nested scopes.
        g = g.rename(columns={
            "samples": "Active Samples",
            # depth 0 — top-level per-tree scope (non-zero only at depth=0 of
            # each tree). The DFS analogue of the BFS top-level scope.
            "TreeTrain":                    "TreeTrain",
            # depth 1 — under TreeTrain.
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
                "-SampleProjection" if symmetric_depthwise
                else "----SampleProjection"),
            "ProjectionEvaluate": (
                "-ApplyProjection" if symmetric_depthwise
                else "----ApplyProjection"),
            "EvaluateProj":                 "----EvaluateProj",
            # Axis-aligned candidate-selection/search tail under FindBestCondition.
            "GetCandidateAttributes": (
                "---GetCandidateAttributes" if symmetric_depthwise
                else "----GetCandidateAttributes"),
            "GetCandidateAttributesAssign": (
                "----GetCandidateAttributesAssign" if symmetric_depthwise
                else "-----GetCandidateAttributesAssign"),
            "GetCandidateAttributesShuffle": (
                "----GetCandidateAttributesShuffle" if symmetric_depthwise
                else "-----GetCandidateAttributesShuffle"),
            "GetCandidateAttributesNumToTest": (
                "----GetCandidateAttributesNumToTest" if symmetric_depthwise
                else "-----GetCandidateAttributesNumToTest"),
            "ColumnWithCast":               "----ColumnWithCast",
            # ApplyProjection sub-phases. In symmetric depthwise AP they are
            # under the TreeTrain-level ApplyProjection; otherwise they are
            # displayed under the node-local ApplyProjection scope.
            "SymBuildBag": (
                "--SymBuildBag" if symmetric_depthwise else "-----SymBuildBag"),
            "SymSortBag": (
                "--SymSortBag" if symmetric_depthwise else "-----SymSortBag"),
            "SymSweep": (
                "--SymSweep" if symmetric_depthwise else "-----SymSweep"),
            "Dw1PreSize":                   "-----Dw1PreSize",
            "Dw1Sweep":                     "-----Dw1Sweep",
            # depth 5 — EvaluateProj's two split-finder paths.
            "CartPath":                     "-----CartPath",
            "HistoPath":                    "-----HistoPath",
            # depth 6 — Dw1Sweep sub-phases (sum to Dw1Sweep modulo ctor/glue).
            # -DDW1_SHARED_ROWS: merged per-block bag build + sort (replaces the
            # per-node gather; sits beside the Bucket/Scatter/ColWalk siblings).
            "Dw1SharedBag":                 "------Dw1SharedBag",
            "Dw1SweepColWalk":              "------Dw1SweepColWalk",
            # depth 7 — ColWalk sub-loops (group-by-node pass + bag scatter pass).
            "Dw1ColWalkGroupByNode":        "-------Dw1ColWalkGroupByNode",
            "Dw1ColWalkBagScatter":         "-------Dw1ColWalkBagScatter",
            # depth 8 — innermost scatter FMA, nested in Dw1ColWalkBagScatter.
            "Dw1ColWalkSlabAccum":          "--------Dw1ColWalkSlabAccum",
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

        # Reorder columns as a strict pre-order (DFS) traversal of the
        # CHRONO scope hierarchy, so reading left-to-right walks the call
        # tree parent→child. Each scope is listed immediately before its
        # own children; the dash prefixes already encode nesting depth.
        # GPU kernels and the Sym*/Dw1* fused-Apply phases are placed under
        # ApplyProjection because they are alternative implementations of
        # that scope (selected at build/runtime). Columns missing for the
        # current mode are filtered out below; all-zero columns are
        # zero-dropped further down.
        desired_order = [
            "tree", "depth", "nodes", "Active Samples",

            # GBT session-level scopes (flat, no dashes — they wrap the whole
            # boosting run, above any per-tree TreeTrain, and are non-zero only
            # in the single placeholder row). GbtTrainTree is the GBT-side
            # wrapper around all decision_tree::Train calls (⊇ Σ TreeTrain);
            # the rest are the per-iteration / one-shot GBT loop phases. Absent
            # for Bagging/RF runs, where they zero-drop.
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
            # depth 1 — under TreeTrain.
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
            "-----Dw1PreSize",
            "-----Dw1Sweep",
            "------Dw1SharedBag",
            "------Dw1SweepColWalk",
            "-------Dw1ColWalkGroupByNode",
            "-------Dw1ColWalkBagScatter",
            "--------Dw1ColWalkSlabAccum",
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
    exp = f"{gpu_mode_label}{ensemble_label}{a.feature_split_type} | {a.numerical_split_type}"
    
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
    # Raw timing CSVs (and their sibling .log) go under a `raw/` subfolder of
    # the <dataset_name> dir; avg_per_depth.py writes the per-depth averages up
    # in the <dataset_name> dir itself (the raw/ vs. parent split is what now
    # distinguishes the two — the averages no longer carry a suffix).
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

        # Prefer the training-only wall time logged by random_forest.cc
        # ("Training block took: X s"). The subprocess wall time also
        # includes dataset generation, learner config, GPU init, etc., so
        # it overstates the training cost. Fall back to subprocess wall
        # time if the marker is missing (e.g. binary crashed before the
        # log line, or chrono build dropped the marker).
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
        # Loading/Init = harness pre-train wall (flag parsing, dataspec
        # inference, synthetic generation, tfrecord load) + the dataset load
        # that happens inside TrainWithStatus ("Dataset load took" markers:
        # abstract_learner.cc for csv/column, the harness for csv/row).
        # Post-processing = harness Training+Post-Processing wall − dataset
        # load − learner training block (OOB, validation, model finalize).
        # A manual exit() after the learner's training-block log (used to
        # skip RF post-processing) drops the harness end marker → "(skipped)".
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
            ("Machine", f"{utils.get_cpu_model_proc()} (nproc={os.cpu_count()})"),
            ("Machine serial", utils.get_machine_serial()),
            ("Helper invocation", helper_invocation),
            ("Bazel build", utils.last_build_cmd if utils.last_build_cmd else "(build skipped)"),
            ("Binary command", binary_cmd_str),
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
