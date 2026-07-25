#pragma once
// Chrono profiling has one always-on coarse base plus three INDEPENDENT axes,
// selected by bazel build flags (see .bazelrc):
//   --config=coarse_chrono_profile         : -DCHRONO_PROFILE=1
//   --config=fine_chrono_applyprojection   : -DCHRONO_PROFILE=1 -DFINE_CHRONO_AP
//   --config=fine_chrono_evaluateprojection: -DCHRONO_PROFILE=1 -DFINE_CHRONO_EP
//   --config=nodewise_chrono               : -DCHRONO_PROFILE=1 -DNODEWISE_CHRONO
//
//   CHRONO_PROFILE undefined : no profiling — every macro compiles to nothing.
//   CHRONO_PROFILE defined   : the coarse base is active. The top-level scopes
//               (TreeTrain, NodeTrain, SampleProjection, EvaluateProj,
//               ProjectionEvaluate, BfsNodeLoop) plus all node-bookkeeping /
//               split-manager / GBT scopes fire via the CHRONO_*_COARSE family.
//   FINE_CHRONO_AP : also fire the inner scopes of ProjectionEvaluator::Evaluate
//               and its fused-apply variants (symmetric / depthwise_1pass), via
//               the CHRONO_*_AP family.
//   FINE_CHRONO_EP : also fire the inner scopes of EvaluateProjection (the
//               histogram / Cart split search), via the CHRONO_*_EP family.
//   NODEWISE_CHRONO : keeps the coarse (tree, depth) aggregation exactly as is,
//               and ADDITIONALLY records one row per (node, projection) call to
//               ProjectionEvaluator::Evaluate at a small set of gated depths.
//               Answers "does a small subset of nodes carry most of the
//               ApplyProjection cost?", which the depth-aggregated tables
//               cannot. See the NODEWISE_CHRONO section below.
// The three axes are independent — each includes coarse but not the others;
// FINE-everywhere = pass both fine configs together. The legacy numeric
// `CHRONO_PROFILE >= 2` gate and the plain CHRONO_SCOPE / CHRONO_BEGIN family
// are kept defined purely so the untouched GPU code still compiles; no config
// sets level 2, so those GPU scopes stay dormant.
#ifdef CHRONO_PROFILE

#include <array>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <thread>
#include <vector>

#ifdef NODEWISE_CHRONO
#include <algorithm>
#include <cstdlib>
#include <fstream>
#include <string>
#include <tuple>
#endif

namespace yggdrasil_decision_forests::chrono_prof {

// ---------- enum + fallback global --------------------------------
enum FuncId {
  kTreeTrain = 0,
  kProjectionEvaluate,

  kSortFillExampleBucketSet,
  kSortScanSplits,
  kSortInitBuckets,
  kSortFillBuckets,
  kSortFinalizeBuckets,
  kSortFeatures,
  kSortLabels,
  kScanPresorted,

  kHistogramSetup,
  kMinMaxNumerical,
  kAssignSamplesToHistogram,
  kSelectBestThresholdHistogram,
  kGetCandidateAttributes,
  kGetCandidateAttributesAssign,
  kGetCandidateAttributesShuffle,
  kGetCandidateAttributesNumToTest,
  kColumnWithCast,

  // GBT-level scopes (set in gradient_boosted_trees.cc). These accumulate
  // to global_stats[id] because GBT does not open a TreeScope around its
  // per-iter work — TreeScope is only set by random_forest.cc per tree.
  kGbtStartup,
  kGbtPreprocessDataset,
  kGbtUpdateGradients,
  kGbtSampleExamples,
  kGbtTrainTree,
  kGbtUpdatePredictions,
  kGbtValidationEval,
  kGbtFinalize,

  // CPU-side scopes around GPU dispatch. Kept as-is.
  kGpuInit,
  kGpuCsrFlatten,
  kGpuMutexWait,
  kGpuSampleProjectionsBatch,

  // Per-stage GPU timings, measured via cudaEvent_t inside each bridge.
  // Named after the helper function bracketing the stage. Mutually
  // exclusive by mode (zero-drop hides the unused ones per run).
  kGpuApplyColumnADD,           // ApplyProjectionColumnADD (nodewise apply)
  kGpuApplyColumnADDMultiNode,  // ApplyProjectionColumnADDMultiNode (fused depthwise apply + segmented min/max)
  kGpuRandomHistogram,          // RandomHistogram (Random hist stage)
  kGpuSplitHistogram,           // HistogramSplit (Random split stage)
  kGpuSortIndices,              // ThrustSortIndicesOnly (Exact sort stage)
  kGpuExactSplit,               // ExactSplit (Exact gain/argmax stage)
  kGpuOther,                    // Residual = bridge total − Σ tracked stages

  // Sub-phases of ApplyProjectionsSymmetricDepthwiseAP. Only emitted when
  // compiled with -DSYMMETRIC_DEPTHWISE_AP; left at zero otherwise so the
  // enum values stay stable across builds.
  kSymBuildBag,   // Pre-size slabs (+ fallback concat / root-span copy)
  kSymSortBag,    // O(bag) incremental relabel of the previous depth's sorted
                  // bag (fallback: K32V32 VQSort rebuild)
  kSymSweep,      // K bag-wide stride-1 projection sweeps (the hot loop)

  // Sub-phases of ApplyProjectionsDepthwise1Pass. Only emitted when compiled
  // with -DDEPTHWISE_1_PASS; zero otherwise. The Sweep scope wraps the
  // ConcurrentForLoop synchronization point in the caller thread, so
  // wall-clock attribution is preserved even when num_threads > 1.
  kDw1PreSize,    // proj_prefix sum + per-node out_projected[n].assign(...)
  kDw1Sweep,      // ProjectionEvaluator ctor + kernel dispatch (Q tasks)

  // Sub-phases of the Dw1 Sweep, nested inside kDw1Sweep so the children sum
  // back to it (same convention as kEvaluateProj ≈ kCartPath + kHistoPath).
  // All fire once per task (≈ tasks.size() times), never per-row, so the
  // ScopedTimer clock-read overhead stays negligible. Splits the column-centric
  // block into bucketing vs. the gather/FMA hot loop to locate the 93%.
  kDw1SweepColWalk,  // gather/FMA hot loop: o[i] += w * col[sel_ptr[i]]
  kDw1ColWalkGroupByNode,  // inner loop 1: group sorted entries by node into ref_proj/ref_w
  kDw1ColWalkBagScatter,   // inner loop 2: bag pass scatter-accumulate into out_projected
  kDw1SweepBig,      // EvaluateNodeProjMajor path (oversized single node)
  kDw1SweepGeneric,  // !direct fallback (EvaluateProjectionRowsGeneric)
  kDw1SharedBag,     // -DDW1_SHARED_ROWS: per-block merged-bag build + sort
                     // (the stride-1-read colwalk variant; the sweep itself
                     //  still accrues to kDw1SweepColWalk for A/B comparison)

  // Per-node bookkeeping scopes inside NodeTrain /
  // FindBestConditionSparseObliqueTemplate.
  kNodeTrain,
  kFindBestCondition,
  kObliqueSplitSearch,
  kAxisAlignedSplitSearch,
  kSampleProjection,
  kSplitExamplesInPlace,
  kSetLeafValue,

  // Partition of FindBestConditionConcurrentManager (training.cc), the manager
  // used by GBT whenever deployment.num_threads > 1. These four fire on the
  // *manager* thread (i.e. inside a TreeScope), so they land in the per-tree /
  // per-depth tables and satisfy
  //     kFindBestCondition ≈ Setup + Submit + Wait + Process
  // up to the manager's own residual (request struct init, RNG discard).
  //
  // kSplitManagerSetup   : label-stats casts, SplitterWorkRequestCommon, cache
  //                        resizes, GetCandidateAttributes (kGetCandidateAttributes
  //                        nests inside it).
  // kSplitManagerSubmit  : build_request + processor.Submit for both the
  //                        oblique and the axis-aligned scheduling loops.
  // kSplitManagerWait    : processor.GetResult() — the dominant term; it is the
  //                        wall-time the manager spends blocked on the workers.
  // kSplitManagerProcess : recording a response + the in-order processing loop.
  kSplitManagerSetup,
  kSplitManagerSubmit,
  kSplitManagerWait,
  kSplitManagerProcess,

  // Worker-side CPU time, measured inside FindBestConditionFromSplitterWorkRequest
  // on the split-finder threads. Those threads have tls_ctx.cur_tree == -1, so
  // add_time routes them to global_stats (atomic) — they are CPU-time summed
  // over workers, NOT wall-time, and are deliberately kept out of the per-tree /
  // per-depth tables (multiple workers serve one node concurrently, and the
  // by_depth writes are non-atomic — see add_time). These two account for what
  // fills kSplitManagerWait:
  //     ΣSplitWorkerOblique ≈ FindObliqueSetup + SampleProjection +
  //                           ProjectionEvaluate + EvaluateProj (+ dispatch)
  //     kSplitManagerWait   ≲ (SplitWorkerOblique + SplitWorkerAxisAligned)
  //                           / effective_parallelism
  kSplitWorkerOblique,
  kSplitWorkerAxisAligned,

  // Sub-scopes of kObliqueSplitSearch, closing the residual gap between
  // kObliqueSplitSearch and Σ(SampleProjection + ProjectionEvaluate +
  // histogram/sort scopes) observed at ~20% on trunk 3M×4096.
  //
  // kFindObliqueSetup: per-node setup inside FindBestConditionSparseObliqueTemplate
  //   — ProjectionEvaluator ctor, ExtractLabels copy, dense_example_idxs alloc +
  //   iota — fires once per NodeTrain call (not K times).
  // kEvaluateProj: per-K-projection call to EvaluateProjection (the wrapper that
  //   dispatches to FindSplitLabel*FeatureNumericalHistogram / Cart). Captures
  //   the scaffolding around the existing kHistogramSetup / kSortScanSplits /
  //   kSelectBestThresholdHistogram sub-scopes (effective_internal_config copy,
  //   template dispatch, DCHECK loop in debug builds).
  kFindObliqueSetup,
  kEvaluateProj,

  // Sub-scope of kEvaluateProj, isolating BuildCountLogCountTable(total_sum) at
  // training.cc:2324. The lookup-table is rebuilt from scratch per (projection,
  // node) call — std::vector<double>(N_node + 1) allocation + N_node std::log
  // evaluations — sitting in the uncovered gap between kAssignSamplesToHistogram
  // and kSelectBestThresholdHistogram. Identified as the prime suspect for the
  // ~16s residual inside kEvaluateProj on trunk 3M×4096.
  kEntropyTableSetup,

  // Sub-scope of kEvaluateProj wrapping FindSplitLabelClassificationFeature-
  // NumericalCart at training.cc:2398. Used by the dynamic-fallback EXACT path
  // for nodes below dt_config.dynamic_split_threshold examples. The outer
  // dispatch (feature_filler lambda, EffectiveStrategy, sorting_strategy
  // branching, Filler/Initializer construction) plus the FindBestSplitFlat-
  // Highway buffer-resize and bucket InitializeAndZero are uncovered today; the
  // inner kSortFillBuckets / kSortFeatures / kSortScanSplits scopes nest inside
  // and are still per-phase measured.
  kCartPath,

  // Sub-scope of kCartPath wrapping the feature_filler immediate lambda at the
  // top of FindSplitLabelClassificationFeatureNumericalCart — LOCAL_IMPUTATION
  // check, EffectiveStrategy (config-proto reads), FeatureNumericalBucket::
  // Filler construction. Isolates the pre-dispatch setup from the unnamed
  // remainder of CartPath (leaf scopes' own clock-read overhead + StatusOr
  // return machinery).
  kCartSetup,

  // Sub-scope of kEvaluateProj wrapping the whole body of FindSplitLabel-
  // ClassificationFeatureNumericalHistogram at training.cc:2172, symmetric to
  // kCartPath. Captures what the inner kHistogramSetup / kAssignSamplesToHist /
  // kEntropyTableSetup / kSelectBestThresholdHistogram scopes miss: the reverse
  // cumulative sweep over candidate_splits, the scalar entropy setup, and the
  // destructors of candidate_splits + count_log_count (the latter is a
  // vector<double>(N_node+1) free per call). With it, EvaluateProj ≈ CartPath +
  // HistoPath up to dispatch overhead.
  kHistoPath,

  // BFS-only scheduler scope. Emitted by GrowTreeLocalBFS to characterize
  // the BFS scheduling overhead in isolation from any fused-Apply work.
  // kBfsNodeLoop fires only on -DBFS_ONLY and wraps the per-node NodeTrain
  // dispatch in the fallback path (i.e. without shared projections or fused
  // Apply), so it isolates the cost of running K projections per node under
  // BFS order vs. DFS. (The frontier pop-loop that drains node_queue into
  // depth_batch is not chrono'd — measured at <0.1 s for 3M rows.)
  kBfsNodeLoop,

  kNumFuncs
};

inline std::array<std::atomic<uint64_t>, kNumFuncs> global_stats{};

// ---------- per-thread context ------------------------------------
struct TlsCtx { int cur_tree = -1; int cur_depth = -1; };
inline thread_local TlsCtx tls_ctx;

// ---------- helper typedefs ---------------------------------------
using FuncArray = std::array<uint64_t, kNumFuncs>;
using DepthVec  = std::vector<FuncArray>;

// ---------- immortal singletons -----------------------------------
inline std::vector<DepthVec>& time_ns() {
  static auto* p = new std::vector<DepthVec>();
  return *p;
}
inline std::vector<DepthVec>& call_cnt() {
  static auto* p = new std::vector<DepthVec>();
  return *p;
}
inline std::vector<std::vector<uint64_t>>& node_cnt() {
  static auto* p = new std::vector<std::vector<uint64_t>>();
  return *p;
}
inline std::vector<std::vector<uint64_t>>& sample_cnt() {
  static auto* p = new std::vector<std::vector<uint64_t>>();
  return *p;
}
inline std::vector<std::thread::id>& tree_thread_id() {
  static auto* p = new std::vector<std::thread::id>();
  return *p;
}

// ================== NODEWISE_CHRONO: per-node AP records ==================
// The coarse tables above aggregate ApplyProjection over all nodes at a depth,
// so they cannot say whether the cost is spread evenly across a depth's nodes
// or carried by a handful of them. This axis adds a second sink on the SAME
// clock read: every ProjectionEvaluator::Evaluate call at a gated depth also
// deposits one record. No extra steady_clock::now() calls are introduced, so
// the measured ApplyProjection interval is bit-for-bit the one the coarse
// tables see (checked at dump time — see DumpNodewiseApCsv's self-check).
//
// One record per NODE: the node's Evaluate calls are accumulated in thread-
// local state and emitted once, when the node's NodeTrain scope closes. `nnz`
// and `ap_ns` are therefore sums over the node's projections, and the identity
// n_gathers = n_rows * nnz survives the summation (the row set is fixed within
// a node). Nothing is pushed from inside the projection loop.
//
// Volume is controlled by the gate (default: tree 0, depths 5/10/15/20 — 55k
// records ≈ 1.8 MB on HIGGS). Gating a whole tree at every depth is ~2.6M
// records ≈ 80 MB, which is why the gate defaults to a depth ladder.
//
// Runtime knobs (read once, before the thread pool starts):
//   YDF_NODEWISE_TREE    tree index to record; -1 = every tree. Default 0.
//   YDF_NODEWISE_DEPTHS  comma-separated depth list, or "*"/"all" for every
//                        depth. Default "5,10,15,20".
//   YDF_NODEWISE_OUT     output CSV path. Default "nodewise_ap.csv".
//   YDF_NODEWISE_RESERVE records reserved per recording tree. Default 1<<20
//                        for a single tree, 1<<16 when recording all trees.
#ifdef NODEWISE_CHRONO

// One row of the output CSV. 24 B; n_rows fits uint32 because it is bounded by
// the bag size, itself bounded by UnsignedExampleIdx (uint32).
struct NodeApRec {
  uint64_t node_id;    // heap index — see NodewiseChildId
  uint64_t ap_ns;      // Σ Evaluate intervals, same values the coarse sink got
  uint32_t depth;      // root = 1, so depth d holds at most 2^(d-1) nodes
  uint32_t n_rows;     // selected_examples.size() for this node
  uint32_t nnz;        // Σ projection.size() = features gathered per row
  uint32_t num_proj;   // Evaluate calls for this node (0 ⇒ AP never ran)
};

// Node identity: the heap index of the node in the binary tree. The root (which
// the growers enter at depth 1) is 1, and the children of i are 2i (negative)
// and 2i+1 (positive) — so depth d holds ids [2^(d-1), 2^d), and the id's binary
// digits after the leading 1 spell the root→node branch path.
//
// 0 means "unknown" and propagates down: a subtree grown without a rooted
// parent id (only the fused kernels' BFS→DFS handoff, which this axis does not
// support anyway), or a depth past 63, where the index no longer fits.
inline uint64_t NodewiseChildId(uint64_t parent_id, bool positive,
                                int32_t child_depth) {
  if (parent_id == 0 || child_depth > 63) return 0;
  return 2 * parent_id + (positive ? 1 : 0);
}

inline std::vector<std::vector<NodeApRec>>& node_ap_recs() {
  static auto* p = new std::vector<std::vector<NodeApRec>>();
  return *p;
}

inline constexpr int kMaxNodewiseDepth = 256;

struct NodewiseGate {
  int tree = 0;  // -1 = all trees
  bool any_depth = false;
  std::array<bool, kMaxNodewiseDepth> depth{};
  std::string depths_spec;
  std::string out_path;
  size_t reserve = 0;
};

// Parsed once (function-local static ⇒ thread-safe init). Call it once from the
// single-threaded setup path so no worker ever pays the initialization.
inline const NodewiseGate& NodewiseGateConfig() {
  static const NodewiseGate* g = [] {
    auto* p = new NodewiseGate();
    const char* tree_env = std::getenv("YDF_NODEWISE_TREE");
    p->tree = (tree_env != nullptr) ? std::atoi(tree_env) : 0;

    const char* depths_env = std::getenv("YDF_NODEWISE_DEPTHS");
    p->depths_spec = (depths_env != nullptr) ? depths_env : "5,10,15,20";
    if (p->depths_spec == "*" || p->depths_spec == "all") {
      p->any_depth = true;
    } else {
      size_t pos = 0;
      while (pos <= p->depths_spec.size()) {
        size_t comma = p->depths_spec.find(',', pos);
        if (comma == std::string::npos) comma = p->depths_spec.size();
        if (comma > pos) {
          const int v = std::atoi(p->depths_spec.substr(pos, comma - pos).c_str());
          if (v >= 0 && v < kMaxNodewiseDepth) p->depth[v] = true;
        }
        pos = comma + 1;
      }
    }

    const char* out_env = std::getenv("YDF_NODEWISE_OUT");
    p->out_path = (out_env != nullptr) ? out_env : "nodewise_ap.csv";

    const char* reserve_env = std::getenv("YDF_NODEWISE_RESERVE");
    p->reserve = (reserve_env != nullptr)
                     ? static_cast<size_t>(std::atoll(reserve_env))
                     : (p->tree >= 0 ? (size_t{1} << 20) : (size_t{1} << 16));
    return p;
  }();
  return *g;
}

inline bool NodewiseTreeSelected(int tree) {
  if (tree < 0) return false;
  const auto& g = NodewiseGateConfig();
  return g.tree < 0 || tree == g.tree;
}

inline bool NodewiseRecording(int tree, int depth) {
  if (!NodewiseTreeSelected(tree)) return false;
  if (tree >= static_cast<int>(node_ap_recs().size())) return false;
  const auto& g = NodewiseGateConfig();
  if (g.any_depth) return true;
  return depth >= 0 && depth < kMaxNodewiseDepth && g.depth[depth];
}

// Pre-allocates the recording trees' buffers so no push_back inside the tree
// loop can trigger a multi-MB realloc mid-measurement.
inline void ReserveNodewiseApRecs() {
  const auto& g = NodewiseGateConfig();
  for (int t = 0; t < static_cast<int>(node_ap_recs().size()); ++t) {
    if (NodewiseTreeSelected(t)) node_ap_recs()[t].reserve(g.reserve);
  }
}

// Per-thread nodewise state. Kept separate from TlsCtx so the struct the rest
// of the profiler reads stays byte-identical across build configs.
struct NodewiseCtx {
  bool recording = false;
  int tree = -1;
  uint64_t node_id = 0;
  uint64_t ap_ns = 0;         // accumulated over the current node's projections
  uint32_t depth = 0;
  uint32_t n_rows = 0;
  uint32_t nnz = 0;           // accumulated over the current node's projections
  uint32_t num_proj = 0;      // Evaluate calls so far within the current node
};
inline thread_local NodewiseCtx nw_ctx;

#endif  // NODEWISE_CHRONO

// ---------- add_time ----------------------------------------------
inline void add_time(int tree, int depth, FuncId id, uint64_t dt_ns) {
  if (tree < 0 || tree >= static_cast<int>(time_ns().size())) {
    global_stats[id].fetch_add(dt_ns, std::memory_order_relaxed);
    return;
  }
  auto& by_depth = time_ns()[tree];
  if (depth >= static_cast<int>(by_depth.size()))
    by_depth.resize(depth + 1);
  by_depth[depth][id] += dt_ns;          // single-threaded write

  auto& cnt_by_depth = call_cnt()[tree];
  if (depth >= static_cast<int>(cnt_by_depth.size()))
    cnt_by_depth.resize(depth + 1);
  cnt_by_depth[depth][id] += 1;
}

// ---------- ScopedTimer -------------------------------------------
// RAII timer: reads the clock on construction and accumulates the elapsed
// time into (cur_tree, cur_depth, id) on destruction.
class ScopedTimer {
 public:
  explicit ScopedTimer(FuncId id)
      : id_(id), start_(std::chrono::steady_clock::now()) {}
  ~ScopedTimer() {
    const uint64_t dt_ns =
        std::chrono::duration_cast<std::chrono::nanoseconds>(
            std::chrono::steady_clock::now() - start_).count();
    add_time(tls_ctx.cur_tree, tls_ctx.cur_depth, id_, dt_ns);
  }
 private:
  FuncId id_;
  std::chrono::steady_clock::time_point start_;
};

class ScopedTopTimer {
 public:
  explicit ScopedTopTimer(FuncId id)
      : id_(id),
        tree_(tls_ctx.cur_tree),
        start_(std::chrono::steady_clock::now()) {}
  ~ScopedTopTimer() {
    const uint64_t dt_ns =
        std::chrono::duration_cast<std::chrono::nanoseconds>(
            std::chrono::steady_clock::now() - start_).count();
    add_time(tree_, 0, id_, dt_ns);
  }

 private:
  FuncId id_;
  int tree_;
  std::chrono::steady_clock::time_point start_;
};

// ---------- NODEWISE_CHRONO scopes --------------------------------
#ifdef NODEWISE_CHRONO

// Drop-in replacement for ScopedTimer around ProjectionEvaluator::Evaluate.
// Deposits into the coarse (tree, depth) table exactly as ScopedTimer does, and
// — only at gated depths — also accumulates into the current node's record.
// The accumulation happens AFTER the closing clock read, so it is outside the
// interval it describes; it is three adds and no allocation.
class ScopedNodewiseApTimer {
 public:
  ScopedNodewiseApTimer(FuncId id, uint32_t n_rows, uint32_t nnz)
      : id_(id),
        n_rows_(n_rows),
        nnz_(nnz),
        start_(std::chrono::steady_clock::now()) {}
  ~ScopedNodewiseApTimer() {
    const uint64_t dt_ns =
        std::chrono::duration_cast<std::chrono::nanoseconds>(
            std::chrono::steady_clock::now() - start_).count();
    add_time(tls_ctx.cur_tree, tls_ctx.cur_depth, id_, dt_ns);
    if (nw_ctx.recording) {
      nw_ctx.ap_ns += dt_ns;
      nw_ctx.nnz += nnz_;
      // Every projection of a node sees the same row set; taking it from here
      // rather than from NodeTrain keeps n_rows the count Evaluate actually
      // walked, which is what n_gathers must be built from.
      nw_ctx.n_rows = n_rows_;
      ++nw_ctx.num_proj;
    }
  }

 private:
  FuncId id_;
  uint32_t n_rows_;
  uint32_t nnz_;
  std::chrono::steady_clock::time_point start_;
};

// Opened once per NodeTrain call: arms/disarms recording for the node, holds
// the accumulators its Evaluate calls add into, and emits the node's single
// record on exit. Nodes that never reached ApplyProjection (leaves, empty
// nodes, early returns) still emit a row, with num_proj = nnz = ap_ns = 0 —
// dropping them would silently shrink the denominator and inflate every
// concentration statistic.
class NodewiseNodeScope {
 public:
  NodewiseNodeScope(int tree, int depth, uint64_t node_id, uint32_t n_rows)
      : prev_(nw_ctx) {
    nw_ctx.tree = tree;
    nw_ctx.node_id = node_id;
    nw_ctx.depth = static_cast<uint32_t>(depth < 0 ? 0 : depth);
    nw_ctx.n_rows = n_rows;  // overwritten by the first Evaluate, if any
    nw_ctx.ap_ns = 0;
    nw_ctx.nnz = 0;
    nw_ctx.num_proj = 0;
    nw_ctx.recording = NodewiseRecording(tree, depth);
  }
  ~NodewiseNodeScope() {
    if (nw_ctx.recording) {
      node_ap_recs()[nw_ctx.tree].push_back({nw_ctx.node_id, nw_ctx.ap_ns,
                                             nw_ctx.depth, nw_ctx.n_rows,
                                             nw_ctx.nnz, nw_ctx.num_proj});
    }
    // NodeTrain does not nest today, but restore anyway so a nested caller
    // cannot corrupt the parent's node identity or steal its accumulators.
    nw_ctx = prev_;
  }

 private:
  NodewiseCtx prev_;
};

struct NodewiseDumpStats {
  bool ok = false;
  size_t records = 0;
  size_t selfcheck_cells = 0;
  size_t selfcheck_mismatches = 0;
  std::string path;
};

// Writes the records as CSV and cross-checks them against the coarse table.
// Both sinks are fed the same dt_ns by the same timer, so for every recorded
// (tree, depth) the sums must match EXACTLY; a mismatch means AP time reached
// the coarse table from a path that bypasses ScopedNodewiseApTimer (the fused
// symmetric / depthwise-1pass kernels do exactly that, so expect mismatches if
// this axis is combined with them).
inline NodewiseDumpStats DumpNodewiseApCsv() {
  const auto& g = NodewiseGateConfig();
  NodewiseDumpStats stats;
  stats.path = g.out_path;

  for (const auto& recs : node_ap_recs()) stats.records += recs.size();

  std::ofstream out(g.out_path);
  if (!out.is_open()) return stats;

  out << "# nodewise_ap format=2 tree_filter=" << g.tree
      << " depths=" << g.depths_spec << " records=" << stats.records << "\n";
  out << "tree,depth,node_id,n_rows,nnz,n_gathers,ap_ns,num_proj\n";
  for (size_t t = 0; t < node_ap_recs().size(); ++t) {
    // Records arrive in tree-growth order; sorting by (depth, node_id) is off
    // the measured path and makes a depth's nodes contiguous and left-to-right.
    auto& recs = node_ap_recs()[t];
    std::sort(recs.begin(), recs.end(),
              [](const NodeApRec& a, const NodeApRec& b) {
                return std::tie(a.depth, a.node_id) <
                       std::tie(b.depth, b.node_id);
              });
    for (const auto& r : recs) {
      out << t << ',' << r.depth << ',' << r.node_id << ',' << r.n_rows << ','
          << r.nnz << ',' << static_cast<uint64_t>(r.n_rows) * r.nnz << ','
          << r.ap_ns << ',' << r.num_proj << '\n';
    }
  }
  out.flush();
  stats.ok = out.good();

  for (size_t t = 0; t < node_ap_recs().size(); ++t) {
    if (node_ap_recs()[t].empty()) continue;
    std::vector<uint64_t> sum_by_depth;
    std::vector<char> seen;
    for (const auto& r : node_ap_recs()[t]) {
      if (r.depth >= sum_by_depth.size()) {
        sum_by_depth.resize(r.depth + 1, 0);
        seen.resize(r.depth + 1, 0);
      }
      sum_by_depth[r.depth] += r.ap_ns;
      seen[r.depth] = 1;
    }
    for (size_t d = 0; d < sum_by_depth.size(); ++d) {
      if (!seen[d]) continue;
      ++stats.selfcheck_cells;
      const uint64_t coarse =
          (t < time_ns().size() && d < time_ns()[t].size())
              ? time_ns()[t][d][kProjectionEvaluate]
              : 0;
      if (coarse != sum_by_depth[d]) ++stats.selfcheck_mismatches;
    }
  }
  return stats;
}

#endif  // NODEWISE_CHRONO

// ---------- Tree / Depth scopes -----------------------------------
struct TreeScope {
  explicit TreeScope(int tree) {
    tls_ctx.cur_tree = tree;
    tls_ctx.cur_depth = 0;
    if (tree >= 0 && tree < static_cast<int>(tree_thread_id().size()))
      tree_thread_id()[tree] = std::this_thread::get_id();
#ifdef NODEWISE_CHRONO
    // node_seq is per-tree; a thread trains many trees in sequence.
    nw_ctx = NodewiseCtx{};
    nw_ctx.tree = tree;
#endif
  }
  ~TreeScope() {
    tls_ctx.cur_tree = -1;
#ifdef NODEWISE_CHRONO
    nw_ctx.recording = false;
    nw_ctx.tree = -1;
#endif
  }
};
struct DepthScope   { DepthScope()  { ++tls_ctx.cur_depth; }
                      ~DepthScope() { --tls_ctx.cur_depth; } };

// ---------- small macros ------------------------------------------
#define YDF_PP_CAT_INNER(a,b) a##b
#define YDF_PP_CAT(a,b)       YDF_PP_CAT_INNER(a,b)

// ===== Coarse tier ================================================
// Active whenever CHRONO_PROFILE is defined (levels 1 and 2). Only the
// top-level scopes listed in the file header use these, so the coarse build
// keeps just enough instrumentation to attribute time per (tree, depth)
// without the per-sub-scope clock-read overhead.
#define CHRONO_SCOPE_COARSE(ID) \
  yggdrasil_decision_forests::chrono_prof::ScopedTimer \
      YDF_PP_CAT(_chrono_ctimer_, __LINE__)(ID)
#define CHRONO_SCOPE_COARSE_TOP(ID) \
  yggdrasil_decision_forests::chrono_prof::ScopedTopTimer \
      YDF_PP_CAT(_chrono_ctop_timer_, __LINE__)(ID)
#define CHRONO_BEGIN_COARSE(name)                                       \
  const auto YDF_PP_CAT(_chrono_cbegin_, name) =                        \
      std::chrono::steady_clock::now()
#define CHRONO_END_COARSE(name, id)                                     \
  ::yggdrasil_decision_forests::chrono_prof::add_time(                  \
      ::yggdrasil_decision_forests::chrono_prof::tls_ctx.cur_tree,      \
      ::yggdrasil_decision_forests::chrono_prof::tls_ctx.cur_depth,     \
      (id),                                                             \
      std::chrono::duration_cast<std::chrono::nanoseconds>(             \
          std::chrono::steady_clock::now() -                            \
          YDF_PP_CAT(_chrono_cbegin_, name))                            \
          .count())

// ===== Axis: nodewise ApplyProjection records =====================
// CHRONO_SCOPE_NODEWISE_AP degrades to the plain coarse scope when the axis is
// off, so the single call site in ProjectionEvaluator::Evaluate stays correct
// (and free) in every other build. N_ROWS / NNZ are then never evaluated.
#ifdef NODEWISE_CHRONO
#define CHRONO_SCOPE_NODEWISE_AP(ID, N_ROWS, NNZ)                       \
  ::yggdrasil_decision_forests::chrono_prof::ScopedNodewiseApTimer      \
      YDF_PP_CAT(_chrono_nw_ap_timer_, __LINE__)(                       \
          (ID), static_cast<uint32_t>(N_ROWS), static_cast<uint32_t>(NNZ))
#define CHRONO_NODEWISE_NODE(TREE, DEPTH, NODE_ID, N_ROWS)              \
  ::yggdrasil_decision_forests::chrono_prof::NodewiseNodeScope          \
      YDF_PP_CAT(_chrono_nw_node_, __LINE__)(                           \
          (TREE), (DEPTH), (NODE_ID), static_cast<uint32_t>(N_ROWS))
#else  // nodewise axis off: keep the coarse deposit, drop the record
#define CHRONO_SCOPE_NODEWISE_AP(ID, N_ROWS, NNZ) CHRONO_SCOPE_COARSE(ID)
#define CHRONO_NODEWISE_NODE(TREE, DEPTH, NODE_ID, N_ROWS)
#endif  // NODEWISE_CHRONO

// ===== Fine axis: ApplyProjection =================================
// Active only under -DFINE_CHRONO_AP. Wraps the inner scopes of
// ProjectionEvaluator::Evaluate and its fused-apply variants (symmetric /
// depthwise_1pass). Inert otherwise, so a plain coarse or EP-fine build pays
// zero overhead for these.
#ifdef FINE_CHRONO_AP
#define CHRONO_SCOPE_AP(ID) \
  yggdrasil_decision_forests::chrono_prof::ScopedTimer \
      YDF_PP_CAT(_chrono_ap_timer_, __LINE__)(ID)
#define CHRONO_SCOPE_AP_TOP(ID) \
  yggdrasil_decision_forests::chrono_prof::ScopedTopTimer \
      YDF_PP_CAT(_chrono_ap_top_timer_, __LINE__)(ID)
#define CHRONO_BEGIN_AP(name)                                           \
  const auto YDF_PP_CAT(_chrono_ap_begin_, name) =                      \
      std::chrono::steady_clock::now()
#define CHRONO_END_AP(name, id)                                         \
  ::yggdrasil_decision_forests::chrono_prof::add_time(                  \
      ::yggdrasil_decision_forests::chrono_prof::tls_ctx.cur_tree,      \
      ::yggdrasil_decision_forests::chrono_prof::tls_ctx.cur_depth,     \
      (id),                                                             \
      std::chrono::duration_cast<std::chrono::nanoseconds>(             \
          std::chrono::steady_clock::now() -                            \
          YDF_PP_CAT(_chrono_ap_begin_, name))                          \
          .count())
#else  // AP fine axis off: inert
#define CHRONO_SCOPE_AP(ID)
#define CHRONO_SCOPE_AP_TOP(ID)
#define CHRONO_BEGIN_AP(name)
#define CHRONO_END_AP(name, id)
#endif  // FINE_CHRONO_AP

// ===== Fine axis: EvaluateProjection ==============================
// Active only under -DFINE_CHRONO_EP. Wraps the inner scopes of
// EvaluateProjection (histogram / Cart split search). Inert otherwise.
#ifdef FINE_CHRONO_EP
#define CHRONO_SCOPE_EP(ID) \
  yggdrasil_decision_forests::chrono_prof::ScopedTimer \
      YDF_PP_CAT(_chrono_ep_timer_, __LINE__)(ID)
#define CHRONO_SCOPE_EP_TOP(ID) \
  yggdrasil_decision_forests::chrono_prof::ScopedTopTimer \
      YDF_PP_CAT(_chrono_ep_top_timer_, __LINE__)(ID)
#define CHRONO_BEGIN_EP(name)                                           \
  const auto YDF_PP_CAT(_chrono_ep_begin_, name) =                      \
      std::chrono::steady_clock::now()
#define CHRONO_END_EP(name, id)                                         \
  ::yggdrasil_decision_forests::chrono_prof::add_time(                  \
      ::yggdrasil_decision_forests::chrono_prof::tls_ctx.cur_tree,      \
      ::yggdrasil_decision_forests::chrono_prof::tls_ctx.cur_depth,     \
      (id),                                                             \
      std::chrono::duration_cast<std::chrono::nanoseconds>(             \
          std::chrono::steady_clock::now() -                            \
          YDF_PP_CAT(_chrono_ep_begin_, name))                          \
          .count())
#else  // EP fine axis off: inert
#define CHRONO_SCOPE_EP(ID)
#define CHRONO_SCOPE_EP_TOP(ID)
#define CHRONO_BEGIN_EP(name)
#define CHRONO_END_EP(name, id)
#endif  // FINE_CHRONO_EP

// ===== Legacy fine tier (GPU only) ================================
// Active only at level >= 2. No config sets level 2 anymore; kept solely so the
// untouched GPU code (oblique_gpu.cc) still compiles. Dormant under all current
// configs.
#if CHRONO_PROFILE >= 2
#define CHRONO_SCOPE(ID) \
  yggdrasil_decision_forests::chrono_prof::ScopedTimer \
      YDF_PP_CAT(_chrono_timer_, __LINE__)(ID)
#define CHRONO_SCOPE_TOP(ID) \
  yggdrasil_decision_forests::chrono_prof::ScopedTopTimer \
      YDF_PP_CAT(_chrono_top_timer_, __LINE__)(ID)

// Manual begin/end variant for spans that can't be wrapped in `{}` because
// variables declared inside need to outlive the scope. Lost on early
// return paths (RAII variant fires on stack unwind; this one doesn't).
#define CHRONO_BEGIN(name)                                              \
  const auto YDF_PP_CAT(_chrono_begin_, name) =                         \
      std::chrono::steady_clock::now()
#define CHRONO_END(name, id)                                            \
  ::yggdrasil_decision_forests::chrono_prof::add_time(                  \
      ::yggdrasil_decision_forests::chrono_prof::tls_ctx.cur_tree,      \
      ::yggdrasil_decision_forests::chrono_prof::tls_ctx.cur_depth,     \
      (id),                                                             \
      std::chrono::duration_cast<std::chrono::nanoseconds>(             \
          std::chrono::steady_clock::now() -                            \
          YDF_PP_CAT(_chrono_begin_, name))                             \
          .count())
#else  // coarse build: fine-grained scopes are inert
#define CHRONO_SCOPE(ID)
#define CHRONO_SCOPE_TOP(ID)
#define CHRONO_BEGIN(name)
#define CHRONO_END(name, id)
#endif  // CHRONO_PROFILE >= 2

}  // namespace yggdrasil_decision_forests::chrono_prof
#else  // CHRONO_PROFILE undefined: no profiling at all
#define CHRONO_SCOPE(ID)
#define CHRONO_SCOPE_TOP(ID)
#define CHRONO_BEGIN(name)
#define CHRONO_END(name, id)
#define CHRONO_SCOPE_COARSE(ID)
#define CHRONO_SCOPE_COARSE_TOP(ID)
#define CHRONO_BEGIN_COARSE(name)
#define CHRONO_END_COARSE(name, id)
#define CHRONO_SCOPE_AP(ID)
#define CHRONO_SCOPE_AP_TOP(ID)
#define CHRONO_BEGIN_AP(name)
#define CHRONO_END_AP(name, id)
#define CHRONO_SCOPE_EP(ID)
#define CHRONO_SCOPE_EP_TOP(ID)
#define CHRONO_BEGIN_EP(name)
#define CHRONO_END_EP(name, id)
#define CHRONO_SCOPE_NODEWISE_AP(ID, N_ROWS, NNZ)
#define CHRONO_NODEWISE_NODE(TREE, DEPTH, NODE_ID, N_ROWS)
#endif
