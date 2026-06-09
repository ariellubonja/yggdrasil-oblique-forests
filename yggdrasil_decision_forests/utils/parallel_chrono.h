#pragma once
#ifdef CHRONO_ENABLED
#include <array>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <thread>
#include <vector>

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
  kAxisAlignedColumnFetch,

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
  kSymBuildBag,   // Pre-size slabs + concat per-node selected_examples → bag
  kSymSortBag,    // stable_sort(perm) + materialize sorted bag/node_of_bag
  kSymSweep,      // K bag-wide stride-1 projection sweeps (the hot loop)

  // Sub-phases of ApplyProjectionsDepthwise1Pass. Only emitted when compiled
  // with -DDEPTHWISE_1_PASS; zero otherwise. The Sweep scope wraps the
  // ConcurrentForLoop synchronization point in the caller thread, so
  // wall-clock attribution is preserved even when num_threads > 1.
  kDw1PreSize,    // proj_prefix sum + per-node out_projected[n].assign(...)
  kDw1Sweep,      // ProjectionEvaluator ctor + kernel dispatch (Q tasks)

  // Sub-phases of ApplyProjectionsProjectionMatrixControl. Only emitted when
  // compiled with -DPROJECTION_MATRIX_CONTROL; zero otherwise. The PMC path
  // interleaves PreSize and Sweep per-node (unlike Dw1, which does PreSize
  // for all nodes up front, then Sweep for all nodes), so both scopes are
  // accumulated across the per-node loop. The ProjectionEvaluator ctor is
  // not covered by either sub-phase, only by the outer kProjectionEvaluate.
  kPmcPreSize,    // per-node out_projected[n].assign(...)
  kPmcSweep,      // per-node rows-outer / projs-inner triple loop

  // Per-node bookkeeping scopes inside NodeTrain /
  // FindBestConditionSparseObliqueTemplate.
  kNodeTrain,
  kFindBestCondition,
  kObliqueSplitSearch,
  kAxisAlignedSplitSearch,
  kSampleProjection,
  kSplitExamplesInPlace,
  kSetLeafValue,

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

  // BFS-only scheduler scopes. Emitted by GrowTreeLocalBFS to characterize
  // the BFS scheduling overhead in isolation from any fused-Apply work.
  // kBfsFrontier fires for any BFS-routed build (depthwise_1pass, symmetric_*,
  // bfs_only) and measures the inner pop-loop that drains node_queue into
  // depth_batch at each level. kBfsNodeLoop fires only on -DBFS_ONLY and
  // wraps the per-node NodeTrain dispatch in the fallback path (i.e. without
  // shared projections or fused Apply), so it isolates the cost of running
  // K projections per node under BFS order vs. DFS.
  kBfsFrontier,
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
// When `enabled` is false the timer is inert: no clock reads, no add_time.
// This lets a call site disable measurement for a code path it doesn't care
// about (e.g. the axis-aligned splitter that still runs alongside oblique).
class ScopedTimer {
 public:
  explicit ScopedTimer(FuncId id, bool enabled = true)
      : id_(id),
        enabled_(enabled),
        start_(enabled ? std::chrono::steady_clock::now()
                       : std::chrono::steady_clock::time_point{}) {}
  ~ScopedTimer() {
    if (!enabled_) return;
    const uint64_t dt_ns =
        std::chrono::duration_cast<std::chrono::nanoseconds>(
            std::chrono::steady_clock::now() - start_).count();
    add_time(tls_ctx.cur_tree, tls_ctx.cur_depth, id_, dt_ns);
  }
 private:
  FuncId id_;
  bool enabled_;
  std::chrono::steady_clock::time_point start_;
};

class ScopedTopTimer {
 public:
  explicit ScopedTopTimer(FuncId id, bool enabled = true)
      : id_(id),
        enabled_(enabled),
        tree_(tls_ctx.cur_tree),
        start_(enabled ? std::chrono::steady_clock::now()
                       : std::chrono::steady_clock::time_point{}) {}
  ~ScopedTopTimer() {
    if (!enabled_) return;
    const uint64_t dt_ns =
        std::chrono::duration_cast<std::chrono::nanoseconds>(
            std::chrono::steady_clock::now() - start_).count();
    add_time(tree_, 0, id_, dt_ns);
  }

 private:
  FuncId id_;
  bool enabled_;
  int tree_;
  std::chrono::steady_clock::time_point start_;
};

// ---------- Tree / Depth scopes -----------------------------------
struct TreeScope {
  explicit TreeScope(int tree) {
    tls_ctx.cur_tree = tree;
    tls_ctx.cur_depth = 0;
    if (tree >= 0 && tree < static_cast<int>(tree_thread_id().size()))
      tree_thread_id()[tree] = std::this_thread::get_id();
  }
  ~TreeScope() { tls_ctx.cur_tree = -1; }
};
struct DepthScope   { DepthScope()  { ++tls_ctx.cur_depth; }
                      ~DepthScope() { --tls_ctx.cur_depth; } };

// ---------- small macros ------------------------------------------
#define YDF_PP_CAT_INNER(a,b) a##b
#define YDF_PP_CAT(a,b)       YDF_PP_CAT_INNER(a,b)
#define CHRONO_SCOPE(ID) \
  yggdrasil_decision_forests::chrono_prof::ScopedTimer \
      YDF_PP_CAT(_chrono_timer_, __LINE__)(ID)
#define CHRONO_SCOPE_TOP(ID) \
  yggdrasil_decision_forests::chrono_prof::ScopedTopTimer \
      YDF_PP_CAT(_chrono_top_timer_, __LINE__)(ID)
// Like CHRONO_SCOPE, but only measures when COND is true. When false the
// timer collects nothing (no clock reads, no accumulation).
#define CHRONO_SCOPE_IF(COND, ID) \
  yggdrasil_decision_forests::chrono_prof::ScopedTimer \
      YDF_PP_CAT(_chrono_timer_, __LINE__)((ID), (COND))

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

}  // namespace yggdrasil_decision_forests::chrono_prof
#else
#define CHRONO_SCOPE(ID)
#define CHRONO_SCOPE_TOP(ID)
#define CHRONO_SCOPE_IF(COND, ID)
#define CHRONO_BEGIN(name)
#define CHRONO_END(name, id)
#endif
