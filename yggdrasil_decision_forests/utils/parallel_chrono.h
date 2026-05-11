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
  kCartFinderSetup,
  kGetCandidateAttributes,
  kAxisAlignedCandidateLoop,
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
#define CHRONO_SCOPE_TOP(ID) CHRONO_SCOPE(ID)

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
#define CHRONO_BEGIN(name)
#define CHRONO_END(name, id)
#endif