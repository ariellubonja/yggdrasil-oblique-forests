#include "yggdrasil_decision_forests/learner/decision_tree/oblique_cpu_depthwise_1pass.h"

#ifdef DEPTHWISE_1_PASS

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <iterator>
#include <string>
#include <utility>
#include <vector>

#include "absl/log/check.h"
#include "absl/status/status.h"
#include "absl/types/span.h"
#include "yggdrasil_decision_forests/learner/decision_tree/oblique.h"
#include "yggdrasil_decision_forests/utils/concurrency.h"
#include "yggdrasil_decision_forests/utils/parallel_chrono.h"

namespace yggdrasil_decision_forests::model::decision_tree {

namespace {

// Column-centric depthwise sweep.
//
// At mid depths the frontier holds thousands of small nodes, and the union of
// their sampled columns covers the feature space many times over: total
// column references N*K*d >> F. The previous kernel visited columns in
// (node, projection) order — K*d random column hops per node, the same DRAM
// pattern as the nodewise baseline. This kernel inverts the loop: group a
// block of consecutive nodes, bucket every (node, projection, weight)
// reference by column, then walk the touched columns once in ascending
// order. Each column's gathers become one ordered pass over the block's row
// bands (page/TLB/DRAM-row friendly) instead of being scattered across the
// whole depth.
//
// Blocks are sized so the block's output slabs stay cache-resident, because
// the column order revisits each slab ~K*d times. Nodes too large for a
// block keep the projection-major order (one streaming write of their slab),
// mirroring the big/small dispatch of Dynamic_Row_Col_Major.

// Output-slab budget (floats) per column-centric block. Also the cutoff
// above which a node is processed projection-major. Tunable for experiments.
size_t Dw1BlockFloats() {
  static const size_t value = [] {
    const char* e = std::getenv("YDF_DW1_BLOCK_FLOATS");
    // Measured at 3M×4096: 4 MiB 55.0 s, 16 MiB 49.6, 64 MiB 47.0,
    // 256 MiB 45.6 — column sharing dominates output-slab residency, with
    // diminishing returns past 64 MiB. Default 256 MiB (64 Mi floats).
    return e != nullptr ? static_cast<size_t>(std::strtoull(e, nullptr, 10))
                        : static_cast<size_t>(64) << 20;
  }();
  return value;
}

// Override for dw1's internal parallelism. The caller passes
// deployment.num_threads(), but when random_forest already trains
// num_threads trees concurrently, each tree spawning its own per-depth
// ThreadPool oversubscribes the machine (num_threads^2 runnable threads +
// a sync barrier per depth). YDF_DW1_THREADS=1 forces the serial kernel
// within each tree; unset keeps the caller's value.
int Dw1ThreadsOverride(int num_threads) {
  static const int value = [] {
    const char* e = std::getenv("YDF_DW1_THREADS");
    return e != nullptr ? static_cast<int>(std::strtol(e, nullptr, 10)) : -1;
  }();
  return value > 0 ? value : num_threads;
}

struct ColEntry {
  int32_t node;   // index within the depth batch
  int32_t proj;   // projection index within the node
  int32_t col;    // attribute idx
  float weight;
};

struct Dw1Task {
  size_t begin_node;
  size_t end_node;  // exclusive; big nodes come as [n, n+1) with big=true
  bool big;
};

// Per-worker scratch, reused across blocks.
struct Dw1Scratch {
  std::vector<ColEntry> entries;
  std::vector<ColEntry> sorted;
  std::vector<int32_t> touched;    // touched column ids, sorted ascending
  std::vector<int32_t> col_count;  // per-column counters, sized max_attr+1
};

// Projection-major kernel for one node (big nodes / fallback). Direct column
// pointers; the per-load AttributeValue branch chain is hoisted out.
inline void EvaluateNodeProjMajor(
    const internal::ProjectionEvaluator& evaluator,
    const std::vector<internal::Projection>& projs,
    const UnsignedExampleIdx* sel_ptr, size_t rows_n, float* out_ptr) {
  struct FeatRef {
    const float* col;
    float weight;
#ifdef ENABLE_APPLYPROJECTION_ISNAN
    float na;
#endif
  };
  std::vector<FeatRef> feats;
  for (size_t p = 0; p < projs.size(); ++p) {
    feats.clear();
    for (const auto& feat : projs[p]) {
      feats.push_back({evaluator.AttributeData(feat.attribute_idx),
                       feat.weight
#ifdef ENABLE_APPLYPROJECTION_ISNAN
                       ,
                       evaluator.NaReplacementValue(feat.attribute_idx)
#endif
      });
    }
    float* o = out_ptr + p * rows_n;
    for (size_t i = 0; i < rows_n; ++i) {
      const UnsignedExampleIdx ex = sel_ptr[i];
      float acc = 0.f;
      for (const auto& f : feats) {
        float v = f.col[ex];
#ifdef ENABLE_APPLYPROJECTION_ISNAN
        if (std::isnan(v)) v = f.na;
#endif
        acc += f.weight * v;
      }
      o[i] = acc;
    }
  }
}

// Original per-(node, projection) kernel via AttributeValue. Only used when
// direct column pointers are unavailable (alternate dataset layouts).
inline void EvaluateProjectionRowsGeneric(
    const internal::ProjectionEvaluator& evaluator,
    const internal::Projection& proj, const UnsignedExampleIdx* sel_ptr,
    size_t rows_n, float* out_for_proj) {
  for (size_t i = 0; i < rows_n; ++i) {
    const UnsignedExampleIdx ex = sel_ptr[i];
    float acc = 0.f;
    for (const auto& feat : proj) {
      float v = evaluator.AttributeValue(feat.attribute_idx, ex);
#ifdef ENABLE_APPLYPROJECTION_ISNAN
      if (std::isnan(v)) v = evaluator.NaReplacementValue(feat.attribute_idx);
#endif
      acc += feat.weight * v;
    }
    out_for_proj[i] = acc;
  }
}

}  // namespace

absl::Status ApplyProjectionsDepthwise1Pass(
    const dataset::VerticalDataset& train_dataset,
    const google::protobuf::RepeatedField<int32_t>& numerical_features,
    absl::Span<const absl::Span<const UnsignedExampleIdx>>
        selected_examples_per_node,
    absl::Span<const std::vector<internal::Projection>> projections_per_node,
    absl::Span<std::vector<float>> out_projected, int num_threads) {
  num_threads = Dw1ThreadsOverride(num_threads);
  const size_t N = selected_examples_per_node.size();
  DCHECK_EQ(N, projections_per_node.size());
  DCHECK_EQ(N, out_projected.size());

  CHRONO_SCOPE(::yggdrasil_decision_forests::chrono_prof::kProjectionEvaluate);

  if (N == 0) return absl::OkStatus();

  int max_attr = 0;
  for (const auto attribute_idx : numerical_features) {
    max_attr = std::max(max_attr, attribute_idx);
  }

  // ── Phase 1: PreSize ──────────────────────────────────────────────
  // Slab pre-size (zero-init: the column sweep accumulates) + task build.
  std::vector<Dw1Task> tasks;
  {
    CHRONO_SCOPE(::yggdrasil_decision_forests::chrono_prof::kDw1PreSize);
    const size_t budget = Dw1BlockFloats();
    size_t blk_begin = 0;
    size_t blk_floats = 0;
    for (size_t n = 0; n < N; ++n) {
      const size_t slab =
          selected_examples_per_node[n].size() * projections_per_node[n].size();
      out_projected[n].assign(slab, 0.f);
      if (slab > budget) {
        if (blk_begin < n) tasks.push_back({blk_begin, n, /*big=*/false});
        tasks.push_back({n, n + 1, /*big=*/true});
        blk_begin = n + 1;
        blk_floats = 0;
      } else if (blk_floats + slab > budget) {
        if (blk_begin < n) tasks.push_back({blk_begin, n, /*big=*/false});
        blk_begin = n;
        blk_floats = slab;
      } else {
        blk_floats += slab;
      }
    }
    if (blk_begin < N) tasks.push_back({blk_begin, N, /*big=*/false});
  }

  // ── Phase 2: Sweep ────────────────────────────────────────────────
  {
    CHRONO_SCOPE(::yggdrasil_decision_forests::chrono_prof::kDw1Sweep);
    internal::ProjectionEvaluator evaluator(train_dataset, numerical_features);

    // Direct column pointers exist only for the default VerticalDataset
    // layout; alternate trunk layouts fall back to the generic kernel.
    bool direct = true;
    for (const auto attribute_idx : numerical_features) {
      if (evaluator.AttributeData(attribute_idx) == nullptr) {
        direct = false;
        break;
      }
    }

    const auto run_task = [&](const Dw1Task& task, Dw1Scratch& scratch) {
      if (!direct) {
        for (size_t n = task.begin_node; n < task.end_node; ++n) {
          const auto sel = selected_examples_per_node[n];
          const auto& projs = projections_per_node[n];
          for (size_t p = 0; p < projs.size(); ++p) {
            EvaluateProjectionRowsGeneric(
                evaluator, projs[p], sel.data(), sel.size(),
                out_projected[n].data() + p * sel.size());
          }
        }
        return;
      }
      if (task.big) {
        const size_t n = task.begin_node;
        const auto sel = selected_examples_per_node[n];
        EvaluateNodeProjMajor(evaluator, projections_per_node[n], sel.data(),
                              sel.size(), out_projected[n].data());
        return;
      }

      // Column-centric block: bucket references by column, then walk the
      // touched columns ascending.
      auto& entries = scratch.entries;
      auto& sorted = scratch.sorted;
      auto& touched = scratch.touched;
      auto& col_count = scratch.col_count;
      if (col_count.size() < static_cast<size_t>(max_attr) + 1) {
        col_count.assign(static_cast<size_t>(max_attr) + 1, 0);
      }
      entries.clear();
      touched.clear();

      for (size_t n = task.begin_node; n < task.end_node; ++n) {
        const auto& projs = projections_per_node[n];
        for (size_t p = 0; p < projs.size(); ++p) {
          for (const auto& feat : projs[p]) {
            entries.push_back({static_cast<int32_t>(n),
                               static_cast<int32_t>(p), feat.attribute_idx,
                               feat.weight});
            if (col_count[feat.attribute_idx]++ == 0) {
              touched.push_back(feat.attribute_idx);
            }
          }
        }
      }
      std::sort(touched.begin(), touched.end());

      // Counting sort by column: col_count becomes the running fill cursor.
      size_t offset = 0;
      for (const int32_t c : touched) {
        const int32_t cnt = col_count[c];
        col_count[c] = static_cast<int32_t>(offset);
        offset += cnt;
      }
      sorted.resize(entries.size());
      for (const auto& e : entries) {
        sorted[col_count[e.col]++] = e;
      }

      size_t pos = 0;
      for (const int32_t c : touched) {
        const float* col = evaluator.AttributeData(c);
#ifdef ENABLE_APPLYPROJECTION_ISNAN
        const float na = evaluator.NaReplacementValue(c);
#endif
        const size_t end = static_cast<size_t>(col_count[c]);
        for (; pos < end; ++pos) {
          const ColEntry& e = sorted[pos];
          const auto sel = selected_examples_per_node[e.node];
          const size_t rows_n = sel.size();
          const UnsignedExampleIdx* sel_ptr = sel.data();
          float* o = out_projected[e.node].data() + e.proj * rows_n;
          const float w = e.weight;
          for (size_t i = 0; i < rows_n; ++i) {
            float v = col[sel_ptr[i]];
#ifdef ENABLE_APPLYPROJECTION_ISNAN
            if (std::isnan(v)) v = na;
#endif
            o[i] += w * v;
          }
        }
        col_count[c] = 0;  // reset for the next block
      }
    };

    if (num_threads <= 1 || tasks.size() == 1) {
      Dw1Scratch scratch;
      for (const auto& task : tasks) run_task(task, scratch);
    } else {
      const size_t num_blocks =
          std::min<size_t>(static_cast<size_t>(num_threads), tasks.size());
      utils::concurrency::ThreadPool pool(
          num_threads, {.name_prefix = std::string("depthwise_1pass")});
      utils::concurrency::ConcurrentForLoop(
          num_blocks, &pool, tasks.size(),
          [&](size_t /*block_idx*/, size_t begin, size_t end) {
            Dw1Scratch scratch;
            for (size_t t = begin; t < end; ++t) run_task(tasks[t], scratch);
          });
    }
  }

  return absl::OkStatus();
}

}  // namespace yggdrasil_decision_forests::model::decision_tree

#endif  // DEPTHWISE_1_PASS
