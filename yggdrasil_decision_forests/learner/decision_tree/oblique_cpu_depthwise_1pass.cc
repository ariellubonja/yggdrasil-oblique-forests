#include "yggdrasil_decision_forests/learner/decision_tree/oblique_cpu_depthwise_1pass.h"

#ifdef DEPTHWISE_1_PASS

#include <algorithm>
#include <cmath>
#include <cstddef>
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

// Inner kernel: process all `rows_n` examples of one (node, projection) pair.
inline void EvaluateProjectionRows(
    const internal::ProjectionEvaluator& evaluator,
    const internal::Projection& proj,
    const UnsignedExampleIdx* sel_ptr, size_t rows_n,
    float* out_for_proj) {
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
    absl::Span<std::vector<float>> out_projected,
    int num_threads) {
  const size_t N = selected_examples_per_node.size();
  DCHECK_EQ(N, projections_per_node.size());
  DCHECK_EQ(N, out_projected.size());

  // Outer scope: total ProjectionEvaluate budget. Sub-phase scopes below
  // partition this into PreSize / Sweep so parallel_chrono.py can break
  // ApplyProjection down per depth.
  CHRONO_SCOPE(::yggdrasil_decision_forests::chrono_prof::kProjectionEvaluate);

  if (N == 0) return absl::OkStatus();

  // ── Phase 1: PreSize ──────────────────────────────────────────────
  // Prefix sum over (n, p) pairs + per-node output-slab pre-size. The
  // slab assign is the allocation-heavy half; the prefix sum is O(N).
  // Each "task" in the parallel decomposition is one (node, projection):
  // one full EvaluateProjectionRows call processing all rows of that node
  // for that projection. Total tasks Q = sum_n P_n.
  std::vector<size_t> proj_prefix(N + 1, 0);
  size_t Q = 0;
  {
    CHRONO_SCOPE(::yggdrasil_decision_forests::chrono_prof::kDw1PreSize);
    for (size_t n = 0; n < N; ++n) {
      proj_prefix[n + 1] = proj_prefix[n] + projections_per_node[n].size();
    }
    Q = proj_prefix[N];

    // Pre-size each per-node output slab. The kernel writes directly into
    // out_projected[n].data() + p * rows_n; no flat-out scatter.
    for (size_t n = 0; n < N; ++n) {
      out_projected[n].assign(
          selected_examples_per_node[n].size() *
              projections_per_node[n].size(),
          0.f);
    }
  }

  // ── Phase 2: Sweep ────────────────────────────────────────────────
  // ProjectionEvaluator construction + kernel dispatch. When num_threads
  // > 1, the scope spans the ConcurrentForLoop synchronization point in
  // the caller thread, so wall-clock attribution is preserved even though
  // worker-thread tls_ctx is unset (workers' inner scopes, if any, would
  // fall through to global_stats).
  {
    CHRONO_SCOPE(::yggdrasil_decision_forests::chrono_prof::kDw1Sweep);
    internal::ProjectionEvaluator evaluator(train_dataset, numerical_features);

    // Decode (node, projection-within-node) from a global task index q.
    // O(log N) per task, but tasks are coarse (one per projection -- dozens
    // per level at most), so the decode cost is negligible.
    auto decode = [&](size_t q) {
      const auto it = std::upper_bound(proj_prefix.begin(),
                                       proj_prefix.end(), q);
      const size_t n = static_cast<size_t>(
          std::distance(proj_prefix.begin(), it) - 1);
      const size_t p = q - proj_prefix[n];
      return std::pair<size_t, size_t>{n, p};
    };

    const auto kernel = [&](size_t begin, size_t end) {
      if (begin >= end) return;
      auto [n, p] = decode(begin);
      size_t rows_n = selected_examples_per_node[n].size();
      size_t P_n = projections_per_node[n].size();
      const UnsignedExampleIdx* sel_ptr = selected_examples_per_node[n].data();
      const internal::Projection* projs_ptr =
          projections_per_node[n].data();
      float* out_ptr = out_projected[n].data();

      for (size_t q = begin; q < end; ++q) {
        EvaluateProjectionRows(evaluator, projs_ptr[p], sel_ptr, rows_n,
                               out_ptr + p * rows_n);
        ++p;
        if (p == P_n) {
          p = 0;
          ++n;
          if (n < N) {
            rows_n = selected_examples_per_node[n].size();
            P_n = projections_per_node[n].size();
            sel_ptr = selected_examples_per_node[n].data();
            projs_ptr = projections_per_node[n].data();
            out_ptr = out_projected[n].data();
          }
        }
      }
    };

    if (Q > 0) {
      if (num_threads <= 1) {
        kernel(0, Q);
      } else {
        const size_t num_blocks =
            std::min<size_t>(static_cast<size_t>(num_threads), Q);
        utils::concurrency::ThreadPool pool(
            num_threads, {.name_prefix = std::string("depthwise_1pass")});
        utils::concurrency::ConcurrentForLoop(
            num_blocks, &pool, Q,
            [&](size_t /*block_idx*/, size_t begin, size_t end) {
              kernel(begin, end);
            });
      }
    }
  }

  return absl::OkStatus();
}

}  // namespace yggdrasil_decision_forests::model::decision_tree

#endif  // DEPTHWISE_1_PASS
