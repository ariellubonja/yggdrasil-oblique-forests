#include "yggdrasil_decision_forests/learner/decision_tree/oblique_cpu_depthwise_1pass.h"

#ifdef DEPTHWISE_1_PASS

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <iterator>
#include <string>
#include <vector>

#include "absl/log/check.h"
#include "absl/status/status.h"
#include "absl/types/span.h"
#include "yggdrasil_decision_forests/learner/decision_tree/oblique.h"
#include "yggdrasil_decision_forests/utils/concurrency.h"
#include "yggdrasil_decision_forests/utils/parallel_chrono.h"

namespace yggdrasil_decision_forests::model::decision_tree {

namespace {

// Inner-loop kernel for one (node n, projection p) pair processing
// `rows_n` examples. Uses a 4-wide row-block structure: 4 independent
// accumulators, 4 parallel `attr[ex]` loads per item — exposes 4-way ILP
// to the OOO scheduler so up to 4 DRAM misses are in flight per
// projection item, hiding load latency that the natural 1-row form
// serializes through the FMA chain. Tail < 4 rows handled scalar.
inline void EvaluateProjectionRowBlocks(
    const internal::ProjectionEvaluator& evaluator,
    const internal::Projection& proj,
    const UnsignedExampleIdx* sel_ptr, size_t rows_n,
    float* out_for_proj) {
  size_t i = 0;
  for (; i + 4 <= rows_n; i += 4) {
    const UnsignedExampleIdx ex0 = sel_ptr[i + 0];
    const UnsignedExampleIdx ex1 = sel_ptr[i + 1];
    const UnsignedExampleIdx ex2 = sel_ptr[i + 2];
    const UnsignedExampleIdx ex3 = sel_ptr[i + 3];
    float acc0 = 0.f, acc1 = 0.f, acc2 = 0.f, acc3 = 0.f;
    for (const auto& feat : proj) {
      const std::vector<float>& col =
          evaluator.AttributeValues(feat.attribute_idx);
      const float w = feat.weight;
      float v0 = col[ex0], v1 = col[ex1], v2 = col[ex2], v3 = col[ex3];
#ifndef YDF_BENCH_SKIP_ISNAN
      const float na = evaluator.NaReplacementValue(feat.attribute_idx);
      if (std::isnan(v0)) v0 = na;
      if (std::isnan(v1)) v1 = na;
      if (std::isnan(v2)) v2 = na;
      if (std::isnan(v3)) v3 = na;
#endif
      acc0 += w * v0;
      acc1 += w * v1;
      acc2 += w * v2;
      acc3 += w * v3;
    }
    out_for_proj[i + 0] = acc0;
    out_for_proj[i + 1] = acc1;
    out_for_proj[i + 2] = acc2;
    out_for_proj[i + 3] = acc3;
  }
  for (; i < rows_n; ++i) {
    const UnsignedExampleIdx ex = sel_ptr[i];
    float acc = 0.f;
    for (const auto& feat : proj) {
      float v = evaluator.AttributeValues(feat.attribute_idx)[ex];
#ifndef YDF_BENCH_SKIP_ISNAN
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

  CHRONO_SCOPE(::yggdrasil_decision_forests::chrono_prof::kProjectionEvaluate);

  if (N == 0) return absl::OkStatus();

  // Prefix sum over (n, p) pairs: global_proj_prefix[n+1] = Σ_{n' ≤ n} P_{n'}.
  // Each "task" in the parallel decomposition is one (node, projection) —
  // i.e. one full `EvaluateProjectionRowBlocks` call processing all rows
  // of that node for that projection. Total tasks Q = Σ_n P_n.
  std::vector<size_t> proj_prefix(N + 1, 0);
  for (size_t n = 0; n < N; ++n) {
    proj_prefix[n + 1] = proj_prefix[n] + projections_per_node[n].size();
  }
  const size_t Q = proj_prefix[N];

  // Pre-size each per-node output slab. The kernel writes directly into
  // `out_projected[n].data() + p * rows_n`; no flat_out scatter.
  for (size_t n = 0; n < N; ++n) {
    out_projected[n].assign(
        selected_examples_per_node[n].size() *
            projections_per_node[n].size(),
        0.f);
  }

  internal::ProjectionEvaluator evaluator(train_dataset, numerical_features);

  // Decode a (node, projection-within-node) pair from a global task index.
  // O(log N) per task, but tasks are coarse (one per projection — typically
  // dozens per level), so the decode cost is negligible.
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
      EvaluateProjectionRowBlocks(evaluator, projs_ptr[p], sel_ptr, rows_n,
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
      pool.StartWorkers();
      utils::concurrency::ConcurrentForLoop(
          num_blocks, &pool, Q,
          [&](size_t /*block_idx*/, size_t begin, size_t end) {
            kernel(begin, end);
          });
    }
  }

  return absl::OkStatus();
}

}  // namespace yggdrasil_decision_forests::model::decision_tree

#endif  // DEPTHWISE_1_PASS
