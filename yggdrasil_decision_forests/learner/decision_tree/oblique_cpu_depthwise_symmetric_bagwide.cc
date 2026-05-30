#include "yggdrasil_decision_forests/learner/decision_tree/oblique_cpu_depthwise_symmetric_bagwide.h"

#ifdef DEPTHWISE_SYMMETRIC_BAGWIDE

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <numeric>
#include <vector>

#include "absl/log/check.h"
#include "absl/status/status.h"
#include "absl/types/span.h"
#include "yggdrasil_decision_forests/learner/decision_tree/oblique.h"
#include "yggdrasil_decision_forests/utils/parallel_chrono.h"

namespace yggdrasil_decision_forests::model::decision_tree {

absl::Status ApplyProjectionsDepthwiseSymmetricBagwide(
    const dataset::VerticalDataset& train_dataset,
    const google::protobuf::RepeatedField<int32_t>& numerical_features,
    absl::Span<const absl::Span<const UnsignedExampleIdx>>
        selected_examples_per_node,
    absl::Span<const internal::Projection> shared_projections,
    absl::Span<std::vector<float>> out_projected) {
  CHRONO_SCOPE(::yggdrasil_decision_forests::chrono_prof::kProjectionEvaluate);

  const size_t N = selected_examples_per_node.size();
  DCHECK_EQ(N, out_projected.size());
  if (N == 0) return absl::OkStatus();

  const size_t K = shared_projections.size();
  if (K == 0) {
    for (size_t n = 0; n < N; ++n) out_projected[n].clear();
    return absl::OkStatus();
  }

  // Per-node row counts + slab pre-size.
  std::vector<size_t> rows_n(N);
  size_t bag_size = 0;
  for (size_t n = 0; n < N; ++n) {
    rows_n[n] = selected_examples_per_node[n].size();
    bag_size += rows_n[n];
    out_projected[n].assign(K * rows_n[n], 0.f);
  }
  if (bag_size == 0) return absl::OkStatus();

  // Build the bag (concat per-node selected_examples + parallel node-id),
  // then sort by example_idx so projections sweep stride-1 in column space.
  // Each per-node selected_examples is already individually sorted ascending.
  std::vector<UnsignedExampleIdx> bag(bag_size);
  std::vector<uint32_t> node_of_bag(bag_size);
  {
    size_t cur = 0;
    for (size_t n = 0; n < N; ++n) {
      for (const UnsignedExampleIdx e : selected_examples_per_node[n]) {
        bag[cur] = e;
        node_of_bag[cur] = static_cast<uint32_t>(n);
        ++cur;
      }
    }
  }

  // Sort bag by example_idx, carrying node_of_bag with it.
  // Because per-node lists were already sorted, within each (sorted) node
  // group the relative order of that node's examples is preserved — that's
  // what makes the write_cursor[n]++ on the consumer side give the correct
  // pos_in_node[i] for slab[k * rows_n + pos].
  std::vector<uint32_t> perm(bag_size);
  std::iota(perm.begin(), perm.end(), 0u);
  std::stable_sort(perm.begin(), perm.end(),
                   [&bag](uint32_t a, uint32_t b) { return bag[a] < bag[b]; });
  {
    std::vector<UnsignedExampleIdx> sorted_bag(bag_size);
    std::vector<uint32_t> sorted_node_of_bag(bag_size);
    for (size_t i = 0; i < bag_size; ++i) {
      sorted_bag[i] = bag[perm[i]];
      sorted_node_of_bag[i] = node_of_bag[perm[i]];
    }
    bag.swap(sorted_bag);
    node_of_bag.swap(sorted_node_of_bag);
  }

  internal::ProjectionEvaluator evaluator(train_dataset, numerical_features);

  // K bag-wide sweeps. For each projection k:
  //  - Hoist per-item col pointers + weights + NA values out of the i-loop.
  //  - Walk sorted bag in stride-1 order. Per example: compute value, route
  //    to its owning node's slab via a write_cursor[].
  std::vector<uint32_t> write_cursor(N);
  for (size_t k = 0; k < K; ++k) {
    std::fill(write_cursor.begin(), write_cursor.end(), 0u);

    const auto& proj = shared_projections[k];
    const size_t M = proj.size();
    if (M == 0) continue;

    std::vector<const float*> col_ptrs(M);
    std::vector<float> ws(M);
#ifndef YDF_BENCH_SKIP_ISNAN
    std::vector<float> nas(M);
#endif
    for (size_t m = 0; m < M; ++m) {
      col_ptrs[m] = evaluator.AttributeValues(proj[m].attribute_idx).data();
      ws[m] = proj[m].weight;
#ifndef YDF_BENCH_SKIP_ISNAN
      nas[m] = evaluator.NaReplacementValue(proj[m].attribute_idx);
#endif
    }

    for (size_t i = 0; i < bag_size; ++i) {
      const UnsignedExampleIdx ex = bag[i];
      float value = 0.f;
      for (size_t m = 0; m < M; ++m) {
        float v = col_ptrs[m][ex];
#ifndef YDF_BENCH_SKIP_ISNAN
        if (std::isnan(v)) v = nas[m];
#endif
        value += ws[m] * v;
      }
      const uint32_t n = node_of_bag[i];
      const uint32_t pos = write_cursor[n]++;
      out_projected[n][k * rows_n[n] + pos] = value;
    }
  }

  return absl::OkStatus();
}

}  // namespace yggdrasil_decision_forests::model::decision_tree

#endif  // DEPTHWISE_SYMMETRIC_BAGWIDE
