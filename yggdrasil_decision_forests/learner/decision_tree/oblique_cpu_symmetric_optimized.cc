#include "yggdrasil_decision_forests/learner/decision_tree/oblique_cpu_symmetric_optimized.h"

#ifdef SYMMETRIC_OPTIMIZED

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <vector>

#include "absl/log/check.h"
#include "absl/status/status.h"
#include "absl/types/span.h"
#include "yggdrasil_decision_forests/learner/decision_tree/oblique.h"
#include "yggdrasil_decision_forests/learner/decision_tree/oblique_cpu_depthwise_bag.h"
#include "yggdrasil_decision_forests/utils/parallel_chrono.h"

namespace yggdrasil_decision_forests::model::decision_tree {

absl::Status ApplyProjectionsSymmetricOptimized(
    const dataset::VerticalDataset& train_dataset,
    const google::protobuf::RepeatedField<int32_t>& numerical_features,
    absl::Span<const absl::Span<const UnsignedExampleIdx>>
        selected_examples_per_node,
    absl::Span<const internal::Projection> shared_projections,
    absl::Span<const int32_t> prev_first_child,
    SymmetricBagState* bag_state,
    absl::Span<std::vector<float>> out_projected) {
  // Outer scope: total ProjectionEvaluate budget. Sub-phase scopes below
  // partition this into BuildBag / SortBag / Sweep so parallel_chrono.py
  // can break ApplyProjection down per depth.
  CHRONO_SCOPE_COARSE(::yggdrasil_decision_forests::chrono_prof::kProjectionEvaluate);

  const size_t N = selected_examples_per_node.size();
  if (N == 0) return absl::OkStatus();

  const size_t K = shared_projections.size();
  if (K == 0) {
    for (size_t n = 0; n < N; ++n) out_projected[n].clear();
    // The bag for this depth batch was never materialized, so the state
    // cannot seed the next depth's relabel.
    bag_state->valid = false;
    return absl::OkStatus();
  }

  // ── Phase 1: BuildBag 
  // Per-node row counts + slab storage. Not 0-init bcs. whole vector is overwritten
  const bool single_node = (N == 1);

  std::vector<size_t> rows_n(N);
  std::vector<float*> out_ptr(N, nullptr);
  size_t bag_size = 0;
  {
    CHRONO_SCOPE_AP(::yggdrasil_decision_forests::chrono_prof::kSymBuildBag);
    for (size_t n = 0; n < N; ++n) {
      rows_n[n] = selected_examples_per_node[n].size();
      bag_size += rows_n[n];
      out_projected[n].reserve(K * rows_n[n]);
      out_ptr[n] = out_projected[n].data();
    }
    if (bag_size == 0) {
      bag_state->valid = false;
      return absl::OkStatus();
    }
  }

  // ── Phase 2: obtain the sorted bag 
  // Delegated to the shared depth-bag module: incremental O(bag) relabel of
  // the previous depth's sorted bag in the steady state, concat + VQSort
  // fallback otherwise (see oblique_cpu_depthwise_bag.{h,cc}). Billed to
  // kSymBuildBag / kSymSortBag exactly as before via the kSymmetric tag. On
  // return bag_state->{bag,node_of_bag} describe this depth and .valid is set.
  AdvanceDepthBag(selected_examples_per_node, prev_first_child, bag_size,
                  DepthBagChrono::kSymmetric, bag_state);
  const UnsignedExampleIdx* bag_data = bag_state->bag.data();

  // ── Phase 3: Sweep ────────────────────────────────────────────────
  // K bag-wide sweeps. For each projection k:
  //  - Hoist per-item col pointers + weights + NA values out of the i-loop.
  //  - Walk the (sorted) bag in stride-1 order. Per example: compute value,
  //    route to its owning node's slab (written through the reserved-only
  //    raw pointer out_ptr[n], never operator[] — the slab .size() is 0).
  // The N == 1 case writes sequentially (bag order == node 0's slab order)
  // with no node_of_bag / write_cursor indirection; the N > 1 case routes
  // each result via the per-node write cursor. The single_node test is
  // hoisted out of the hot i-loop.
  {
    CHRONO_SCOPE_AP(::yggdrasil_decision_forests::chrono_prof::kSymSweep);
    internal::ProjectionEvaluator evaluator(train_dataset, numerical_features);

    // AdvanceDepthBag always sizes node_of_bag to the bag (all zeros for a
    // single-node depth); the single-node sweep below skips node routing
    // purely as a performance fast path.
    const uint32_t* node_of_bag = bag_state->node_of_bag.data();

    std::vector<uint32_t> write_cursor;
    if (!single_node) write_cursor.assign(N, 0u);

    for (size_t k = 0; k < K; ++k) {
      const auto& proj = shared_projections[k];
      const size_t M = proj.size();
      // SampleProjection guarantees nnz >= 1; an empty projection here would
      // leave this k's slab region uninitialized, but the finder skips empty
      // projections (oblique.cc), so it is never read. Defensive.
      if (M == 0) continue;

      std::vector<const float*> col_ptrs(M);
      std::vector<float> ws(M);
      for (size_t m = 0; m < M; ++m) {
        col_ptrs[m] = evaluator.AttributeData(proj[m].attribute_idx);
        ws[m] = proj[m].weight;
      }

      if (single_node) {
        float* out = out_ptr[0] + k * rows_n[0];
        for (size_t i = 0; i < bag_size; ++i) {
          const UnsignedExampleIdx ex = bag_data[i];
          float value = 0.f;
          for (size_t m = 0; m < M; ++m) {
            float v = col_ptrs[m] != nullptr
                          ? col_ptrs[m][ex]
                          : evaluator.AttributeValue(proj[m].attribute_idx, ex);
            value += ws[m] * v;
          }
          out[i] = value;
        }
      } else {
        std::fill(write_cursor.begin(), write_cursor.end(), 0u);
        for (size_t i = 0; i < bag_size; ++i) {
          const UnsignedExampleIdx ex = bag_data[i];
          float value = 0.f;
          for (size_t m = 0; m < M; ++m) {
            float v = col_ptrs[m] != nullptr
                          ? col_ptrs[m][ex]
                          : evaluator.AttributeValue(proj[m].attribute_idx, ex);
            value += ws[m] * v;
          }
          const uint32_t n = node_of_bag[i];
          const uint32_t pos = write_cursor[n]++;
          out_ptr[n][k * rows_n[n] + pos] = value;
        }
      }
    }
  }

  return absl::OkStatus();
}

}  // namespace yggdrasil_decision_forests::model::decision_tree

#endif  // SYMMETRIC_OPTIMIZED
