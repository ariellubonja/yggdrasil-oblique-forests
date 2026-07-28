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
  AdvanceDepthBag(selected_examples_per_node, prev_first_child, bag_size,
                  DepthBagChrono::kSymmetric, bag_state);
  const UnsignedExampleIdx* bag_data = bag_state->bag.data();

  // ── Phase 3: Sweep 
  {
    CHRONO_SCOPE_AP(::yggdrasil_decision_forests::chrono_prof::kSymSweep);
    internal::ProjectionEvaluator evaluator(train_dataset, numerical_features);
    // The sweep below walks raw column pointers, which exist only for the
    // default column-major VerticalDataset layout. Alternate layouts
    // (row-major mirror) are not supported yet.
    if (!evaluator.IsColumnMajor()) {
      return absl::UnimplementedError(
          "The symmetric-optimized oblique kernel requires the column-major "
          "VerticalDataset layout. Alternate dataset layouts (e.g. "
          "--config=row_major_dataset_layout with --dataset_layout=row) are "
          "not supported with --config=symmetric_optimized yet.");
    }

    // AdvanceDepthBag always sizes node_of_bag to the bag, so the routed sweep
    // below handles any N (a 1-node depth routes everything to node 0).
    const uint32_t* node_of_bag = bag_state->node_of_bag.data();

    std::vector<uint32_t> write_cursor(N, 0u);

    for (size_t k = 0; k < K; ++k) {
      CHRONO_BEGIN_AP(sym_col_setup);
      const auto& proj = shared_projections[k];
      const size_t M = proj.size();
      // SampleProjection guarantees nnz >= 1; but the finder skips empty
      // projections (oblique.cc), so it is never read. Defensive.
      if (M == 0) continue;

      std::vector<const float*> col_ptrs(M);
      std::vector<float> ws(M);
      for (size_t m = 0; m < M; ++m) {
        col_ptrs[m] = evaluator.AttributeData(proj[m].attribute_idx);
        ws[m] = proj[m].weight;
      }

      std::fill(write_cursor.begin(), write_cursor.end(), 0u);
      CHRONO_END_AP(sym_col_setup,
                    ::yggdrasil_decision_forests::chrono_prof::kSymSweepColSetup);

      CHRONO_BEGIN_AP(sym_main_loop);
      for (size_t i = 0; i < bag_size; ++i) {
        const UnsignedExampleIdx ex = bag_data[i];
        float value = 0.f;
        for (size_t m = 0; m < M; ++m) {
          value += ws[m] * col_ptrs[m][ex];
        }
        const uint32_t n = node_of_bag[i];
        const uint32_t pos = write_cursor[n]++;
        out_ptr[n][k * rows_n[n] + pos] = value;
      }
      CHRONO_END_AP(sym_main_loop,
                    ::yggdrasil_decision_forests::chrono_prof::kSymSweepMainLoop);
    }
  }

  return absl::OkStatus();
}

}

#endif
