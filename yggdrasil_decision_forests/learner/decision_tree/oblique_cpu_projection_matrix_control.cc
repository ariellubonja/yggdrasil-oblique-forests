#include "yggdrasil_decision_forests/learner/decision_tree/oblique_cpu_projection_matrix_control.h"

#ifdef PROJECTION_MATRIX_CONTROL

#include <cmath>
#include <cstddef>
#include <cstdlib>
#include <vector>

#include "absl/log/check.h"
#include "absl/status/status.h"
#include "absl/types/span.h"
#include "yggdrasil_decision_forests/learner/decision_tree/oblique.h"
#include "yggdrasil_decision_forests/utils/parallel_chrono.h"

namespace yggdrasil_decision_forests::model::decision_tree {

namespace {

// Big-node cutoff shared with depthwise_1_pass (same env knob and default),
// so PMC and dw1 split big-vs-small nodes identically. Nodes whose slab
// exceeds this run the same projection-major kernel in both builds; smaller
// nodes run the same feature-major accumulate loop — PMC in node order, dw1
// in column-sorted order. That ordering is the single ablation variable.
size_t PmcNodeBudgetFloats() {
  static const size_t value = [] {
    const char* e = std::getenv("YDF_DW1_BLOCK_FLOATS");
    return e != nullptr ? static_cast<size_t>(std::strtoull(e, nullptr, 10))
                        : static_cast<size_t>(64) << 20;
  }();
  return value;
}

}  // namespace

absl::Status ApplyProjectionsProjectionMatrixControl(
    const dataset::VerticalDataset& train_dataset,
    const google::protobuf::RepeatedField<int32_t>& numerical_features,
    absl::Span<const absl::Span<const UnsignedExampleIdx>>
        selected_examples_per_node,
    absl::Span<const std::vector<internal::Projection>> projections_per_node,
    absl::Span<std::vector<float>> out_projected) {
  const size_t N = selected_examples_per_node.size();
  DCHECK_EQ(N, projections_per_node.size());
  DCHECK_EQ(N, out_projected.size());

  // Outer scope: total ProjectionEvaluate budget. Sub-phase scopes below
  // partition the per-node work into PmcPreSize / PmcSweep, accumulated
  // across nodes. The ProjectionEvaluator ctor sits outside both sub-scopes,
  // so kProjectionEvaluate − (kPmcPreSize + kPmcSweep) ≈ ctor cost.
  CHRONO_SCOPE(::yggdrasil_decision_forests::chrono_prof::kProjectionEvaluate);

  internal::ProjectionEvaluator evaluator(train_dataset, numerical_features);

  for (size_t n = 0; n < N; ++n) {
    const auto selected = selected_examples_per_node[n];
    const auto& projs = projections_per_node[n];
    const size_t rows_n = selected.size();
    const size_t P = projs.size();

    auto& out = out_projected[n];
    out.assign(P * rows_n, 0.f);
    

    // Per-node kernel dispatch: the depthwise_1_pass control — dw1's exact
    // kernel work executed in (node, projection, feature) order instead of
    // column-sorted order (no column sharing). dw1 minus this = the
    // column-sharing benefit (the entry build + counting sort are intrinsic
    // costs of sharing and stay charged to dw1)
    {
      CHRONO_SCOPE(::yggdrasil_decision_forests::chrono_prof::kPmcSweep);
      if (evaluator.AttributeData(numerical_features.Get(0)) != nullptr) {
        // Same kernels as dw1, same big/small split, but the
        // (node, projection, feature) references execute in node order — no
        // cross-node column grouping.
        if (P * rows_n > PmcNodeBudgetFloats()) {
          // Big node: identical to dw1's projection-major path — per
          // projection, hoist column pointers, one dot product per row,
          // single write (no accumulate).
          struct FeatRef {
            const float* col;
            float weight;
          };
          std::vector<FeatRef> feats;
          for (size_t p = 0; p < P; ++p) {
            feats.clear();
            for (const auto& feat : projs[p]) {
              feats.push_back({evaluator.AttributeData(feat.attribute_idx),
                               feat.weight});
            }
            float* o = &out[p * rows_n];
            for (size_t i = 0; i < rows_n; ++i) {
              const UnsignedExampleIdx ex = selected[i];
              float acc = 0.f;
              for (const auto& f : feats) {
                float v = f.col[ex];
                acc += f.weight * v;
              }
              o[i] = acc;
            }
          }
        } else {
          // Small node: identical to dw1's per-entry execution — one column
          // at a time, accumulate sweep over the node's rows — but in
          // (node, projection, feature) order instead of column-sorted.
          for (size_t p = 0; p < P; ++p) {
            float* o = &out[p * rows_n];
            for (const auto& feat : projs[p]) {
              const float* col = evaluator.AttributeData(feat.attribute_idx);
              const float w = feat.weight;
              for (size_t i = 0; i < rows_n; ++i) {
                float v = col[selected[i]];
                o[i] += w * v;
              }
            }
          }
        }
      } else {
        // No direct column pointers (alternate layouts without active
        // stores): generic per-(node, projection) loop, same as dw1's
        // generic fallback.
        for (size_t p = 0; p < P; ++p) {
          float* o = &out[p * rows_n];
          for (size_t i = 0; i < rows_n; ++i) {
            const UnsignedExampleIdx ex = selected[i];
            float value = 0;
            for (const auto& feat : projs[p]) {
              float v = evaluator.AttributeValue(feat.attribute_idx, ex);
#ifdef ENABLE_APPLYPROJECTION_ISNAN
              if (std::isnan(v)) {
                v = evaluator.NaReplacementValue(feat.attribute_idx);
              }
#endif
              value += v * feat.weight;
            }
            o[i] = value;
          }
        }
      }
    }
  }

  return absl::OkStatus();
}

}  // namespace yggdrasil_decision_forests::model::decision_tree

#endif  // PROJECTION_MATRIX_CONTROL
