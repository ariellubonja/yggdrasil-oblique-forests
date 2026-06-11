#include "yggdrasil_decision_forests/learner/decision_tree/oblique_cpu_projection_matrix_control.h"

#ifdef PROJECTION_MATRIX_CONTROL

#include <cmath>
#include <cstddef>
#include <cstdlib>
#include <limits>
#include <vector>

#include "absl/log/check.h"
#include "absl/status/status.h"
#include "absl/types/span.h"
#include "yggdrasil_decision_forests/learner/decision_tree/oblique.h"
#include "yggdrasil_decision_forests/utils/parallel_chrono.h"

namespace yggdrasil_decision_forests::model::decision_tree {

namespace {

// Dynamic_Row_Col_Major dispatch threshold: nodes with at most this many
// selected rows take the row-major path; larger nodes take the column-major
// path. Experiment knob, read once. Unset => row-major for every node.
size_t RowMajorMaxRows() {
  static const size_t value = [] {
    const char* e = std::getenv("YDF_RM_MAX_ROWS");
    return e != nullptr ? static_cast<size_t>(std::strtoull(e, nullptr, 10))
                        : std::numeric_limits<size_t>::max();
  }();
  return value;
}

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
    {
      CHRONO_SCOPE(::yggdrasil_decision_forests::chrono_prof::kPmcPreSize);
      out.assign(P * rows_n, 0.f);
    }

    // Per-node kernel dispatch: Dynamic_Row_Col_Major treatment paths when
    // the bf16/fp32 dual stores are active (alternate trunk layouts), else
    // the depthwise_1_pass control: dw1's exact kernel work executed in node
    // order instead of column-sorted order (no column sharing).
    {
      CHRONO_SCOPE(::yggdrasil_decision_forests::chrono_prof::kPmcSweep);
      const auto* bf16_rows = evaluator.Bf16Rows();
      const auto* bf16_cols = evaluator.Bf16Cols();
      if (bf16_rows != nullptr &&
          (rows_n <= RowMajorMaxRows() || bf16_cols == nullptr)) {
        // Row-major bf16: one pass per row serves all P projections; the
        // row's feature loads share pages/lines instead of P*d isolated
        // DRAM gathers.
        for (size_t i = 0; i < rows_n; ++i) {
          const uint16_t* row = bf16_rows->row_ptr(selected[i]);
          for (size_t p = 0; p < P; ++p) {
            float acc = 0.f;
            for (const auto& feat : projs[p]) {
              float v = dataset::Bf16ToFloat(row[feat.attribute_idx]);
#ifdef ENABLE_APPLYPROJECTION_ISNAN
              if (std::isnan(v)) {
                v = evaluator.NaReplacementValue(feat.attribute_idx);
              }
#endif
              acc += feat.weight * v;
            }
            out[p * rows_n + i] = acc;
          }
        }
      } else if (bf16_cols != nullptr) {
        // Column-major bf16 for large/shallow nodes: per projection, walk
        // only the needed columns sequentially-ish; 4-row blocks expose
        // independent accumulators (V2-rev3 pattern).
        for (size_t p = 0; p < P; ++p) {
          float* o = &out[p * rows_n];
          size_t i = 0;
          for (; i + 4 <= rows_n; i += 4) {
            float a0 = 0.f, a1 = 0.f, a2 = 0.f, a3 = 0.f;
            for (const auto& feat : projs[p]) {
              const uint16_t* col = bf16_cols->col_ptr(feat.attribute_idx);
              const float w = feat.weight;
              a0 += w * dataset::Bf16ToFloat(col[selected[i + 0]]);
              a1 += w * dataset::Bf16ToFloat(col[selected[i + 1]]);
              a2 += w * dataset::Bf16ToFloat(col[selected[i + 2]]);
              a3 += w * dataset::Bf16ToFloat(col[selected[i + 3]]);
            }
            o[i] = a0;
            o[i + 1] = a1;
            o[i + 2] = a2;
            o[i + 3] = a3;
          }
          for (; i < rows_n; ++i) {
            float acc = 0.f;
            for (const auto& feat : projs[p]) {
              acc += feat.weight * dataset::Bf16ToFloat(
                                       bf16_cols->col_ptr(
                                           feat.attribute_idx)[selected[i]]);
            }
            o[i] = acc;
          }
        }
      } else if (const auto* fp32_rows = evaluator.Fp32Rows();
                 fp32_rows != nullptr &&
                 (rows_n <= RowMajorMaxRows() ||
                  evaluator.Fp32Cols() == nullptr)) {
        // Row-major fp32: same dispatch as the bf16 hybrid, full precision.
        // Accumulation order must match EvalConditionOblique's scalar loop
        // (split application) bit-for-bit; icx fast-math may otherwise
        // reassociate one loop and not the other, flipping examples whose
        // projected value straddles the threshold by ulps.
#pragma clang fp reassociate(off)
        for (size_t i = 0; i < rows_n; ++i) {
          const float* row = fp32_rows->row_ptr(selected[i]);
          for (size_t p = 0; p < P; ++p) {
            float acc = 0.f;
            for (const auto& feat : projs[p]) {
              float v = row[feat.attribute_idx];
#ifdef ENABLE_APPLYPROJECTION_ISNAN
              if (std::isnan(v)) {
                v = evaluator.NaReplacementValue(feat.attribute_idx);
              }
#endif
              acc += feat.weight * v;
            }
            out[p * rows_n + i] = acc;
          }
        }
      } else if (const auto* fp32_cols = evaluator.Fp32Cols();
                 fp32_cols != nullptr) {
        // Same strict-order requirement as the row-major fp32 kernel above:
        // per-example accumulation must match split application bit-for-bit.
#pragma clang fp reassociate(off)
        for (size_t p = 0; p < P; ++p) {
          float* o = &out[p * rows_n];
          size_t i = 0;
          for (; i + 4 <= rows_n; i += 4) {
            float a0 = 0.f, a1 = 0.f, a2 = 0.f, a3 = 0.f;
            for (const auto& feat : projs[p]) {
              const float* col = fp32_cols->col_ptr(feat.attribute_idx);
              const float w = feat.weight;
              a0 += w * col[selected[i + 0]];
              a1 += w * col[selected[i + 1]];
              a2 += w * col[selected[i + 2]];
              a3 += w * col[selected[i + 3]];
            }
            o[i] = a0;
            o[i + 1] = a1;
            o[i + 2] = a2;
            o[i + 3] = a3;
          }
          for (; i < rows_n; ++i) {
            float acc = 0.f;
            for (const auto& feat : projs[p]) {
              acc += feat.weight *
                     fp32_cols->col_ptr(feat.attribute_idx)[selected[i]];
            }
            o[i] = acc;
          }
        }
      } else if (evaluator.AttributeData(
                     numerical_features.Get(0)) != nullptr) {
        // depthwise_1_pass control. Same kernels as dw1, same big/small
        // split, but the (node, projection, feature) references execute in
        // node order — no cross-node column grouping. dw1 minus this =
        // column-sharing benefit (the entry build + counting sort are
        // intrinsic costs of sharing and stay charged to dw1).
        if (P * rows_n > PmcNodeBudgetFloats()) {
          // Big node: identical to dw1's projection-major path — per
          // projection, hoist column pointers, one dot product per row,
          // single write (no accumulate).
          struct FeatRef {
            const float* col;
            float weight;
#ifdef ENABLE_APPLYPROJECTION_ISNAN
            float na;
#endif
          };
          std::vector<FeatRef> feats;
          for (size_t p = 0; p < P; ++p) {
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
            float* o = &out[p * rows_n];
            for (size_t i = 0; i < rows_n; ++i) {
              const UnsignedExampleIdx ex = selected[i];
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
        } else {
          // Small node: identical to dw1's per-entry execution — one column
          // at a time, accumulate sweep over the node's rows — but in
          // (node, projection, feature) order instead of column-sorted.
          for (size_t p = 0; p < P; ++p) {
            float* o = &out[p * rows_n];
            for (const auto& feat : projs[p]) {
              const float* col = evaluator.AttributeData(feat.attribute_idx);
              const float w = feat.weight;
#ifdef ENABLE_APPLYPROJECTION_ISNAN
              const float na =
                  evaluator.NaReplacementValue(feat.attribute_idx);
#endif
              for (size_t i = 0; i < rows_n; ++i) {
                float v = col[selected[i]];
#ifdef ENABLE_APPLYPROJECTION_ISNAN
                if (std::isnan(v)) v = na;
#endif
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
