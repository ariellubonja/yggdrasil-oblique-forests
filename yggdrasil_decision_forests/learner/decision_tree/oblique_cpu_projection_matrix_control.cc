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

    // Rows outer, projections inner. For a fixed row, its feature loads
    // stay hot in cache while all P projections consume them; the baseline
    // layout re-sweeps the node's row range once per projection.
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
      } else {
        for (size_t i = 0; i < rows_n; ++i) {
          const UnsignedExampleIdx ex = selected[i];
          for (size_t p = 0; p < P; ++p) {
            float acc = 0.f;
            for (const auto& feat : projs[p]) {
              float v = evaluator.AttributeValue(feat.attribute_idx, ex);
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
      }
    }
  }

  return absl::OkStatus();
}

}  // namespace yggdrasil_decision_forests::model::decision_tree

#endif  // PROJECTION_MATRIX_CONTROL
