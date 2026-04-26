#include "yggdrasil_decision_forests/learner/decision_tree/oblique_cpu_nodewise_proj_matrix.h"

#ifdef NODEWISE_PROJ_MATRIX

#include <cmath>
#include <cstddef>
#include <vector>

#include "absl/log/check.h"
#include "absl/status/status.h"
#include "absl/types/span.h"
#include "yggdrasil_decision_forests/learner/decision_tree/oblique.h"
#include "yggdrasil_decision_forests/utils/parallel_chrono.h"

namespace yggdrasil_decision_forests::model::decision_tree {

absl::Status ApplyProjectionsNodewiseProjMatrix(
    const dataset::VerticalDataset& train_dataset,
    const google::protobuf::RepeatedField<int32_t>& numerical_features,
    absl::Span<const absl::Span<const UnsignedExampleIdx>>
        selected_examples_per_node,
    absl::Span<const std::vector<internal::Projection>> projections_per_node,
    absl::Span<std::vector<float>> out_projected) {
  const size_t N = selected_examples_per_node.size();
  DCHECK_EQ(N, projections_per_node.size());
  DCHECK_EQ(N, out_projected.size());

  // Tagged with kProjectionEvaluate so benchmark CSVs can diff V1 vs. V2 and
  // the baseline per-node projection loop directly at the same tag.
  CHRONO_SCOPE(::yggdrasil_decision_forests::chrono_prof::kProjectionEvaluate);

  internal::ProjectionEvaluator evaluator(train_dataset, numerical_features);

  for (size_t n = 0; n < N; ++n) {
    const auto selected = selected_examples_per_node[n];
    const auto& projs = projections_per_node[n];
    const size_t rows_n = selected.size();
    const size_t P = projs.size();

    auto& out = out_projected[n];
    out.assign(P * rows_n, 0.f);

    // Rows outer, projections inner. For a fixed row, its feature loads stay
    // hot in cache while all P projections consume them; the baseline layout
    // re-sweeps the node's row range once per projection.
    for (size_t i = 0; i < rows_n; ++i) {
      const UnsignedExampleIdx ex = selected[i];
      for (size_t p = 0; p < P; ++p) {
        float acc = 0.f;
        for (const auto& feat : projs[p]) {
          float v = evaluator.AttributeValues(feat.attribute_idx)[ex];
#ifndef YDF_BENCH_SKIP_ISNAN
          if (std::isnan(v)) v = evaluator.NaReplacementValue(feat.attribute_idx);
#endif
          acc += feat.weight * v;
        }
        out[p * rows_n + i] = acc;
      }
    }
  }

  return absl::OkStatus();
}

}  // namespace yggdrasil_decision_forests::model::decision_tree

#endif  // NODEWISE_PROJ_MATRIX
