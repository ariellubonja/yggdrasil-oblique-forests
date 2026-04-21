// Fused per-level CPU ApplyProjection for Oblique Random Forests.
//
// The current per-node path (oblique.cc::FindBestConditionSparseObliqueTemplate)
// evaluates projections one at a time: for each projection p, sweep every row
// in the node. At depth d with P projections per node, a level's dataset slice
// is touched P * 2^d times, even though every training example lives in exactly
// one node per BFS level — one pass over the data could produce every
// (node, projection) output.
//
// ApplyProjectionsFusedLevel reorders the work so each row's feature loads are
// reused across all P projections for that row's node. Output shape matches
// what EvaluateProjection already consumes: a per-node contiguous slab of
// floats, row-minor within projection (projected[p * rows_n + i]).
//
// Feeding those slabs into per-node NodeTrain via
// InternalTrainConfig::depthwise_cpu_projected_values causes
// FindBestConditionSparseObliqueTemplate to skip its own Apply step and run
// split-finding directly over the pre-computed values.

#ifndef YGGDRASIL_DECISION_FORESTS_LEARNER_DECISION_TREE_OBLIQUE_CPU_DEPTHWISE_H_
#define YGGDRASIL_DECISION_FORESTS_LEARNER_DECISION_TREE_OBLIQUE_CPU_DEPTHWISE_H_

#include <vector>

#include "absl/status/status.h"
#include "absl/types/span.h"
#include "google/protobuf/repeated_field.h"
#include "yggdrasil_decision_forests/dataset/types.h"
#include "yggdrasil_decision_forests/dataset/vertical_dataset.h"
#include "yggdrasil_decision_forests/learner/decision_tree/oblique_types.h"

namespace yggdrasil_decision_forests::model::decision_tree {

// Fills out_projected[n] with a (P_n * rows_n)-float slab, laid out row-minor
// within projection: slab[p * rows_n + i] = <projections_per_node[n][p],
// features[selected_examples_per_node[n][i]]>, with NaN inputs replaced by the
// dataset-level feature mean (same convention as ProjectionEvaluator::Evaluate).
//
// Preconditions:
//   out_projected.size() == selected_examples_per_node.size()
//                        == projections_per_node.size() == N.
//   Each out_projected[n] is resized/overwritten internally; its prior
//   contents are ignored.
//
// Single-threaded V1. Per-node outer loop; within a node, rows outer and
// projections inner so each row's feature reads are amortized across all P
// projections for the node.
absl::Status ApplyProjectionsFusedLevel(
    const dataset::VerticalDataset& train_dataset,
    const google::protobuf::RepeatedField<int32_t>& numerical_features,
    absl::Span<const absl::Span<const UnsignedExampleIdx>>
        selected_examples_per_node,
    absl::Span<const std::vector<internal::Projection>> projections_per_node,
    absl::Span<std::vector<float>> out_projected);

}  // namespace yggdrasil_decision_forests::model::decision_tree

#endif  // YGGDRASIL_DECISION_FORESTS_LEARNER_DECISION_TREE_OBLIQUE_CPU_DEPTHWISE_H_
