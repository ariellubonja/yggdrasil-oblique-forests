// V1 fused-per-level CPU ApplyProjection for Oblique Random Forests:
// per-node rows-outer / projections-inner matrix fill.
//
// The baseline per-node path
// (oblique.cc::FindBestConditionSparseObliqueTemplate) evaluates projections
// one at a time: for each projection p, sweep every row in the node. At depth
// d with P projections per node, a level's dataset slice is touched P * 2^d
// times even though every training example lives in exactly one node per BFS
// level.
//
// V1 reorders the work *within each node* so each row's feature loads are
// reused across all P projections for that row's node. It does NOT fuse
// across nodes — the outer loop still walks the level's nodes serially on
// the caller thread, so multi-node work is not parallelized.
//
// V2 (ApplyProjectionsDepthwise1Pass, oblique_cpu_depthwise_1pass.h) is the
// truly fused variant: a single-pass kernel across all (row, projection)
// tasks at the level, scheduled across threads with contention-free writes.
// The two variants are mutually exclusive at build time
// (NODEWISE_PROJ_MATRIX vs. DEPTHWISE_1_PASS; see label.h).
//
// Output contract (shared with V2): out_projected[n] is a (P_n * rows_n)-
// float slab, row-minor within projection — slab[p * rows_n + i] =
// <projections_per_node[n][p], features[selected_examples_per_node[n][i]]>,
// with NaN inputs replaced by the dataset-level feature mean (same
// convention as ProjectionEvaluator::Evaluate).
//
// Feeding those slabs into per-node NodeTrain via
// InternalTrainConfig::precomputed_projected_values causes
// FindBestConditionSparseObliqueTemplate to skip its own Apply step and run
// split-finding directly over the pre-computed values.

#ifndef YGGDRASIL_DECISION_FORESTS_LEARNER_DECISION_TREE_OBLIQUE_CPU_NODEWISE_PROJ_MATRIX_H_
#define YGGDRASIL_DECISION_FORESTS_LEARNER_DECISION_TREE_OBLIQUE_CPU_NODEWISE_PROJ_MATRIX_H_

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
// features[selected_examples_per_node[n][i]]>, with NaN inputs replaced by
// the dataset-level feature mean (same convention as
// ProjectionEvaluator::Evaluate).
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
absl::Status ApplyProjectionsNodewiseProjMatrix(
    const dataset::VerticalDataset& train_dataset,
    const google::protobuf::RepeatedField<int32_t>& numerical_features,
    absl::Span<const absl::Span<const UnsignedExampleIdx>>
        selected_examples_per_node,
    absl::Span<const std::vector<internal::Projection>> projections_per_node,
    absl::Span<std::vector<float>> out_projected);

}  // namespace yggdrasil_decision_forests::model::decision_tree

#endif  // YGGDRASIL_DECISION_FORESTS_LEARNER_DECISION_TREE_OBLIQUE_CPU_NODEWISE_PROJ_MATRIX_H_
