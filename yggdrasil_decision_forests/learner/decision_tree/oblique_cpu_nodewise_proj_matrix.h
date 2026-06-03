// Nodewise fused-per-level CPU ApplyProjection for Oblique Random Forests:
// per-node rows-outer / projections-inner matrix fill.
//
// Ported from branch `1-pass-AP-CPU` (commit 98ed1c66). Used as the
// control counterpart to Depthwise (DEPTHWISE_1_PASS): Nodewise reorders the
// work *within each node* so each row's feature loads are reused across all P
// projections for that row's node, but it does NOT fuse across nodes -- the
// outer loop walks the level's nodes serially on the caller thread, so
// multi-node work is not parallelized.
//
// Depthwise (ApplyProjectionsDepthwise1Pass, oblique_cpu_depthwise_1pass.h)
// is the fused variant: a single-pass kernel across all (node, projection)
// tasks at the level, scheduled across threads with contention-free writes.
//
// The two variants and the symmetric kernels are mutually exclusive at
// build time (NODEWISE_PROJ_MATRIX vs. DEPTHWISE_1_PASS vs.
// SYMMETRIC_DEPTHWISE_AP / SYMMETRIC_NODEWISE_CONTROL).
//
// Output contract (shared with Depthwise and the symmetric kernel):
// out_projected[n] is a (P_n * rows_n)-float slab, row-minor within
// projection -- slab[p * rows_n + i] = <projections_per_node[n][p],
// features[selected_examples_per_node[n][i]]>, with NaN inputs replaced
// by the dataset-level feature mean when ENABLE_APPLYPROJECTION_ISNAN is
// defined.

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

// Single-threaded Nodewise control. Per-node outer loop; within a node, rows outer and
// projections inner so each row's feature reads are amortized across all
// P projections for the node.
absl::Status ApplyProjectionsNodewiseProjMatrix(
    const dataset::VerticalDataset& train_dataset,
    const google::protobuf::RepeatedField<int32_t>& numerical_features,
    absl::Span<const absl::Span<const UnsignedExampleIdx>>
        selected_examples_per_node,
    absl::Span<const std::vector<internal::Projection>> projections_per_node,
    absl::Span<std::vector<float>> out_projected);

}  // namespace yggdrasil_decision_forests::model::decision_tree

#endif  // YGGDRASIL_DECISION_FORESTS_LEARNER_DECISION_TREE_OBLIQUE_CPU_NODEWISE_PROJ_MATRIX_H_
