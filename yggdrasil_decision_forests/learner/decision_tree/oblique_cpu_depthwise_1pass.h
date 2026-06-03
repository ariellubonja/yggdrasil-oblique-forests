// Depthwise fused-per-level CPU ApplyProjection for Oblique Random Forests:
// single-pass kernel across all (node, projection) tasks at a tree level.
//
// Ported from branch `1-pass-AP-CPU` (commit 98ed1c66). This path carries the
// depth scheduler only; row-block inner-kernel unrolling is a separate possible
// implementation improvement and is intentionally not included here.
//
// Design contrasted with Nodewise (ApplyProjectionsNodewiseProjMatrix,
// oblique_cpu_nodewise_proj_matrix.h): Nodewise walks the level's nodes
// serially on one thread, filling a per-node (rows x projections) matrix in
// each iteration. Depthwise flattens the level's (node, projection) pairs into
// a prefix-summed task range so every (node, projection) maps to a unique
// global task index. A thread pool iterates that task range; each task
// processes all rows of one (node, projection) pair and writes into its own
// contention-free per-node slab slot. Node-level parallelism falls out
// implicitly -- no special-case logic for N == 1 vs. N large.
//
// Mutually exclusive at build time with Nodewise (NODEWISE_PROJ_MATRIX) and
// with the symmetric-trees variants (SYMMETRIC_DEPTHWISE_AP /
// SYMMETRIC_NODEWISE_CONTROL).
//
// Output contract (shared with Nodewise and the symmetric kernel):
// out_projected[n] is a (P_n * rows_n)-float slab, row-minor within
// projection -- slab[p * rows_n + i] = <projections_per_node[n][p],
// features[selected_examples_per_node[n][i]]>, with NaN inputs replaced
// by the dataset-level feature mean when ENABLE_APPLYPROJECTION_ISNAN is
// defined.

#ifndef YGGDRASIL_DECISION_FORESTS_LEARNER_DECISION_TREE_OBLIQUE_CPU_DEPTHWISE_1PASS_H_
#define YGGDRASIL_DECISION_FORESTS_LEARNER_DECISION_TREE_OBLIQUE_CPU_DEPTHWISE_1PASS_H_

#include <vector>

#include "absl/status/status.h"
#include "absl/types/span.h"
#include "google/protobuf/repeated_field.h"
#include "yggdrasil_decision_forests/dataset/types.h"
#include "yggdrasil_decision_forests/dataset/vertical_dataset.h"
#include "yggdrasil_decision_forests/learner/decision_tree/oblique_types.h"

namespace yggdrasil_decision_forests::model::decision_tree {

// Fused-per-level Apply. `num_threads <= 1` runs the kernel inline on the
// caller thread (no thread pool); larger values dispatch (node, projection)
// tasks across a thread pool with contention-free per-slab writes.
absl::Status ApplyProjectionsDepthwise1Pass(
    const dataset::VerticalDataset& train_dataset,
    const google::protobuf::RepeatedField<int32_t>& numerical_features,
    absl::Span<const absl::Span<const UnsignedExampleIdx>>
        selected_examples_per_node,
    absl::Span<const std::vector<internal::Projection>> projections_per_node,
    absl::Span<std::vector<float>> out_projected,
    int num_threads);

}  // namespace yggdrasil_decision_forests::model::decision_tree

#endif  // YGGDRASIL_DECISION_FORESTS_LEARNER_DECISION_TREE_OBLIQUE_CPU_DEPTHWISE_1PASS_H_
