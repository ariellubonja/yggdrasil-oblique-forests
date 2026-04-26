// V2 fused-per-level CPU ApplyProjection for Oblique Random Forests:
// single-pass kernel across all (row, projection) tasks at a tree level.
//
// Design contrasted with V1 (ApplyProjectionsNodewiseProjMatrix,
// oblique_cpu_nodewise_proj_matrix.h): V1 walks the level's nodes serially
// on one thread, filling a per-node (rows × projections) matrix in each
// iteration. V2 flattens the level's per-node selected-example lists and
// per-node projection lists into prefix-summed task ranges so every
// (node, row, projection) triplet maps to a unique global task index. A
// thread pool iterates that task range; each task does one projection read
// + one write to its own output slot, contention-free. Node-level
// parallelism falls out implicitly — no special-case logic for N = 1 vs.
// N large.
//
// The two variants are mutually exclusive at build time
// (NODEWISE_PROJ_MATRIX vs. DEPTHWISE_1_PASS; see label.h).
//
// Output contract (shared with V1): out_projected[n] is a (P_n * rows_n)-
// float slab, row-minor within projection — slab[p * rows_n + i] =
// <projections_per_node[n][p], features[selected_examples_per_node[n][i]]>,
// with NaN inputs replaced by the dataset-level feature mean.

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

// Same output contract as V1 (ApplyProjectionsNodewiseProjMatrix). Takes
// one extra `num_threads` parameter controlling the thread pool used by
// the single-pass kernel; num_threads <= 1 runs the kernel inline on the
// caller thread (still correct, no parallelism). See header comment above
// for the design contrast.
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
