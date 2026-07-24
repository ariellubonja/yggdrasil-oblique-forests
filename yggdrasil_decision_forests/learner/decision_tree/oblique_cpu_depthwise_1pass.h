// Depthwise fused-per-level CPU ApplyProjection for Oblique Random Forests:
// single-pass kernel across all (node, projection) tasks at a tree level.
//
// Ported from branch `1-pass-AP-CPU` (commit 98ed1c66). This path carries the
// depth scheduler only; row-block inner-kernel unrolling is a separate possible
// implementation improvement and is intentionally not included here.
//
// Depthwise groups the level's nodes into blocks and
// buckets every (node, projection, weight) reference by column, then sweeps
// each touched column once across the block (column sharing across nodes).
// It runs single-threaded on the caller thread: RandomForest already trains
// one tree per thread, so an internal pool here would only oversubscribe.
//
// Mutually exclusive at build time with the symmetric-trees variant
// (SYMMETRIC_DEPTHWISE_AP).
//
// Output contract (shared with the symmetric kernel):
// out_projected[n] is a (P_n * rows_n)-float slab, row-minor within
// projection -- slab[p * rows_n + i] = <projections_per_node[n][p],
// features[selected_examples_per_node[n][i]]>, with NaN inputs replaced
// by the dataset-level feature mean when ENABLE_ISNAN is
// defined.

#ifndef YGGDRASIL_DECISION_FORESTS_LEARNER_DECISION_TREE_OBLIQUE_CPU_DEPTHWISE_1PASS_H_
#define YGGDRASIL_DECISION_FORESTS_LEARNER_DECISION_TREE_OBLIQUE_CPU_DEPTHWISE_1PASS_H_

#include <vector>

#include "absl/status/status.h"
#include "absl/types/span.h"
#include "google/protobuf/repeated_field.h"
#include "yggdrasil_decision_forests/dataset/types.h"
#include "yggdrasil_decision_forests/dataset/vertical_dataset.h"
#include "yggdrasil_decision_forests/learner/decision_tree/oblique_cpu_depthwise_bag.h"
#include "yggdrasil_decision_forests/learner/decision_tree/oblique_types.h"

namespace yggdrasil_decision_forests::model::decision_tree {

// Fused-per-level Apply. Runs single-threaded on the caller thread:
// RandomForest already trains one tree per thread, so parallelizing here
// would oversubscribe the machine. The whole level's (node, projection)
// work is processed inline.
//
// `prev_first_child` + `bag_state` drive the shared-rows colwalk's depth bag
// (see oblique_cpu_depthwise_bag.h): the DW1_SHARED_ROWS build relabels the
// previous depth's sorted bag into this depth's, then distributes it into
// per-block ex-sorted arenas. The col-sharing build (no DW1_SHARED_ROWS)
// compiles the same signature and ignores both arguments — its path has no
// bag and must not pay for one. Pass an empty span + nullptr from that path.
//
// `current_depth` is used only by the per-depth column-stats debug print
// (YDF_DW1_COL_STATS); the kernel itself is depth-agnostic.
absl::Status ApplyProjectionsDepthwise1Pass(
    const dataset::VerticalDataset& train_dataset,
    const google::protobuf::RepeatedField<int32_t>& numerical_features,
    absl::Span<const absl::Span<const UnsignedExampleIdx>>
        selected_examples_per_node,
    absl::Span<const std::vector<internal::Projection>> projections_per_node,
    absl::Span<const int32_t> prev_first_child, DepthBagState* bag_state,
    absl::Span<std::vector<float>> out_projected, int32_t current_depth);

}  // namespace yggdrasil_decision_forests::model::decision_tree

#endif  // YGGDRASIL_DECISION_FORESTS_LEARNER_DECISION_TREE_OBLIQUE_CPU_DEPTHWISE_1PASS_H_
