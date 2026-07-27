// Depthwise fused-per-level CPU ApplyProjection: buckets the level's refs by
// column, sweeps each touched column once. Excludes SYMMETRIC_DEPTHWISE_AP.
// Output (as symmetric): slab[p*rows_n + i] = <proj[n][p], features[sel[i]]>.

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

// Fused-per-level Apply, single-threaded (RF already trains one tree/thread).
// The frontier is the depth's HOT nodes only; the driver pre-advanced
// `bag_state` over those rows, labelled by hot index. `current_depth`: debug.
absl::Status ApplyProjectionsDepthwise1Pass(
    const dataset::VerticalDataset& train_dataset,
    const google::protobuf::RepeatedField<int32_t>& numerical_features,
    absl::Span<const absl::Span<const UnsignedExampleIdx>>
        selected_examples_per_node,
    absl::Span<const std::vector<internal::Projection>> projections_per_node,
    DepthBagState* bag_state, absl::Span<std::vector<float>> out_projected,
    int32_t current_depth);

}  // namespace yggdrasil_decision_forests::model::decision_tree

#endif  // YGGDRASIL_DECISION_FORESTS_LEARNER_DECISION_TREE_OBLIQUE_CPU_DEPTHWISE_1PASS_H_
