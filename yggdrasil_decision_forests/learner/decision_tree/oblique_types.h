// Shared type definitions for oblique split-finding.
// Extracted to avoid circular includes between oblique.h and oblique_gpu.h.

#ifndef YGGDRASIL_DECISION_FORESTS_LEARNER_DECISION_TREE_OBLIQUE_TYPES_H_
#define YGGDRASIL_DECISION_FORESTS_LEARNER_DECISION_TREE_OBLIQUE_TYPES_H_

#include <vector>

namespace yggdrasil_decision_forests::model::decision_tree {

// Per-node winning split descriptor returned by the full-GPU split path
// (ObliqueGpuComputer::FindBestSplitNodewise / FindBestSplitDepthwise). All
// fields are set if a valid split was found; best_gain < 0 signals
// "no split found" (e.g. all examples fell on one side).
struct BestSplitResult {
  int   best_proj_idx = -1;              // index into the node's projection span
  int   best_bin_idx = -1;               // right-side bin index
  float best_gain = -1.0f;                // Gini / entropy gain
  float best_threshold = 0.0f;           // materialized split threshold
  int   num_pos_training_examples = 0;   // right-side example count
};

namespace internal {

// A projection is defined as \sum features[projection[i].index] *
// projection[i].weight;
struct AttributeAndWeight {
  int attribute_idx;
  float weight;
};
typedef std::vector<AttributeAndWeight> Projection;

}  // namespace internal
}  // namespace yggdrasil_decision_forests::model::decision_tree

#endif  // YGGDRASIL_DECISION_FORESTS_LEARNER_DECISION_TREE_OBLIQUE_TYPES_H_
