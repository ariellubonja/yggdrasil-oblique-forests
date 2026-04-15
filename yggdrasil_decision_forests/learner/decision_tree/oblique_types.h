// Shared type definitions for oblique split-finding.
// Extracted to avoid circular includes between oblique.h and oblique_gpu.h.

#ifndef YGGDRASIL_DECISION_FORESTS_LEARNER_DECISION_TREE_OBLIQUE_TYPES_H_
#define YGGDRASIL_DECISION_FORESTS_LEARNER_DECISION_TREE_OBLIQUE_TYPES_H_

#include <vector>

namespace yggdrasil_decision_forests::model::decision_tree::internal {

// A projection is defined as \sum features[projection[i].index] *
// projection[i].weight;
struct AttributeAndWeight {
  int attribute_idx;
  float weight;
};
typedef std::vector<AttributeAndWeight> Projection;

}  // namespace yggdrasil_decision_forests::model::decision_tree::internal

#endif  // YGGDRASIL_DECISION_FORESTS_LEARNER_DECISION_TREE_OBLIQUE_TYPES_H_
