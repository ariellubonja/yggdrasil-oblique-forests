// Shared type definitions for oblique split-finding.
// Extracted to keep CUDA TUs (oblique_gpu_kernels.cu.cc, randomprojection.cu)
// free of the absl/protobuf-heavy transitive includes pulled in by oblique.h.

#ifndef YGGDRASIL_DECISION_FORESTS_LEARNER_DECISION_TREE_OBLIQUE_TYPES_H_
#define YGGDRASIL_DECISION_FORESTS_LEARNER_DECISION_TREE_OBLIQUE_TYPES_H_

#include <cstdint>
#include <vector>

namespace yggdrasil_decision_forests::model::decision_tree {

// Subtree-scoped compact feature cache for sparse-oblique training
// (--config=subtree_gather). When a node's example count first drops to
// RowMajorMaxRows() (YDF_RM_MAX_ROWS, default 5000) or below, that node's
// example list becomes the current "block". Feature columns are gathered
// lazily — on the first projection that touches them — into dense
// block-local arrays; descendant nodes reuse the gathered columns through
// the example->slot map. This recovers the deep-node locality win of the
// Dynamic_Row_Col_Major dual store without a second full copy of the
// dataset: live memory is at most YDF_SG_BUDGET_MB per thread.
//
// Validity is epoch-tagged and checked per node, so any traversal order is
// correct; DFS is what makes it fast (one block serves a whole subtree
// before the next one is materialized).
struct SubtreeGatherCache {
  static constexpr uint32_t kSlotBits = 16;
  static constexpr uint32_t kSlotMask = (1u << kSlotBits) - 1;
  static constexpr uint32_t kMaxEpoch = (1u << 16) - 1;

  // example_idx -> (epoch << kSlotBits | slot). An entry belongs to the
  // current block iff its epoch tag matches `epoch`. Sized to the dataset's
  // nrow on first use.
  std::vector<uint32_t> slot_of_example;
  // Current block id in [1, kMaxEpoch]; 0 = no block. When the tag space
  // wraps, slot_of_example is cleared.
  uint32_t epoch = 0;
  // Examples of the current block, in slot order. May contain duplicates
  // under bootstrap sampling; duplicate rows gather identical values, so
  // which slot the map retains is irrelevant.
  std::vector<uint32_t> block_rows;
  // Per-attribute gathered column (block_rows.size() values each) and the
  // epoch it was gathered in. Indexed by raw dataspec attribute index.
  std::vector<std::vector<float>> cols;
  std::vector<uint32_t> col_epoch;
  // Per-node scratch: slot of each selected example of the node being split.
  std::vector<uint32_t> node_slots;
  // Bytes gathered in the current epoch (budget gate) and bytes retained
  // across epochs as vector capacity (sweep gate).
  uint64_t gathered_bytes = 0;
  uint64_t retained_bytes = 0;
};

// Per-node winning split descriptor returned by the full-GPU split path
// (ObliqueGpuComputer::FindBestSplitNodewise / FindBestSplitDepthwise). All
// fields are set if a valid split was found; best_gain < 0 signals
// "no split found" (e.g. all examples fell on one side).
struct BestSplitResult {
  int best_proj_idx = -1;             // index into the node's projection span
  int best_bin_idx = -1;              // right-side bin index
  float best_gain = -1.0f;            // Gini / entropy gain
  float best_threshold = 0.0f;        // materialized split threshold
  int num_pos_training_examples = 0;  // right-side example count
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
