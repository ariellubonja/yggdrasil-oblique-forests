// Shared depth-bag machinery for the fused-per-level CPU ApplyProjection
// kernels (symmetric-trees `SYMMETRIC_DEPTHWISE_AP` and the DW1 shared-rows
// colwalk, i.e. `DEPTHWISE_1_PASS` without `DW1_COLWALK_CONTROL`; the colwalk
// control has no bag). Both kernels need the same object: the current BFS
// depth's example-sorted bag together with, per bag entry, the owning
// depth-batch node index. This module owns that state (DepthBagState) and the
// single routine that advances it one depth (AdvanceDepthBag).
//
// Incremental sorted bag: the depth-d sorted bag never needs a sort in the
// steady state. SplitExamplesInPlace stably partitions each parent's sorted
// span, so merging all of depth d's node spans reproduces depth d-1's sorted
// bag minus the rows of nodes that leafed out -- i.e. the sorted row order is
// the (already sorted) bootstrap bag filtered to surviving rows, at every
// depth. AdvanceDepthBag therefore keeps the previous depth's sorted
// (example, node) sequence in DepthBagState and derives this depth's bag with
// one O(bag) comparison-free relabel pass (RelabelBagForNewDepth in the .cc):
// per surviving entry, the owning node advances parent -> child, where the
// child is identified by a single equality test against the negative child's
// next-unconsumed span element (every bootstrap copy of a row takes the same
// side of the split, so a row id lives in exactly one child span). This
// replaces the per-depth concat + VQSort that was ~half of the fused-kernel
// ApplyProjection time on HIGGS. The concat + K32V32 VQSort remains as the
// fallback whenever the state or the parent->child mapping is unavailable or
// fails the pass's self-validation.

#ifndef YGGDRASIL_DECISION_FORESTS_LEARNER_DECISION_TREE_OBLIQUE_CPU_DEPTHWISE_BAG_H_
#define YGGDRASIL_DECISION_FORESTS_LEARNER_DECISION_TREE_OBLIQUE_CPU_DEPTHWISE_BAG_H_

#include <cstddef>
#include <cstdint>
#include <vector>

#include "absl/types/span.h"
#include "yggdrasil_decision_forests/dataset/types.h"

namespace yggdrasil_decision_forests::model::decision_tree {

// Cross-depth state for the incremental sorted bag (see the file header).
// Owned by the BFS driver (one instance per tree; buffers are typically
// thread_local at the call site so their capacity amortizes across trees on
// the same pool thread). All vectors are managed by AdvanceDepthBag; the
// driver only resets `valid` at tree start.
struct DepthBagState {
  // The sorted-by-example bag of the depth most recently processed, and the
  // owning node's depth-batch index per entry. `node_of_bag` is always sized
  // to `bag` (all zeros for a single-node depth), so consumers read it
  // unconditionally.
  std::vector<UnsignedExampleIdx> bag;
  std::vector<uint32_t> node_of_bag;
  // Ping-pong scratch swapped with bag/node_of_bag by the relabel pass.
  std::vector<UnsignedExampleIdx> bag_scratch;
  std::vector<uint32_t> node_scratch;
  // Per-new-node span-consumption cursors for the relabel membership test.
  std::vector<size_t> cursor;
  // True iff bag/node_of_bag describe the previous call's depth batch.
  bool valid = false;
};

// Per-variant chrono billing selector for AdvanceDepthBag's two internal
// phases. Defined unconditionally: the chrono FuncId enum only exists under
// CHRONO_PROFILE, so the kernels cannot name the kSym*/kDw1* ids directly in
// non-profiling builds. AdvanceDepthBag maps this tag to the right FuncIds
// internally, preserving today's per-variant attribution.
enum class DepthBagChrono {
  // Symmetric: build (concat / root-span copy) -> kSymBuildBag,
  //            relabel + VQSort fallback       -> kSymSortBag.
  kSymmetric,
  // DW1 shared-rows: build and relabel/sort both -> kDw1SharedBag.
  kDw1SharedRows,
};

// Advances *state to the depth described by `selected_examples_per_node` (the
// per-node example partitions, each individually sorted ascending). On return
// state->bag / state->node_of_bag hold this depth's example-sorted bag and
// per-entry owning node index, and state->valid is true. `bag_size` must equal
// the sum of the spans' sizes and be > 0 (callers handle the empty-bag case).
//
// Fast path: O(bag) relabel of the previous depth's bag when state->valid and
// prev_first_child is non-empty (see RelabelBagForNewDepth + the file header).
// Fallback: N == 1 span copy, else concat + K32V32 pack + VQSort by example id.
//
// `prev_first_child` maps the previous depth batch to this one: entry p is the
// batch index of node p's negative child (positive child = value + 1), or -1
// if node p became a leaf. Children are pushed (neg, pos) per split parent in
// parent order, so indices are consecutive pairs. Pass an empty span to force
// the fallback (first depth, or driver-side validation failed).
//
// MUST be called from inside a kernel's kProjectionEvaluate scope (never the
// driver) so bag time stays inside ApplyProjection and the invariant
// TreeTrain = NodeTrain + ApplyProjection + SampleProjection holds.
void AdvanceDepthBag(
    absl::Span<const absl::Span<const UnsignedExampleIdx>>
        selected_examples_per_node,
    absl::Span<const int32_t> prev_first_child, size_t bag_size,
    DepthBagChrono billing, DepthBagState* state);

#if defined(DEPTHWISE_1_PASS) && !defined(DW1_COLWALK_CONTROL)
// Hot-nodes variant of AdvanceDepthBag (see .bazelrc:depthwise_1_pass): the bag
// covers only the depth's HOT nodes, so it is smaller than the depth's example
// set and its labels index the HOT arrays (0..K-1), which is what the kernel
// reads. Called by the BFS driver (not by the kernel) because only the driver
// holds the full-domain node spans the relabel walks.
//
// The parent -> child hop still runs in the FULL node domain -- a hot parent's
// rows may land in a cold child, and the membership test consumes the child
// spans in order -- so the pass takes the full spans plus two index maps:
//   `prev_hot_to_full` : previous depth's hot index -> previous batch index,
//                        i.e. the inverse of the labels stored in state->bag.
//   `hot_of_node`      : this depth's batch index -> hot index, or -1 if cold.
// A row whose new owner is cold is dropped from the bag, exactly like a row
// whose parent leafed out. This is consistent because the gate is monotone
// (child rows <= parent rows => a hot node's parent is hot), so every hot node's
// rows are present in the previous hot bag and the relabel keeps telescoping.
//
// `hot_selected_examples_per_node` is the hot subsequence of the full spans (in
// hot-index order) and is used only by the concat+VQSort fallback.
// `hot_bag_size` must equal the sum of those spans' sizes and be > 0.
//
// Same billing rule as AdvanceDepthBag, one level up: the driver must wrap this
// call in a kProjectionEvaluate scope (the kernel no longer advances the bag in
// this build) so bag time stays inside ApplyProjection.
void AdvanceDepthBagHot(
    absl::Span<const absl::Span<const UnsignedExampleIdx>>
        selected_examples_per_node,
    absl::Span<const absl::Span<const UnsignedExampleIdx>>
        hot_selected_examples_per_node,
    absl::Span<const int32_t> prev_first_child,
    absl::Span<const uint32_t> prev_hot_to_full,
    absl::Span<const int32_t> hot_of_node, size_t hot_bag_size,
    DepthBagState* state);
#endif  // DEPTHWISE_1_PASS && !DW1_COLWALK_CONTROL

}  // namespace yggdrasil_decision_forests::model::decision_tree

#endif  // YGGDRASIL_DECISION_FORESTS_LEARNER_DECISION_TREE_OBLIQUE_CPU_DEPTHWISE_BAG_H_
