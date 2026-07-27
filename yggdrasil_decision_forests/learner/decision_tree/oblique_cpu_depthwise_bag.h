// Shared depth-bag machinery for the fused-per-level CPU AP kernels: the BFS
// depth's example-sorted bag + owning node index per entry. Spans stay sorted
// across splits, so an O(bag) relabel replaces the per-depth concat + VQSort.

#ifndef YGGDRASIL_DECISION_FORESTS_LEARNER_DECISION_TREE_OBLIQUE_CPU_DEPTHWISE_BAG_H_
#define YGGDRASIL_DECISION_FORESTS_LEARNER_DECISION_TREE_OBLIQUE_CPU_DEPTHWISE_BAG_H_

#include <cstddef>
#include <cstdint>
#include <vector>

#include "absl/types/span.h"
#include "yggdrasil_decision_forests/dataset/types.h"

namespace yggdrasil_decision_forests::model::decision_tree {

// Cross-depth state for the incremental sorted bag (see the file header). One
// per tree, buffers thread_local at the call site. AdvanceDepthBag manages
// every vector; the driver only resets `valid` at tree start.
struct DepthBagState {
  // Most recently processed depth's sorted-by-example bag + owning batch index
  // per entry. `node_of_bag` is always bag-sized (zeros for a single-node
  // depth), so consumers read it unconditionally.
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

// Per-variant chrono billing selector for AdvanceDepthBag's two phases. Defined
// unconditionally because the FuncId enum only exists under CHRONO_PROFILE, so
// kernels cannot name kSym*/kDw1* ids directly; AdvanceDepthBag maps the tag.
enum class DepthBagChrono {
  // Symmetric: build (concat / root-span copy) -> kSymBuildBag,
  //            relabel + VQSort fallback       -> kSymSortBag.
  kSymmetric,
  // DW1 shared-rows: build and relabel/sort both -> kDw1SharedBag.
  kDw1SharedRows,
};

// Advances *state to the depth given by `selected_examples_per_node` (each
// sorted ascending; `bag_size` = Σ sizes, > 0) via the O(bag) relabel, else span
// copy or concat + VQSort. Call inside a kernel's kProjectionEvaluate scope.
void AdvanceDepthBag(
    absl::Span<const absl::Span<const UnsignedExampleIdx>>
        selected_examples_per_node,
    // Previous batch index p -> batch index of p's negative child (positive =
    // +1), -1 if p leafed out. Empty forces the fallback.
    absl::Span<const int32_t> prev_first_child, size_t bag_size,
    DepthBagChrono billing, DepthBagState* state);

#if defined(DEPTHWISE_1_PASS) && !defined(DW1_COLWALK_CONTROL)
// Hot-nodes variant: bag covers only the depth's HOT nodes, labelled by hot
// index; driver-called (it holds the full spans), in a kProjectionEvaluate
// scope. Cold-child rows drop; monotone row gate keeps that telescoping.
void AdvanceDepthBagHot(
    // Full-domain spans: the parent -> child hop runs in the full node domain.
    absl::Span<const absl::Span<const UnsignedExampleIdx>>
        selected_examples_per_node,
    // Hot subsequence of the above; used only by the concat+VQSort fallback.
    absl::Span<const absl::Span<const UnsignedExampleIdx>>
        hot_selected_examples_per_node,
    absl::Span<const int32_t> prev_first_child,
    // Previous hot index -> previous batch index (inverse of state->bag labels).
    absl::Span<const uint32_t> prev_hot_to_full,
    // This depth's batch index -> hot index, or -1 if cold.
    absl::Span<const int32_t> hot_of_node, size_t hot_bag_size,
    DepthBagState* state);
#endif  // DEPTHWISE_1_PASS && !DW1_COLWALK_CONTROL

}  // namespace yggdrasil_decision_forests::model::decision_tree

#endif  // YGGDRASIL_DECISION_FORESTS_LEARNER_DECISION_TREE_OBLIQUE_CPU_DEPTHWISE_BAG_H_
