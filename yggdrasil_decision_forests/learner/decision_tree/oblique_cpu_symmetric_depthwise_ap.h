// CatBoost-style symmetric-trees ApplyProjection kernel for Sparse Oblique
// Random Forests.
//
// Invariant exploited: at depth d in a BFS tree-growing schedule, all 2^d
// nodes share the same K candidate projections (drawn once per depth, not
// per node). The union of nodes' selected_examples at depth d == the tree's
// bag.
//
// Per-projection cost is then converted from K * sum-over-nodes(rows_n)
// scattered per-node gathers of column data to K bag-wide stride-1 sweeps:
// for each projection k, walk the bag in example-sorted order, compute
// projection k once per example, and route each result to the owning node's
// per-node slab via a write cursor.
//
// Incremental sorted bag (SymmetricBagState): the depth-d sorted bag never
// needs a sort. SplitExamplesInPlace stably partitions each parent's sorted
// span, so merging all of depth d's node spans reproduces depth d-1's sorted
// bag minus the rows of nodes that leafed out — i.e. the sorted row order is
// the (already sorted) bootstrap bag filtered to surviving rows, at every
// depth. The kernel therefore keeps the previous depth's sorted
// (example, node) sequence in `SymmetricBagState` and derives this depth's
// bag with one O(bag) comparison-free relabel pass: per surviving entry, the
// owning node advances parent -> child, where the child is identified by a
// single equality test against the negative child's next-unconsumed span
// element (every bootstrap copy of a row takes the same side, so a row id
// lives in exactly one child span). This replaces the per-depth concat +
// VQSort (was ~half of symmetric ApplyProjection time on HIGGS). The full
// concat+sort remains as the fallback whenever the state or the
// parent->child mapping is unavailable or fails validation.
//
// Output contract: for each node n, out_projected[n] holds a (K * rows_n)-
// float slab where slab[k * rows_n + i] = projection k applied to node n's
// i-th selected example. IMPORTANT: the slab lives in *reserved capacity*
// only — the kernel writes it through raw pointers and skips the (fully
// overwritten) zero-fill, so out_projected[n].size() stays 0. The caller
// must therefore build the consuming span from out_projected[n].data() with
// explicit length K * rows_n, not from the vector's size. The consumer is
// FindBestConditionSparseObliqueTemplate in oblique.cc (gated by
// SYMMETRIC_DEPTHWISE_AP).

#ifndef YGGDRASIL_DECISION_FORESTS_LEARNER_DECISION_TREE_OBLIQUE_CPU_SYMMETRIC_DEPTHWISE_AP_H_
#define YGGDRASIL_DECISION_FORESTS_LEARNER_DECISION_TREE_OBLIQUE_CPU_SYMMETRIC_DEPTHWISE_AP_H_

#include <cstdint>
#include <vector>

#include "absl/status/status.h"
#include "absl/types/span.h"
#include "google/protobuf/repeated_field.h"
#include "yggdrasil_decision_forests/dataset/types.h"
#include "yggdrasil_decision_forests/dataset/vertical_dataset.h"
#include "yggdrasil_decision_forests/learner/decision_tree/oblique_types.h"

namespace yggdrasil_decision_forests::model::decision_tree {

// Cross-depth state for the incremental sorted bag (see the file header).
// Owned by the BFS driver (one instance per tree; buffers are typically
// thread_local at the call site so their capacity amortizes across trees on
// the same pool thread). All vectors are managed by the kernel; the driver
// only resets `valid` at tree start.
struct SymmetricBagState {
  // The sorted-by-example bag of the depth most recently processed, and the
  // owning node's depth-batch index per entry. `node_of_bag` may be empty
  // when the depth had a single node (all entries implicitly node 0).
  std::vector<UnsignedExampleIdx> bag;
  std::vector<uint32_t> node_of_bag;
  // Ping-pong scratch swapped with bag/node_of_bag by the relabel pass.
  std::vector<UnsignedExampleIdx> bag_scratch;
  std::vector<uint32_t> node_scratch;
  // Per-new-node span-consumption cursors for the relabel membership test.
  std::vector<size_t> cursor;
  // True iff bag/node_of_bag describe the previous kernel call's depth batch.
  bool valid = false;
};

// Symmetric bag-wide kernel. `selected_examples_per_node` are the per-node
// example partitions (each individually sorted ascending). `shared_projections`
// is the K-element vector of projections shared across all nodes at the
// current depth. `out_projected[n]` gets K * rows_n[n] floats of *reserved*
// capacity filled with projection values (its .size() stays 0 — see the
// output-contract note above).
//
// `prev_first_child` maps the *previous* depth batch to this one: entry p is
// the batch index of node p's negative child (positive child = value + 1),
// or -1 if node p became a leaf. Children are pushed (neg, pos) per split
// parent in parent order, so indices are consecutive pairs. Pass an empty
// span when the mapping is unknown (first depth, or driver-side validation
// failed); the kernel then rebuilds the bag with the concat+sort fallback.
// `bag_state` must be non-null; the kernel reads the previous depth's bag
// from it and leaves this depth's bag in it.
absl::Status ApplyProjectionsSymmetricDepthwiseAP(
    const dataset::VerticalDataset& train_dataset,
    const google::protobuf::RepeatedField<int32_t>& numerical_features,
    absl::Span<const absl::Span<const UnsignedExampleIdx>>
        selected_examples_per_node,
    absl::Span<const internal::Projection> shared_projections,
    absl::Span<const int32_t> prev_first_child,
    SymmetricBagState* bag_state,
    absl::Span<std::vector<float>> out_projected);

}  // namespace yggdrasil_decision_forests::model::decision_tree

#endif  // YGGDRASIL_DECISION_FORESTS_LEARNER_DECISION_TREE_OBLIQUE_CPU_SYMMETRIC_DEPTHWISE_AP_H_
