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
// Incremental sorted bag: the depth-d sorted bag never needs a sort in the
// steady state — the previous depth's sorted (example, node) sequence is
// relabelled parent -> child in one O(bag) comparison-free pass. The state
// object and that pass now live in the shared depth-bag module
// (oblique_cpu_depthwise_bag.h / .cc), where the mechanism is documented in
// full; SymmetricBagState below is an alias of that module's DepthBagState.
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
#include "yggdrasil_decision_forests/learner/decision_tree/oblique_cpu_depthwise_bag.h"
#include "yggdrasil_decision_forests/learner/decision_tree/oblique_types.h"

namespace yggdrasil_decision_forests::model::decision_tree {

// The cross-depth sorted-bag state is now shared with the DW1 shared-rows
// kernel; see oblique_cpu_depthwise_bag.h. Kept as an alias so existing call
// sites (this kernel and its BFS driver) read unchanged.
using SymmetricBagState = DepthBagState;

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
