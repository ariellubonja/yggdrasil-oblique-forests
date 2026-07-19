#include "yggdrasil_decision_forests/learner/decision_tree/oblique_cpu_symmetric_depthwise_ap.h"

#ifdef SYMMETRIC_DEPTHWISE_AP

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <vector>

#include "absl/log/check.h"
#include "absl/status/status.h"
#include "absl/types/span.h"
#include "hwy/base.h"
#include "hwy/contrib/sort/vqsort.h"
#include "yggdrasil_decision_forests/learner/decision_tree/oblique.h"
#include "yggdrasil_decision_forests/utils/parallel_chrono.h"

namespace yggdrasil_decision_forests::model::decision_tree {

namespace {

// Derives the new depth's sorted (example, node) bag from the previous
// depth's, in one streaming pass and zero comparisons-for-order (see the
// header): each surviving entry keeps its position in example-sorted order
// (a stable partition of a sorted list is sorted, so a depth's bag is the
// previous depth's bag minus leafed-out rows); only the owning-node label
// advances parent -> child. The child is identified by one equality test
// against the negative child's next-unconsumed span element: a row id lives
// in exactly one child span (all bootstrap copies of a row take the same
// side of the split), and each child span is the sorted subsequence of the
// parent's entries, so it is consumed strictly in order.
//
// Self-validating: every entry's claimed child span element is checked, and
// any mismatch (as well as count mismatches) returns false, in which case
// the caller falls back to the full concat+sort rebuild. Writes the result
// into the scratch buffers and swaps them in on success.
bool RelabelBagForNewDepth(
    absl::Span<const absl::Span<const UnsignedExampleIdx>>
        selected_examples_per_node,
    absl::Span<const int32_t> prev_first_child, const size_t new_bag_size,
    SymmetricBagState* s) {
  const size_t N = selected_examples_per_node.size();
  const size_t old_size = s->bag.size();
  s->bag_scratch.resize(new_bag_size);
  s->node_scratch.resize(new_bag_size);
  s->cursor.assign(N, 0);

  const UnsignedExampleIdx* old_bag = s->bag.data();
  // Empty node_of_bag == the previous depth had a single node (batch idx 0).
  const uint32_t* old_node =
      s->node_of_bag.empty() ? nullptr : s->node_of_bag.data();
  const uint32_t prev_n = static_cast<uint32_t>(prev_first_child.size());

  size_t out = 0;
  for (size_t i = 0; i < old_size; ++i) {
    const uint32_t p = old_node != nullptr ? old_node[i] : 0u;
    if (p >= prev_n) return false;
    const int32_t c = prev_first_child[p];
    if (c < 0) continue;  // Parent became a leaf: the row leaves the bag.
    const UnsignedExampleIdx ex = old_bag[i];

    uint32_t nc = static_cast<uint32_t>(c);
    const auto neg = selected_examples_per_node[c];
    if (!(s->cursor[c] < neg.size() && neg[s->cursor[c]] == ex)) nc = c + 1;
    if (nc >= N) return false;
    size_t& cur = s->cursor[nc];
    const auto child = selected_examples_per_node[nc];
    if (cur >= child.size() || child[cur] != ex) return false;
    ++cur;

    if (out >= new_bag_size) return false;
    s->bag_scratch[out] = ex;
    s->node_scratch[out] = nc;
    ++out;
  }
  if (out != new_bag_size) return false;

  s->bag.swap(s->bag_scratch);
  s->node_of_bag.swap(s->node_scratch);
  DCHECK(std::is_sorted(s->bag.begin(), s->bag.end()));
  return true;
}

}  // namespace

absl::Status ApplyProjectionsSymmetricDepthwiseAP(
    const dataset::VerticalDataset& train_dataset,
    const google::protobuf::RepeatedField<int32_t>& numerical_features,
    absl::Span<const absl::Span<const UnsignedExampleIdx>>
        selected_examples_per_node,
    absl::Span<const internal::Projection> shared_projections,
    absl::Span<const int32_t> prev_first_child,
    SymmetricBagState* bag_state,
    absl::Span<std::vector<float>> out_projected) {
  // Outer scope: total ProjectionEvaluate budget. Sub-phase scopes below
  // partition this into BuildBag / SortBag / Sweep so parallel_chrono.py
  // can break ApplyProjection down per depth.
  CHRONO_SCOPE_COARSE(::yggdrasil_decision_forests::chrono_prof::kProjectionEvaluate);

  DCHECK(bag_state != nullptr);
  const size_t N = selected_examples_per_node.size();
  DCHECK_EQ(N, out_projected.size());
  if (N == 0) return absl::OkStatus();

  const size_t K = shared_projections.size();
  if (K == 0) {
    for (size_t n = 0; n < N; ++n) out_projected[n].clear();
    // The bag for this depth batch was never materialized, so the state
    // cannot seed the next depth's relabel.
    bag_state->valid = false;
    return absl::OkStatus();
  }

  // ── Phase 1: BuildBag ─────────────────────────────────────────────
  // Per-node row counts + slab storage.
  //
  // Slab storage is *reserved*, not zero-filled. The Sweep below writes
  // every cell of every non-empty projection's slab exactly once (each bag
  // entry routes to one cell via the per-node write cursor), so the old
  // out_projected[n].assign(K * rows_n, 0.f) was 100 % overwritten — 264 MB
  // of dead writes per depth on HIGGS, 384 MB on trunk-4096. We reserve
  // capacity only (no value-init) and write through raw pointers. As a
  // consequence out_projected[n].size() stays 0; the consumer builds its
  // span from .data() with explicit length K * rows_n (see training.cc,
  // and the output-contract note in the header).
  const bool single_node = (N == 1);
  std::vector<size_t> rows_n(N);
  std::vector<float*> out_ptr(N, nullptr);
  size_t bag_size = 0;
  {
    CHRONO_SCOPE(::yggdrasil_decision_forests::chrono_prof::kSymBuildBag);
    for (size_t n = 0; n < N; ++n) {
      rows_n[n] = selected_examples_per_node[n].size();
      bag_size += rows_n[n];
      out_projected[n].reserve(K * rows_n[n]);  // capacity only, no zero-fill
      out_ptr[n] = out_projected[n].data();
    }
    if (bag_size == 0) {
      bag_state->valid = false;
      return absl::OkStatus();
    }
  }

  // ── Phase 2: obtain the sorted bag ────────────────────────────────
  // Fast path (billed to kSymSortBag, whose per-depth column it replaces):
  // relabel the previous depth's sorted bag in one O(bag) streaming pass —
  // no concat, no sort (see RelabelBagForNewDepth and the header note).
  //
  // Fallback (first depth of a tree, missing parent->child mapping, or a
  // failed relabel validation): rebuild from the node spans. N == 1 is a
  // straight copy of the (already sorted) single span; N > 1 concats the
  // spans + node ids (kSymBuildBag), packs each (example_idx, node_id) pair
  // into hwy::K32V32 (key=ex, value=node) and VQSorts by key ascending
  // (kSymSortBag). VQSort is unstable, but tie-breaking is invariant here:
  // by the BFS depth-cohort property every copy of one example_idx lives in
  // the same node, so all ties carry the same value field — reordering
  // within a tie group is a no-op. (History: stable_sort → counting sort
  // (~3.5× on HIGGS but O(total_n) walk lost at deep depths) → VQSort,
  // which beat both across the board — and is now the cold path only.)
  bool have_bag = false;
  if (bag_state->valid && !prev_first_child.empty()) {
    CHRONO_SCOPE(::yggdrasil_decision_forests::chrono_prof::kSymSortBag);
    have_bag = RelabelBagForNewDepth(selected_examples_per_node,
                                     prev_first_child, bag_size, bag_state);
  }
  if (!have_bag) {
    if (single_node) {
      // Copy the span into the state: the rolling buffer is repartitioned in
      // place by this depth's SplitExamplesInPlace, so the span's contents
      // will not survive to seed the next depth's relabel. One bag-sized
      // copy per tree (root level), vs. a concat+VQSort per depth saved.
      CHRONO_SCOPE(::yggdrasil_decision_forests::chrono_prof::kSymBuildBag);
      bag_state->bag.assign(selected_examples_per_node[0].begin(),
                            selected_examples_per_node[0].end());
      bag_state->node_of_bag.clear();  // implicit: all entries node 0
    } else {
      auto& bag = bag_state->bag;
      auto& node_of_bag = bag_state->node_of_bag;
      {
        CHRONO_SCOPE(::yggdrasil_decision_forests::chrono_prof::kSymBuildBag);
        bag.resize(bag_size);
        node_of_bag.resize(bag_size);
        size_t cur = 0;
        for (size_t n = 0; n < N; ++n) {
          for (const UnsignedExampleIdx e : selected_examples_per_node[n]) {
            bag[cur] = e;
            node_of_bag[cur] = static_cast<uint32_t>(n);
            ++cur;
          }
        }
      }
      {
        CHRONO_SCOPE(::yggdrasil_decision_forests::chrono_prof::kSymSortBag);
        static_assert(sizeof(UnsignedExampleIdx) == sizeof(uint32_t),
                      "K32V32 path assumes 32-bit example_idx");

        thread_local std::vector<hwy::K32V32> hwy_buf;
        hwy_buf.resize(bag_size);

        for (size_t i = 0; i < bag_size; ++i) {
          hwy_buf[i].key = static_cast<uint32_t>(bag[i]);
          hwy_buf[i].value = node_of_bag[i];
        }

        hwy::VQSort(hwy_buf.data(), bag_size, hwy::SortAscending());

        for (size_t i = 0; i < bag_size; ++i) {
          bag[i] = static_cast<UnsignedExampleIdx>(hwy_buf[i].key);
          node_of_bag[i] = hwy_buf[i].value;
        }
      }
    }
  }
  bag_state->valid = true;
  const UnsignedExampleIdx* bag_data = bag_state->bag.data();

  // ── Phase 3: Sweep ────────────────────────────────────────────────
  // K bag-wide sweeps. For each projection k:
  //  - Hoist per-item col pointers + weights + NA values out of the i-loop.
  //  - Walk the (sorted) bag in stride-1 order. Per example: compute value,
  //    route to its owning node's slab (written through the reserved-only
  //    raw pointer out_ptr[n], never operator[] — the slab .size() is 0).
  // The N == 1 case writes sequentially (bag order == node 0's slab order)
  // with no node_of_bag / write_cursor indirection; the N > 1 case routes
  // each result via the per-node write cursor. The single_node test is
  // hoisted out of the hot i-loop.
  {
    CHRONO_SCOPE(::yggdrasil_decision_forests::chrono_prof::kSymSweep);
    internal::ProjectionEvaluator evaluator(train_dataset, numerical_features);

    // Filled for every N > 1 path (concat rebuild and relabel alike); only
    // the single-node sweep below skips node routing entirely.
    const uint32_t* node_of_bag = bag_state->node_of_bag.data();

    std::vector<uint32_t> write_cursor;
    if (!single_node) write_cursor.assign(N, 0u);

    for (size_t k = 0; k < K; ++k) {
      const auto& proj = shared_projections[k];
      const size_t M = proj.size();
      // SampleProjection guarantees nnz >= 1; an empty projection here would
      // leave this k's slab region uninitialized, but the finder skips empty
      // projections (oblique.cc), so it is never read. Defensive.
      if (M == 0) continue;

      std::vector<const float*> col_ptrs(M);
      std::vector<float> ws(M);
#ifdef ENABLE_ISNAN
      std::vector<float> nas(M);
#endif
      for (size_t m = 0; m < M; ++m) {
        col_ptrs[m] = evaluator.AttributeData(proj[m].attribute_idx);
        ws[m] = proj[m].weight;
#ifdef ENABLE_ISNAN
        nas[m] = evaluator.NaReplacementValue(proj[m].attribute_idx);
#endif
      }

      if (single_node) {
        float* out = out_ptr[0] + k * rows_n[0];
        for (size_t i = 0; i < bag_size; ++i) {
          const UnsignedExampleIdx ex = bag_data[i];
          float value = 0.f;
          for (size_t m = 0; m < M; ++m) {
            float v = col_ptrs[m] != nullptr
                          ? col_ptrs[m][ex]
                          : evaluator.AttributeValue(proj[m].attribute_idx, ex);
#ifdef ENABLE_ISNAN
            if (std::isnan(v)) v = nas[m];
#endif
            value += ws[m] * v;
          }
          out[i] = value;
        }
      } else {
        std::fill(write_cursor.begin(), write_cursor.end(), 0u);
        for (size_t i = 0; i < bag_size; ++i) {
          const UnsignedExampleIdx ex = bag_data[i];
          float value = 0.f;
          for (size_t m = 0; m < M; ++m) {
            float v = col_ptrs[m] != nullptr
                          ? col_ptrs[m][ex]
                          : evaluator.AttributeValue(proj[m].attribute_idx, ex);
#ifdef ENABLE_ISNAN
            if (std::isnan(v)) v = nas[m];
#endif
            value += ws[m] * v;
          }
          const uint32_t n = node_of_bag[i];
          const uint32_t pos = write_cursor[n]++;
          out_ptr[n][k * rows_n[n] + pos] = value;
        }
      }
    }
  }

  return absl::OkStatus();
}

}  // namespace yggdrasil_decision_forests::model::decision_tree

#endif  // SYMMETRIC_DEPTHWISE_AP
