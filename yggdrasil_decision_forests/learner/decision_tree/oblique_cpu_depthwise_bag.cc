#include "yggdrasil_decision_forests/learner/decision_tree/oblique_cpu_depthwise_bag.h"

#if defined(SYMMETRIC_DEPTHWISE_AP) || defined(DEPTHWISE_1_PASS)

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <vector>

#include "absl/log/check.h"
#include "absl/types/span.h"
#include "hwy/base.h"
#include "hwy/contrib/sort/vqsort.h"
#include "yggdrasil_decision_forests/dataset/types.h"
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
    DepthBagState* s) {
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

void AdvanceDepthBag(
    absl::Span<const absl::Span<const UnsignedExampleIdx>>
        selected_examples_per_node,
    absl::Span<const int32_t> prev_first_child, const size_t bag_size,
    const DepthBagChrono billing, DepthBagState* state) {
  DCHECK(state != nullptr);
  DCHECK_GT(bag_size, 0u);
  const size_t N = selected_examples_per_node.size();

  // Resolve the two chrono sub-scope ids for this caller. The FuncId enum
  // only exists under CHRONO_PROFILE; without -DFINE_CHRONO_AP the
  // CHRONO_SCOPE_AP macro is inert, so the ids are [[maybe_unused]].
#ifdef CHRONO_PROFILE
  [[maybe_unused]] const ::yggdrasil_decision_forests::chrono_prof::FuncId
      build_id = billing == DepthBagChrono::kSymmetric
                     ? ::yggdrasil_decision_forests::chrono_prof::kSymBuildBag
                     : ::yggdrasil_decision_forests::chrono_prof::kDw1SharedBag;
  [[maybe_unused]] const ::yggdrasil_decision_forests::chrono_prof::FuncId
      sort_id = billing == DepthBagChrono::kSymmetric
                    ? ::yggdrasil_decision_forests::chrono_prof::kSymSortBag
                    : ::yggdrasil_decision_forests::chrono_prof::kDw1SharedBag;
#endif

  // ── Fast path: incremental relabel of the previous depth's sorted bag ──
  // Billed to sort_id (whose per-depth column it replaces). Self-validating;
  // any failure drops through to the fallback below.
  bool have_bag = false;
  if (state->valid && !prev_first_child.empty()) {
    CHRONO_SCOPE_AP(sort_id);
    have_bag = RelabelBagForNewDepth(selected_examples_per_node,
                                     prev_first_child, bag_size, state);
  }

  // ── Fallback: rebuild from the node spans ──────────────────────────────
  // N == 1 is a straight copy of the (already sorted) single span. N > 1
  // concats the spans + node ids (build_id), packs each (example_idx, node_id)
  // pair into hwy::K32V32 (key=ex, value=node) and VQSorts by key ascending
  // (sort_id). VQSort is unstable, but tie-breaking is invariant here: by the
  // BFS depth-cohort property every copy of one example_idx lives in the same
  // node, so all ties carry the same value field -- reordering within a tie
  // group is a no-op.
  if (!have_bag) {
    if (N == 1) {
      // Copy the span into the state: the rolling buffer is repartitioned in
      // place by this depth's SplitExamplesInPlace, so the span's contents
      // will not survive to seed the next depth's relabel.
      CHRONO_SCOPE_AP(build_id);
      state->bag.assign(selected_examples_per_node[0].begin(),
                        selected_examples_per_node[0].end());
      state->node_of_bag.clear();  // implicit: all entries node 0
    } else {
      auto& bag = state->bag;
      auto& node_of_bag = state->node_of_bag;
      {
        CHRONO_SCOPE_AP(build_id);
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
        CHRONO_SCOPE_AP(sort_id);
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
  state->valid = true;
}

}  // namespace yggdrasil_decision_forests::model::decision_tree

#endif  // defined(SYMMETRIC_DEPTHWISE_AP) || defined(DEPTHWISE_1_PASS)
