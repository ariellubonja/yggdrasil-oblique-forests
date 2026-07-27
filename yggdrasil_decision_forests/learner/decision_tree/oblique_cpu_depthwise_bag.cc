#include "yggdrasil_decision_forests/learner/decision_tree/oblique_cpu_depthwise_bag.h"

#if defined(SYMMETRIC_OPTIMIZED) || defined(DEPTHWISE_1_PASS)

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

// Derives the new depth's sorted (example, node) bag from the previous one in
// one pass, no order comparisons: entries keep their position, the node label
// hops parent -> child. Self-validating; false => caller rebuilds from spans.
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
  DCHECK_EQ(s->node_of_bag.size(), old_size);
  const uint32_t* old_node = s->node_of_bag.data();
  const uint32_t prev_n = static_cast<uint32_t>(prev_first_child.size());

  size_t out = 0;
  for (size_t i = 0; i < old_size; ++i) {
    const uint32_t p = old_node[i];
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

#if defined(DEPTHWISE_1_PASS) && !defined(DW1_COLWALK_CONTROL)
// Hot-nodes variant of RelabelBagForNewDepth: same pass, but the bag's labels
// are hot indices while the parent -> child hop and cursors stay in the full
// node domain. Two extra index lookups, plus a drop when the owner is cold.
bool RelabelBagForNewDepthHot(
    absl::Span<const absl::Span<const UnsignedExampleIdx>>
        selected_examples_per_node,
    absl::Span<const int32_t> prev_first_child,
    absl::Span<const uint32_t> prev_hot_to_full,
    absl::Span<const int32_t> hot_of_node, const size_t new_bag_size,
    DepthBagState* s) {
  const size_t N = selected_examples_per_node.size();
  const size_t old_size = s->bag.size();
  s->bag_scratch.resize(new_bag_size);
  s->node_scratch.resize(new_bag_size);
  s->cursor.assign(N, 0);

  const UnsignedExampleIdx* old_bag = s->bag.data();
  DCHECK_EQ(s->node_of_bag.size(), old_size);
  DCHECK_EQ(hot_of_node.size(), N);
  const uint32_t* old_node = s->node_of_bag.data();
  const uint32_t prev_hot_n = static_cast<uint32_t>(prev_hot_to_full.size());
  const uint32_t prev_n = static_cast<uint32_t>(prev_first_child.size());

  size_t out = 0;
  for (size_t i = 0; i < old_size; ++i) {
    const uint32_t ph = old_node[i];
    if (ph >= prev_hot_n) return false;
    const uint32_t p = prev_hot_to_full[ph];
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
    // The cursor advances for cold children too: it is what identifies the
    // owning child of the NEXT rows of the same parent.
    ++cur;

    const int32_t kc = hot_of_node[nc];
    if (kc < 0) continue;  // New owner is cold: the row leaves the bag.

    if (out >= new_bag_size) return false;
    s->bag_scratch[out] = ex;
    s->node_scratch[out] = static_cast<uint32_t>(kc);
    ++out;
  }
  if (out != new_bag_size) return false;

  s->bag.swap(s->bag_scratch);
  s->node_of_bag.swap(s->node_scratch);
  DCHECK(std::is_sorted(s->bag.begin(), s->bag.end()));
  return true;
}
#endif

// Full rebuild when the relabel is unavailable or fails: N == 1 copies the
// sorted span, N > 1 concats then VQSorts (example, node) as hwy::K32V32 —
// unstable but tie-invariant. Labels = positions, so hot spans give hot labels.
void RebuildBagFromSpans(
    absl::Span<const absl::Span<const UnsignedExampleIdx>>
        selected_examples_per_node,
    const size_t bag_size, const DepthBagChrono billing, DepthBagState* state) {
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
  const size_t N = selected_examples_per_node.size();
  if (N == 1) {
    // Copy the span into the state: the rolling buffer is repartitioned in
    // place by this depth's SplitExamplesInPlace, so the span's contents
    // will not survive to seed the next depth's relabel.
    CHRONO_SCOPE_AP(build_id);
    state->bag.assign(selected_examples_per_node[0].begin(),
                      selected_examples_per_node[0].end());
    state->node_of_bag.assign(state->bag.size(), 0u);
    return;
  }
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

void AdvanceDepthBag(
    absl::Span<const absl::Span<const UnsignedExampleIdx>>
        selected_examples_per_node,
    absl::Span<const int32_t> prev_first_child, const size_t bag_size,
    const DepthBagChrono billing, DepthBagState* state) {
  DCHECK(state != nullptr);
  DCHECK_GT(bag_size, 0u);

  // Resolve the relabel's chrono sub-scope id for this caller. The FuncId enum
  // only exists under CHRONO_PROFILE; without -DFINE_CHRONO_AP the
  // CHRONO_SCOPE_AP macro is inert, so the id is [[maybe_unused]].
#ifdef CHRONO_PROFILE
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
  if (!have_bag) {
    RebuildBagFromSpans(selected_examples_per_node, bag_size, billing, state);
  }
  state->valid = true;
}

#if defined(DEPTHWISE_1_PASS) && !defined(DW1_COLWALK_CONTROL)
void AdvanceDepthBagHot(
    absl::Span<const absl::Span<const UnsignedExampleIdx>>
        selected_examples_per_node,
    absl::Span<const absl::Span<const UnsignedExampleIdx>>
        hot_selected_examples_per_node,
    absl::Span<const int32_t> prev_first_child,
    absl::Span<const uint32_t> prev_hot_to_full,
    absl::Span<const int32_t> hot_of_node, const size_t hot_bag_size,
    DepthBagState* state) {
  DCHECK(state != nullptr);
  DCHECK_GT(hot_bag_size, 0u);
  DCHECK_EQ(hot_of_node.size(), selected_examples_per_node.size());

  // Fast path. Needs both maps: the previous bag's labels are hot indices of
  // the previous depth, which only prev_hot_to_full can resolve.
  bool have_bag = false;
  if (state->valid && !prev_first_child.empty() && !prev_hot_to_full.empty()) {
    CHRONO_SCOPE_AP(::yggdrasil_decision_forests::chrono_prof::kDw1SharedBag);
    have_bag = RelabelBagForNewDepthHot(
        selected_examples_per_node, prev_first_child, prev_hot_to_full,
        hot_of_node, hot_bag_size, state);
  }

  // Fallback: rebuild from the HOT spans only -- their positions are exactly
  // the hot labels the kernel expects.
  if (!have_bag) {
    RebuildBagFromSpans(hot_selected_examples_per_node, hot_bag_size,
                        DepthBagChrono::kDw1SharedRows, state);
  }
  state->valid = true;
}
#endif  // DEPTHWISE_1_PASS && !DW1_COLWALK_CONTROL

}  // namespace yggdrasil_decision_forests::model::decision_tree

#endif  // defined(SYMMETRIC_OPTIMIZED) || defined(DEPTHWISE_1_PASS)
