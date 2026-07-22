#include "yggdrasil_decision_forests/learner/decision_tree/oblique_cpu_depthwise_1pass.h"

#ifdef DEPTHWISE_1_PASS

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <vector>

#include "absl/log/check.h"
#include "absl/status/status.h"
#include "absl/types/span.h"
#include "yggdrasil_decision_forests/learner/decision_tree/oblique.h"
#include "yggdrasil_decision_forests/learner/decision_tree/oblique_cpu_depthwise_bag.h"
#include "yggdrasil_decision_forests/utils/parallel_chrono.h"

namespace yggdrasil_decision_forests::model::decision_tree {

namespace {

// Column-centric depthwise sweep.
//
// At mid depths the frontier holds thousands of small nodes, and the union of
// their sampled columns covers the feature space many times over: total
// column references N*K*d >> F. The previous kernel visited columns in
// (node, projection) order — K*d random column hops per node, the same DRAM
// pattern as the nodewise baseline. This kernel inverts the loop: group a
// block of consecutive nodes, bucket every (node, projection, weight)
// reference by column, then walk the touched columns once in ascending
// order. Each column's gathers become one ordered pass over the block's row
// bands (page/TLB/DRAM-row friendly) instead of being scattered across the
// whole depth.
//
// Blocks are sized so the block's output slabs stay cache-resident, because
// the column order revisits each slab ~K*d times. Nodes too large for a
// block keep the projection-major order (one streaming write of their slab),
// mirroring the big/small dispatch of Dynamic_Row_Col_Major.

// Output-slab budget (floats) per column-centric block. Also the cutoff
// above which a node is processed projection-major. Tunable for experiments.
size_t Dw1BlockFloats() {
  static const size_t value = [] {
    const char* e = std::getenv("YDF_DW1_BLOCK_FLOATS");
    // Measured at 3M×4096: 4 MiB 55.0 s, 16 MiB 49.6, 64 MiB 47.0,
    // 256 MiB 45.6 — column sharing dominates output-slab residency, with
    // diminishing returns past 64 MiB. Default 256 MiB (64 Mi floats).
    return e != nullptr ? static_cast<size_t>(std::strtoull(e, nullptr, 10))
                        : static_cast<size_t>(64) << 20;
  }();
  return value;
}

struct ColEntry {
  int32_t node;   // index within the depth batch
  int32_t proj;   // projection index within the node
  int32_t col;    // attribute idx
  float weight;
};

struct Dw1Task {
  size_t begin_node;
  size_t end_node;  // exclusive; big nodes come as [n, n+1) with big=true
  bool big;
};

// Per-worker scratch, reused across blocks.
struct Dw1Scratch {
  std::vector<ColEntry> entries;
  std::vector<ColEntry> sorted;
  std::vector<int32_t> touched;    // touched column ids, sorted ascending
  std::vector<int32_t> col_count;  // per-column counters, sized max_attr+1
#ifdef DW1_SHARED_ROWS
  std::vector<int32_t> node_ref_off;   // per block-node: start in ref_*, else -1
  std::vector<int32_t> node_ref_cnt;   // per block-node: #refs to current column
  std::vector<int32_t> node_local;     // per block-node: rows seen this column
  std::vector<int32_t> ref_proj;       // current column's (proj, weight) runs,
  std::vector<float> ref_w;            //   grouped by node (node-ascending)
  std::vector<int32_t> col_touched;    // block-node ids touched this column
#endif
};

// Projection-major kernel for one node (big nodes / fallback). Direct column
// pointers; the per-load AttributeValue branch chain is hoisted out.
inline void EvaluateNodeProjMajor(
    const internal::ProjectionEvaluator& evaluator,
    const std::vector<internal::Projection>& projs,
    const UnsignedExampleIdx* sel_ptr, size_t rows_n, float* out_ptr) {
  struct FeatRef {
    const float* col;
    float weight;
  };
  std::vector<FeatRef> feats;
  for (size_t p = 0; p < projs.size(); ++p) {
    feats.clear();
    for (const auto& feat : projs[p]) {
      feats.push_back({evaluator.AttributeData(feat.attribute_idx),
                       feat.weight});
    }
    float* o = out_ptr + p * rows_n;
    for (size_t i = 0; i < rows_n; ++i) {
      const UnsignedExampleIdx ex = sel_ptr[i];
      float acc = 0.f;
      for (const auto& f : feats) {
        float v = f.col[ex];
        acc += f.weight * v;
      }
      o[i] = acc;
    }
  }
}

// Original per-(node, projection) kernel via AttributeValue. Only used when
// direct column pointers are unavailable (alternate dataset layouts).
inline void EvaluateProjectionRowsGeneric(
    const internal::ProjectionEvaluator& evaluator,
    const internal::Projection& proj, const UnsignedExampleIdx* sel_ptr,
    size_t rows_n, float* out_for_proj) {
  for (size_t i = 0; i < rows_n; ++i) {
    const UnsignedExampleIdx ex = sel_ptr[i];
    float acc = 0.f;
    for (const auto& feat : proj) {
      float v = evaluator.AttributeValue(feat.attribute_idx, ex);
      acc += feat.weight * v;
    }
    out_for_proj[i] = acc;
  }
}

}  // namespace


// TODO avoid entirely for big nodes - only run in mid-tree
absl::Status ApplyProjectionsDepthwise1Pass(
    const dataset::VerticalDataset& train_dataset,
    const google::protobuf::RepeatedField<int32_t>& numerical_features,
    absl::Span<const absl::Span<const UnsignedExampleIdx>>
        selected_examples_per_node,
    absl::Span<const std::vector<internal::Projection>> projections_per_node,
    absl::Span<const int32_t> prev_first_child, DepthBagState* bag_state,
    absl::Span<std::vector<float>> out_projected) {

  CHRONO_SCOPE_COARSE(::yggdrasil_decision_forests::chrono_prof::kProjectionEvaluate);
  const size_t N = selected_examples_per_node.size();
  if (N == 0) return absl::OkStatus();

  /* #region Performance minions. All these take <10% of AP time */

  int max_attr = 0;
  for (const auto attribute_idx : numerical_features) {
    max_attr = std::max(max_attr, attribute_idx);
  }

#ifdef DW1_SHARED_ROWS
  // Per-block ex-sorted arenas: the depth bag distributed into one contiguous
  // slice per small block. SoA (arena_ex/arena_node, node ids block-local) so
  // the colwalk reads it without the `- begin_node` subtract. thread_local so
  // capacity amortizes across depths/trees on this pool thread. block_of_node
  // maps a depth-batch node to its small-block id (-1 for big / fallback
  // nodes); block_off holds prefix offsets; block_begin[b] is block b's first
  // depth-batch node index.
  thread_local std::vector<int32_t> block_of_node;
  thread_local std::vector<size_t> block_begin;
  thread_local std::vector<size_t> block_off;
  thread_local std::vector<size_t> block_cursor;
  thread_local std::vector<UnsignedExampleIdx> arena_ex;
  thread_local std::vector<int32_t> arena_node;
  size_t total_rows = 0;
  for (size_t n = 0; n < N; ++n) {
    total_rows += selected_examples_per_node[n].size();
  }
#endif

  /* #region ── Phase 1: PreSize --- ~5% of AP time */
  // Slab pre-size (zero-init: the column sweep accumulates) + task build
  std::vector<Dw1Task> tasks;
  {
    CHRONO_SCOPE(::yggdrasil_decision_forests::chrono_prof::kDw1PreSize);
    const size_t budget = Dw1BlockFloats();
    size_t blk_begin = 0;
    size_t blk_floats = 0;
    for (size_t n = 0; n < N; ++n) {
      // Accounts for variable n. projections / node
      const size_t slab =
          selected_examples_per_node[n].size() * projections_per_node[n].size();
      out_projected[n].assign(slab, 0.f);

      // TODO Ariel When is slab over budget? What happens then?
      // Also why is there 3 paths in this if {}
      if (slab > budget) {
        if (blk_begin < n) tasks.push_back({blk_begin, n, /*big=*/false});
        tasks.push_back({n, n + 1, /*big=*/true});
        blk_begin = n + 1;
        blk_floats = 0;
      } else if (blk_floats + slab > budget) {
        if (blk_begin < n) tasks.push_back({blk_begin, n, /*big=*/false});
        blk_begin = n;
        blk_floats = slab;
      } else {
        blk_floats += slab;
      }
    }
    if (blk_begin < N) tasks.push_back({blk_begin, N, /*big=*/false});
  }

#ifdef DW1_SHARED_ROWS
  // ── Depth bag + per-block distribution (billed to kDw1SharedBag) ────
  // Obtain this depth's example-sorted (bag, node_of_bag) once for the whole
  // frontier (O(bag) relabel of the previous depth's bag in the steady state,
  // concat + VQSort fallback otherwise), then split it into per-small-block
  // ex-sorted arenas. This replaces the old per-block bag build + VQSort inside
  // run_task: one relabel + one stable distribution pass per depth instead of a
  // sort per block. Big nodes' rows stay in the bag (the next depth's relabel
  // needs every surviving row) but are skipped by the distribution (they run
  // projection-major in run_task).
  AdvanceDepthBag(selected_examples_per_node, prev_first_child, total_rows,
                  DepthBagChrono::kDw1SharedRows, bag_state);
  {
    CHRONO_SCOPE(::yggdrasil_decision_forests::chrono_prof::kDw1SharedBag);
    // Assign small-block ids + sizes (big tasks -> block_of_node[n] == -1).
    block_of_node.assign(N, -1);
    block_begin.clear();
    block_off.clear();
    block_off.push_back(0);
    size_t arena_total = 0;
    for (const auto& task : tasks) {
      if (task.big) continue;
      const int32_t b = static_cast<int32_t>(block_begin.size());
      size_t blk_rows = 0;
      for (size_t n = task.begin_node; n < task.end_node; ++n) {
        block_of_node[n] = b;
        blk_rows += selected_examples_per_node[n].size();
      }
      block_begin.push_back(task.begin_node);
      arena_total += blk_rows;
      block_off.push_back(arena_total);
    }

    // Stable distribution: walk the ex-sorted depth bag once, routing each
    // entry to its block's arena. Within a block a node's rows keep ascending
    // example order (the depth bag is ex-sorted and rows are disjoint across
    // nodes), so the colwalk's running-counter slab-slot recovery is preserved.
    arena_ex.resize(arena_total);
    arena_node.resize(arena_total);
    block_cursor.resize(block_begin.size());
    for (size_t b = 0; b < block_begin.size(); ++b) {
      block_cursor[b] = block_off[b];
    }
    const UnsignedExampleIdx* bag = bag_state->bag.data();
    // node_of_bag is empty only for a single-node depth; the fused DW1 branch
    // requires depth_batch.size() > 1, so it is populated here, but guard
    // defensively (single node -> all entries node 0).
    const bool single = bag_state->node_of_bag.empty();
    const uint32_t* nob =
        single ? nullptr : bag_state->node_of_bag.data();
    for (size_t i = 0; i < total_rows; ++i) {
      const uint32_t n = single ? 0u : nob[i];
      const int32_t b = block_of_node[n];
      if (b < 0) continue;  // big node: processed projection-major, no arena
      const size_t w = block_cursor[b]++;
      arena_ex[w] = bag[i];
      arena_node[w] = static_cast<int32_t>(n - block_begin[b]);  // block-local
    }
  }
#endif  // DW1_SHARED_ROWS

/* #endregion */

  // ── Phase 2: Sweep ────────────────────────────────────────────────
  // Takes the majority of ApplyProjection time: 9.052292408 for PreSize vs.	138.5347208 for Sweep
  {
    CHRONO_SCOPE(::yggdrasil_decision_forests::chrono_prof::kDw1Sweep);
    internal::ProjectionEvaluator evaluator(train_dataset, numerical_features);

    // Direct column pointers exist only for the default VerticalDataset
    // layout; alternate trunk layouts fall back to the generic kernel.

    // TODO make the code error out if incompatible layout. don't need fallback
    bool direct = true;
    for (const auto attribute_idx : numerical_features) {
      if (evaluator.AttributeData(attribute_idx) == nullptr) {
        direct = false;
        break;
      }
    }

    const auto run_task = [&](const Dw1Task& task, Dw1Scratch& scratch) {
      if (!direct) {
        CHRONO_SCOPE(::yggdrasil_decision_forests::chrono_prof::kDw1SweepGeneric);
        for (size_t n = task.begin_node; n < task.end_node; ++n) {
          const auto sel = selected_examples_per_node[n];
          const auto& projs = projections_per_node[n];
          for (size_t p = 0; p < projs.size(); ++p) {
            EvaluateProjectionRowsGeneric(
                evaluator, projs[p], sel.data(), sel.size(),
                out_projected[n].data() + p * sel.size());
          }
        }
        return;
      }

      // TODO what is this?
      if (task.big) {
        CHRONO_SCOPE(::yggdrasil_decision_forests::chrono_prof::kDw1SweepBig);
        const size_t n = task.begin_node;
        const auto sel = selected_examples_per_node[n];
        EvaluateNodeProjMajor(evaluator, projections_per_node[n], sel.data(),
                              sel.size(), out_projected[n].data());
        return;
      }

      // Column-centric block: bucket references by column, then walk the
      // touched columns ascending.
      auto& entries = scratch.entries;
      auto& sorted = scratch.sorted;
      auto& touched = scratch.touched;
      auto& col_count = scratch.col_count;
      if (col_count.size() < static_cast<size_t>(max_attr) + 1) {
        col_count.assign(static_cast<size_t>(max_attr) + 1, 0);
      }
      entries.clear();
      touched.clear();

      // <= 2% of AP time
      for (size_t n = task.begin_node; n < task.end_node; ++n) {
        const auto& projs = projections_per_node[n];
        for (size_t p = 0; p < projs.size(); ++p) {
          for (const auto& feat : projs[p]) {
            entries.push_back({static_cast<int32_t>(n),
                               static_cast<int32_t>(p), feat.attribute_idx,
                               feat.weight});
            if (col_count[feat.attribute_idx]++ == 0) {
              touched.push_back(feat.attribute_idx);
            }
          }
        }
      }
      std::sort(touched.begin(), touched.end());

      // Counting sort by column: col_count becomes the running fill cursor.
      // <1% of AP time
      size_t offset = 0;
      for (const int32_t c : touched) {
        const int32_t cnt = col_count[c];
        col_count[c] = static_cast<int32_t>(offset);
        offset += cnt;
      }
      sorted.resize(entries.size());
      for (const auto& e : entries) {
        sorted[col_count[e.col]++] = e;
      }

/* #endregion */

/* #region SHARED_ROWS */

// TODO BEFORE ANYTHING! RUN CHRONO ON SHARED_ROWS AP
#ifdef DW1_SHARED_ROWS
      // Shared-rows colwalk. This block's example-sorted bag was already
      // distributed from the depth bag into arena[a_begin, a_end) (SoA
      // arena_ex/arena_node, node ids block-local) before the task loop — no
      // per-block bag build or sort here. Read each touched column in ONE
      // ascending pass over the arena slice and fan each value out (scatter) to
      // every (node,projection) of its owning node referencing the column: the
      // gather path's per-(node,proj) sequential writes become random writes
      // into the block's output slabs. This is the sparse-reads-for-sparse-
      // writes trade; YDF_DW1_BLOCK_FLOATS bounds the scatter-write working set,
      // so shrink it if the slabs spill LLC.
      const int32_t begin_node_i = static_cast<int32_t>(task.begin_node);
      const size_t block_nodes = task.end_node - task.begin_node;
      const int32_t blk = block_of_node[task.begin_node];  // small-block id
      const size_t a_begin = block_off[blk];
      const size_t a_end = block_off[blk + 1];

      auto& node_ref_off = scratch.node_ref_off;
      auto& node_ref_cnt = scratch.node_ref_cnt;
      if (node_ref_off.size() < block_nodes) {
        // -1 == "owning node has no ref to the current column". The per-column
        // reset below restores every touched slot to -1, so the whole
        // [0, block_nodes) range is -1 at each column boundary.
        node_ref_off.assign(block_nodes, -1);
        node_ref_cnt.assign(block_nodes, 0);
        scratch.node_local.assign(block_nodes, 0);
      }

      CHRONO_BEGIN(dw1_sweep_colwalk); // ~86% of AP runtime
      {
        auto& ref_proj = scratch.ref_proj;
        auto& ref_w = scratch.ref_w;
        auto& col_touched = scratch.col_touched;
        auto& node_local = scratch.node_local;
        size_t pos = 0;

        for (const int32_t c : touched) {
          const float* col = evaluator.AttributeData(c);
          const size_t slice_begin = pos;
          const size_t slice_end = static_cast<size_t>(col_count[c]);
          pos = slice_end;

          // Group column c's refs by node. Entries were bucketed node-major, so
          // within the slice the node id is non-decreasing and each node's refs
          // form one contiguous run [off, off+cnt) in ref_proj / ref_w. Entries
          // still carry the global node id, so map to block-local here.
          ref_proj.resize(slice_end - slice_begin);
          ref_w.resize(slice_end - slice_begin);
          col_touched.clear();
          int32_t prev_node = -1;
          CHRONO_BEGIN(dw1_colwalk_group_by_node); // This operation is free - ~0% cost
          for (size_t k = slice_begin; k < slice_end; ++k) {
            const ColEntry& e = sorted[k];
            const size_t kk = k - slice_begin;
            ref_proj[kk] = e.proj;
            ref_w[kk] = e.weight;
            const int32_t bn = e.node - begin_node_i;
            if (e.node != prev_node) {
              node_ref_off[bn] = static_cast<int32_t>(kk);
              node_ref_cnt[bn] = 0;
              node_local[bn] = 0;  // restart this node's row counter for col c
              col_touched.push_back(bn);
              prev_node = e.node;
            }
            ++node_ref_cnt[bn];
          }
          CHRONO_END(dw1_colwalk_group_by_node,
                     ::yggdrasil_decision_forests::chrono_prof::kDw1ColWalkGroupByNode);

          // One ascending pass over the block's arena slice: dense read of col,
          // scatter write of each contribution to its node's projections
          // referencing c. arena_node is already block-local (no subtract).
          CHRONO_BEGIN(dw1_colwalk_bag_scatter);
          for (size_t j = a_begin; j < a_end; ++j) {
            const int32_t bn = arena_node[j];
            const int32_t off = node_ref_off[bn];
            if (off < 0) continue;  // owning node has no projection on column c

            const float v = col[arena_ex[j]];

            const int32_t cnt = node_ref_cnt[bn];
            // The arena slice is ex-sorted and each node's selected_examples is
            // sorted ascending, so its rows are visited in slab order: the
            // running counter IS the row's local slot. Advances once per node
            // row (every entry that reaches here), so it stays in lockstep.
            const int32_t local = node_local[bn]++;
            const size_t node = static_cast<size_t>(bn) + task.begin_node;
            CHRONO_BEGIN(dw1_colwalk_slab_ptr);
            float* slab = out_projected[node].data();
            CHRONO_END(dw1_colwalk_slab_ptr,
                       ::yggdrasil_decision_forests::chrono_prof::kDw1ColWalkSlabPtr);
            const size_t rows_n = selected_examples_per_node[node].size();

            {
              CHRONO_SCOPE(
                  ::yggdrasil_decision_forests::chrono_prof::kDw1ColWalkSlabAccum);
              for (int32_t t = 0; t < cnt; ++t) {
                slab[static_cast<size_t>(ref_proj[off + t]) * rows_n + local] +=
                    ref_w[off + t] * v;
              }
            }
          }

          CHRONO_END(dw1_colwalk_bag_scatter,
                     ::yggdrasil_decision_forests::chrono_prof::kDw1ColWalkBagScatter);

          for (const int32_t bn : col_touched) node_ref_off[bn] = -1;
          col_count[c] = 0;  // reset for the next block
        }
      }
      CHRONO_END(dw1_sweep_colwalk,
                 ::yggdrasil_decision_forests::chrono_prof::kDw1SweepColWalk);
#else
/* #endregion */

/* #region Col sharing only */
// Slower than BFS by <= 15%
// Takeaway: column sharing via cache residency doesn't work at scale.
      CHRONO_BEGIN(dw1_sweep_colwalk); // ~93% of the ApplyProjection time in non-Shared-Rows. In Shared-rows, ~66%
      size_t pos = 0;
      for (const int32_t c : touched) {
        const float* col = evaluator.AttributeData(c);
        // What does end do? pos is not reset to 0 after loop termination
        const size_t end = static_cast<size_t>(col_count[c]);
        for (; pos < end; ++pos) {
          // TODO sorted carries the weight. But we can simply sample the weight when we need it
          //  TODO would dropping the weight make it more cache friendly?

          // Sorted : sorted by selected column, regardless of node/projection
          const ColEntry& e = sorted[pos];
          const auto sel = selected_examples_per_node[e.node];
          // I think this node's bag size
          const size_t rows_n = sel.size();
          // TODO check whether the row sparsity here is the reason for misses
          const UnsignedExampleIdx* sel_ptr = sel.data();

          float* o = out_projected[e.node].data() + e.proj * rows_n;
          const float w = e.weight;

          for (size_t i = 0; i < rows_n; ++i) { // Critical section. check access patterns here
            // This read should cache miss each time: sel_ptr is still per-node, not union across nodes
            float v = col[sel_ptr[i]];
            // This may get impacted by union across nodes. Will no longer be sequential
            o[i] += w * v;
            // TODO later: leverage SIMD here
          }
        }
        col_count[c] = 0;  // reset for the next block
      }
/* #endregion */
      CHRONO_END(dw1_sweep_colwalk,
                 ::yggdrasil_decision_forests::chrono_prof::kDw1SweepColWalk);
#endif  // DW1_SHARED_ROWS
    };

    // Single-threaded by design: RandomForest already trains one tree per
    // thread, so the cores are busy with sibling trees. dw1 runs inline on
    // the caller thread like every other tree-internal kernel — spawning a
    // per-depth pool here would only oversubscribe (num_threads^2 runnable
    // threads + a barrier per depth).
    Dw1Scratch scratch;
    for (const auto& task : tasks) run_task(task, scratch);
  }

  return absl::OkStatus();
}

}  // namespace yggdrasil_decision_forests::model::decision_tree

#endif  // DEPTHWISE_1_PASS
