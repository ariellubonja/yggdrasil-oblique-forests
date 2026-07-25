#include "yggdrasil_decision_forests/learner/decision_tree/oblique_cpu_depthwise_1pass.h"

#ifdef DEPTHWISE_1_PASS

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <mutex>
#include <sstream>
#include <string>
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
// pattern as the nodewise baseline. This kernel inverts the loop: bucket
// every (node, projection, weight) reference of the whole depth by column,
// then walk the touched columns once in ascending order. Each column's
// gathers become one ordered pass over the depth's row bands
// (page/TLB/DRAM-row friendly) instead of being scattered across the whole
// depth.

struct ColEntry {
  int32_t node;   // index within the depth batch
  int32_t proj;   // projection index within the node
  int32_t col;    // attribute idx
  float weight;
};

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

absl::Status ApplyProjectionsDepthwise1Pass(
    const dataset::VerticalDataset& train_dataset,
    const google::protobuf::RepeatedField<int32_t>& numerical_features,
    absl::Span<const absl::Span<const UnsignedExampleIdx>>
        selected_examples_per_node,
    absl::Span<const std::vector<internal::Projection>> projections_per_node,
    absl::Span<const int32_t> prev_first_child, DepthBagState* bag_state,
    absl::Span<std::vector<float>> out_projected, int32_t current_depth) {

  CHRONO_SCOPE_COARSE(::yggdrasil_decision_forests::chrono_prof::kProjectionEvaluate);
  const size_t N = selected_examples_per_node.size();
  if (N == 0) return absl::OkStatus();

  /* #region Performance minions. All these take <10% of AP time */

  int max_attr = 0;
  for (const auto attribute_idx : numerical_features) {
    max_attr = std::max(max_attr, attribute_idx);
  }

#ifdef DW1_SHARED_ROWS
  size_t total_rows = 0;
  for (size_t n = 0; n < N; ++n) {
    total_rows += selected_examples_per_node[n].size();
  }
#endif

  // ── Phase 1: PreSize --- ~5% of AP time
  // Slab pre-size (zero-init: the column sweep accumulates)
  {
    CHRONO_SCOPE_AP(::yggdrasil_decision_forests::chrono_prof::kDw1PreSize);
    for (size_t n = 0; n < N; ++n) {
      // Accounts for variable n. projections / node
      const size_t slab =
          selected_examples_per_node[n].size() * projections_per_node[n].size();
      out_projected[n].assign(slab, 0.f);
    }
  }

#if defined(DW1_SHARED_ROWS) && !defined(DW1_HOT_NODES)
  // ── Depth bag ──────────────────────────────────────────────────────
  // Obtain this depth's example-sorted (bag, node_of_bag) once for the whole
  // frontier (O(bag) relabel of the previous depth's bag in the steady state,
  // concat + VQSort fallback otherwise). The colwalk below reads it directly.
  AdvanceDepthBag(selected_examples_per_node, prev_first_child, total_rows,
                  DepthBagChrono::kDw1SharedRows, bag_state);
#endif  // DW1_SHARED_ROWS && !DW1_HOT_NODES
#if defined(DW1_HOT_NODES)
  // Hot-nodes build: the spans handed in are the depth's HOT nodes only, and
  // the bag (over exactly those rows, labelled by hot index) was advanced by
  // the BFS driver before this call -- only the driver holds the full-domain
  // spans + hot/full index maps the relabel needs
  DCHECK(bag_state != nullptr && bag_state->valid);
  DCHECK_EQ(bag_state->bag.size(), total_rows);
#endif  // DW1_HOT_NODES

  // ── Phase 2: Sweep ────────────────────────────────────────────────
  // Takes the majority of ApplyProjection time: 9.052292408 for PreSize vs.	138.5347208 for Sweep
  {
    CHRONO_SCOPE_AP(::yggdrasil_decision_forests::chrono_prof::kDw1Sweep);
    internal::ProjectionEvaluator evaluator(train_dataset, numerical_features);

    // Direct column pointers exist only for the default VerticalDataset
    // layout; alternate trunk layouts fall back to the generic kernel.

    // TODO make the code error out if incompatible layout. don't need fallback
    bool col_major_dataset = true;
    for (const auto attribute_idx : numerical_features) {
      if (evaluator.AttributeData(attribute_idx) == nullptr) {
        col_major_dataset = false;
        break;
      }
    }

    if (!col_major_dataset) {
      CHRONO_SCOPE_AP(::yggdrasil_decision_forests::chrono_prof::kDw1SweepGeneric);
      for (size_t n = 0; n < N; ++n) {
        const auto sel = selected_examples_per_node[n];
        const auto& projs = projections_per_node[n];
        for (size_t p = 0; p < projs.size(); ++p) {
          EvaluateProjectionRowsGeneric(
              evaluator, projs[p], sel.data(), sel.size(),
              out_projected[n].data() + p * sel.size());
        }
      }
      return absl::OkStatus();
    }

    // Column-centric sweep: bucket references by column, then walk the
    // touched columns ascending.
    std::vector<ColEntry> entries;
    std::vector<ColEntry> sorted;
    std::vector<int32_t> touched;    // touched column ids, sorted ascending
    std::vector<int32_t> col_count(static_cast<size_t>(max_attr) + 1, 0);

    // <= 2% of AP time
    for (size_t n = 0; n < N; ++n) {
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

    // Trees are trained one per thread and depth increases monotonically
    // within a tree, so a thread-local counter bumped whenever the depth
    // stops increasing labels the tree — without depending on the chrono
    // build (tls_ctx only exists under -DCHRONO_PROFILE). Shared by the two
    // debug blocks below.
    int stats_tree;
    {
      static thread_local int tls_tree = -1;
      static thread_local int32_t prev_depth = -1;
      if (current_depth <= prev_depth) ++tls_tree;
      if (tls_tree < 0) tls_tree = 0;
      prev_depth = current_depth;
      stats_tree = tls_tree;
    }

    // ── Per-node example counts at one depth, dumped to a file ─────────
    // At every depth (default) append one line per node of that depth with its
    // example count, to YDF_DW1_NODE_SIZES_FILE (default dw1_node_sizes.txt).
    // Restrict to a single depth with YDF_DW1_NODE_SIZES_DEPTH=<d>; off with
    // YDF_DW1_NODE_SIZES_DEPTH=-1.
    {
      // -2 = every depth (the default), -1 = off, >=0 = that depth only.
      static const int32_t kNodeSizesDepth = [] {
        const char* e = std::getenv("YDF_DW1_NODE_SIZES_DEPTH");
        return e == nullptr ? -2 : static_cast<int32_t>(std::atoi(e));
      }();
      static const std::string kNodeSizesFile = [] {
        const char* e = std::getenv("YDF_DW1_NODE_SIZES_FILE");
        return std::string(e == nullptr ? "dw1_node_sizes.txt" : e);
      }();
      if (kNodeSizesDepth == -2 || current_depth == kNodeSizesDepth) {
        std::ostringstream os;
        size_t depth_examples = 0;
        for (size_t n = 0; n < N; ++n) {
          depth_examples += selected_examples_per_node[n].size();
        }
        os << "----DEPTH " << current_depth << "---- tree=" << stats_tree
           << " nodes=" << N << " examples=" << depth_examples << "\n";
        for (size_t n = 0; n < N; ++n) {
          os << "n" << n << ": " << selected_examples_per_node[n].size()
             << " examples\n";
        }
        // Trees run in parallel: one locked append per depth keeps the
        // per-tree blocks whole.
        static std::mutex mu;
        std::lock_guard<std::mutex> lock(mu);
        std::ofstream f(kNodeSizesFile, std::ios::app);
        f << os.str();
      }
    }

    // ── Per-depth column reference stats (debug print) ─────────────────
    // For every column: in how many nodes of this depth it appears, and how
    // many elements are read from it — Σ over the distinct nodes referencing
    // the column of that node's row count (a node referencing the column from
    // several projections re-reads the same examples, so it counts once).
    // One column per line, under a ----DEPTH X---- header; the whole block is
    // buffered and emitted in one write so parallel trees don't interleave.
    // Silence with YDF_DW1_COL_STATS=0; add untouched columns (c17: 0) with
    // YDF_DW1_COL_STATS_ALL=1.
    {
      static const bool kColStats = [] {
        const char* e = std::getenv("YDF_DW1_COL_STATS");
        return e == nullptr || std::string(e) != "0";
      }();
      static const bool kColStatsAll = [] {
        const char* e = std::getenv("YDF_DW1_COL_STATS_ALL");
        return e != nullptr && std::string(e) == "1";
      }();
      if (kColStats) {
        size_t depth_examples = 0;
        for (size_t n = 0; n < N; ++n) {
          depth_examples += selected_examples_per_node[n].size();
        }
        std::ostringstream os;
        os << "\n----DEPTH " << current_depth << "---- [DW1 colstats] tree="
           << stats_tree << " nodes=" << N << " examples=" << depth_examples
           << " touched_cols=" << touched.size() << "/"
           << numerical_features.size() << " refs=" << entries.size() << "\n\n";

        size_t next_touched = 0;  // walks `touched` to fill untouched gaps
        size_t slice_begin = 0;   // start of the current column's slice
        for (const int32_t c : touched) {
          if (kColStatsAll) {
            for (int32_t u = (next_touched == 0 ? 0 : touched[next_touched - 1] + 1);
                 u < c; ++u) {
              os << "c" << u << ": 0\n";
            }
          }
          ++next_touched;
          const size_t slice_end = static_cast<size_t>(col_count[c]);
          size_t examples = 0;
          int32_t nodes = 0, prev_node = -1;
          for (size_t k = slice_begin; k < slice_end; ++k) {
            if (sorted[k].node != prev_node) {  // slice is node-major
              ++nodes;
              examples += selected_examples_per_node[sorted[k].node].size();
              prev_node = sorted[k].node;
            }
          }
          os << "c" << c << ": " << nodes << " nodes, " << examples
             << " examples\n";
          slice_begin = slice_end;
        }
        if (kColStatsAll && !touched.empty()) {
          for (int32_t u = touched.back() + 1; u <= max_attr; ++u) {
            os << "c" << u << ": 0\n";
          }
        }
        std::cout << os.str() << std::flush;
      }
    }

/* #endregion */

/* #region SHARED_ROWS */

#ifdef DW1_SHARED_ROWS
    // Shared-rows colwalk. Read each touched column in ONE ascending pass
    // over the depth's example-sorted bag and fan each value out (scatter)
    // to every (node,projection) of its owning node referencing the column:
    // the gather path's per-(node,proj) sequential writes become random
    // writes into the output slabs. This is the sparse-reads-for-sparse-
    // writes trade.

    // Where the first element of ref_proj that node n has
    std::vector<int32_t> node_ref_off(N, -1);  // per node: start in ref_*, else -1
    // How many projections of node n reference column c
    std::vector<int32_t> node_ref_cnt(N, 0);   // per node: #refs to current column
    
    // How many examples of node n have been processed so far
    std::vector<int32_t> node_local(N, 0);     // per node: rows seen this column

    std::vector<int32_t> ref_proj;             // current column's (proj, weight)
    std::vector<float> ref_w;                  //   runs, grouped by node
    std::vector<int32_t> col_touched;          // node ids touched this column

    const UnsignedExampleIdx* bag = bag_state->bag.data();
    // AdvanceDepthBag always sizes node_of_bag to the bag (all zeros for a
    // single-node batch), so the kernel reads it unconditionally and is
    // agnostic of batch shape/depth.
    DCHECK_EQ(bag_state->node_of_bag.size(), total_rows);
    const uint32_t* nob = bag_state->node_of_bag.data();

    CHRONO_BEGIN_AP(dw1_sweep_colwalk); // ~86% of AP runtime
    {
      size_t pos = 0;

      for (const int32_t c : touched) {
        const float* col = evaluator.AttributeData(c);
        const size_t slice_begin = pos;
        const size_t slice_end = static_cast<size_t>(col_count[c]);
        pos = slice_end;

        // Group column c's refs by node. Entries were bucketed node-major, so
        // within the slice the node id is non-decreasing and each node's refs
        // form one contiguous run [off, off+cnt) in ref_proj / ref_w.
        ref_proj.resize(slice_end - slice_begin);
        ref_w.resize(slice_end - slice_begin);
        col_touched.clear();
        int32_t prev_node = -1;
        CHRONO_BEGIN_AP(dw1_colwalk_group_by_node); // This operation is free - ~0% cost
        for (size_t k = slice_begin; k < slice_end; ++k) {
          const ColEntry& e = sorted[k];
          const size_t kk = k - slice_begin;
          ref_proj[kk] = e.proj;
          ref_w[kk] = e.weight;
          const int32_t n = e.node;
          if (n != prev_node) {
            node_ref_off[n] = static_cast<int32_t>(kk);
            node_ref_cnt[n] = 0;
            node_local[n] = 0;  // restart this node's row counter for col c
            col_touched.push_back(n);
            prev_node = n;
          }
          ++node_ref_cnt[n];
        }
        CHRONO_END_AP(dw1_colwalk_group_by_node,
                   ::yggdrasil_decision_forests::chrono_prof::kDw1ColWalkGroupByNode);

        // std::cout << "node_ref_off node_ref_cnt" << std::endl;
        // for (int i = 0 ; i < 25; i++) {
        //     std::cout << node_ref_off[i] << " " << node_ref_cnt[i] << std::endl;
        // }

        // One ascending pass over the depth bag: dense read of col, scatter
        // write of each contribution to its node's projections referencing c.
        CHRONO_BEGIN_AP(dw1_colwalk_bag_scatter);
        for (size_t s = 0; s < total_rows; ++s) { // loop over examples
          const int32_t n = static_cast<int32_t>(nob[s]);
          const int32_t off = node_ref_off[n];
          if (off < 0) continue;  // owning node has no projection on column c

          const float v = col[bag[s]];

          const int32_t cnt = node_ref_cnt[n];
          // The bag is ex-sorted and each node's selected_examples is sorted
          // ascending, so its rows are visited in slab order: the running
          // counter is the row's local slot. Advances once per node row
          // (every entry that reaches here), so it stays in lockstep.
          const int32_t local = node_local[n]++;
          float* slab = out_projected[n].data();
          const size_t rows_n = selected_examples_per_node[n].size();

          for (int32_t t = 0; t < cnt; ++t) {
            slab[static_cast<size_t>(ref_proj[off + t]) * rows_n + local] +=
                ref_w[off + t] * v;
          }
        }

        CHRONO_END_AP(dw1_colwalk_bag_scatter,
                   ::yggdrasil_decision_forests::chrono_prof::kDw1ColWalkBagScatter);

        for (const int32_t n : col_touched) node_ref_off[n] = -1;
      }
    }
    CHRONO_END_AP(dw1_sweep_colwalk,
               ::yggdrasil_decision_forests::chrono_prof::kDw1SweepColWalk);
#else
/* #endregion */

/* #region Col sharing only */
// Slower than BFS by <= 15%
// Takeaway: column sharing via cache residency doesn't work at scale.
    CHRONO_BEGIN_AP(dw1_sweep_colwalk); // ~93% of the ApplyProjection time in non-Shared-Rows. In Shared-rows, ~66%
    size_t pos = 0;
    for (const int32_t c : touched) {
      const float* col = evaluator.AttributeData(c);
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
    }
/* #endregion */
    CHRONO_END_AP(dw1_sweep_colwalk,
               ::yggdrasil_decision_forests::chrono_prof::kDw1SweepColWalk);
#endif  // DW1_SHARED_ROWS
  }

  return absl::OkStatus();
}

}  // namespace yggdrasil_decision_forests::model::decision_tree

#endif  // DEPTHWISE_1_PASS
