// Motivation: at mid depths the frontier's column references N*K*d >> F, yet
// the nodewise kernel hops columns in (node, projection) order. This one buckets
// the depth's refs by column and walks each touched column once, ascending.


/* #region Imports */
#include "yggdrasil_decision_forests/learner/decision_tree/oblique_cpu_depthwise_1pass.h"

#ifdef DEPTHWISE_1_PASS

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

#include "absl/log/check.h"
#include "absl/status/status.h"
#include "absl/types/span.h"
#include "hwy/contrib/sort/vqsort.h"
#include "yggdrasil_decision_forests/learner/decision_tree/oblique.h"
#include "yggdrasil_decision_forests/learner/decision_tree/oblique_cpu_depthwise_bag.h"
#include "yggdrasil_decision_forests/utils/parallel_chrono.h"
/* #endregion */

namespace yggdrasil_decision_forests::model::decision_tree {

namespace {

// Print per-depth node & column stats?
constexpr bool kDw1DebugStats = false;


struct ColEntry {
  int32_t node;   // index within the depth batch
  int32_t proj;   // projection index within the node
  int32_t col;    // attribute idx
  float weight;
};

}

absl::Status ApplyProjectionsDepthwise1Pass(
    const dataset::VerticalDataset& train_dataset,
    const google::protobuf::RepeatedField<int32_t>& numerical_features,
    absl::Span<const absl::Span<const UnsignedExampleIdx>>
        selected_examples_per_node,
    absl::Span<const std::vector<internal::Projection>> projections_per_node,
    DepthBagState* bag_state, absl::Span<std::vector<float>> out_projected,
    int32_t current_depth) {
  CHRONO_SCOPE_COARSE(::yggdrasil_decision_forests::chrono_prof::kProjectionEvaluate);

  internal::ProjectionEvaluator evaluator(train_dataset, numerical_features);
  // The column-centric sweep below walks raw column pointers, which exist
  // only for the default column-major VerticalDataset layout. Alternate
  // layouts (row-major mirror) are not supported yet.
  if (!evaluator.IsColumnMajor()) {
    return absl::UnimplementedError(
        "The depthwise-1-pass oblique kernel requires the column-major "
        "VerticalDataset layout. Alternate dataset layouts (e.g. "
        "--config=row_major_dataset_layout with --dataset_layout=row) are "
        "not supported with --config=depthwise_1_pass yet.");
  }

  const size_t N = selected_examples_per_node.size();
  if (N == 0) return absl::OkStatus();

  /* #region Performance minions. All these take <10% of AP time */

  // Only used to size the col_count histogram below. Taken from the feature
  // indices, not ncol(): the label is not always the rightmost column (it is
  // column 0 in HIGGS_with_header.csv, last in the synthetic specs).
  const int max_attr =
      *std::max_element(numerical_features.begin(), numerical_features.end());

#ifndef DW1_COLWALK_CONTROL
  size_t total_rows = 0;
  for (size_t n = 0; n < N; ++n) {
    total_rows += selected_examples_per_node[n].size();
  }
#endif

  // ── Phase 1: PreSize --- ~5% of AP time
  // Slab pre-size (zero-init: the column sweep accumulates)
  {
    CHRONO_SCOPE_AP(::yggdrasil_decision_forests::chrono_prof::kDw1PreSize);

    // Reserve write space (0-filled). ||nodes|| * ||projections|| * (each node's n. examples)
    for (size_t n = 0; n < N; ++n) {
      // Accounts for variable n. projections / node
      const size_t slab =
          selected_examples_per_node[n].size() * projections_per_node[n].size();
      out_projected[n].assign(slab, 0.f);
    }
  }

  // ── Phase 2: Sweep. Takes the majority of ApplyProjection time
  {
    CHRONO_SCOPE_AP(::yggdrasil_decision_forests::chrono_prof::kDw1Sweep);

    // Column-centric sweep: bucket references by column, then walk the
    // touched columns ascending.
    // Begin/end (not RAII) because the buffers below outlive the setup phase.
    CHRONO_BEGIN_AP(dw1_sweep_setup);
    std::vector<ColEntry> entries;
    std::vector<ColEntry> sorted;
    std::vector<int32_t> touched_cols;    // touched column ids, sorted ascending
    std::vector<int32_t> col_count(static_cast<size_t>(max_attr) + 1, 0);

    // Fill in entries quadruple: (node, proj, col_id, weight). <= 2% of AP time
    for (size_t node = 0; node < N; ++node) {      
      for (size_t p = 0; p < projections_per_node[node].size(); ++p) {
        for (const auto& feat : projections_per_node[node][p]) {
          // feat.weight can just be sampled here. No need to be stored. But probably memory size is low
          // Also will need to be sampled in regular ApplyProjection for nodes that disqualify from DW1
          entries.push_back({static_cast<int32_t>(node), static_cast<int32_t>(p),
                             feat.attribute_idx, feat.weight});
          
          // Why?
          if (col_count[feat.attribute_idx]++ == 0) {
            touched_cols.push_back(feat.attribute_idx);
          }
        }
      }
    }
    hwy::VQSort(touched_cols.data(), touched_cols.size(), hwy::SortAscending());

    // Sort entries by column: nodes that share columns get "scheduled" together.
    // By itself it gives column cache locality
    size_t offset = 0;
    for (const int32_t col_id : touched_cols) {
      const int32_t cnt = col_count[col_id];
      col_count[col_id] = static_cast<int32_t>(offset);
      offset += cnt;
    }
    sorted.resize(entries.size());
    for (const auto& e : entries) {
      sorted[col_count[e.col]++] = e;
    }

    // Print stats about row occupancy and column hit count
    if constexpr (kDw1DebugStats) {
// assumes Single threaded training
    static int stats_tree = 0;
    {
      static int32_t prev_depth = -1;
      if (current_depth <= prev_depth) ++stats_tree;
      prev_depth = current_depth;
    }

    // ── Per-node example counts, appended to DW1_NODE_SIZES_FILE (default
    // dw1_node_sizes.txt), one line per node per depth. DW1_NODE_SIZES_DEPTH
    // picks a single depth; -1 turns it off.
    {
      // -2 = every depth (the default), -1 = off, >=0 = that depth only.
      static const int32_t kNodeSizesDepth = [] {
        const char* e = std::getenv("DW1_NODE_SIZES_DEPTH");
        return e == nullptr ? -2 : static_cast<int32_t>(std::atoi(e));
      }();
      static const std::string kNodeSizesFile = [] {
        const char* e = std::getenv("DW1_NODE_SIZES_FILE");
        return std::string(e == nullptr ? "dw1_node_sizes.txt" : e);
      }();
      if (kNodeSizesDepth == -2 || current_depth == kNodeSizesDepth) {
        size_t depth_examples = 0;
        for (size_t n = 0; n < N; ++n) {
          depth_examples += selected_examples_per_node[n].size();
        }
        std::ofstream f(kNodeSizesFile, std::ios::app);
        f << "----DEPTH " << current_depth << "---- tree=" << stats_tree
          << " nodes=" << N << " examples=" << depth_examples << "\n";
        for (size_t n = 0; n < N; ++n) {
          f << "n" << n << ": " << selected_examples_per_node[n].size()
            << " examples\n";
        }
      }
    }

    // ── Per-depth column reference stats: per column, #nodes of this depth
    // referencing it and Σ of those nodes' row counts (a node counts once).
    // DW1_COL_STATS=0|summary, DW1_COL_STATS_ALL=1 adds untouched columns.
    {
      static const std::string kColStatsMode = [] {
        const char* e = std::getenv("DW1_COL_STATS");
        return std::string(e == nullptr ? "" : e);
      }();
      static const bool kColStats = kColStatsMode != "0";
      static const bool kColStatsPerCol = kColStats && kColStatsMode != "summary";
      static const bool kColStatsAll = [] {
        const char* e = std::getenv("DW1_COL_STATS_ALL");
        return e != nullptr && std::string(e) == "1";
      }();
      static const std::string kColShareFile = [] {
        const char* e = std::getenv("DW1_COL_SHARE_OUT");
        return std::string(e == nullptr ? "" : e);
      }();
      if (kColStats || !kColShareFile.empty()) {
        size_t depth_examples = 0;
        for (size_t n = 0; n < N; ++n) {
          depth_examples += selected_examples_per_node[n].size();
        }
        if (kColStatsPerCol) {
          std::cout << "\n----DEPTH " << current_depth
                    << "---- [DW1 colstats] tree=" << stats_tree
                    << " nodes=" << N << " examples=" << depth_examples
                    << " touched_cols=" << touched_cols.size() << "/"
                    << numerical_features.size() << " refs=" << entries.size()
                    << "\n\n";
        }

        size_t pairs = 0;       // Σ_c distinct nodes referencing c
        size_t useful = 0;      // Σ_c rows of those nodes
        size_t nodewise = 0;    // Σ_refs rows of the referencing node
        size_t shared_cols = 0; // columns referenced by >= 2 nodes
        int32_t max_nodes_col = 0;

        size_t next_touched = 0;  // walks `touched` to fill untouched gaps
        size_t slice_begin = 0;   // start of the current column's slice
        for (const int32_t c : touched_cols) {
          if (kColStatsPerCol && kColStatsAll) {
            for (int32_t u = (next_touched == 0 ? 0 : touched_cols[next_touched - 1] + 1);
                 u < c; ++u) {
              std::cout << "c" << u << ": 0\n";
            }
          }
          ++next_touched;
          const size_t slice_end = static_cast<size_t>(col_count[c]);
          size_t examples = 0;
          int32_t nodes = 0, prev_node = -1;
          for (size_t k = slice_begin; k < slice_end; ++k) {
            const size_t rows_k =
                selected_examples_per_node[sorted[k].node].size();
            nodewise += rows_k;  // every ref: the stock path re-gathers
            if (sorted[k].node != prev_node) {  // slice is node-major
              ++nodes;
              examples += rows_k;
              prev_node = sorted[k].node;
            }
          }
          pairs += static_cast<size_t>(nodes);
          useful += examples;
          if (nodes >= 2) ++shared_cols;
          max_nodes_col = std::max(max_nodes_col, nodes);
          if (kColStatsPerCol) {
            std::cout << "c" << c << ": " << nodes << " nodes, " << examples
                      << " examples\n";
          }
          slice_begin = slice_end;
        }
        if (kColStatsPerCol && kColStatsAll && !touched_cols.empty()) {
          for (int32_t u = touched_cols.back() + 1; u <= max_attr; ++u) {
            std::cout << "c" << u << ": 0\n";
          }
        }

        const size_t swept = touched_cols.size() * depth_examples;
        const double share =
            touched_cols.empty() ? 0.0
                            : static_cast<double>(pairs) / touched_cols.size();
        const double eff =
            swept == 0 ? 0.0 : static_cast<double>(useful) / swept;
        const double amort =
            swept == 0 ? 0.0 : static_cast<double>(nodewise) / swept;
        if (kColStats) {
          std::cout << "[DW1 colshare] tree=" << stats_tree
                    << " depth=" << current_depth << " nodes=" << N
                    << " rows=" << depth_examples << " cols=" << touched_cols.size()
                    << "/" << numerical_features.size()
                    << " refs=" << entries.size() << " pairs=" << pairs
                    << " share=" << share << " shared_cols=" << shared_cols
                    << " max_nodes_col=" << max_nodes_col
                    << " useful=" << useful << " swept=" << swept
                    << " eff=" << eff << " nodew=" << nodewise
                    << " amort=" << amort << "\n"
                    << std::flush;
        }
        if (!kColShareFile.empty()) {
          // The first write of the process truncates so a run starts from a
          // clean file.
          static bool share_header = false;
          std::ofstream f(kColShareFile, share_header ? std::ios::app
                                                      : std::ios::trunc);
          if (!share_header) {
            f << "tree,depth,nodes,rows,cols_touched,num_features,refs,pairs,"
                 "share,shared_cols,max_nodes_col,useful,swept,eff,nodewise,"
                 "amort\n";
            share_header = true;
          }
          f << stats_tree << "," << current_depth << "," << N << ","
            << depth_examples << "," << touched_cols.size() << ","
            << numerical_features.size() << "," << entries.size() << ","
            << pairs << "," << share << "," << shared_cols << ","
            << max_nodes_col << "," << useful << "," << swept << "," << eff
            << "," << nodewise << "," << amort << "\n";
        }
      }
    }
    }

/* #endregion */

/* #region SHARED_ROWS */

#ifndef DW1_COLWALK_CONTROL
    // Read each touched column in ONE depthwise bag pass and scatter each value to the 
    // (node,projection) addesses that reference the column. Trades sparse reads for sparse writes.

    // Where the first element of ref_proj that node n has
    std::vector<int32_t> node_ref_off(N, -1);  // per node: start in ref_*, else -1
    // How many projections of node n reference column c
    std::vector<int32_t> node_ref_cnt(N, 0);   // per node: #refs to current column
    
    // How many examples of node n have been processed so far
    std::vector<int32_t> node_local(N, 0);     // per node: rows seen this column

    std::vector<int32_t> ref_proj;             // current column's (proj, weight)
    std::vector<float> ref_w;                  //   runs, grouped by node
    std::vector<int32_t> cols_nodes;          // node ids that have this column

    const UnsignedExampleIdx* bag = bag_state->bag.data();
    // AdvanceDepthBag always sizes node_of_bag to the bag (all zeros for a
    // single-node batch), so the kernel reads it unconditionally and is
    // agnostic of batch shape/depth.
    const uint32_t* nob = bag_state->node_of_bag.data();

    CHRONO_END_AP(dw1_sweep_setup,
              ::yggdrasil_decision_forests::chrono_prof::kDw1SweepSetup);

    CHRONO_BEGIN_AP(dw1_sweep_colwalk); // ~86% of AP runtime
    {
      size_t pos = 0;

      for (const int32_t col_id : touched_cols) {
        CHRONO_BEGIN_AP(dw1_colwalk_group_by_node); // This operation is free - ~0% cost

        const size_t slice_begin = pos;
        const size_t slice_end = static_cast<size_t>(col_count[col_id]);
        pos = slice_end;

        // Group column c's refs by node. Entries were bucketed node-major, so
        // within the slice the node id is non-decreasing and each node's refs
        // form one contiguous run [off, off+cnt) in ref_proj / ref_w.
        ref_proj.resize(slice_end - slice_begin);
        ref_w.resize(slice_end - slice_begin);
        cols_nodes.clear();
        int32_t prev_node = -1;
        
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
            cols_nodes.push_back(n);
            prev_node = n;
          }
          ++node_ref_cnt[n];
        }
        CHRONO_END_AP(dw1_colwalk_group_by_node,
                   ::yggdrasil_decision_forests::chrono_prof::kDw1ColWalkGroupByNode);


        // One ascending pass over the depth bag: dense read of col, scatter
        // write of each contribution to its node's projections referencing c.
        CHRONO_BEGIN_AP(dw1_colwalk_bag_scatter);
        const float* col_data = evaluator.AttributeData(col_id);

        for (size_t s = 0; s < total_rows; ++s) { // loop over examples
          const int32_t n = static_cast<int32_t>(nob[s]);
          const int32_t off = node_ref_off[n];
          if (off < 0) continue;  // owning node has no projection on column c

          const float v = col_data[bag[s]];

          const int32_t cnt = node_ref_cnt[n];
          // Bag and each node's selected_examples are sorted ascending, so rows
          // arrive in slab order and this running counter is the local slot.
          // It advances once per node row, so it stays in lockstep.
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

        for (const int32_t n : cols_nodes) node_ref_off[n] = -1;
      }
    }
    CHRONO_END_AP(dw1_sweep_colwalk,
               ::yggdrasil_decision_forests::chrono_prof::kDw1SweepColWalk);
#else
/* #endregion */

/* #region Col sharing DW1 - Control for Column cache locality */
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
#endif  // !DW1_COLWALK_CONTROL
  }

  return absl::OkStatus();
}

}  // namespace yggdrasil_decision_forests::model::decision_tree

#endif  // DEPTHWISE_1_PASS
