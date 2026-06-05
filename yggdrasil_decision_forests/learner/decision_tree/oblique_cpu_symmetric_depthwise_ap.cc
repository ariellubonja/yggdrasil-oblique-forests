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

absl::Status ApplyProjectionsSymmetricDepthwiseAP(
    const dataset::VerticalDataset& train_dataset,
    const google::protobuf::RepeatedField<int32_t>& numerical_features,
    absl::Span<const absl::Span<const UnsignedExampleIdx>>
        selected_examples_per_node,
    absl::Span<const internal::Projection> shared_projections,
    absl::Span<std::vector<float>> out_projected) {
  // Outer scope: total ProjectionEvaluate budget. Sub-phase scopes below
  // partition this into BuildBag / SortBag / Sweep so parallel_chrono.py
  // can break ApplyProjection down per depth.
  CHRONO_SCOPE(::yggdrasil_decision_forests::chrono_prof::kProjectionEvaluate);

  const size_t N = selected_examples_per_node.size();
  DCHECK_EQ(N, out_projected.size());
  if (N == 0) return absl::OkStatus();

  const size_t K = shared_projections.size();
  if (K == 0) {
    for (size_t n = 0; n < N; ++n) out_projected[n].clear();
    return absl::OkStatus();
  }

  // ── Phase 1: BuildBag ─────────────────────────────────────────────
  // Per-node row counts + slab pre-size, then concat per-node
  // selected_examples (+ parallel node-id) into one flat bag.
  std::vector<size_t> rows_n(N);
  size_t bag_size = 0;
  std::vector<UnsignedExampleIdx> bag;
  std::vector<uint32_t> node_of_bag;
  {
    CHRONO_SCOPE(::yggdrasil_decision_forests::chrono_prof::kSymBuildBag);
    for (size_t n = 0; n < N; ++n) {
      rows_n[n] = selected_examples_per_node[n].size();
      bag_size += rows_n[n];
      out_projected[n].assign(K * rows_n[n], 0.f);
    }
    if (bag_size == 0) return absl::OkStatus();

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

  // ── Phase 2: SortBag (Highway VQSort over packed K32V32 pairs) ────
  // Sort bag by example_idx (carrying node_of_bag along) so that Sweep's
  // K column reads are stride-1 in example index.
  //
  // We pack each (example_idx, node_id) pair into hwy::K32V32 (key=ex,
  // value=node), VQSort by key ascending, then unpack. The buffer is
  // thread-local to amortize the bag_size-sized allocation / first-touch
  // pages across depth calls and across trees on the same thread.
  //
  // VQSort is unstable, but tie-breaking is invariant here: by the BFS
  // depth-cohort property every copy of one example_idx lives in the
  // same node, so all ties carry the same value field — reordering
  // within a tie group is a no-op. The previous code used stable_sort;
  // the materialized sorted bag here is bitwise identical for our
  // workload (modulo the trivial value-equal reordering).
  //
  // The previous implementation tried a counting sort over [0, total_n)
  // with thread-local count_at / node_at. It beat stable_sort by ~3.5×
  // on HIGGS but had a fixed O(total_n) walk that lost at deep depths,
  // so a dense/sparse gate was needed. VQSort scales with bag_size, so
  // one code path handles every depth — and beats counting sort across
  // the board because Highway's SIMD radix kernel is faster per element
  // than the scatter+walk Counting needs.
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

  // ── Phase 3: Sweep ────────────────────────────────────────────────
  // K bag-wide sweeps. For each projection k:
  //  - Hoist per-item col pointers + weights + NA values out of the i-loop.
  //  - Walk sorted bag in stride-1 order. Per example: compute value, route
  //    to its owning node's slab via a write_cursor[].
  {
    CHRONO_SCOPE(::yggdrasil_decision_forests::chrono_prof::kSymSweep);
    internal::ProjectionEvaluator evaluator(train_dataset, numerical_features);

    std::vector<uint32_t> write_cursor(N);
    for (size_t k = 0; k < K; ++k) {
      std::fill(write_cursor.begin(), write_cursor.end(), 0u);

      const auto& proj = shared_projections[k];
      const size_t M = proj.size();
      if (M == 0) continue;

      std::vector<const float*> col_ptrs(M);
      std::vector<float> ws(M);
#ifdef ENABLE_APPLYPROJECTION_ISNAN
      std::vector<float> nas(M);
#endif
      for (size_t m = 0; m < M; ++m) {
        col_ptrs[m] = evaluator.AttributeData(proj[m].attribute_idx);
        ws[m] = proj[m].weight;
#ifdef ENABLE_APPLYPROJECTION_ISNAN
        nas[m] = evaluator.NaReplacementValue(proj[m].attribute_idx);
#endif
      }

      for (size_t i = 0; i < bag_size; ++i) {
        const UnsignedExampleIdx ex = bag[i];
        float value = 0.f;
        for (size_t m = 0; m < M; ++m) {
          float v = col_ptrs[m] != nullptr
                        ? col_ptrs[m][ex]
                        : evaluator.AttributeValue(proj[m].attribute_idx, ex);
#ifdef ENABLE_APPLYPROJECTION_ISNAN
          if (std::isnan(v)) v = nas[m];
#endif
          value += ws[m] * v;
        }
        const uint32_t n = node_of_bag[i];
        const uint32_t pos = write_cursor[n]++;
        out_projected[n][k * rows_n[n] + pos] = value;
      }
    }
  }

  return absl::OkStatus();
}

}  // namespace yggdrasil_decision_forests::model::decision_tree

#endif  // SYMMETRIC_DEPTHWISE_AP
