// Counts duplicate sparse-oblique projections per node with the real
// internal::SampleProjection (BINARY weights, no axis-aligned pass).
// Usage: projection_overlap <out.csv> <nodes> <name:F[:W]>...  (W = workers)

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <unordered_set>
#include <vector>

#include "yggdrasil_decision_forests/dataset/data_spec.pb.h"
#include "yggdrasil_decision_forests/learner/abstract_learner.pb.h"
#include "yggdrasil_decision_forests/learner/decision_tree/decision_tree.pb.h"
#include "yggdrasil_decision_forests/learner/decision_tree/oblique.h"
#include "yggdrasil_decision_forests/utils/random.h"

namespace ydf = yggdrasil_decision_forests;
namespace dt = ydf::model::decision_tree;

struct Profile {
  const char* name;
  float exponent;
  int max_projections;
  float density;
};

// kdd28_ydf and kdd28_hao of params/kdd28_params.json.
static const Profile kProfiles[] = {{"ydf", 2.f, 6000, 2.f},
                                    {"hao", 1.f, 1000, 1.5f}};

// Key of a projection. `canonical` flips signs so the first weight is +1
// (w and -w give the same set of splits).
static std::string Key(const dt::internal::Projection& p, bool canonical) {
  const bool flip = canonical && p.front().weight < 0;
  std::string key;
  for (const auto& item : p) {
    const bool negative = (item.weight < 0) != flip;
    key += std::to_string(item.attribute_idx);
    key += negative ? '-' : '+';
  }
  return key;
}

int main(int argc, char** argv) {
  if (argc < 4) {
    fprintf(stderr, "usage: %s <out.csv> <nodes> <name:F>...\n", argv[0]);
    return 2;
  }
  FILE* out = fopen(argv[1], "w");
  if (out == nullptr) {
    perror(argv[1]);
    return 2;
  }
  const int num_nodes = atoi(argv[2]);
  fprintf(out,
          "dataset,F_global,workers,F_local,profile,P,nodes,mean_nnz,singleton_draws,"
          "distinct_singletons,distinct_exact,distinct_signeq,"
          "dup_frac_exact,dup_frac_signeq,max_nnz\n");

  for (int a = 3; a < argc; a++) {
    const std::string arg = argv[a];
    const auto colon = arg.find(':');
    const std::string name = arg.substr(0, colon);
    const int global_features = atoi(arg.c_str() + colon + 1);
    const auto colon2 = arg.find(':', colon + 1);
    const int workers =
        colon2 == std::string::npos ? 1 : atoi(arg.c_str() + colon2 + 1);
    // Worker 0 of W with disjoint round-robin columns: ceil(F / W) of them.
    const int num_features = (global_features + workers - 1) / workers;

    ydf::dataset::proto::DataSpecification data_spec;
    ydf::model::proto::TrainingConfigLinking config_link;
    std::vector<int> features(num_features);
    for (int f = 0; f < num_features; f++) {
      data_spec.add_columns()->set_type(ydf::dataset::proto::NUMERICAL);
      config_link.add_numerical_features(f);
      features[f] = f;
    }

    for (const auto& profile : kProfiles) {
      dt::proto::DecisionTreeTrainingConfig dt_config;
      auto* so = dt_config.mutable_sparse_oblique_split();
      so->set_num_projections_exponent(profile.exponent);
      so->set_max_num_projections(profile.max_projections);
      so->set_projection_density_factor(profile.density);
      so->mutable_binary();
      // Count from the global F, split over workers (ProjectionsForWorker,
      // worker 0). Density from the local F (SampleNodeProjections).
      const int total = dt::GetNumProjections(dt_config, global_features);
      const int num_projections = total / workers + (total % workers > 0);
      const float density =
          std::clamp(so->projection_density_factor() / num_features, 0.f, 1.f);

      ydf::utils::RandomEngine random(1234);
      double sum_nnz = 0, sum_single = 0, sum_distinct_single = 0;
      double sum_exact = 0, sum_signeq = 0;
      size_t max_nnz = 0;
      dt::internal::Projection projection;
      for (int node = 0; node < num_nodes; node++) {
        std::unordered_set<std::string> exact, signeq, singles;
        for (int p = 0; p < num_projections; p++) {
          int8_t monotonic;
          dt::internal::SampleProjection(
              features, dt_config, data_spec, config_link, density,
              &projection, &monotonic, &random,
              /*exclude_single_feature=*/so->include_axis_aligned_splits());
          sum_nnz += projection.size();
          max_nnz = std::max(max_nnz, projection.size());
          exact.insert(Key(projection, false));
          const std::string canonical = Key(projection, true);
          signeq.insert(canonical);
          if (projection.size() == 1) {
            sum_single++;
            singles.insert(canonical);
          }
        }
        sum_exact += exact.size();
        sum_signeq += signeq.size();
        sum_distinct_single += singles.size();
      }
      const double draws = static_cast<double>(num_nodes) * num_projections;
      fprintf(out, "%s,%d,%d,%d,%s,%d,%d,%.4f,%.2f,%.2f,%.2f,%.2f,%.4f,%.4f,%zu\n",
              name.c_str(), global_features, workers, num_features, profile.name, num_projections,
              num_nodes, sum_nnz / draws, sum_single / num_nodes,
              sum_distinct_single / num_nodes, sum_exact / num_nodes,
              sum_signeq / num_nodes, 1.0 - sum_exact / draws,
              1.0 - sum_signeq / draws, max_nnz);
      fflush(out);
    }
  }
  fclose(out);
  return 0;
}
