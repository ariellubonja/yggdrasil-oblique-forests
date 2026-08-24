// Benchmark harness for the May-2025 upstream YDF baseline (branch
// ydf-may-2025). Port of the fork's examples/train_oblique_forest.cc: same
// flag names, same trunk generator (bit-identical datasets), same log lines
// (parse_log_to_csv.py anchors), trimmed to what this code base supports.
// Not available here (upstream has no such code): the Dynamic histogram split
// types, --dynamic_split_threshold, --dataset_layout=row, single-pass CSV
// load, RM_MAX_ROWS / NO_EARLY_EXIT shortcuts.

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <memory>
#include <random>
#include <string>
#include <thread>
#include <utility>

#include "absl/flags/flag.h"
#include "absl/flags/parse.h"
#include "absl/status/statusor.h"

#include "yggdrasil_decision_forests/dataset/data_spec.h"
#include "yggdrasil_decision_forests/dataset/data_spec.pb.h"
#include "yggdrasil_decision_forests/dataset/data_spec_inference.h"
#include "yggdrasil_decision_forests/dataset/vertical_dataset.h"
#include "yggdrasil_decision_forests/dataset/vertical_dataset_io.h"
#include "yggdrasil_decision_forests/learner/decision_tree/oblique.h"
#include "yggdrasil_decision_forests/learner/gradient_boosted_trees/gradient_boosted_trees.pb.h"
#include "yggdrasil_decision_forests/learner/learner_library.h"
#include "yggdrasil_decision_forests/learner/random_forest/random_forest.pb.h"
#include "yggdrasil_decision_forests/metric/metric.h"
#include "yggdrasil_decision_forests/metric/metric.pb.h"
#include "yggdrasil_decision_forests/metric/report.h"
#include "yggdrasil_decision_forests/model/model_library.h"
#include "yggdrasil_decision_forests/utils/logging.h"
#include "yggdrasil_decision_forests/utils/random.h"

/* #region ABSL Flags (names match the fork's harness) */
ABSL_FLAG(std::string, input_mode, "", "Data input mode: csv or trunk.");
ABSL_FLAG(std::string, train_csv, "./benchmarks/data/HIGGS_with_header.csv",
          "Path to training CSV file (for csv mode).");
ABSL_FLAG(std::string, test_csv, "",
          "Optional held-out test CSV (csv mode only).");
ABSL_FLAG(std::string, label_col, "class", "Name of label column.");
ABSL_FLAG(std::string, model_out_dir, "",
          "Path to output trained model directory (optional).");
ABSL_FLAG(int, num_threads, -1, "Number of threads to use.");
ABSL_FLAG(int, num_trees, 240, "Number of trees in the random forest.");
ABSL_FLAG(int, tree_depth, -1, "Maximum depth of trees (-1 for unlimited).");

ABSL_FLAG(std::string, feature_split_type, "Oblique",
          "'Axis Aligned' or 'Oblique'.");
ABSL_FLAG(int, max_num_projections, 1000,
          "Maximum number of projections for oblique splits.");
ABSL_FLAG(float, projection_density_factor, 1.5f,
          "Projection density factor.");
ABSL_FLAG(float, num_projections_exponent, .5,
          "Exponent to determine number of projections.");

ABSL_FLAG(std::string, ensemble_method, "Bagging",
          "'Bagging' (Random Forest) or 'Boosting' (GBT).");
ABSL_FLAG(float, shrinkage, 0.1f, "Learning rate for boosting.");
ABSL_FLAG(std::string, growing_strategy, "Local",
          "'Local' or 'GlobalBestFirst'.");
ABSL_FLAG(bool, bootstrap_training_dataset, true,
          "Whether to use bootstrap sampling (Bagging only).");
ABSL_FLAG(bool, compute_oob_performances, false,
          "Whether to compute out-of-bag performances.");

ABSL_FLAG(int64_t, rows, 4096, "Number of examples (trunk mode).");
ABSL_FLAG(int, cols, 4096, "Number of numerical features (trunk mode).");
ABSL_FLAG(uint32_t, seed, 1, "PRNG seed (trunk generation and training).");

// Upstream May-2025 has EXACT, HISTOGRAM_RANDOM and HISTOGRAM_EQUAL_WIDTH
// only; the Dynamic variants are fork-side inventions and rejected here.
ABSL_FLAG(std::string, numerical_split_type, "Exact",
          "'Exact', 'Random' or 'Equal Width'.");
ABSL_FLAG(int, histogram_num_bins, 64, "Number of bins for histogram splits.");

using namespace yggdrasil_decision_forests;
/* #endregion */

/* #region Trunk synthetic dataset — byte-identical to the fork's generator */
dataset::proto::DataSpecification MakeSyntheticSpec(
    int cols, int64_t rows, const std::string& label_col) {
  dataset::proto::DataSpecification spec;
  for (int c = 0; c < cols; ++c) {
    auto* f = spec.add_columns();
    f->set_name("x" + std::to_string(c));
    f->set_type(dataset::proto::NUMERICAL);
  }
  auto* lbl = spec.add_columns();
  lbl->set_name(label_col);
  lbl->set_type(dataset::proto::CATEGORICAL);
  lbl->mutable_categorical()->set_number_of_unique_values(3);  // OOD, 1, 2
  lbl->mutable_categorical()->set_is_already_integerized(true);
  spec.set_created_num_rows(rows);
  return spec;
}

dataset::VerticalDataset MakeTrunkDataset(
    const dataset::proto::DataSpecification& spec, int64_t rows, int cols,
    uint32_t seed) {
  dataset::VerticalDataset ds;
  ds.set_data_spec(spec);
  CHECK_OK(ds.CreateColumnsFromDataspec());
  ds.Resize(rows);
  using RNG = std::minstd_rand;

  constexpr int kNInformative = 256;
  const int ninform = std::min(kNInformative, cols);

  std::vector<float> mu0(cols, 0.f), mu1(cols, 0.f);
  for (int j = 0; j < ninform; ++j) {
    const float f = 1.f / std::sqrt(float(j + 1));
    mu0[j] = -f;
    mu1[j] = f;
  }

  for (int j = 0; j < cols; ++j) {
    std::seed_seq seq{seed, static_cast<uint32_t>(j)};
    RNG rng(seq);
    std::normal_distribution<float> normal(0.0f, 1.0f);
    auto* col = ds.MutableColumnWithCast<
        dataset::VerticalDataset::NumericalColumn>(j)->mutable_values();
    for (int64_t i = 0; i < rows; ++i) {
      const bool cls1 = (i >= rows / 2);
      const float mean = cls1 ? mu1[j] : mu0[j];
      (*col)[i] = mean + normal(rng);
    }
  }

  auto* y = ds.MutableColumnWithCast<
      dataset::VerticalDataset::CategoricalColumn>(cols)->mutable_values();
  for (int64_t i = 0; i < rows; ++i) (*y)[i] = (i >= rows / 2) ? 2 : 1;

  return ds;
}
/* #endregion */

int main(int argc, char** argv) {
  const auto t_main_start = std::chrono::high_resolution_clock::now();
  absl::ParseCommandLine(argc, argv);
  const auto mode = absl::GetFlag(FLAGS_input_mode);
  if (mode.empty()) {
    std::cerr << "Error: --input_mode is required. Use csv or trunk.\n";
    return 1;
  }

  dataset::proto::DataSpecification data_spec;
  std::unique_ptr<dataset::VerticalDataset> tf_ds;
  dataset::VerticalDataset* ds_ptr = nullptr;
  std::string csv_path;
  std::string label_col;

  if (mode == "csv") {
    csv_path = absl::GetFlag(FLAGS_train_csv);
    label_col = absl::GetFlag(FLAGS_label_col);
    if (csv_path.empty() || label_col.empty()) {
      std::cerr << "Error: --train_csv and --label_col required in csv mode.\n";
      return 1;
    }
    LOG(INFO) << "Inferring DataSpec from CSV: " << csv_path;
    dataset::proto::DataSpecificationGuide guide;
    auto* col_guide = guide.add_column_guides();
    col_guide->set_column_name_pattern(label_col);
    col_guide->set_type(dataset::proto::ColumnType::CATEGORICAL);
    dataset::CreateDataSpec("csv:" + csv_path,
                            /*require_same_dataset_fields=*/false, guide,
                            &data_spec);
  } else if (mode == "trunk") {
    LOG(INFO) << "Generating trunk synthetic dataset: rows="
              << absl::GetFlag(FLAGS_rows)
              << ", cols=" << absl::GetFlag(FLAGS_cols);
    label_col = "y";
    data_spec = MakeSyntheticSpec(absl::GetFlag(FLAGS_cols),
                                  absl::GetFlag(FLAGS_rows), label_col);
    auto ds = MakeTrunkDataset(data_spec, absl::GetFlag(FLAGS_rows),
                               absl::GetFlag(FLAGS_cols),
                               absl::GetFlag(FLAGS_seed));
    tf_ds = std::make_unique<dataset::VerticalDataset>(std::move(ds));
    ds_ptr = tf_ds.get();
  } else {
    std::cerr << "Unknown input_mode: " << mode << ". Use csv or trunk.\n";
    return 1;
  }

  // Configure learner.
  model::proto::TrainingConfig train_config;
  train_config.set_task(model::proto::Task::CLASSIFICATION);
  train_config.set_label(label_col);

  model::proto::DeploymentConfig deploy_config;
  const int num_threads_flag = absl::GetFlag(FLAGS_num_threads);
  if (num_threads_flag > 0) {
    LOG(INFO) << "Running with " << num_threads_flag
              << " threads, as requested.";
    deploy_config.set_num_threads(num_threads_flag);
  } else if (num_threads_flag == -1) {
    unsigned int cpu_count = std::thread::hardware_concurrency();
    if (cpu_count == 0) cpu_count = 1;
    LOG(INFO) << "-1 (automatic) threads requested. " << cpu_count
              << " threads set.";
    deploy_config.set_num_threads(cpu_count);
  } else {
    std::cerr << "Invalid --num_threads: " << num_threads_flag << "\n";
    return 1;
  }

  const std::string ensemble_method = absl::GetFlag(FLAGS_ensemble_method);
  model::decision_tree::proto::DecisionTreeTrainingConfig* dt_config;
  if (ensemble_method == "Bagging") {
    train_config.set_learner("RANDOM_FOREST");
    auto& rf = *train_config.MutableExtension(
        model::random_forest::proto::random_forest_config);
    rf.set_num_trees(absl::GetFlag(FLAGS_num_trees));
    rf.set_bootstrap_training_dataset(
        absl::GetFlag(FLAGS_bootstrap_training_dataset));
    rf.set_bootstrap_size_ratio(1.0);
    rf.set_winner_take_all_inference(false);
    rf.set_compute_oob_performances(
        absl::GetFlag(FLAGS_compute_oob_performances));
    dt_config = rf.mutable_decision_tree();
  } else if (ensemble_method == "Boosting") {
    train_config.set_learner("GRADIENT_BOOSTED_TREES");
    auto& gbt = *train_config.MutableExtension(
        model::gradient_boosted_trees::proto::gradient_boosted_trees_config);
    gbt.set_num_trees(absl::GetFlag(FLAGS_num_trees));
    gbt.set_shrinkage(absl::GetFlag(FLAGS_shrinkage));
    gbt.set_validation_set_ratio(0);
    gbt.set_early_stopping(
        model::gradient_boosted_trees::proto::
            GradientBoostedTreesTrainingConfig::NONE);
    dt_config = gbt.mutable_decision_tree();
  } else {
    std::cerr << "Unknown ensemble_method: " << ensemble_method << "\n";
    return 1;
  }

  train_config.set_random_seed(absl::GetFlag(FLAGS_seed));

  int tree_depth = absl::GetFlag(FLAGS_tree_depth);
  if (ensemble_method == "Boosting" && tree_depth == -1) tree_depth = 6;
  dt_config->set_max_depth(tree_depth);
  dt_config->set_min_examples(ensemble_method == "Boosting" ? 5 : 1);

  const auto growing_strategy = absl::GetFlag(FLAGS_growing_strategy);
  if (growing_strategy == "GlobalBestFirst") {
    dt_config->mutable_growing_strategy_best_first_global()->set_max_num_nodes(
        -1);
  } else if (growing_strategy != "Local") {
    std::cerr << "Unknown growing_strategy: " << growing_strategy << "\n";
    return 1;
  }

  const std::string feature_split_type =
      absl::GetFlag(FLAGS_feature_split_type);
  if (feature_split_type == "Oblique") {
    LOG(INFO) << "Configuring oblique splits";
    auto* sos = dt_config->mutable_sparse_oblique_split();
    sos->set_max_num_projections(absl::GetFlag(FLAGS_max_num_projections));
    sos->set_projection_density_factor(
        absl::GetFlag(FLAGS_projection_density_factor));
    sos->set_num_projections_exponent(
        absl::GetFlag(FLAGS_num_projections_exponent));
  } else if (feature_split_type == "Axis Aligned") {
    LOG(INFO) << "Using axis-aligned splits";
    auto* sos = dt_config->mutable_sparse_oblique_split();
    sos->set_max_num_projections(absl::GetFlag(FLAGS_max_num_projections));
    sos->set_num_projections_exponent(
        absl::GetFlag(FLAGS_num_projections_exponent));
    int num_numerical_features = 0;
    for (int i = 0; i < data_spec.columns_size(); ++i) {
      if (data_spec.columns(i).type() == dataset::proto::NUMERICAL) {
        num_numerical_features++;
      }
    }
    const int num_candidates = model::decision_tree::GetNumProjections(
        *dt_config, num_numerical_features);
    dt_config->set_num_candidate_attributes(num_candidates);
    dt_config->clear_sparse_oblique_split();
    LOG(INFO) << "Num Candidate Attributes: " << num_candidates;
  } else {
    std::cerr << "Unknown feature_split_type: " << feature_split_type << "\n";
    return 1;
  }

  auto* numerical_split = dt_config->mutable_numerical_split();
  const std::string hist_type = absl::GetFlag(FLAGS_numerical_split_type);
  if (hist_type == "Exact") {
    numerical_split->set_type(
        model::decision_tree::proto::NumericalSplit::EXACT);
    LOG(INFO) << "Using exact splitting";
  } else if (hist_type == "Random") {
    numerical_split->set_type(
        model::decision_tree::proto::NumericalSplit::HISTOGRAM_RANDOM);
    numerical_split->set_num_candidates(absl::GetFlag(FLAGS_histogram_num_bins));
    LOG(INFO) << "Using histogram splitting: Random with "
              << absl::GetFlag(FLAGS_histogram_num_bins) << " bins";
  } else if (hist_type == "Equal Width") {
    numerical_split->set_type(
        model::decision_tree::proto::NumericalSplit::HISTOGRAM_EQUAL_WIDTH);
    numerical_split->set_num_candidates(absl::GetFlag(FLAGS_histogram_num_bins));
    LOG(INFO) << "Using histogram splitting: Equal Width with "
              << absl::GetFlag(FLAGS_histogram_num_bins) << " bins";
  } else {
    std::cerr << "Unknown numerical_split_type for the May-2025 baseline: "
              << hist_type << ". Use 'Exact', 'Random' or 'Equal Width'.\n";
    return 1;
  }

  std::unique_ptr<model::AbstractLearner> learner;
  CHECK_OK(model::GetLearner(train_config, &learner, deploy_config));

  {
    const std::chrono::duration<double> init_dur =
        std::chrono::high_resolution_clock::now() - t_main_start;
    LOG(INFO) << "train_oblique_forest wall-time - Loading/Init (pre-train): "
              << init_dur.count() << "s";
  }
  const auto start = std::chrono::high_resolution_clock::now();
  absl::StatusOr<std::unique_ptr<model::AbstractModel>> model_or;
  if (mode == "csv") {
    model_or = learner->TrainWithStatus("csv:" + csv_path, data_spec);
  } else {
    model_or = learner->TrainWithStatus(*ds_ptr);
  }
  if (!model_or.ok()) {
    std::cerr << "Training failed: " << model_or.status().message()
              << std::endl;
    return 1;
  }
  auto model_ptr = std::move(model_or.value());
  const std::chrono::duration<double> dur =
      std::chrono::high_resolution_clock::now() - start;
  LOG(INFO) << "train_oblique_forest wall-time - Training + Post-Processing: "
            << dur.count() << "s";

  const std::string out_dir = absl::GetFlag(FLAGS_model_out_dir);
  if (!out_dir.empty()) {
    const auto save_status = model::SaveModel(out_dir, *model_ptr);
    if (!save_status.ok()) {
      std::cerr << "Could not save model: " << save_status.message()
                << std::endl;
      return 1;
    }
    LOG(INFO) << "Model saved to: " << out_dir;
  }

  const std::string test_csv = absl::GetFlag(FLAGS_test_csv);
  if (!test_csv.empty()) {
    if (mode != "csv") {
      LOG(ERROR) << "--test_csv is only supported in csv mode; skipping.";
    } else {
      dataset::VerticalDataset test_ds;
      const absl::Status load_status = dataset::LoadVerticalDataset(
          "csv:" + test_csv, model_ptr->data_spec(), &test_ds);
      if (!load_status.ok()) {
        LOG(ERROR) << "Could not load test CSV '" << test_csv
                   << "': " << load_status.message();
      } else {
        metric::proto::EvaluationOptions eval_options;
        eval_options.set_task(model::proto::Task::CLASSIFICATION);
        utils::RandomEngine rnd(absl::GetFlag(FLAGS_seed));
        auto eval_or =
            model_ptr->EvaluateWithStatus(test_ds, eval_options, &rnd);
        if (!eval_or.ok()) {
          LOG(ERROR) << "Test-set evaluation failed: "
                     << eval_or.status().message();
        } else {
          const float test_acc = metric::Accuracy(eval_or.value());
          const float test_logloss = metric::LogLoss(eval_or.value());
          double test_auc = -1.0;
          const auto& eval_cls = eval_or.value().classification();
          if (eval_cls.rocs_size() > 0) {
            test_auc = eval_cls.rocs(eval_cls.rocs_size() - 1).auc();
          }
          LOG(INFO) << "Test-set evaluation on " << test_csv << " ("
                    << test_ds.nrow() << " rows): test-accuracy:" << test_acc
                    << " test-auc:" << test_auc
                    << " test-logloss:" << test_logloss;
        }
      }
    }
  }

  return 0;
}
