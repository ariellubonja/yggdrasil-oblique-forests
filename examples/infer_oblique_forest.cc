// Inference counterpart of train_oblique_forest.cc: trains (or loads) a forest,
// then prints which serving engine YDF picks for it and why, with a focus on
// the two QuickScorer implementations (quick_scorer_extended.{h,cc} = Legacy
// AVX2 intrinsics, quick_scorer_extended_hwy.cc = Highway).
//
// When does each run (register_engines.cc):
//   * QuickScorer is compatible only with a GBT model that has no oblique
//     conditions, <= 64 leaves per tree, global-imputation
//     "is_higher" conditions and a binary-classification / regression / ranking
//     task. A Random Forest never uses it.
//   * Both QuickScorer engines are then compatible; the ranking between them is
//     fixed at COMPILE time: Highway wins iff YDF_USE_DYNAMIC_DISPATCH is
//     defined or the static HWY_TARGET is AVX2 or better (-march=haswell and
//     up); otherwise Legacy wins. Legacy only vectorizes when the binary has
//     __AVX2__ and the CPU supports it, else it runs the scalar sequential loop.
//
// Build / run:
//   bazel build -c opt //examples:infer_oblique_forest
//   bazel-bin/examples/infer_oblique_forest --input_mode=trunk \
//       --ensemble_method=Boosting --feature_split_type="Axis Aligned"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <limits>
#include <memory>
#include <optional>
#include <random>
#include <string>
#include <thread>
#include <vector>

#include "absl/base/config.h"
#include "absl/flags/flag.h"
#include "absl/flags/parse.h"
#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_format.h"
#include "absl/strings/str_join.h"
#include "hwy/highway.h"
#include "hwy/targets.h"

#include "yggdrasil_decision_forests/dataset/data_spec.h"
#include "yggdrasil_decision_forests/dataset/data_spec.pb.h"
#include "yggdrasil_decision_forests/dataset/data_spec_inference.h"
#include "yggdrasil_decision_forests/dataset/vertical_dataset.h"
#include "yggdrasil_decision_forests/dataset/vertical_dataset_io.h"
#include "yggdrasil_decision_forests/learner/gradient_boosted_trees/gradient_boosted_trees.pb.h"
#include "yggdrasil_decision_forests/learner/learner_library.h"
#include "yggdrasil_decision_forests/learner/random_forest/random_forest.pb.h"
#include "yggdrasil_decision_forests/model/decision_tree/decision_forest_interface.h"
#include "yggdrasil_decision_forests/model/decision_tree/decision_tree.h"
#include "yggdrasil_decision_forests/model/gradient_boosted_trees/gradient_boosted_trees.h"
#include "yggdrasil_decision_forests/model/model_library.h"
#include "yggdrasil_decision_forests/serving/decision_forest/quick_scorer_extended.h"
#include "yggdrasil_decision_forests/serving/decision_forest/register_engines.h"
#include "yggdrasil_decision_forests/serving/example_set.h"
#include "yggdrasil_decision_forests/serving/fast_engine.h"

// ----------------- Flags (same names as train_oblique_forest) -----------------
ABSL_FLAG(std::string, input_mode, "trunk",
          "Data input mode: 'csv' or 'trunk' (synthetic).");
ABSL_FLAG(std::string, train_csv, "",
          "Path to training CSV file (csv mode). Also used for inference "
          "unless --test_csv is given.");
ABSL_FLAG(std::string, test_csv, "",
          "Optional CSV to run inference on (csv mode).");
ABSL_FLAG(std::string, label_col, "class", "Label column name (csv mode).");
ABSL_FLAG(std::string, model_dir, "",
          "If set, load this model (e.g. written by train_oblique_forest "
          "--model_out_dir) instead of training one.");
ABSL_FLAG(int, num_threads, -1, "Number of training threads.");
ABSL_FLAG(int, num_trees, 100, "Number of trees.");
ABSL_FLAG(int, tree_depth, -1,
          "Max tree depth. -1 = 6 for Boosting, unlimited for Bagging. "
          "QuickScorer needs <= 64 leaves per tree.");
ABSL_FLAG(int, min_examples, -1,
          "Min examples per leaf. -1 = 5 for Boosting, 1 for Bagging.");
ABSL_FLAG(std::string, feature_split_type, "Oblique",
          "'Axis Aligned' or 'Oblique'.");
ABSL_FLAG(std::string, ensemble_method, "Boosting",
          "'Bagging' (Random Forest) or 'Boosting' (GBT).");
ABSL_FLAG(float, shrinkage, 0.1f, "GBT shrinkage.");
ABSL_FLAG(int64_t, rows, 100000, "Number of examples (trunk mode).");
ABSL_FLAG(int, cols, 32, "Number of numerical features (trunk mode).");
ABSL_FLAG(uint32_t, seed, 1, "Random seed.");
// Inference-only flags.
ABSL_FLAG(std::string, force_engine, "",
          "If set, also report on BuildFastEngine(<this engine name>).");
ABSL_FLAG(int, batch_size, 1024, "Examples per Predict() call.");
ABSL_FLAG(int, num_runs, 5, "Timing repetitions per engine (best is kept).");
ABSL_FLAG(int64_t, infer_rows, 100000,
          "Max number of dataset rows to run inference on.");

using namespace yggdrasil_decision_forests;

namespace {

using QsModel = serving::decision_forest::internal::QuickScorerExtendedModel;

const char kLegacy[] = "GradientBoostedTreesQuickScorerExtended";
const char kHighway[] = "GradientBoostedTreesQuickScorerExtendedHighway";

// Same data as train_oblique_forest's trunk generator (serial version).
dataset::VerticalDataset MakeTrunkDataset(
    const dataset::proto::DataSpecification& spec, int64_t rows, int cols,
    uint32_t seed) {
  dataset::VerticalDataset ds;
  ds.set_data_spec(spec);
  CHECK_OK(ds.CreateColumnsFromDataspec());
  ds.set_nrow(rows);
  for (int j = 0; j < cols; ++j) {
    const float f = j < 256 ? 1.f / std::sqrt(float(j + 1)) : 0.f;
    std::seed_seq seq{seed, static_cast<uint32_t>(j)};
    std::minstd_rand rng(seq);
    std::normal_distribution<float> normal(0.0f, 1.0f);
    auto* col = ds.MutableColumnWithCast<
                      dataset::VerticalDataset::NumericalColumn>(j)
                    ->mutable_values();
    col->resize(rows);
    for (int64_t i = 0; i < rows; ++i) {
      (*col)[i] = ((i >= rows / 2) ? f : -f) + normal(rng);
    }
  }
  auto* y = ds.MutableColumnWithCast<
                  dataset::VerticalDataset::CategoricalColumn>(cols)
                ->mutable_values();
  y->resize(rows);
  for (int64_t i = 0; i < rows; ++i) (*y)[i] = (i >= rows / 2) ? 2 : 1;
  return ds;
}

dataset::proto::DataSpecification MakeSyntheticSpec(int cols, int64_t rows) {
  dataset::proto::DataSpecification spec;
  for (int c = 0; c < cols; ++c) {
    auto* f = spec.add_columns();
    f->set_name("x" + std::to_string(c));
    f->set_type(dataset::proto::NUMERICAL);
  }
  auto* lbl = spec.add_columns();
  lbl->set_name("y");
  lbl->set_type(dataset::proto::CATEGORICAL);
  lbl->mutable_categorical()->set_number_of_unique_values(3);  // OOD, 1, 2
  lbl->mutable_categorical()->set_is_already_integerized(true);
  spec.set_created_num_rows(rows);
  return spec;
}

// What the compiler baked in: this decides Highway vs Legacy.
void PrintBuildInfo() {
  std::cout << "\n=== Build (decides Highway vs Legacy at compile time) ===\n";
  std::cout << "HWY_TARGET (static)        : " << hwy::TargetName(HWY_TARGET)
            << "  [u64 lanes = examples per QuickScorer iteration: "
            << hwy::HWY_NAMESPACE::Lanes(
                   hwy::HWY_NAMESPACE::ScalableTag<uint64_t>())
            << "]\n";
  std::cout << "HWY_TARGET <= HWY_AVX2     : "
            << (HWY_TARGET <= HWY_AVX2 ? "yes" : "no") << "\n";
#ifdef YDF_USE_DYNAMIC_DISPATCH
  std::cout << "YDF_USE_DYNAMIC_DISPATCH   : defined\n";
#else
  std::cout << "YDF_USE_DYNAMIC_DISPATCH   : not defined\n";
#endif
#ifdef __AVX2__
  std::cout << "__AVX2__ (Legacy SIMD path): compiled in\n";
#else
  std::cout << "__AVX2__ (Legacy SIMD path): NOT compiled in -> Legacy would "
               "run the scalar sequential loop\n";
#endif
  std::vector<std::string> cpu_targets;
  for (int64_t t = hwy::SupportedTargets(); t != 0; t &= t - 1) {
    cpu_targets.push_back(hwy::TargetName(t & -t));
  }
  std::cout << "CPU-supported hwy targets  : "
            << absl::StrJoin(cpu_targets, " ") << "\n";
#if defined(YDF_USE_DYNAMIC_DISPATCH) || HWY_TARGET <= HWY_AVX2
  std::cout << "Compiled-in QuickScorer preference: Highway > Legacy\n";
#else
  std::cout << "Compiled-in QuickScorer preference: Legacy > Highway\n";
#endif
}

// Mirrors GradientBoostedTreesQuickScorerFastEngineFactory::IsCompatible, but
// says which requirement fails.
void PrintQuickScorerEligibility(const model::AbstractModel& model) {
  std::cout << "\n=== Model / QuickScorer eligibility ===\n";
  const auto* forest =
      dynamic_cast<const model::DecisionForestInterface*>(&model);
  const auto* gbt = dynamic_cast<
      const model::gradient_boosted_trees::GradientBoostedTreesModel*>(&model);
  std::cout << "model type                 : " << model.name() << "\n";
  if (forest == nullptr) return;
  size_t max_leaves = 0;
  for (const auto& tree : forest->decision_trees()) {
    max_leaves = std::max<size_t>(max_leaves, tree->NumLeafs());
  }
  const bool no_oblique = forest->CheckStructure(
      model::decision_tree::CheckStructureOptions::NoObliqueConditions());
  const bool global_imputation =
      forest->CheckStructure({/*.global_imputation_is_higher =*/false});
  const auto yn = [](bool ok) { return ok ? "ok" : "FAIL"; };
  std::cout << "is a GBT                   : " << yn(gbt != nullptr) << "\n";
  std::cout << "trees                      : " << forest->num_trees()
            << " (max " << QsModel::kMaxTrees << ") "
            << yn(forest->num_trees() <= QsModel::kMaxTrees) << "\n";
  std::cout << "max leaves per tree        : " << max_leaves << " (max "
            << QsModel::kMaxLeafs << ") " << yn(max_leaves <= QsModel::kMaxLeafs)
            << "\n";
  std::cout << "no oblique conditions      : " << yn(no_oblique) << "\n";
  std::cout << "global-imputation structure: " << yn(global_imputation) << "\n";
}

// One line saying which QuickScorer code path an engine name maps to.
std::string DescribeEngine(const std::string& name) {
  if (name == kHighway) {
#ifdef YDF_USE_DYNAMIC_DISPATCH
    return "QuickScorer HIGHWAY (dynamic dispatch)";
#else
    return absl::StrCat("QuickScorer HIGHWAY (static ",
                        hwy::TargetName(HWY_TARGET), ")");
#endif
  }
  if (name == kLegacy) {
#if defined(__AVX2__)
#if ABSL_HAVE_BUILTIN(__builtin_cpu_supports)
    if (!__builtin_cpu_supports("avx2")) {
      return "QuickScorer LEGACY (scalar sequential: CPU lacks AVX2)";
    }
#endif
    return "QuickScorer LEGACY (AVX2 intrinsics, 4 examples/iter)";
#else
    return "QuickScorer LEGACY (scalar sequential: built without AVX2)";
#endif
  }
  return "not QuickScorer";
}

struct EngineTiming {
  double ns_per_example = 0;
  std::vector<float> predictions;
};

EngineTiming TimeEngine(const serving::FastEngine& engine,
                        const dataset::VerticalDataset& ds, int64_t num_rows,
                        int batch_size, int num_runs) {
  // Convert the examples outside the timer.
  std::vector<std::unique_ptr<serving::AbstractExampleSet>> batches;
  std::vector<int> batch_rows;
  for (int64_t begin = 0; begin < num_rows; begin += batch_size) {
    const int64_t end = std::min<int64_t>(begin + batch_size, num_rows);
    batches.push_back(engine.AllocateExamples(end - begin));
    batch_rows.push_back(end - begin);
    CHECK_OK(serving::CopyVerticalDatasetToAbstractExampleSet(
        ds, begin, end, engine.features(), batches.back().get()));
  }
  EngineTiming out;
  out.ns_per_example = std::numeric_limits<double>::infinity();
  std::vector<float> batch_predictions;
  for (int run = 0; run < num_runs; ++run) {
    out.predictions.clear();
    const auto t0 = std::chrono::steady_clock::now();
    for (size_t b = 0; b < batches.size(); ++b) {
      engine.Predict(*batches[b], batch_rows[b], &batch_predictions);
      out.predictions.insert(out.predictions.end(), batch_predictions.begin(),
                             batch_predictions.end());
    }
    const std::chrono::duration<double, std::nano> dt =
        std::chrono::steady_clock::now() - t0;
    out.ns_per_example = std::min(out.ns_per_example, dt.count() / num_rows);
  }
  return out;
}

absl::StatusOr<std::unique_ptr<model::AbstractModel>> TrainModel(
    const dataset::VerticalDataset& ds, const std::string& label_col) {
  model::proto::TrainingConfig train_config;
  train_config.set_task(model::proto::Task::CLASSIFICATION);
  train_config.set_label(label_col);
  train_config.set_random_seed(absl::GetFlag(FLAGS_seed));

  model::proto::DeploymentConfig deploy_config;
  const int num_threads = absl::GetFlag(FLAGS_num_threads);
  deploy_config.set_num_threads(
      num_threads > 0 ? num_threads
                      : std::max(1u, std::thread::hardware_concurrency()));

  const std::string ensemble_method = absl::GetFlag(FLAGS_ensemble_method);
  const bool boosting = ensemble_method == "Boosting";
  model::decision_tree::proto::DecisionTreeTrainingConfig* dt_config;
  if (ensemble_method == "Bagging") {
    train_config.set_learner("RANDOM_FOREST");
    auto& rf = *train_config.MutableExtension(
        model::random_forest::proto::random_forest_config);
    rf.set_num_trees(absl::GetFlag(FLAGS_num_trees));
    rf.set_winner_take_all_inference(false);
    rf.set_compute_oob_performances(false);
    dt_config = rf.mutable_decision_tree();
  } else if (boosting) {
    train_config.set_learner("GRADIENT_BOOSTED_TREES");
    auto& gbt = *train_config.MutableExtension(
        model::gradient_boosted_trees::proto::gradient_boosted_trees_config);
    gbt.set_num_trees(absl::GetFlag(FLAGS_num_trees));
    gbt.set_shrinkage(absl::GetFlag(FLAGS_shrinkage));
    gbt.set_validation_set_ratio(0);
    gbt.set_early_stopping(model::gradient_boosted_trees::proto::
                               GradientBoostedTreesTrainingConfig::NONE);
    dt_config = gbt.mutable_decision_tree();
  } else {
    return absl::InvalidArgumentError(
        "Unknown ensemble_method. Use 'Bagging' or 'Boosting'.");
  }

  int tree_depth = absl::GetFlag(FLAGS_tree_depth);
  if (boosting && tree_depth == -1) tree_depth = 6;
  dt_config->set_max_depth(tree_depth);
  const int min_examples = absl::GetFlag(FLAGS_min_examples);
  dt_config->set_min_examples(min_examples >= 1 ? min_examples
                                                : (boosting ? 5 : 1));

  const std::string split_type = absl::GetFlag(FLAGS_feature_split_type);
  if (split_type == "Oblique") {
    dt_config->mutable_sparse_oblique_split();
  } else if (split_type != "Axis Aligned") {
    return absl::InvalidArgumentError(
        "Unknown feature_split_type. Use 'Axis Aligned' or 'Oblique'.");
  }

  std::unique_ptr<model::AbstractLearner> learner;
  RETURN_IF_ERROR(model::GetLearner(train_config, &learner, deploy_config));
  return learner->TrainWithStatus(ds);
}

}  // namespace

int main(int argc, char** argv) {
  absl::ParseCommandLine(argc, argv);
  // The RF learner exit(0)s right after training unless this is set; we need
  // the model back.
  setenv("NO_EARLY_EXIT", "1", /*overwrite=*/1);

  PrintBuildInfo();

  // 1) Model (loaded) and dataset.
  const std::string mode = absl::GetFlag(FLAGS_input_mode);
  const std::string model_dir = absl::GetFlag(FLAGS_model_dir);
  std::unique_ptr<model::AbstractModel> model_ptr;
  if (!model_dir.empty()) {
    CHECK_OK(model::LoadModel(model_dir, &model_ptr));
  }

  dataset::VerticalDataset train_ds;
  std::string label_col;
  if (mode == "csv") {
    label_col = absl::GetFlag(FLAGS_label_col);
    const std::string csv_path = absl::GetFlag(FLAGS_train_csv);
    if (csv_path.empty()) {
      std::cerr << "Error: --train_csv is required in csv mode.\n";
      return 1;
    }
    dataset::proto::DataSpecification data_spec;
    if (model_ptr) {
      data_spec = model_ptr->data_spec();
    } else {
      dataset::proto::DataSpecificationGuide guide;
      auto* col_guide = guide.add_column_guides();
      col_guide->set_column_name_pattern(label_col);
      col_guide->set_type(dataset::proto::ColumnType::CATEGORICAL);
      dataset::CreateDataSpec("csv:" + csv_path,
                              /*require_same_dataset_fields=*/false, guide,
                              &data_spec);
    }
    CHECK_OK(
        dataset::LoadVerticalDataset("csv:" + csv_path, data_spec, &train_ds));
  } else if (mode == "trunk") {
    label_col = "y";
    // With a loaded model, generate the columns its dataspec expects.
    const int cols = model_ptr ? model_ptr->data_spec().columns_size() - 1
                               : absl::GetFlag(FLAGS_cols);
    const int64_t rows = absl::GetFlag(FLAGS_rows);
    train_ds = MakeTrunkDataset(
        model_ptr ? model_ptr->data_spec() : MakeSyntheticSpec(cols, rows),
        rows, cols, absl::GetFlag(FLAGS_seed));
  } else {
    std::cerr << "Unknown input_mode: " << mode << ". Use csv or trunk.\n";
    return 1;
  }

  // 2) Train unless a model was loaded.
  if (!model_ptr) {
    auto model_or = TrainModel(train_ds, label_col);
    if (!model_or.ok()) {
      std::cerr << "Training failed: " << model_or.status().message() << "\n";
      return 1;
    }
    model_ptr = std::move(model_or.value());
  }

  // 3) Inference dataset.
  dataset::VerticalDataset test_ds_storage;
  const dataset::VerticalDataset* infer_ds = &train_ds;
  const std::string test_csv = absl::GetFlag(FLAGS_test_csv);
  if (mode == "csv" && !test_csv.empty()) {
    CHECK_OK(dataset::LoadVerticalDataset("csv:" + test_csv,
                                          model_ptr->data_spec(),
                                          &test_ds_storage));
    infer_ds = &test_ds_storage;
  }
  const int64_t num_rows =
      std::min<int64_t>(infer_ds->nrow(), absl::GetFlag(FLAGS_infer_rows));

  PrintQuickScorerEligibility(*model_ptr);

  // 4) Which engine runs.
  std::cout << "\n=== Engines ===\n";
  const auto engine_names = model_ptr->ListCompatibleFastEngineNames();
  std::cout << "compatible (best first)    : "
            << absl::StrJoin(engine_names, ", ") << "\n";
  if (engine_names.empty()) {
    std::cout << "selected                   : none (slow generic Predict)\n";
    return 0;
  }
  const std::string selected = engine_names.front();
  std::cout << "selected by BuildFastEngine: " << selected << "\n";
  std::cout << "=> " << DescribeEngine(selected) << "\n";
  const std::string force_engine = absl::GetFlag(FLAGS_force_engine);
  if (!force_engine.empty()) {
    const auto forced = model_ptr->BuildFastEngine(force_engine);
    std::cout << "--force_engine=" << force_engine << ": "
              << (forced.ok() ? DescribeEngine(force_engine)
                              : std::string(forced.status().message()))
              << "\n";
  }

  // 5) Run every compatible engine: time + agreement with the selected one.
  std::cout << "\n=== Inference: " << num_rows << " examples, batch "
            << absl::GetFlag(FLAGS_batch_size) << ", best of "
            << absl::GetFlag(FLAGS_num_runs) << ", 1 thread ===\n";
  std::vector<float> reference;
  for (const auto& name : engine_names) {
    auto engine_or = model_ptr->BuildFastEngine(name);
    if (!engine_or.ok()) {
      std::cout << name << ": could not be built: "
                << engine_or.status().message() << "\n";
      continue;
    }
    const EngineTiming timing =
        TimeEngine(*engine_or.value(), *infer_ds, num_rows,
                   absl::GetFlag(FLAGS_batch_size),
                   absl::GetFlag(FLAGS_num_runs));
    if (reference.empty()) reference = timing.predictions;
    float max_diff = 0;
    if (timing.predictions.size() == reference.size()) {
      for (size_t i = 0; i < reference.size(); ++i) {
        max_diff =
            std::max(max_diff, std::abs(timing.predictions[i] - reference[i]));
      }
    } else {
      max_diff = std::numeric_limits<float>::quiet_NaN();
    }
    std::cout << absl::StrFormat(
        "%-48s %9.1f ns/example  max|diff vs selected|=%.3g  [%s]%s\n", name,
        timing.ns_per_example, max_diff, DescribeEngine(name),
        name == selected ? "  <-- selected" : "");
  }
#ifdef YDF_USE_DYNAMIC_DISPATCH
  std::cout << "Highway dynamic dispatch resolved to: "
            << hwy::TargetName(hwy::DispatchedTarget()) << "\n";
#endif
  return 0;
}
