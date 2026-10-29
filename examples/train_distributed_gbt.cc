/*
 * Copyright 2022 Google LLC.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     https://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

// Single-process demo of DISTRIBUTED_GRADIENT_BOOSTED_TREES.
//
// The workers run in-process ("MULTI_THREAD" distribute implementation), so the
// whole feature-distributed protocol (dataset cache creation, per-worker
// feature ownership, FindSplits / MergeBestSplits / EvaluateSplits /
// ShareSplits) is exercised inside one debuggable binary. Useful breakpoints:
//   learner/distributed_gradient_boosted_trees/distributed_gradient_boosted_trees.cc
//     (manager: RunIteration, EmitFindSplits, MergeBestSplits, EmitEvaluateSplits)
//   learner/distributed_gradient_boosted_trees/worker.cc  (worker: FindSplits)
//   learner/distributed_decision_tree/training.cc         (TreeBuilder)
//
// Usage:
//   bazel build -c opt //examples:train_distributed_gbt
//   bazel-bin/examples/train_distributed_gbt --alsologtostderr \
//     --num_workers=4 --num_trees=20
//
#include <cstdint>
#include <filesystem>
#include <string>
#include <system_error>

#include "absl/flags/flag.h"
#include "absl/log/log.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_split.h"
#include "absl/strings/substitute.h"
#include "absl/time/clock.h"
#include "absl/time/time.h"
#include "google/protobuf/text_format.h"
#include "yggdrasil_decision_forests/dataset/data_spec.h"
#include "yggdrasil_decision_forests/dataset/data_spec.pb.h"
#include "yggdrasil_decision_forests/dataset/data_spec_inference.h"
#include "yggdrasil_decision_forests/dataset/vertical_dataset.h"
#include "yggdrasil_decision_forests/dataset/vertical_dataset_io.h"
#include "yggdrasil_decision_forests/learner/abstract_learner.pb.h"
#include "yggdrasil_decision_forests/learner/distributed_gradient_boosted_trees/distributed_gradient_boosted_trees.pb.h"
#include "yggdrasil_decision_forests/learner/learner_library.h"
#include "yggdrasil_decision_forests/metric/metric.h"
#include "yggdrasil_decision_forests/metric/metric.pb.h"
#include "yggdrasil_decision_forests/metric/report.h"
#include "yggdrasil_decision_forests/model/abstract_model.h"
#include "yggdrasil_decision_forests/model/model_library.h"
#include "yggdrasil_decision_forests/model/decision_tree/decision_tree.h"
#include "yggdrasil_decision_forests/model/gradient_boosted_trees/gradient_boosted_trees.h"
#include "yggdrasil_decision_forests/utils/filesystem.h"
#include "yggdrasil_decision_forests/utils/logging.h"
#include "yggdrasil_decision_forests/utils/random.h"

ABSL_FLAG(std::string, train_csv, "benchmarks/data/HIGGS_train_10500k.csv",
          "Training CSV (label column --label).");
ABSL_FLAG(std::string, test_csv, "benchmarks/data/HIGGS_test_500k.csv",
          "Test CSV for the final evaluation.");
ABSL_FLAG(std::string, label, "class",
          "Label column. Forced to CATEGORICAL in the dataspec guide (HIGGS "
          "stores it as 0.0/1.0 floats).");
ABSL_FLAG(int64_t, dataspec_scan_rows, 1000000,
          "Rows scanned to accumulate dataspec statistics (-1 = all). Keeps "
          "the 7.6 GB HIGGS scan short in the debug loop.");
ABSL_FLAG(std::string, work_dir, "/tmp/ydf_dgbt_demo",
          "Working directory: dataset cache, checkpoints, model.");
ABSL_FLAG(int, num_workers, 4, "Number of in-process workers.");
ABSL_FLAG(int, num_trees, 30, "Number of boosted trees.");
ABSL_FLAG(bool, worker_logs, true, "Print the per-worker logs.");
ABSL_FLAG(bool, force_pure_serving, false, "Unused; kept for symmetry.");
ABSL_FLAG(bool, fresh_train, true,
          "Delete previous checkpoints from work_dir before training so every "
          "run replays training from iteration 0. The dataset cache is kept "
          "and reused (the expensive part on HIGGS).");
ABSL_FLAG(int, num_threads, 6, "Threads per worker (split search).");
ABSL_FLAG(int, max_depth, 6, "Maximum tree depth.");
ABSL_FLAG(std::string, split_axis, "AXIS_ALIGNED",
          "AXIS_ALIGNED or SPARSE_OBLIQUE. SPARSE_OBLIQUE = in-worker oblique "
          "splits: each worker samples projections of the columns it owns.");
ABSL_FLAG(double, sparse_oblique_num_projections_exponent, 0.5,
          "Projections per node = ceil(F^exponent); 0.5 = sqrt(F).");
ABSL_FLAG(double, sparse_oblique_projection_density_factor, 1.5,
          "Expected number of non-zero weights per projection.");
ABSL_FLAG(std::string, sparse_oblique_weights, "BINARY",
          "BINARY or CONTINUOUS projection weights.");
ABSL_FLAG(int, sparse_oblique_max_num_projections, 1000,
          "Cap on projections per node (proto default 1000). The paper's "
          "P = F arms pass INT_MAX so wide datasets (Bioresponse, F = 1776) "
          "are not truncated.");
ABSL_FLAG(bool, sparse_oblique_local_projection_count, false,
          "If true, F in ceil(F^exponent) is the worker's own numerical "
          "candidate count instead of the dataset's numerical feature count.");
ABSL_FLAG(double, column_replication, 0.0,
          "In-worker oblique: fraction in [0,1] of the numerical columns a "
          "worker does not own that it also holds and samples projections "
          "from. 0 = disjoint F/W columns per worker, 1 = every worker holds "
          "all F columns.");
ABSL_FLAG(int64_t, seed, -1,
          "Training random seed (TrainingConfig.random_seed); <0 keeps the "
          "library default.");
ABSL_FLAG(bool, force_numerical_features, false,
          "Dataspec guide: every non-label column is NUMERICAL and 0/1 columns "
          "are not detected as BOOLEAN (the dataset policy's fully numeric "
          "input). Fails on string columns.");
ABSL_FLAG(bool, sparse_oblique_ship_columns, false,
          "Naive cross-worker projections: every worker samples from all the "
          "numerical columns and fetches the ones it does not own from their "
          "owners at every layer (in_worker_oblique.ship_foreign_columns). "
          "Communication-cost baseline of the disjoint-column mode.");
ABSL_FLAG(std::string, grpc_worker_addresses, "",
          "Comma-separated host:port of running grpc_worker_main processes. If "
          "set, they replace the in-process MULTI_THREAD workers and "
          "--num_workers is ignored (manager stays in this process).");
ABSL_FLAG(std::string, model_out_dir, "",
          "If set, the trained model is saved there (for bit-identity checks "
          "between builds / thread counts).");
ABSL_FLAG(int64_t, max_unique_values_for_discretized_numerical, -1,
          "If >= 1, overrides the cache's discretization threshold (1 keeps "
          "every non-constant column exact). Needs a fresh cache dir.");

namespace ydf = yggdrasil_decision_forests;

int main(int argc, char** argv) {
  InitLogging(argv[0], &argc, &argv, true);

  const std::string work_dir = absl::GetFlag(FLAGS_work_dir);
  const std::string label = absl::GetFlag(FLAGS_label);
  const auto train_path = absl::StrCat("csv:", absl::GetFlag(FLAGS_train_csv));
  const auto test_path = absl::StrCat("csv:", absl::GetFlag(FLAGS_test_csv));
  QCHECK_OK(file::RecursivelyCreateDir(work_dir, file::Defaults()));

  // 1. Dataspec (the manager needs it before the workers build their cache).
  // The HIGGS label is 0.0/1.0 floats, so force it to CATEGORICAL; everything
  // else stays inferred (28 NUMERICAL features).
  ydf::dataset::proto::DataSpecificationGuide guide;
  guide.set_max_num_scanned_rows_to_accumulate_statistics(
      absl::GetFlag(FLAGS_dataspec_scan_rows));
  auto* label_guide = guide.add_column_guides();
  label_guide->set_column_name_pattern(absl::StrCat("^", label, "$"));
  label_guide->set_type(ydf::dataset::proto::ColumnType::CATEGORICAL);
  if (absl::GetFlag(FLAGS_force_numerical_features)) {
    guide.set_detect_boolean_as_numerical(true);
    guide.mutable_default_column_guide()->set_type(
        ydf::dataset::proto::ColumnType::NUMERICAL);
  }
  const auto data_spec = ydf::dataset::CreateDataSpec(train_path, guide).value();
  LOG(INFO) << "Dataspec:\n" << ydf::dataset::PrintHumanReadable(data_spec);

  // 2. Training configuration: the distributed learner.
  ydf::model::proto::TrainingConfig train_config;
  train_config.set_learner("DISTRIBUTED_GRADIENT_BOOSTED_TREES");
  train_config.set_task(ydf::model::proto::Task::CLASSIFICATION);
  train_config.set_label(label);
  if (absl::GetFlag(FLAGS_seed) >= 0) {
    train_config.set_random_seed(absl::GetFlag(FLAGS_seed));
  }
  auto* dgbt = train_config.MutableExtension(
      ydf::model::distributed_gradient_boosted_trees::proto::
          distributed_gradient_boosted_trees_config);
  dgbt->set_worker_logs(absl::GetFlag(FLAGS_worker_logs));
  dgbt->set_checkpoint_interval_trees(1000);  // No checkpointing in the demo.
  dgbt->mutable_gbt()->set_num_trees(absl::GetFlag(FLAGS_num_trees));
  auto* dt = dgbt->mutable_gbt()->mutable_decision_tree();
  dt->set_max_depth(absl::GetFlag(FLAGS_max_depth));
  if (absl::GetFlag(FLAGS_max_unique_values_for_discretized_numerical) >= 1) {
    dgbt->mutable_create_cache()->set_max_unique_values_for_discretized_numerical(
        absl::GetFlag(FLAGS_max_unique_values_for_discretized_numerical));
  }
  const std::string split_axis = absl::GetFlag(FLAGS_split_axis);
  if (split_axis == "SPARSE_OBLIQUE") {
    auto* oblique = dt->mutable_sparse_oblique_split();
    oblique->set_num_projections_exponent(
        absl::GetFlag(FLAGS_sparse_oblique_num_projections_exponent));
    oblique->set_projection_density_factor(
        absl::GetFlag(FLAGS_sparse_oblique_projection_density_factor));
    oblique->set_max_num_projections(
        absl::GetFlag(FLAGS_sparse_oblique_max_num_projections));
    if (absl::GetFlag(FLAGS_sparse_oblique_weights) == "CONTINUOUS") {
      oblique->mutable_continuous();
    } else {
      oblique->mutable_binary();
    }
    dgbt->mutable_in_worker_oblique()->set_num_projections_from_local_features(
        absl::GetFlag(FLAGS_sparse_oblique_local_projection_count));
    dgbt->mutable_in_worker_oblique()->set_column_replication(
        absl::GetFlag(FLAGS_column_replication));
    dgbt->mutable_in_worker_oblique()->set_ship_foreign_columns(
        absl::GetFlag(FLAGS_sparse_oblique_ship_columns));
  } else {
    QCHECK_EQ(split_axis, "AXIS_ALIGNED") << "Unknown --split_axis";
  }

  // 3. Deployment configuration: where the workers live. "MULTI_THREAD" runs
  // them as threads of this process; swapping in "GRPC" (see
  // examples/distributed_training.sh) is the only change needed for real
  // multi-machine training.
  ydf::model::proto::DeploymentConfig deployment;
  deployment.set_num_threads(absl::GetFlag(FLAGS_num_threads));
  deployment.set_cache_path(file::JoinPath(work_dir, "cache"));
  // try_resume_training=true keeps the working directory stable across runs:
  // with false, the learner appends a random+timestamp subdir to cache_path
  // (distributed_gradient_boosted_trees.cc, "Set Working directory"), so the
  // dataset cache is rebuilt from scratch on every run. With true, the cache
  // is reused via its "done" marker (dataset_cache.cc). --fresh_train deletes
  // the checkpoints so training itself still restarts at iteration 0.
  deployment.set_try_resume_training(true);
  if (absl::GetFlag(FLAGS_fresh_train)) {
    // file::RecursivelyDelete is fs::remove in the default filesystem (fails
    // on non-empty dirs), so use remove_all directly.
    std::error_code ec;
    std::filesystem::remove_all(file::JoinPath(work_dir, "cache", "checkpoint"),
                                ec);
    QCHECK(!ec) << "Deleting previous checkpoint: " << ec.message();
  }
  const std::string grpc_addresses = absl::GetFlag(FLAGS_grpc_worker_addresses);
  if (grpc_addresses.empty()) {
    QCHECK(google::protobuf::TextFormat::ParseFromString(
        absl::Substitute(R"pb(
                           implementation_key: "MULTI_THREAD"
                           [yggdrasil_decision_forests.distribute.proto
                                .multi_thread] { num_workers: $0 }
                         )pb",
                         absl::GetFlag(FLAGS_num_workers)),
        deployment.mutable_distribute()));
  } else {
    // Workers are separate processes (one grpc_worker_main each); the
    // manager runs here. Worker-to-worker traffic then really crosses a
    // socket, which is what the column-shipping measurements need.
    std::string addresses_pbtxt;
    for (const auto address : absl::StrSplit(grpc_addresses, ',')) {
      absl::SubstituteAndAppend(&addresses_pbtxt, "addresses: \"$0\" ",
                                address);
    }
    QCHECK(google::protobuf::TextFormat::ParseFromString(
        absl::Substitute(R"pb(
                           implementation_key: "GRPC"
                           [yggdrasil_decision_forests.distribute.proto.grpc] {
                             grpc_addresses { $0 }
                           }
                         )pb",
                         addresses_pbtxt),
        deployment.mutable_distribute()));
  }

  // 4. Train. The learner reads the dataset *by path*: each worker reads the
  // shards it owns and builds its slice of the dataset cache.
  const auto learner =
      ydf::model::GetLearner(train_config, deployment,
                             file::JoinPath(work_dir, "train_logs"))
          .value();
  const auto begin_train = absl::Now();
  const auto model = learner->TrainWithStatus(train_path, data_spec).value();
  LOG(INFO) << "Training (incl. cache creation) took "
            << absl::Now() - begin_train << " split_axis=" << split_axis;

  const std::string model_out_dir = absl::GetFlag(FLAGS_model_out_dir);
  if (!model_out_dir.empty()) {
    QCHECK_OK(ydf::model::SaveModel(model_out_dir, *model));
    LOG(INFO) << "Model saved to " << model_out_dir;
  }

  // 5. Evaluate + describe.
  ydf::dataset::VerticalDataset test_dataset;
  QCHECK_OK(ydf::dataset::LoadVerticalDataset(test_path, model->data_spec(),
                                              &test_dataset));
  ydf::utils::RandomEngine rnd;
  ydf::metric::proto::EvaluationOptions eval_options;
  eval_options.set_task(model->task());
  const auto evaluation = model->Evaluate(test_dataset, eval_options, &rnd);
  LOG(INFO) << "Evaluation:\n"
            << ydf::metric::TextReport(evaluation).value();
  LOG(INFO) << "Model:\n" << model->DescriptionAndStatistics(false);

  // Split-type census: how many conditions are oblique, and how many of
  // those combine more than one column.
  const auto* gbt = dynamic_cast<
      const ydf::model::gradient_boosted_trees::GradientBoostedTreesModel*>(
      model.get());
  QCHECK(gbt != nullptr);
  int num_conditions = 0, num_oblique = 0, num_multi_column = 0;
  for (const auto& tree : gbt->decision_trees()) {
    tree->IterateOnNodes([&](const ydf::model::decision_tree::NodeWithChildren&
                                 node,
                             const int depth) {
      if (node.IsLeaf()) return;
      num_conditions++;
      const auto& condition = node.node().condition().condition();
      if (condition.has_oblique_condition()) {
        num_oblique++;
        num_multi_column += condition.oblique_condition().attributes_size() > 1;
      }
    });
  }
  LOG(INFO) << "Conditions: " << num_conditions << " total, " << num_oblique
            << " oblique, " << num_multi_column << " with >1 column";
  return 0;
}
