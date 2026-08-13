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
//   bazel build -c opt //examples:distributed_gbt_demo
//   bazel-bin/examples/distributed_gbt_demo --alsologtostderr \
//     --num_workers=4 --num_trees=20
//
#include <cstdint>
#include <filesystem>
#include <string>
#include <system_error>

#include "absl/flags/flag.h"
#include "absl/log/log.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/substitute.h"
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
ABSL_FLAG(int, num_trees, 20, "Number of boosted trees.");
ABSL_FLAG(bool, worker_logs, true, "Print the per-worker logs.");
ABSL_FLAG(bool, force_pure_serving, false, "Unused; kept for symmetry.");
ABSL_FLAG(bool, fresh_train, true,
          "Delete previous checkpoints from work_dir before training so every "
          "run replays training from iteration 0. The dataset cache is kept "
          "and reused (the expensive part on HIGGS).");

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
  const auto data_spec = ydf::dataset::CreateDataSpec(train_path, guide).value();
  LOG(INFO) << "Dataspec:\n" << ydf::dataset::PrintHumanReadable(data_spec);

  // 2. Training configuration: the distributed learner.
  ydf::model::proto::TrainingConfig train_config;
  train_config.set_learner("DISTRIBUTED_GRADIENT_BOOSTED_TREES");
  train_config.set_task(ydf::model::proto::Task::CLASSIFICATION);
  train_config.set_label(label);
  auto* dgbt = train_config.MutableExtension(
      ydf::model::distributed_gradient_boosted_trees::proto::
          distributed_gradient_boosted_trees_config);
  dgbt->set_worker_logs(absl::GetFlag(FLAGS_worker_logs));
  dgbt->set_checkpoint_interval_trees(1000);  // No checkpointing in the demo.
  dgbt->mutable_gbt()->set_num_trees(absl::GetFlag(FLAGS_num_trees));

  // 3. Deployment configuration: where the workers live. "MULTI_THREAD" runs
  // them as threads of this process; swapping in "GRPC" (see
  // examples/distributed_training.sh) is the only change needed for real
  // multi-machine training.
  ydf::model::proto::DeploymentConfig deployment;
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
  QCHECK(google::protobuf::TextFormat::ParseFromString(
      absl::Substitute(R"pb(
                         implementation_key: "MULTI_THREAD"
                         [yggdrasil_decision_forests.distribute.proto
                              .multi_thread] { num_workers: $0 }
                       )pb",
                       absl::GetFlag(FLAGS_num_workers)),
      deployment.mutable_distribute()));

  // 4. Train. The learner reads the dataset *by path*: each worker reads the
  // shards it owns and builds its slice of the dataset cache.
  const auto learner =
      ydf::model::GetLearner(train_config, deployment,
                             file::JoinPath(work_dir, "train_logs"))
          .value();
  const auto model = learner->TrainWithStatus(train_path, data_spec).value();

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
  return 0;
}
