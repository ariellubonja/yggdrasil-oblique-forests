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

// Generic hyperparameters (random_seed, ...) set through SetHyperParameters,
// as the Python API does, must reach the distributed GBT training config.

#include <memory>
#include <optional>
#include <string>

#include "gtest/gtest.h"
#include "yggdrasil_decision_forests/learner/abstract_learner.h"
#include "yggdrasil_decision_forests/learner/abstract_learner.pb.h"
#include "yggdrasil_decision_forests/learner/distributed_gradient_boosted_trees/distributed_gradient_boosted_trees.h"
#include "yggdrasil_decision_forests/learner/distributed_gradient_boosted_trees/distributed_gradient_boosted_trees.pb.h"
#include "yggdrasil_decision_forests/learner/learner_library.h"
#include "yggdrasil_decision_forests/metric/metric.h"
#include "yggdrasil_decision_forests/model/abstract_model.h"
#include "yggdrasil_decision_forests/utils/filesystem.h"
#include "yggdrasil_decision_forests/utils/test.h"
#include "yggdrasil_decision_forests/utils/test_utils.h"

namespace yggdrasil_decision_forests {
namespace model {
namespace distributed_gradient_boosted_trees {
namespace {

model::proto::GenericHyperParameters SeedHParam(const int seed) {
  model::proto::GenericHyperParameters hparams;
  auto* field = hparams.add_fields();
  field->set_name("random_seed");
  field->mutable_value()->set_integer(seed);
  return hparams;
}

// No training: the four generic fields of AbstractLearner must survive.
TEST(SetHyperParameters, KeepsGenericFields) {
  model::proto::TrainingConfig config;
  config.set_learner(DistributedGradientBoostedTreesLearner::kRegisteredName);
  config.set_task(model::proto::Task::CLASSIFICATION);
  config.set_label("income");
  std::unique_ptr<AbstractLearner> learner;
  ASSERT_OK(GetLearner(config, &learner));

  model::proto::GenericHyperParameters hparams = SeedHParam(7);
  auto* f = hparams.add_fields();
  f->set_name("maximum_training_duration_seconds");
  f->mutable_value()->set_real(12.5);
  f = hparams.add_fields();
  f->set_name("maximum_model_size_in_memory_in_bytes");
  f->mutable_value()->set_real(1000);
  f = hparams.add_fields();
  f->set_name("pure_serving_model");
  f->mutable_value()->set_categorical("true");
  ASSERT_OK(learner->SetHyperParameters(hparams));

  const auto& tc = learner->training_config();
  EXPECT_EQ(tc.random_seed(), 7);
  EXPECT_EQ(tc.maximum_training_duration_seconds(), 12.5);
  EXPECT_EQ(tc.maximum_model_size_in_memory_in_bytes(), 1000);
  EXPECT_TRUE(tc.pure_serving_model());
}

// MULTI_THREAD distribution, 2 workers, attribute sampling (seed-dependent).
class DatasetAdult : public utils::TrainAndTestTester {
 public:
  // Trains with the seed set either as a generic hyperparameter (Python API
  // path) or directly on the TrainingConfig (C++ driver path).
  std::unique_ptr<AbstractModel> Train(const std::optional<int> generic_seed,
                                       const std::optional<int> direct_seed,
                                       const std::string& run) {
    generic_parameters_.reset();
    train_config_.clear_random_seed();
    if (generic_seed.has_value()) {
      generic_parameters_ = SeedHParam(*generic_seed);
    }
    if (direct_seed.has_value()) {
      train_config_.set_random_seed(*direct_seed);
    }
    deployment_config_.set_cache_path(
        file::JoinPath(test::TmpDirectory(), "work_" + run));
    TrainAndEvaluateModel();
    LOG(INFO) << "run=" << run
              << " effective random_seed=" << learner_->training_config().random_seed()
              << " accuracy=" << metric::Accuracy(evaluation_);
    return std::move(model_);
  }

 private:
  void SetUp() override {
    train_config_ = PARSE_TEST_PROTO(R"pb(
      learner: "DISTRIBUTED_GRADIENT_BOOSTED_TREES"
      task: CLASSIFICATION
      label: "income"
      [yggdrasil_decision_forests.model.distributed_gradient_boosted_trees.proto
           .distributed_gradient_boosted_trees_config] {
        worker_logs: false
        gbt {
          num_trees: 10
          decision_tree { num_candidate_attributes: 4 }
        }
      }
    )pb");
    *deployment_config_.mutable_distribute() = PARSE_TEST_PROTO(R"pb(
      implementation_key: "MULTI_THREAD"
      [yggdrasil_decision_forests.distribute.proto.multi_thread] {
        num_workers: 2
      }
    )pb");
    dataset_filename_ = "adult_train.csv";
    dataset_test_filename_ = "adult_test.csv";
    preferred_format_type = "csv";
    pass_training_dataset_as_path_ = true;
    num_shards_ = 20;
  }
};

// Control: the seed matters when it is set on the TrainingConfig.
TEST_F(DatasetAdult, DirectSeedChangesModel) {
  const auto m1 = Train(std::nullopt, 1, "direct1");
  const auto m2 = Train(std::nullopt, 2, "direct2");
  EXPECT_NE(m1->DebugCompare(*m2), "");
}

// The bug: a generic random_seed must train the same model as the direct one,
// and different generic seeds must give different models.
TEST_F(DatasetAdult, GenericSeedReachesTraining) {
  const auto direct1 = Train(std::nullopt, 1, "direct1");
  const auto generic1 = Train(1, std::nullopt, "generic1");
  const auto generic2 = Train(2, std::nullopt, "generic2");
  const auto unset = Train(std::nullopt, std::nullopt, "unset");
  EXPECT_EQ(direct1->DebugCompare(*generic1), "")
      << "generic random_seed=1 did not train the random_seed=1 model";
  EXPECT_NE(generic1->DebugCompare(*generic2), "")
      << "generic random_seed=1 and =2 trained identical models";
  EXPECT_NE(generic1->DebugCompare(*unset), "")
      << "generic random_seed=1 trained the default-seed (123456) model";
}

}  // namespace
}  // namespace distributed_gradient_boosted_trees
}  // namespace model
}  // namespace yggdrasil_decision_forests
