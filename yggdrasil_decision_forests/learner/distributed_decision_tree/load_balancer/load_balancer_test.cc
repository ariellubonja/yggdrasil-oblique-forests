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

#include "yggdrasil_decision_forests/learner/distributed_decision_tree/load_balancer/load_balancer.h"

#include "gmock/gmock.h"
#include "absl/log/log.h"
#include "yggdrasil_decision_forests/utils/logging.h"
#include "yggdrasil_decision_forests/utils/test.h"

namespace yggdrasil_decision_forests {
namespace model {
namespace distributed_decision_tree {
namespace {

using test::EqualsProto;
using testing::ElementsAre;

TEST(LoadBalancer, StaticBalancing) {
  distributed_decision_tree::dataset_cache::proto::CacheMetadata
      cache_metadata = PARSE_TEST_PROTO(R"pb(
        columns { categorical { num_values: 15 } }
        columns { numerical { num_unique_values: 20 } }
        columns { numerical { discretized: true num_discretized_values: 10 } }
        columns { boolean {} }
      )pb");
  auto balancer = LoadBalancer::Create(/*features=*/{0, 1, 2, 3},
                                       /*num_workers=*/2, cache_metadata, {})
                      .value();
  EXPECT_THAT(balancer.WorkersPerFeatures(), ElementsAre(1, 0, 0, 1));
}

TEST(LoadBalancer, Replicas) {
  distributed_decision_tree::dataset_cache::proto::CacheMetadata
      cache_metadata = PARSE_TEST_PROTO(R"pb(
        columns { categorical { num_values: 15 } }
        columns { numerical { num_unique_values: 20 } }
        columns { numerical { discretized: true num_discretized_values: 10 } }
        columns { boolean {} }
      )pb");
  // Replicas need static ownership (the manager disables dynamic balancing in
  // oblique mode).
  const proto::LoadBalancerOptions options = PARSE_TEST_PROTO(
      R"pb(
        dynamic_balancing_frequency_iteration: -1
        dynamic_balancing_frequency_seconds: -1
      )pb");
  auto balancer = LoadBalancer::Create(/*features=*/{0, 1, 2, 3},
                                       /*num_workers=*/2, cache_metadata,
                                       options)
                      .value();
  // Owners: worker 0 -> {1, 2}, worker 1 -> {0, 3}.
  EXPECT_THAT(balancer.WorkersPerFeatures(), ElementsAre(1, 0, 0, 1));
  EXPECT_FALSE(balancer.HasReplicas());
  EXPECT_THAT(balancer.FeatureHolders(0).value(), ElementsAre(1));

  // Invalid: replicating an owned feature, a duplicate, an unknown feature.
  EXPECT_FALSE(balancer.SetReplicas({{1}, {}}).ok());
  EXPECT_FALSE(balancer.SetReplicas({{0, 0}, {}}).ok());
  EXPECT_FALSE(balancer.SetReplicas({{7}, {}}).ok());
  EXPECT_FALSE(balancer.SetReplicas({{0}}).ok());

  EXPECT_OK(balancer.SetReplicas({{3, 0}, {2}}));
  EXPECT_TRUE(balancer.HasReplicas());
  EXPECT_THAT(balancer.ReplicasPerWorker(0), ElementsAre(0, 3));
  EXPECT_THAT(balancer.ReplicasPerWorker(1), ElementsAre(2));
  EXPECT_THAT(balancer.HeldFeaturesPerWorker(0), ElementsAre(0, 1, 2, 3));
  EXPECT_THAT(balancer.HeldFeaturesPerWorker(1), ElementsAre(0, 2, 3));
  // Ownership is unchanged.
  EXPECT_THAT(balancer.WorkersPerFeatures(), ElementsAre(1, 0, 0, 1));
  EXPECT_THAT(balancer.FeaturesPerWorker(0), ElementsAre(1, 2));
  // Holders: owner first, then replicators.
  EXPECT_THAT(balancer.FeatureHolders(0).value(), ElementsAre(1, 0));
  EXPECT_THAT(balancer.FeatureHolders(2).value(), ElementsAre(0, 1));
  EXPECT_THAT(balancer.FeatureHolders(1).value(), ElementsAre(0));

  // Explicit source workers: opaque keys are copied into the plan items.
  const auto plan =
      balancer.MakeSplitSharingPlan(/*feature_idxs=*/{-1, -2},
                                    /*src_workers=*/{0, 1})
          .value();
  ASSERT_EQ(plan.rounds_size(), 2);
  for (const auto& request : plan.rounds(0).requests()) {
    for (const auto& item : request.second.items()) {
      ASSERT_EQ(item.features_size(), 1);
      EXPECT_EQ(item.features(0), -1 - item.src_worker());
    }
  }
  EXPECT_FALSE(balancer.MakeSplitSharingPlan({-1}, {5}).ok());
  EXPECT_FALSE(balancer.MakeSplitSharingPlan({-1, -2}, {0}).ok());
}

TEST(LoadBalancer, DynamicBalancing) {
  const dataset_cache::proto::CacheMetadata cache_metadata = PARSE_TEST_PROTO(
      R"pb(
        columns {
          available: true
          numerical { discretized: true num_discretized_values: 6 }
        }
        columns {}
        columns {
          available: true
          categorical { num_values: 9 }
        }
        columns {}
        columns {
          available: true
          categorical { num_values: 8 }
        }
        columns {}
        columns {
          available: true
          categorical { num_values: 7 }
        }
      )pb");

  const proto::LoadBalancerOptions options = PARSE_TEST_PROTO(
      R"pb(
        median_margin_ratio: 1.5
        max_unbalance_ratio: 2
        estimation_window_length: 3
        dynamic_balancing_frequency_iteration: 1
      )pb");

  auto balancer =
      LoadBalancer::Create(/*features=*/{0, 2, 4, 6}, /*num_workers=*/2,
                           cache_metadata, options)
          .value();

  balancer.AddFeatureLoadingDurationMeasurement(1.0);
  balancer.AddFeatureLoadingDurationMeasurement(0.9);
  balancer.AddFeatureLoadingDurationMeasurement(1.1);
  LOG(INFO) << balancer.Info();
  EXPECT_THAT(balancer.WorkersPerFeatures(),
              ElementsAre(1, -1, 0, -1, 1, -1, 0));  // 2 in each workers.

  // Worker #0 is faster than worker #1.
  LOG(INFO) << "Step 1";
  CHECK_OK(balancer.AddWorkDurationMeasurement({{1, 2}, {5, 2}}).status());
  CHECK_OK(balancer.ApplyPendingOrder());
  LOG(INFO) << balancer.Info();
  EXPECT_THAT(balancer.WorkersPerFeatures(),
              ElementsAre(1, -1, 0, -1, 1, -1, 0));  // 2 in each workers.

  LOG(INFO) << "Step 2";
  CHECK_OK(balancer.AddWorkDurationMeasurement({{1, 2}, {5, 2}}).status());
  CHECK_OK(balancer.ApplyPendingOrder());
  LOG(INFO) << balancer.Info();
  EXPECT_THAT(balancer.WorkersPerFeatures(),
              ElementsAre(1, -1, 0, -1, 1, -1, 0));  // 2 in each workers.

  LOG(INFO) << "Step 3";
  CHECK_OK(balancer.AddWorkDurationMeasurement({{1, 2}, {5, 2}}).status());
  CHECK_OK(balancer.ApplyPendingOrder());
  LOG(INFO) << balancer.Info();
  // Transfer of feature expected.
  EXPECT_THAT(balancer.WorkersPerFeatures(),
              ElementsAre(0, -1, 0, -1, 1, -1, 0));  // 3 in worker 0.

  // Now, worker #1 is faster than worker #0.
  LOG(INFO) << "Step 4";
  CHECK_OK(balancer.AddWorkDurationMeasurement({{8, 3}, {1, 1}}).status());
  CHECK_OK(balancer.ApplyPendingOrder());
  LOG(INFO) << balancer.Info();
  EXPECT_THAT(balancer.WorkersPerFeatures(),
              ElementsAre(0, -1, 0, -1, 1, -1, 0));  // 3 in worker 0.

  LOG(INFO) << "Step 5";
  CHECK_OK(balancer.AddWorkDurationMeasurement({{8, 3}, {1, 1}}).status());
  CHECK_OK(balancer.ApplyPendingOrder());
  LOG(INFO) << balancer.Info();
  EXPECT_THAT(balancer.WorkersPerFeatures(),
              ElementsAre(1, -1, 0, -1, 1, -1, 0));  // 2 in worker 0.

  LOG(INFO) << "Step 6";
  CHECK_OK(balancer.AddWorkDurationMeasurement({{4, 2}, {4, 2}}).status());
  CHECK_OK(balancer.ApplyPendingOrder());
  LOG(INFO) << balancer.Info();
  // Two transfers of feature expected.
  EXPECT_THAT(balancer.WorkersPerFeatures(),
              ElementsAre(1, -1, 0, -1, 1, -1, 0));  // 2 in worker 0.
}

TEST(LoadBalancer, MakeSplitSharingPlan) {
  distributed_decision_tree::dataset_cache::proto::CacheMetadata
      cache_metadata = PARSE_TEST_PROTO(R"pb(
        columns { numerical { num_unique_values: 10 } }
        columns { numerical { num_unique_values: 10 } }
        columns { numerical { num_unique_values: 10 } }
        columns { numerical { num_unique_values: 10 } }
        columns { numerical { num_unique_values: 10 } }
        columns { numerical { num_unique_values: 10 } }
        columns { numerical { num_unique_values: 10 } }
        columns { numerical { num_unique_values: 10 } }
        columns { numerical { num_unique_values: 10 } }
        columns { numerical { num_unique_values: 10 } }
        columns { categorical { num_values: 3 } }
      )pb");
  auto balancer =
      LoadBalancer::Create(/*features=*/{0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10},
                           /*num_workers=*/10, cache_metadata, {})
          .value();
  // Features 9 and 10 are owned by worker #0.
  // Feature 8 is owned by worker #1.

  const auto plan_1 = balancer.MakeSplitSharingPlan({9}).value();
  const proto::SplitSharingPlan expected_plan_1 = PARSE_TEST_PROTO(R"pb(
    rounds {
      requests {
        key: 1
        value { items { src_worker: 0 features: 9 } }
      }
      requests {
        key: 2
        value { items { src_worker: 0 features: 9 } }
      }
      requests {
        key: 3
        value { items { src_worker: 0 features: 9 } }
      }
      requests {
        key: 4
        value { items { src_worker: 0 features: 9 } }
      }
    }
    rounds {
      requests {
        key: 0
        value { last_request_of_plan: true }
      }
      requests {
        key: 1
        value { last_request_of_plan: true }
      }
      requests {
        key: 2
        value { last_request_of_plan: true }
      }
      requests {
        key: 3
        value { last_request_of_plan: true }
      }
      requests {
        key: 4
        value { last_request_of_plan: true }
      }
      requests {
        key: 5
        value {
          items { src_worker: 0 features: 9 }
          last_request_of_plan: true
        }
      }
      requests {
        key: 6
        value {
          items { src_worker: 1 features: 9 }
          last_request_of_plan: true
        }
      }
      requests {
        key: 7
        value {
          items { src_worker: 2 features: 9 }
          last_request_of_plan: true
        }
      }
      requests {
        key: 8
        value {
          items { src_worker: 3 features: 9 }
          last_request_of_plan: true
        }
      }
      requests {
        key: 9
        value {
          items { src_worker: 4 features: 9 }
          last_request_of_plan: true
        }
      }
    }
  )pb");
  EXPECT_THAT(plan_1, EqualsProto(expected_plan_1));

  const auto plan_2 = balancer.MakeSplitSharingPlan({8, 9, 10}).value();
  const proto::SplitSharingPlan expected_plan_2 = PARSE_TEST_PROTO(R"pb(
    rounds {
      requests {
        key: 1
        value { items { src_worker: 0 features: 9 features: 10 } }
      }
      requests {
        key: 2
        value { items { src_worker: 0 features: 9 features: 10 } }
      }
      requests {
        key: 3
        value { items { src_worker: 0 features: 9 features: 10 } }
      }
      requests {
        key: 4
        value { items { src_worker: 0 features: 9 features: 10 } }
      }
      requests {
        key: 5
        value { items { src_worker: 1 features: 8 } }
      }
      requests {
        key: 6
        value { items { src_worker: 1 features: 8 } }
      }
      requests {
        key: 7
        value { items { src_worker: 1 features: 8 } }
      }
      requests {
        key: 8
        value { items { src_worker: 1 features: 8 } }
      }
    }
    rounds {
      requests {
        key: 0
        value {
          items { src_worker: 1 features: 8 }
          last_request_of_plan: true
        }
      }
      requests {
        key: 1
        value { last_request_of_plan: true }
      }
      requests {
        key: 2
        value {
          items { src_worker: 5 features: 8 }
          last_request_of_plan: true
        }
      }
      requests {
        key: 3
        value {
          items { src_worker: 6 features: 8 }
          last_request_of_plan: true
        }
      }
      requests {
        key: 4
        value {
          items { src_worker: 7 features: 8 }
          last_request_of_plan: true
        }
      }
      requests {
        key: 5
        value {
          items { src_worker: 0 features: 9 features: 10 }
          last_request_of_plan: true
        }
      }
      requests {
        key: 6
        value {
          items { src_worker: 1 features: 9 features: 10 }
          last_request_of_plan: true
        }
      }
      requests {
        key: 7
        value {
          items { src_worker: 2 features: 9 features: 10 }
          last_request_of_plan: true
        }
      }
      requests {
        key: 8
        value {
          items { src_worker: 3 features: 9 features: 10 }
          last_request_of_plan: true
        }
      }
      requests {
        key: 9
        value {
          items { src_worker: 4 features: 9 features: 10 }
          items { src_worker: 8 features: 8 }
          last_request_of_plan: true
        }
      }
    }
  )pb");
  EXPECT_THAT(plan_2, EqualsProto(expected_plan_2));
}

}  // namespace
}  // namespace distributed_decision_tree
}  // namespace model
}  // namespace yggdrasil_decision_forests
