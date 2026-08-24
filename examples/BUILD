package(
    default_visibility = ["//visibility:public"],
    licenses = ["notice"],
)

# Usage example of Yggdrasil Decision Forests as a library.
#
# Compile and run the example with:
#   bazel build //examples:beginner_cc --config=linux_cpp17 --config=linux_avx2
#   bazel-bin/examples/beginner_cc --alsologtostderr
#
# See the "Using the C++ library" section in the user manual for more details about the API.
# See the "Compile command-line-interface from source" section in the user manual for more details about the compilation flags.
#
cc_binary(
    name = "beginner_cc",
    srcs = ["beginner.cc"],
    # Because this binary is defined in the Yggdrasil project directly, the
    # following dependencies are local. In your project you will probably use
    # a "http_archive" dependency:
    # https://docs.bazel.build/versions/master/repo/http.html#http_archive*
    #
    # See TensorFlow Decision Forests for an example (TF-DF imports YDF):
    #   https://github.com/tensorflow/decision-forests
    deps = [
        "//yggdrasil_decision_forests/dataset:all_dataset_formats",
        "//yggdrasil_decision_forests/dataset:data_spec",
        "//yggdrasil_decision_forests/dataset:data_spec_cc_proto",
        "//yggdrasil_decision_forests/dataset:data_spec_inference",
        "//yggdrasil_decision_forests/dataset:vertical_dataset_io",
        "//yggdrasil_decision_forests/learner:all_learners",
        "//yggdrasil_decision_forests/learner:learner_library",
        "//yggdrasil_decision_forests/metric:report",
        "//yggdrasil_decision_forests/model:model_library",
        "//yggdrasil_decision_forests/utils:filesystem",
        "//yggdrasil_decision_forests/utils:logging",
        "@com_google_absl//absl/flags:flag",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/strings",
    ],
)

cc_binary(
    name = "fast_8bits_numerical",
    srcs = ["fast_8bits_numerical.cc"],
    deps = [
        "//yggdrasil_decision_forests/dataset:all_dataset_formats",
        "//yggdrasil_decision_forests/dataset:data_spec",
        "//yggdrasil_decision_forests/dataset:data_spec_cc_proto",
        "//yggdrasil_decision_forests/dataset:data_spec_inference",
        "//yggdrasil_decision_forests/dataset:example_cc_proto",
        "//yggdrasil_decision_forests/dataset:vertical_dataset",
        "//yggdrasil_decision_forests/dataset:vertical_dataset_io",
        "//yggdrasil_decision_forests/learner:learner_library",
        "//yggdrasil_decision_forests/learner/gradient_boosted_trees",
        "//yggdrasil_decision_forests/metric",
        "//yggdrasil_decision_forests/metric:report",
        "//yggdrasil_decision_forests/model:model_library",
        "//yggdrasil_decision_forests/serving/decision_forest:8bits_numerical_features",
        "//yggdrasil_decision_forests/utils:filesystem",
        "//yggdrasil_decision_forests/utils:logging",
        "@com_google_absl//absl/flags:flag",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/strings",
    ],
)

# Benchmark harness ported from the oblique-forests fork (branch ydf-may-2025).
# Build:
#   bazel build -c opt --config=linux_cpp17 --copt=-march=native --cxxopt=-O3 \
#     //examples:train_oblique_forest
cc_binary(
    name = "train_oblique_forest",
    srcs = ["train_oblique_forest.cc"],
    deps = [
        "//yggdrasil_decision_forests/dataset:all_dataset_formats",
        "//yggdrasil_decision_forests/dataset:data_spec",
        "//yggdrasil_decision_forests/dataset:data_spec_cc_proto",
        "//yggdrasil_decision_forests/dataset:data_spec_inference",
        "//yggdrasil_decision_forests/dataset:vertical_dataset",
        "//yggdrasil_decision_forests/dataset:vertical_dataset_io",
        "//yggdrasil_decision_forests/learner:all_learners",
        "//yggdrasil_decision_forests/learner:learner_library",
        "//yggdrasil_decision_forests/learner/decision_tree:training",
        "//yggdrasil_decision_forests/learner/gradient_boosted_trees:gradient_boosted_trees_cc_proto",
        "//yggdrasil_decision_forests/learner/random_forest:random_forest_cc_proto",
        "//yggdrasil_decision_forests/metric",
        "//yggdrasil_decision_forests/metric:metric_cc_proto",
        "//yggdrasil_decision_forests/metric:report",
        "//yggdrasil_decision_forests/model:model_library",
        "//yggdrasil_decision_forests/utils:logging",
        "//yggdrasil_decision_forests/utils:random",
        "@com_google_absl//absl/flags:flag",
        "@com_google_absl//absl/flags:parse",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
    ],
)
