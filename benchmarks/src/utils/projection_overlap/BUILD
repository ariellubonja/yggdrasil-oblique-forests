package(default_visibility = ["//visibility:private"])

cc_binary(
    name = "projection_overlap",
    srcs = ["projection_overlap.cc"],
    deps = [
        "//yggdrasil_decision_forests/dataset:data_spec_cc_proto",
        "//yggdrasil_decision_forests/learner:abstract_learner_cc_proto",
        "//yggdrasil_decision_forests/learner/decision_tree:decision_tree_cc_proto",
        "//yggdrasil_decision_forests/learner/decision_tree:training",
        "//yggdrasil_decision_forests/utils:random",
    ],
)
