# Tell Bazel where to find the rule definition.
load("@hedron_compile_commands//:refresh_compile_commands.bzl",
     "refresh_compile_commands")

# Generate compile_commands.json only for the target(s) we care about.
#
# --features=-parse_headers,-layering_check: newer rules_cc emits header-only
# "syntax-check" compile actions (e.g. `-xc++-header ... ascii.h`) that the
# Hedron extractor cannot parse (it errors with "No source files found in
# compile args"). Disabling these features during extraction skips those
# actions; it has no effect on normal `bazel build`.
# --host_features=...: the same actions also appear in the exec/host config
# (build tooling like protoc plugins), which --features does NOT cover, so we
# must disable them there too.
_NO_HEADER_PARSE = (
    "--features=-parse_headers --features=-layering_check " +
    "--host_features=-parse_headers --host_features=-layering_check"
)

refresh_compile_commands(
    name = "refresh_compile_commands",
    targets = {
        "//examples:train_oblique_forest": _NO_HEADER_PARSE,
        "//yggdrasil_decision_forests/learner/decision_tree:training": _NO_HEADER_PARSE,
    },
)