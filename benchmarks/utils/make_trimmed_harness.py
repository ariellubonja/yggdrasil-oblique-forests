"""Derive the upstream-YDF harness (branch upstream-bench) from the fork's
examples/train_oblique_forest.cc by removing fork-only features (row-major
store, Dynamic split types, RM_MAX_ROWS, early exit) and adding the
"Training block took:" marker bench_common.sh parses.
Usage: make_trimmed_harness.py examples/train_oblique_forest.cc <worktree>/examples/train_oblique_forest.cc
Every edit is anchored on exact text and asserts one match, so a fork-side
harness change that moves an anchor fails loudly instead of silently."""
import sys, re, pathlib
src = pathlib.Path(sys.argv[1]).read_text(); s = src
def sub1(s, old, new, label):
    n = s.count(old); assert n == 1, f"{label}: expected 1 match, got {n}"; return s.replace(old, new)
def cut(s, start, end, new, label):
    assert s.count(start) == 1, label; i = s.index(start); j = s.index(end, i); return s[:i] + new + s[j:]
s = sub1(s, '#include "yggdrasil_decision_forests/dataset/row_major_feature_matrix.h"\n', '', 'rm include')
s = sub1(s, '''ABSL_FLAG(int, dynamic_split_threshold, 250,
          "When using dynamic histogram splits, switch to exact splitting if "
          "the number of examples at a node is below this threshold. "
          "Set to -1 to disable.");
''', '', 'dynamic flag')
s = sub1(s, '''ABSL_FLAG(std::string, dataset_layout, "column",
          "Hidden dataset-layout experiment: 'column' (default vertical "
          "store) or 'row' (fp32 row-major store feeding the projection "
          "loop). 'row' requires --config=row_major_dataset_layout.");
''', '', 'layout flag')
s = sub1(s, '''ABSL_FLAG(std::string, numerical_split_type, "Dynamic Random Histogram",
          "Type of histogram splitting: 'Exact (no histogramming)', 'Random', 'Equal Width', 'Dynamic Random Histogram' or 'Dynamic Equal Width Histogram.");''',
'''// Upstream-YDF build of the harness: the Dynamic (histogram-with-exact-fallback)
// types, the row-major store and the fork's env knobs are deliberately absent.
ABSL_FLAG(std::string, numerical_split_type, "Exact",
          "Numerical split search: 'Exact', 'Random' or 'Equal Width'.");''', 'split flag')
s = cut(s, 'struct RowMajorTrunkDataset {', '/* #endregion */\n\n/* #region Single-pass CSV load */', '', 'row-major helpers')
s = sub1(s, '''  if (std::getenv("RM_MAX_ROWS") == nullptr) {
    setenv("RM_MAX_ROWS", "5000", /*overwrite=*/0);
  }
  // The RF learner has a benchmark shortcut that exit(0)s right after the
  // training block (skipping model finalization + return) to speed up
  // pure-runtime experiments. Any post-training use of the returned model needs
  // it disabled: held-out test evaluation (--test_csv) or model saving
  // (--model_out_dir). GBT has no such shortcut, so this only affects RF.
  if (!absl::GetFlag(FLAGS_test_csv).empty() ||
      !absl::GetFlag(FLAGS_model_out_dir).empty()) {
    setenv("NO_EARLY_EXIT", "1", /*overwrite=*/1);
  }
  dataset::RowMajorFeatureMatrix::SetActive(nullptr);
''', '', 'env knobs')
s = cut(s, '  else if (mode == "trunk") {', '  else if (mode == "tfrecord") {', '''  else if (mode == "trunk") {
    LOG(INFO) << "Generating " << mode << " synthetic dataset: rows="
              << absl::GetFlag(FLAGS_rows)
              << ", cols=" << absl::GetFlag(FLAGS_cols)
              << ", layout=column";

    label_col = "y";
    data_spec = MakeSyntheticSpec(absl::GetFlag(FLAGS_cols),
                                  absl::GetFlag(FLAGS_rows),
                                  2, label_col);
    auto ds = BuildSyntheticDataset(mode, data_spec,
                                    absl::GetFlag(FLAGS_rows),
                                    absl::GetFlag(FLAGS_cols),
                                    absl::GetFlag(FLAGS_seed));
    tf_ds = std::make_unique<dataset::VerticalDataset>(std::move(ds));
    ds_ptr = tf_ds.get();
  }
''', 'trunk branch')
s = sub1(s, '''    sos->set_dynamic_split_threshold(
        absl::GetFlag(FLAGS_dynamic_split_threshold));
''', '', 'dynamic threshold')
s = cut(s, '  } else if (hist_type == "Dynamic Random Histogram") {', '   else {\n    std::cerr << "Unknown histogram type: "', '  }\n', 'dynamic branches')
s = sub1(s, '''              << ". Use 'Exact', 'Random', 'Equal Width', 'Dynamic Equal Width Histogram' or 'Dynamic Random Histogram'.\\n";''',
            '''              << ". Use 'Exact', 'Random' or 'Equal Width' (Dynamic types "
                 "exist only on the ariellubonja fork).\\n";''', 'hist error msg')
s = cut(s, '  if (mode == "csv") {\n#if defined(ROW_MAJOR_DATASET_LAYOUT)', '  } else {\n    model_or = learner->TrainWithStatus(*ds_ptr);', '''  if (mode == "csv") {
    if (ds_ptr != nullptr) {
      model_or = learner->TrainWithStatus(*ds_ptr);
    } else {
      model_or = learner->TrainWithStatus("csv:" + csv_path, data_spec);
    }
''', 'csv train block')
# Upstream learners do not print the fork's "Training block took:" line that
# bench_common.sh parses, so the harness emits it (TrainWithStatus wall time).
s = sub1(s, '''  LOG(INFO) << "train_oblique_forest wall-time - Training + Post-Processing: " << dur.count() << "s";
''', '''  LOG(INFO) << "train_oblique_forest wall-time - Training + Post-Processing: " << dur.count() << "s";
  // Same marker the fork's learners print; here it is TrainWithStatus wall time.
  LOG(INFO) << "train_oblique_forest.cc Training block took: " << dur.count() << " s";
''', 'training block marker')
s = sub1(s, '''      // The row-major training mirror (if any) must not shadow inference.
      dataset::RowMajorFeatureMatrix::SetActive(nullptr);
''', '', 'test-eval SetActive')
left = [(i+1, l.strip()) for i, l in enumerate(s.splitlines())
        if re.search(r'RowMajor|DYNAMIC_|dynamic_split_threshold|RM_MAX_ROWS|EARLY_EXIT|dataset_layout|ROW_MAJOR', l)]
for ln, l in left: print(f"LEFTOVER {ln}: {l}")
pathlib.Path(sys.argv[2]).write_text(s)
print(f"trimmed harness: {len(src.splitlines())} -> {len(s.splitlines())} lines; leftovers={len(left)}")
