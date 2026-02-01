#!/bin/bash

set -e # Stop on 1st error

histogram_num_bins=256


logdir="benchmarks/results"

ts=$(date +%Y%m%d_%H%M%S)              # time-stamp for uniqueness
logfile="${logdir}/accuracy_runtime_${ts}.log"

numerical_split_type="Exact"

printf "\n\n--------Running all $numerical_split_type w/ $histogram_num_bins 2>&1 | tee -a "$logfile" bins-------\n\n"

./bazel-bin/examples/train_oblique_forest --input_mode csv --train_csv benchmarks/data/HIGGS_with_header.csv --label_col class --num_trees 100 --num_threads -1 --numerical_split_type "$numerical_split_type" --histogram_num_bins "$histogram_num_bins" 2>&1 | tee -a "$logfile"
# ./bazel-bin/examples/train_oblique_forest --input_mode csv --train_csv benchmarks/data/SUSY_with_header.csv --label_col class --num_trees 100 --num_threads -1 --numerical_split_type "$numerical_split_type" --histogram_num_bins "$histogram_num_bins" 2>&1 | tee -a "$logfile"

# ./bazel-bin/examples/train_oblique_forest --input_mode csv --num_trees 100 --num_threads -1 --numerical_split_type "$numerical_split_type" 2>&1 | tee -a "$logfile"
# ./bazel-bin/examples/train_oblique_forest --input_mode csv --num_trees 100 --num_threads -1 --train_csv "benchmarks/data/cc18_binary_csv/task_14965_bank-marketing/repeat0_fold0_sample0_train.csv" --label_col Class --numerical_split_type "$numerical_split_type" 2>&1 | tee -a "$logfile"
# ./bazel-bin/examples/train_oblique_forest --input_mode csv --num_trees 100 --num_threads -1 --train_csv "benchmarks/data/cc18_binary_csv/task_14952_PhishingWebsites/repeat0_fold0_sample0_train.csv" --label_col Result --numerical_split_type "$numerical_split_type" 2>&1 | tee -a "$logfile"
# ./bazel-bin/examples/train_oblique_forest --input_mode csv --num_trees 100 --num_threads -1 --train_csv "benchmarks/data/cc18_binary_csv/task_29_credit-approval/repeat0_fold0_sample0_train.csv" --label_col class --numerical_split_type "$numerical_split_type" 2>&1 | tee -a "$logfile"
# ./bazel-bin/examples/train_oblique_forest --input_mode csv --num_trees 100 --num_threads -1 --train_csv "benchmarks/data/cc18_binary_csv/task_167125_Internet-Advertisements/repeat0_fold0_sample0_train.csv" --label_col class --numerical_split_type "$numerical_split_type" 2>&1 | tee -a "$logfile"

# ./bazel-bin/examples/train_oblique_forest --input_mode trunk --rows 10000 --num_trees 100 --num_threads -1 --numerical_split_type "$numerical_split_type" 2>&1 | tee -a "$logfile"
# ./bazel-bin/examples/train_oblique_forest --input_mode trunk --rows 100000 --num_trees 100 --num_threads -1 --numerical_split_type "$numerical_split_type" 2>&1 | tee -a "$logfile"
# ./bazel-bin/examples/train_oblique_forest --input_mode trunk --rows 1000000 --num_trees 100 --num_threads -1 --numerical_split_type "$numerical_split_type" 2>&1 | tee -a "$logfile"


numerical_split_type="Equal Width"

printf "\n\n--------Running all $numerical_split_type w/ $histogram_num_bins 2>&1 | tee -a "$logfile" bins-------\n\n"

./bazel-bin/examples/train_oblique_forest --input_mode csv --train_csv benchmarks/data/HIGGS_with_header.csv --label_col class --num_trees 100 --num_threads -1 --numerical_split_type "$numerical_split_type" --histogram_num_bins "$histogram_num_bins" 2>&1 | tee -a "$logfile"
# ./bazel-bin/examples/train_oblique_forest --input_mode csv --train_csv benchmarks/data/SUSY_with_header.csv --label_col class --num_trees 100 --num_threads -1 --numerical_split_type "$numerical_split_type" --histogram_num_bins "$histogram_num_bins" 2>&1 | tee -a "$logfile"

# ./bazel-bin/examples/train_oblique_forest --input_mode csv --num_trees 100 --num_threads -1 --numerical_split_type "$numerical_split_type" --histogram_num_bins "$histogram_num_bins" 2>&1 | tee -a "$logfile"
# ./bazel-bin/examples/train_oblique_forest --input_mode csv --num_trees 100 --num_threads -1 --train_csv "benchmarks/data/cc18_binary_csv/task_14965_bank-marketing/repeat0_fold0_sample0_train.csv" --label_col Class --numerical_split_type "$numerical_split_type" --histogram_num_bins "$histogram_num_bins"
# ./bazel-bin/examples/train_oblique_forest --input_mode csv --num_trees 100 --num_threads -1 --train_csv "benchmarks/data/cc18_binary_csv/task_14952_PhishingWebsites/repeat0_fold0_sample0_train.csv" --label_col Result --numerical_split_type "$numerical_split_type" --histogram_num_bins "$histogram_num_bins"
# ./bazel-bin/examples/train_oblique_forest --input_mode csv --num_trees 100 --num_threads -1 --train_csv "benchmarks/data/cc18_binary_csv/task_29_credit-approval/repeat0_fold0_sample0_train.csv" --label_col class --numerical_split_type "$numerical_split_type" --histogram_num_bins "$histogram_num_bins"
# ./bazel-bin/examples/train_oblique_forest --input_mode csv --num_trees 100 --num_threads -1 --train_csv "benchmarks/data/cc18_binary_csv/task_167125_Internet-Advertisements/repeat0_fold0_sample0_train.csv" --label_col class --numerical_split_type "$numerical_split_type" --histogram_num_bins "$histogram_num_bins"

# ./bazel-bin/examples/train_oblique_forest --input_mode trunk --rows 10000 --num_trees 100 --num_threads -1 --numerical_split_type "$numerical_split_type" --histogram_num_bins "$histogram_num_bins" 2>&1 | tee -a "$logfile"
# ./bazel-bin/examples/train_oblique_forest --input_mode trunk --rows 100000 --num_trees 100 --num_threads -1 --numerical_split_type "$numerical_split_type" --histogram_num_bins "$histogram_num_bins" 2>&1 | tee -a "$logfile"
# ./bazel-bin/examples/train_oblique_forest --input_mode trunk --rows 1000000 --num_trees 100 --num_threads -1 --numerical_split_type "$numerical_split_type" --histogram_num_bins "$histogram_num_bins" 2>&1 | tee -a "$logfile"


numerical_split_type="Random"

printf "\n\n--------Running all $numerical_split_type w/ $histogram_num_bins 2>&1 | tee -a "$logfile" bins-------\n\n"

./bazel-bin/examples/train_oblique_forest --input_mode csv --train_csv benchmarks/data/HIGGS_with_header.csv --label_col class --num_trees 100 --num_threads -1 --numerical_split_type "$numerical_split_type" --histogram_num_bins "$histogram_num_bins" 2>&1 | tee -a "$logfile"
# ./bazel-bin/examples/train_oblique_forest --input_mode csv --train_csv benchmarks/data/SUSY_with_header.csv --label_col class --num_trees 100 --num_threads -1 --numerical_split_type "$numerical_split_type" --histogram_num_bins "$histogram_num_bins" 2>&1 | tee -a "$logfile"

# ./bazel-bin/examples/train_oblique_forest --input_mode csv --num_trees 100 --num_threads -1 --numerical_split_type "$numerical_split_type" --histogram_num_bins "$histogram_num_bins" 2>&1 | tee -a "$logfile"
# ./bazel-bin/examples/train_oblique_forest --input_mode csv --num_trees 100 --num_threads -1 --train_csv "benchmarks/data/cc18_binary_csv/task_14965_bank-marketing/repeat0_fold0_sample0_train.csv" --label_col Class --numerical_split_type "$numerical_split_type" --histogram_num_bins "$histogram_num_bins"
# ./bazel-bin/examples/train_oblique_forest --input_mode csv --num_trees 100 --num_threads -1 --train_csv "benchmarks/data/cc18_binary_csv/task_14952_PhishingWebsites/repeat0_fold0_sample0_train.csv" --label_col Result --numerical_split_type "$numerical_split_type" --histogram_num_bins "$histogram_num_bins"
# ./bazel-bin/examples/train_oblique_forest --input_mode csv --num_trees 100 --num_threads -1 --train_csv "benchmarks/data/cc18_binary_csv/task_29_credit-approval/repeat0_fold0_sample0_train.csv" --label_col class --numerical_split_type "$numerical_split_type" --histogram_num_bins "$histogram_num_bins"
# ./bazel-bin/examples/train_oblique_forest --input_mode csv --num_trees 100 --num_threads -1 --train_csv "benchmarks/data/cc18_binary_csv/task_167125_Internet-Advertisements/repeat0_fold0_sample0_train.csv" --label_col class --numerical_split_type "$numerical_split_type" --histogram_num_bins "$histogram_num_bins"

# ./bazel-bin/examples/train_oblique_forest --input_mode trunk --rows 10000 --num_trees 100 --num_threads -1 --numerical_split_type "$numerical_split_type" --histogram_num_bins "$histogram_num_bins" 2>&1 | tee -a "$logfile"
# ./bazel-bin/examples/train_oblique_forest --input_mode trunk --rows 100000 --num_trees 100 --num_threads -1 --numerical_split_type "$numerical_split_type" --histogram_num_bins "$histogram_num_bins" 2>&1 | tee -a "$logfile"
# ./bazel-bin/examples/train_oblique_forest --input_mode trunk --rows 1000000 --num_trees 100 --num_threads -1 --numerical_split_type "$numerical_split_type" --histogram_num_bins "$histogram_num_bins" 2>&1 | tee -a "$logfile"

numerical_split_type="Dynamic Equal Width Histogram"

printf "\n\n--------Running all $numerical_split_type w/ $histogram_num_bins 2>&1 | tee -a "$logfile" bins-------\n\n"

./bazel-bin/examples/train_oblique_forest --input_mode csv --train_csv benchmarks/data/HIGGS_with_header.csv --label_col class --num_trees 100 --num_threads -1 --numerical_split_type "$numerical_split_type" --histogram_num_bins "$histogram_num_bins" 2>&1 | tee -a "$logfile"
# ./bazel-bin/examples/train_oblique_forest --input_mode csv --train_csv benchmarks/data/SUSY_with_header.csv --label_col class --num_trees 100 --num_threads -1 --numerical_split_type "$numerical_split_type" --histogram_num_bins "$histogram_num_bins" 2>&1 | tee -a "$logfile"

# ./bazel-bin/examples/train_oblique_forest --input_mode csv --num_trees 100 --num_threads -1 --numerical_split_type "$numerical_split_type" --histogram_num_bins "$histogram_num_bins" 2>&1 | tee -a "$logfile"
# ./bazel-bin/examples/train_oblique_forest --input_mode csv --num_trees 100 --num_threads -1 --train_csv "benchmarks/data/cc18_binary_csv/task_14965_bank-marketing/repeat0_fold0_sample0_train.csv" --label_col Class --numerical_split_type "$numerical_split_type" --histogram_num_bins "$histogram_num_bins"
# ./bazel-bin/examples/train_oblique_forest --input_mode csv --num_trees 100 --num_threads -1 --train_csv "benchmarks/data/cc18_binary_csv/task_14952_PhishingWebsites/repeat0_fold0_sample0_train.csv" --label_col Result --numerical_split_type "$numerical_split_type" --histogram_num_bins "$histogram_num_bins"
# ./bazel-bin/examples/train_oblique_forest --input_mode csv --num_trees 100 --num_threads -1 --train_csv "benchmarks/data/cc18_binary_csv/task_29_credit-approval/repeat0_fold0_sample0_train.csv" --label_col class --numerical_split_type "$numerical_split_type" --histogram_num_bins "$histogram_num_bins"
# ./bazel-bin/examples/train_oblique_forest --input_mode csv --num_trees 100 --num_threads -1 --train_csv "benchmarks/data/cc18_binary_csv/task_167125_Internet-Advertisements/repeat0_fold0_sample0_train.csv" --label_col class --numerical_split_type "$numerical_split_type" --histogram_num_bins "$histogram_num_bins"

# ./bazel-bin/examples/train_oblique_forest --input_mode trunk --rows 10000 --num_trees 100 --num_threads -1 --numerical_split_type "$numerical_split_type" --histogram_num_bins "$histogram_num_bins" 2>&1 | tee -a "$logfile"
# ./bazel-bin/examples/train_oblique_forest --input_mode trunk --rows 100000 --num_trees 100 --num_threads -1 --numerical_split_type "$numerical_split_type" --histogram_num_bins "$histogram_num_bins" 2>&1 | tee -a "$logfile"
# ./bazel-bin/examples/train_oblique_forest --input_mode trunk --rows 1000000 --num_trees 100 --num_threads -1 --numerical_split_type "$numerical_split_type" --histogram_num_bins "$histogram_num_bins" 2>&1 | tee -a "$logfile"


numerical_split_type="Dynamic Random Histogram"

printf "\n\n--------Running all $numerical_split_type w/ $histogram_num_bins 2>&1 | tee -a "$logfile" bins-------\n\n"

./bazel-bin/examples/train_oblique_forest --input_mode csv --train_csv benchmarks/data/HIGGS_with_header.csv --label_col class --num_trees 100 --num_threads -1 --numerical_split_type "$numerical_split_type" --histogram_num_bins "$histogram_num_bins" 2>&1 | tee -a "$logfile"
# ./bazel-bin/examples/train_oblique_forest --input_mode csv --train_csv benchmarks/data/SUSY_with_header.csv --label_col class --num_trees 100 --num_threads -1 --numerical_split_type "$numerical_split_type" --histogram_num_bins "$histogram_num_bins" 2>&1 | tee -a "$logfile"

# ./bazel-bin/examples/train_oblique_forest --input_mode csv --num_trees 100 --num_threads -1 --numerical_split_type "$numerical_split_type" --histogram_num_bins "$histogram_num_bins" 2>&1 | tee -a "$logfile"
# ./bazel-bin/examples/train_oblique_forest --input_mode csv --num_trees 100 --num_threads -1 --train_csv "benchmarks/data/cc18_binary_csv/task_14965_bank-marketing/repeat0_fold0_sample0_train.csv" --label_col Class --numerical_split_type "$numerical_split_type" --histogram_num_bins "$histogram_num_bins"
# ./bazel-bin/examples/train_oblique_forest --input_mode csv --num_trees 100 --num_threads -1 --train_csv "benchmarks/data/cc18_binary_csv/task_14952_PhishingWebsites/repeat0_fold0_sample0_train.csv" --label_col Result --numerical_split_type "$numerical_split_type" --histogram_num_bins "$histogram_num_bins"
# ./bazel-bin/examples/train_oblique_forest --input_mode csv --num_trees 100 --num_threads -1 --train_csv "benchmarks/data/cc18_binary_csv/task_29_credit-approval/repeat0_fold0_sample0_train.csv" --label_col class --numerical_split_type "$numerical_split_type" --histogram_num_bins "$histogram_num_bins"
# ./bazel-bin/examples/train_oblique_forest --input_mode csv --num_trees 100 --num_threads -1 --train_csv "benchmarks/data/cc18_binary_csv/task_167125_Internet-Advertisements/repeat0_fold0_sample0_train.csv" --label_col class --numerical_split_type "$numerical_split_type" --histogram_num_bins "$histogram_num_bins"

# ./bazel-bin/examples/train_oblique_forest --input_mode trunk --rows 10000 --num_trees 100 --num_threads -1 --numerical_split_type "$numerical_split_type" --histogram_num_bins "$histogram_num_bins" 2>&1 | tee -a "$logfile"
# ./bazel-bin/examples/train_oblique_forest --input_mode trunk --rows 100000 --num_trees 100 --num_threads -1 --numerical_split_type "$numerical_split_type" --histogram_num_bins "$histogram_num_bins" 2>&1 | tee -a "$logfile"
# ./bazel-bin/examples/train_oblique_forest --input_mode trunk --rows 1000000 --num_trees 100 --num_threads -1 --numerical_split_type "$numerical_split_type" --histogram_num_bins "$histogram_num_bins" 2>&1 | tee -a "$logfile"