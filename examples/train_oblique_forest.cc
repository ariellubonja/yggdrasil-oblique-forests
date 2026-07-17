#include <iostream>
#include <string>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <fstream>
#include <limits>
#include <map>
#include <memory>
#include <random>
#include <thread>
#include <utility>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/flags/flag.h"
#include "absl/flags/parse.h"
#include "absl/strings/ascii.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_split.h"

#include "yggdrasil_decision_forests/dataset/data_spec_inference.h"
#include "yggdrasil_decision_forests/dataset/data_spec.h"
#include "yggdrasil_decision_forests/dataset/data_spec.pb.h"
#include "yggdrasil_decision_forests/dataset/row_major_feature_matrix.h"
#include "yggdrasil_decision_forests/dataset/vertical_dataset.h"
#include "yggdrasil_decision_forests/dataset/vertical_dataset_io.h"
#include "yggdrasil_decision_forests/learner/learner_library.h"
#include "yggdrasil_decision_forests/learner/random_forest/random_forest.pb.h"
#include "yggdrasil_decision_forests/learner/gradient_boosted_trees/gradient_boosted_trees.pb.h"
#include "yggdrasil_decision_forests/model/model_library.h"
// #include "yggdrasil_decision_forests/utils/status_macros.h"
#include "yggdrasil_decision_forests/utils/filesystem.h"
#include "yggdrasil_decision_forests/learner/decision_tree/oblique.h"

#include <random>

// WARNING: Abseil does not do input seeding! Multiple runs have non-deterministic outputs
// #include "absl/random/random.h"          // BitGen (xoshiro)
// #include "absl/random/distributions.h"   // Gaussian()


/* #region ABSL Flags */
// Input mode flag: "csv", "synthetic", or "tfrecord"
ABSL_FLAG(std::string, input_mode, "",
          "Data input mode: csv, Uniform synthetic, Trunk Synthetic, or tfrecord.");
// CSV mode flags
ABSL_FLAG(std::string, train_csv, "./benchmarks/data/processed_wise1_data.csv",
          "Path to training CSV file (for csv mode). Must include --label_col.");
// TFRecord mode flags
ABSL_FLAG(std::string, ds_path, "",
          "Path (without extension) to TF-Record file (for tfrecord mode).");
// Common flags
ABSL_FLAG(std::string, label_col, "Cancer Status",
          "Name of label column (used in all modes).");
ABSL_FLAG(std::string, model_out_dir, "",
          "Path to output trained model directory (optional)."
          " If empty, model is not saved.");
ABSL_FLAG(int, num_threads, 1, "Number of threads to use.");
ABSL_FLAG(int, num_trees, 240, "Number of trees in the random forest.");
ABSL_FLAG(int, tree_depth, -1,
          "Maximum depth of trees (-1 for unlimited).");

ABSL_FLAG(std::string, feature_split_type, "Oblique",
          "Type of feature splits in decision trees: 'Axis Aligned' or 'Oblique'.");

// Only consulted when --feature_split_type='Axis Aligned'. Oblique mode is
// already forced to IN_NODE by SetInternalDefaultHyperParameters
// (training.cc:4587), so this flag has no effect there.
ABSL_FLAG(std::string, aa_sorting_strategy, "in_node",
          "Sorting strategy for AA splits: 'in_node' (default — no presort), "
          "'presorted', 'force_presorted', or 'auto'. Presorting allocates a "
          "second N*sizeof(ExampleIdx) buffer per feature (~49 GB on a "
          "3M-row, 4096-feature dataset) and is only useful for EXACT splits, "
          "so keep 'in_node' for AA + Random/Histogram.");

// Oblique split parameters (only used when feature_split_type = "Oblique")
ABSL_FLAG(int, max_num_projections, 1000,
          "Maximum number of projections for oblique splits.");
ABSL_FLAG(float, projection_density_factor, 1.5f,
          "Projection density factor.");
ABSL_FLAG(float, num_projections_exponent, .5,
          "Exponent to determine number of projections.");
ABSL_FLAG(int, dynamic_split_threshold, 250,
          "When using dynamic histogram splits, switch to exact splitting if "
          "the number of examples at a node is below this threshold. "
          "Set to -1 to disable.");

ABSL_FLAG(std::string, ensemble_method, "Bagging",
          "Ensemble method: 'Bagging' (Random Forest) or 'Boosting' (Gradient Boosted Trees/MART).");
ABSL_FLAG(float, shrinkage, 0.1f,
          "Learning rate for boosting (only used when ensemble_method = 'Boosting').");

// Hao uses GlobalBestFirst
ABSL_FLAG(std::string, growing_strategy, "Local",
          "Type of Tree Growing Strategy: 'Local' - depth-first using NodeTrain or 'GlobalBestFirst' - PriorityQueue the nodes based on Score() Gain.");

ABSL_FLAG(bool, bootstrap_training_dataset, true,
          "Whether to use bootstrap sampling of the training dataset (Bagging only). NOTE: NO BOOTSTRAP USES ~50% MORE MEMORY. POSSIBLE BUG");
ABSL_FLAG(bool, compute_oob_performances, false,
          "Whether to compute out-of-bag performances (only for csv mode).");

// Synthetic mode flags
ABSL_FLAG(int64_t, rows, 4096, "Number of examples (for synthetic mode).");
ABSL_FLAG(int, cols, 4096, "Number of numerical features (for synthetic mode).");
ABSL_FLAG(int, label_mod, 2,
          "Number of classes (labels are 1..label_mod, for synthetic mode).");
ABSL_FLAG(uint32_t, seed, 1234,
          "PRNG seed (for deterministic synthetic mode and model training).");
ABSL_FLAG(std::string, dataset_layout, "column",
          "Hidden dataset-layout experiment: 'column' (default vertical "
          "store) or 'row' (fp32 row-major store feeding the projection "
          "loop). 'row' requires --config=row_major_dataset_layout.");

// When true (default), CSV mode reads the file ONCE (build the dataspec from
// the in-memory columns) instead of YDF's default two full reads
// (CreateDataSpec's statistics pass + LoadVerticalDataset). Automatically falls
// back to the two-read path when the fast path does not apply (string labels,
// categorical feature columns, degenerate labels). Set to false to force the
// original two-read behaviour (useful for A/B correctness checks).
ABSL_FLAG(bool, csv_single_pass_load, true,
          "Load CSV in a single file scan instead of YDF's default two reads.");

// Histogram-based splits - Updated to match Yggdrasil implementation
ABSL_FLAG(std::string, numerical_split_type, "Dynamic Random Histogram",
          "Type of histogram splitting: 'Exact (no histogramming)', 'Random', 'Equal Width', 'Dynamic Random Histogram' or 'Dynamic Equal Width Histogram.");
ABSL_FLAG(int, histogram_num_bins, 64,
          "Number of bins for histogram splitting.");
// Commented out until hybrid CPU-GPU offloading is added. For now, GPU vs CPU
// is chosen purely at compile time via --config=oblique_gpu.
// ABSL_FLAG(bool, use_gpu, false,
//           "Use GPU for oblique projection computation.");

using namespace yggdrasil_decision_forests;

/* #endregion */


/* #region Synthetic Dataset Generation */

enum class SynthType { kUniform, kTrunk };

inline SynthType ParseSynthType(const std::string& s) {
  if (s == "uniform") return SynthType::kUniform;
  if (s == "trunk")   return SynthType::kTrunk;
  LOG(FATAL) << "Unknown synthetic type: " << s;
  return SynthType::kUniform;  // never reached; avoid warning
}

// Build a DataSpecification for synthetic data
dataset::proto::DataSpecification MakeSyntheticSpec(
    int cols, int64_t rows, int label_mod, const std::string &label_col) {
  dataset::proto::DataSpecification spec;
  for (int c = 0; c < cols; ++c) {
    auto* f = spec.add_columns();
    f->set_name("x" + std::to_string(c));
    f->set_type(dataset::proto::NUMERICAL);
  }
  auto* lbl = spec.add_columns();
  lbl->set_name(label_col);
  lbl->set_type(dataset::proto::CATEGORICAL);
  lbl->mutable_categorical()->set_number_of_unique_values(3); // OOD, 1, 2
  lbl->mutable_categorical()->set_is_already_integerized(true);
  spec.set_created_num_rows(rows);
  return spec;
}

// TODO Abseil does not give deterministic RNG - fix it!
// ------------------------------------------------------------------ uniform
// dataset::VerticalDataset MakeUniformDataset(
//         const dataset::proto::DataSpecification& spec,
//         int64_t rows, int cols, uint32_t seed) {
//   dataset::VerticalDataset ds;
//   ds.set_data_spec(spec);
//   CHECK_OK(ds.CreateColumnsFromDataspec());
//   ds.Resize(rows);
// #pragma omp parallel for schedule(static)
//   for (int c = 0; c < cols; ++c) {
//     std::seed_seq seq{seed + static_cast<uint32_t>(c)};
//     absl::BitGen gen(seq);
//     auto* col =
//         ds.MutableColumnWithCast<
//             dataset::VerticalDataset::NumericalColumn>(c)->mutable_values();
//     for (auto& v : *col) v = absl::Gaussian<float>(gen);
//   }
//   // labels: round-robin 0,1,…
//   auto* ycol = ds.MutableColumnWithCast<
//       dataset::VerticalDataset::CategoricalColumn>(cols);
//   auto* yval = ycol->mutable_values();
//   for (int64_t i = 0; i < rows; ++i) (*yval)[i] = static_cast<int>((i & 1) + 1);    // 1 or 2, not 0/1
//   return ds;
// }

// -------------------------------------------------------------------- trunk
dataset::VerticalDataset MakeTrunkDataset(const dataset::proto::DataSpecification& spec,
                                          int64_t rows, int cols, uint32_t seed) {
  dataset::VerticalDataset ds;
  ds.set_data_spec(spec);
  CHECK_OK(ds.CreateColumnsFromDataspec());
  ds.Resize(rows);
  using RNG = std::minstd_rand;

  constexpr int kNInformative = 256;
  const int ninform = std::min(kNInformative, cols);

  // Pre–compute the two means for the informative coordinates.
  std::vector<float> mu0(cols, 0.f), mu1(cols, 0.f);
  for (int j = 0; j < ninform; ++j) {
    const float f = 1.f / std::sqrt(float(j + 1));
    mu0[j] = -f;
    mu1[j] =  f;
  }

  // Fill the feature columns -------------------------------------------------
  for (int j = 0; j < cols; ++j) {
    // Deterministic per-column seed
    std::seed_seq seq{seed, static_cast<uint32_t>(j)};
    RNG rng(seq);
    std::normal_distribution<float> normal(0.0f, 1.0f);

    auto* col = ds.MutableColumnWithCast<
        dataset::VerticalDataset::NumericalColumn>(j)->mutable_values();

    for (int64_t i = 0; i < rows; ++i) {
      const bool cls1 = (i >= rows / 2);
      const float mean = cls1 ? mu1[j] : mu0[j];
      (*col)[i] = mean + normal(rng);
    }
  }

  // Fill the label column ----------------------------------------------------
  auto* y = ds.MutableColumnWithCast<
      dataset::VerticalDataset::CategoricalColumn>(cols)->mutable_values();
  for (int64_t i = 0; i < rows; ++i)
    (*y)[i] = (i >= rows / 2) ? 2 : 1;                  // 1-based labels

  return ds;
}

dataset::VerticalDataset BuildSyntheticDataset(
        const std::string& mode,
        const dataset::proto::DataSpecification& spec,
        int64_t rows, int cols, uint32_t seed) {
  // if (mode == "uniform") return MakeUniformDataset(spec, rows, cols, seed);
  if (mode == "trunk")   return MakeTrunkDataset(spec, rows, cols, seed);
  LOG(FATAL) << "Unknown synthetic mode: " << mode;
  return {};  // never reached
}

struct RowMajorTrunkDataset {
  std::unique_ptr<dataset::VerticalDataset> vd;
  std::unique_ptr<dataset::RowMajorFeatureMatrix> matrix;
};

template <typename Matrix>
void FillTrunkMatrix(Matrix* matrix, int64_t rows, int cols, uint32_t seed) {
  using RNG = std::minstd_rand;
  constexpr int kNInformative = 256;
  const int ninform = std::min(kNInformative, cols);

  std::vector<float> mu0(cols, 0.f), mu1(cols, 0.f);
  for (int j = 0; j < ninform; ++j) {
    const float f = 1.f / std::sqrt(float(j + 1));
    mu0[j] = -f;
    mu1[j] = f;
  }

  const int n_threads = std::max(1u, std::thread::hardware_concurrency());
  auto fill_range = [&](int j_begin, int j_end) {
    for (int j = j_begin; j < j_end; ++j) {
      std::seed_seq seq{seed, static_cast<uint32_t>(j)};
      RNG rng(seq);
      std::normal_distribution<float> normal(0.0f, 1.0f);
      const float m0 = mu0[j];
      const float m1 = mu1[j];
      for (int64_t i = 0; i < rows; ++i) {
        const bool cls1 = (i >= rows / 2);
        matrix->Set(i, j, (cls1 ? m1 : m0) + normal(rng));
      }
    }
  };

  std::vector<std::thread> workers;
  workers.reserve(n_threads);
  const int chunk = (cols + n_threads - 1) / n_threads;
  for (int t = 0; t < n_threads; ++t) {
    const int j_begin = t * chunk;
    const int j_end = std::min(cols, j_begin + chunk);
    if (j_begin >= j_end) break;
    workers.emplace_back(fill_range, j_begin, j_end);
  }
  for (auto& worker : workers) worker.join();
}

std::unique_ptr<dataset::VerticalDataset> MakeLabelOnlyTrunkDataset(
    const dataset::proto::DataSpecification& spec, int64_t rows, int cols) {
  auto vd = std::make_unique<dataset::VerticalDataset>();
  vd->set_data_spec(spec);
  CHECK_OK(vd->CreateColumnsFromDataspec());
  vd->set_nrow(rows);

  auto* y_col =
      vd->MutableColumnWithCast<dataset::VerticalDataset::CategoricalColumn>(
          cols);
  y_col->mutable_values()->resize(rows);
  auto* y = y_col->mutable_values();
  for (int64_t i = 0; i < rows; ++i) {
    (*y)[i] = (i >= rows / 2) ? 2 : 1;
  }
  return vd;
}

RowMajorTrunkDataset MakeTrunkDatasetRowMajor(
    const dataset::proto::DataSpecification& spec, int64_t rows, int cols,
    uint32_t seed) {
  auto matrix = std::make_unique<dataset::RowMajorFeatureMatrix>(rows, cols);
  FillTrunkMatrix(matrix.get(), rows, cols, seed);
  return {MakeLabelOnlyTrunkDataset(spec, rows, cols), std::move(matrix)};
}

// Mirrors the numerical columns of a loaded VerticalDataset into a feature
// matrix indexed by raw dataspec column index (non-numerical slots stay
// zero). NaN is replaced by the column mean at copy time — the same
// substitution stock Evaluate applies per lookup — so kernels never see NaN.
// Returns the number of mirrored columns.
template <typename Matrix>
int FillMatrixFromDataset(Matrix* matrix,
                          const dataset::VerticalDataset& ds) {
  const auto& spec = ds.data_spec();
  std::vector<int> numerical_cols;
  for (int col = 0; col < spec.columns_size(); ++col) {
    if (spec.columns(col).type() != dataset::proto::ColumnType::NUMERICAL) {
      continue;
    }
    if (!ds.ColumnWithCastWithStatus<dataset::VerticalDataset::NumericalColumn>(
              col)
             .ok()) {
      continue;
    }
    numerical_cols.push_back(col);
  }

  const int64_t nrow = ds.nrow();
  const int n_threads = std::max(1u, std::thread::hardware_concurrency());
  auto fill_range = [&](size_t begin, size_t end) {
    for (size_t k = begin; k < end; ++k) {
      const int col = numerical_cols[k];
      const auto& values =
          ds.ColumnWithCastWithStatus<dataset::VerticalDataset::NumericalColumn>(
                col)
              .value()
              ->values();
      const float na = spec.columns(col).numerical().mean();
      for (int64_t row = 0; row < nrow; ++row) {
        const float v = values[row];
        matrix->Set(row, col, std::isnan(v) ? na : v);
      }
    }
  };

  std::vector<std::thread> workers;
  const size_t chunk = (numerical_cols.size() + n_threads - 1) / n_threads;
  for (int t = 0; t < n_threads; ++t) {
    const size_t begin = t * chunk;
    const size_t end = std::min(numerical_cols.size(), begin + chunk);
    if (begin >= end) break;
    workers.emplace_back(fill_range, begin, end);
  }
  for (auto& worker : workers) worker.join();
  return static_cast<int>(numerical_cols.size());
}

/* #endregion */

/* #region Single-pass CSV load */
// Loads a CSV into a VerticalDataset with a SINGLE file scan, avoiding YDF's
// default two full reads (CreateDataSpec's statistics pass, then
// LoadVerticalDataset). Strategy:
//   1. Read only the header line to learn the column names.
//   2. Load every column as NUMERICAL in one pass. YDF's numerical CSV parser
//      returns an error on a non-numeric cell, so a string label or categorical
//      feature safely aborts this fast path (caller falls back to two reads).
//   3. Recompute exact numerical statistics from the in-RAM columns.
//   4. Convert the label column NUMERICAL -> binary CATEGORICAL from the *full*
//      in-memory label values. Doing this after the full load (rather than from
//      a truncated inference scan) makes it robust to label-sorted files.
namespace single_pass_csv {

std::string FormatLabelKey(float v) {
  if (std::isfinite(v) && std::floor(v) == v && std::fabs(v) < 1e15f) {
    return absl::StrCat(static_cast<int64_t>(v));
  }
  return absl::StrCat(v);
}

absl::StatusOr<std::vector<std::string>> ReadHeader(const std::string& path) {
  std::ifstream in(path);
  if (!in.is_open()) {
    return absl::NotFoundError(absl::StrCat("Cannot open CSV: ", path));
  }
  std::string line;
  if (!std::getline(in, line)) {
    return absl::InvalidArgumentError(absl::StrCat("Empty CSV: ", path));
  }
  if (!line.empty() && line.back() == '\r') line.pop_back();
  std::vector<std::string> names = absl::StrSplit(line, ',');
  for (auto& n : names) n = std::string(absl::StripAsciiWhitespace(n));
  if (names.empty()) {
    return absl::InvalidArgumentError("CSV header has no columns");
  }
  return names;
}

absl::StatusOr<std::unique_ptr<dataset::VerticalDataset>> Load(
    const std::string& csv_path, const std::string& label_col) {
  auto header_or = ReadHeader(csv_path);
  if (!header_or.ok()) return header_or.status();
  const std::vector<std::string>& header = header_or.value();

  int label_idx = -1;
  for (int i = 0; i < static_cast<int>(header.size()); ++i) {
    if (header[i] == label_col) {
      label_idx = i;
      break;
    }
  }
  if (label_idx < 0) {
    return absl::InvalidArgumentError(absl::StrCat(
        "Label column '", label_col, "' not found in CSV header"));
  }

  // (2) Load every column as NUMERICAL in a single scan.
  dataset::proto::DataSpecification numeric_spec;
  for (const auto& name : header) {
    auto* col = numeric_spec.add_columns();
    col->set_name(name);
    col->set_type(dataset::proto::ColumnType::NUMERICAL);
  }
  auto ds = std::make_unique<dataset::VerticalDataset>();
  const absl::Status load_status =
      dataset::LoadVerticalDataset("csv:" + csv_path, numeric_spec, ds.get());
  if (!load_status.ok()) return load_status;  // e.g. a non-numeric column.

  const int64_t nrow = ds->nrow();
  if (nrow <= 0) return absl::InvalidArgumentError("CSV has no data rows");

  // (4) Convert the label column NUMERICAL -> CATEGORICAL from full data.
  auto label_num_or = ds->numerical_column(label_idx);
  if (!label_num_or.ok()) return label_num_or.status();
  // Copy the values: ReplaceColumn destroys the underlying numerical column.
  const std::vector<float> label_vals = label_num_or.value()->values();

  std::map<float, int64_t> counts;  // value-ordered for determinism.
  int64_t label_nas = 0;
  for (float v : label_vals) {
    if (std::isnan(v)) {
      ++label_nas;
      continue;
    }
    ++counts[v];
  }
  if (counts.size() < 2) {
    return absl::FailedPreconditionError(
        "Label has fewer than 2 distinct values; not a classification label");
  }

  // Vocabulary in frequency-descending order (matches YDF), value-ascending on
  // ties (stable over the value-ordered map).
  std::vector<std::pair<float, int64_t>> vocab(counts.begin(), counts.end());
  std::stable_sort(
      vocab.begin(), vocab.end(),
      [](const std::pair<float, int64_t>& a,
         const std::pair<float, int64_t>& b) { return a.second > b.second; });

  dataset::proto::Column cat_spec;
  cat_spec.set_name(header[label_idx]);
  cat_spec.set_type(dataset::proto::ColumnType::CATEGORICAL);
  auto* cat = cat_spec.mutable_categorical();
  cat->set_is_already_integerized(false);
  cat->set_number_of_unique_values(static_cast<int64_t>(vocab.size()) + 1);
  auto& items = *cat->mutable_items();
  {
    auto& ood = items[dataset::kOutOfDictionaryItemKey];
    ood.set_index(dataset::kOutOfDictionaryItemIndex);
    ood.set_count(label_nas);
  }
  std::map<float, int32_t> value_to_index;
  int64_t most_freq_idx = dataset::kOutOfDictionaryItemIndex;
  int64_t most_freq_count = -1;
  for (size_t i = 0; i < vocab.size(); ++i) {
    const int32_t idx = static_cast<int32_t>(i) + 1;
    auto& item = items[FormatLabelKey(vocab[i].first)];
    item.set_index(idx);
    item.set_count(vocab[i].second);
    value_to_index[vocab[i].first] = idx;
    if (vocab[i].second > most_freq_count) {
      most_freq_count = vocab[i].second;
      most_freq_idx = idx;
    }
  }
  cat->set_most_frequent_value(most_freq_idx);

  auto replaced = ds->ReplaceColumn(label_idx, cat_spec);
  if (!replaced.ok()) return replaced.status();
  auto cat_col_or = ds->mutable_categorical_column(label_idx);
  if (!cat_col_or.ok()) return cat_col_or.status();
  auto* cat_col = cat_col_or.value();
  for (int64_t r = 0; r < nrow; ++r) {
    const float v = label_vals[r];
    if (std::isnan(v)) {
      cat_col->Set(r, dataset::VerticalDataset::CategoricalColumn::kNaValue);
    } else {
      cat_col->Set(r, value_to_index[v]);
    }
  }

  // (3) Recompute exact numerical statistics for feature columns from RAM.
  for (int col_idx = 0; col_idx < ds->data_spec().columns_size(); ++col_idx) {
    if (col_idx == label_idx) continue;
    auto* col_spec = ds->mutable_data_spec()->mutable_columns(col_idx);
    if (col_spec->type() != dataset::proto::ColumnType::NUMERICAL) continue;
    auto num_or = ds->numerical_column(col_idx);
    if (!num_or.ok()) continue;
    const std::vector<float>& vals = num_or.value()->values();
    double sum = 0.0, sum_sq = 0.0;
    int64_t count = 0, nas = 0;
    float min_v = std::numeric_limits<float>::infinity();
    float max_v = -std::numeric_limits<float>::infinity();
    for (float v : vals) {
      if (std::isnan(v)) {
        ++nas;
        continue;
      }
      sum += v;
      sum_sq += static_cast<double>(v) * v;
      ++count;
      min_v = std::min(min_v, v);
      max_v = std::max(max_v, v);
    }
    auto* num = col_spec->mutable_numerical();
    if (count > 0) {
      const double mean = sum / count;
      double var = sum_sq / count - mean * mean;
      if (var < 0) var = 0;
      num->set_mean(mean);
      num->set_standard_deviation(std::sqrt(var));
      num->set_min_value(min_v);
      num->set_max_value(max_v);
    }
    col_spec->set_count_nas(nas);
  }

  ds->mutable_data_spec()->set_created_num_rows(nrow);
  return ds;
}

}  // namespace single_pass_csv

/* #endregion */

int main(int argc, char** argv) {
  const auto t_main_start = std::chrono::high_resolution_clock::now();
  absl::ParseCommandLine(argc, argv);
  if (std::getenv("YDF_RM_MAX_ROWS") == nullptr) {
    setenv("YDF_RM_MAX_ROWS", "5000", /*overwrite=*/0);
  }
  dataset::RowMajorFeatureMatrix::SetActive(nullptr);
  const auto mode = absl::GetFlag(FLAGS_input_mode);

  // Validate required input_mode flag
  if (mode.empty()) {
    std::cerr << "Error: --input_mode is required. Use csv, synthetic, or tfrecord.\n";
    return 1;
  }

  dataset::proto::DataSpecification data_spec;
  std::unique_ptr<dataset::VerticalDataset> tf_ds;
  dataset::VerticalDataset* ds_ptr = nullptr;
  std::string csv_path;
  std::string label_col;

  // 1) Prepare data source based on mode
  if (mode == "csv") {
    csv_path = absl::GetFlag(FLAGS_train_csv);
    label_col = absl::GetFlag(FLAGS_label_col);

    // Validate required CSV parameters
    if (csv_path.empty()) {
      std::cerr << "Error: --train_csv is required in csv mode.\n";
      return 1;
    }
    if (label_col.empty()) {
      std::cerr << "Error: --label_col is required in csv mode.\n";
      return 1;
    }

    bool single_pass_ok = false;
    if (absl::GetFlag(FLAGS_csv_single_pass_load)) {
      LOG(INFO) << "Loading CSV in a single pass: " << csv_path;
      auto ds_or = single_pass_csv::Load(csv_path, label_col);
      if (ds_or.ok()) {
        tf_ds = std::move(ds_or.value());
        ds_ptr = tf_ds.get();
        data_spec = tf_ds->data_spec();
        single_pass_ok = true;
        LOG(INFO) << "Single-pass CSV load complete: " << tf_ds->nrow()
                  << " rows, " << data_spec.columns_size() << " columns.";
      } else {
        LOG(WARNING) << "Single-pass CSV load not applicable ("
                     << ds_or.status().message()
                     << "); falling back to two-pass inference + load.";
      }
    }

    if (!single_pass_ok) {
      LOG(INFO) << "Inferring DataSpec from CSV: " << csv_path;
      dataset::proto::DataSpecificationGuide guide;
      auto* col_guide = guide.add_column_guides();
      col_guide->set_column_name_pattern(label_col);
      col_guide->set_type(dataset::proto::ColumnType::CATEGORICAL);

      dataset::CreateDataSpec(
          "csv:" + csv_path,
          /*require_same_dataset_fields=*/false,
          guide,
          &data_spec);
    }
  }
  else if (mode == "uniform" || mode == "trunk") {
    std::string layout = absl::GetFlag(FLAGS_dataset_layout);
    LOG(INFO) << "Generating " << mode << " synthetic dataset: rows="
              << absl::GetFlag(FLAGS_rows)
              << ", cols=" << absl::GetFlag(FLAGS_cols)
              << ", layout=" << layout;

    label_col = "y";
    data_spec = MakeSyntheticSpec(absl::GetFlag(FLAGS_cols),
                                  absl::GetFlag(FLAGS_rows),
                                  2, label_col);

    static std::unique_ptr<dataset::RowMajorFeatureMatrix> row_major_matrix;
    dataset::RowMajorFeatureMatrix::SetActive(nullptr);

    if (layout == "column") {
      auto ds = BuildSyntheticDataset(mode, data_spec,
                                      absl::GetFlag(FLAGS_rows),
                                      absl::GetFlag(FLAGS_cols),
                                      absl::GetFlag(FLAGS_seed));
      tf_ds = std::make_unique<dataset::VerticalDataset>(std::move(ds));
      ds_ptr = tf_ds.get();
    } else if (layout == "row") {
#if defined(ROW_MAJOR_DATASET_LAYOUT)
      if (mode != "trunk") {
        std::cerr << "--dataset_layout=row only supports trunk synthetic.\n";
        return 1;
      }
      auto rm = MakeTrunkDatasetRowMajor(data_spec,
                                         absl::GetFlag(FLAGS_rows),
                                         absl::GetFlag(FLAGS_cols),
                                         absl::GetFlag(FLAGS_seed));
      tf_ds = std::move(rm.vd);
      row_major_matrix = std::move(rm.matrix);
      dataset::RowMajorFeatureMatrix::SetActive(row_major_matrix.get());
      LOG(INFO) << "Row-major matrix allocated: "
                << (row_major_matrix->bytes() / (1024.0 * 1024.0 * 1024.0))
                << " GiB";
      ds_ptr = tf_ds.get();
#else
      std::cerr << "--dataset_layout=row requires "
                   "--config=row_major_dataset_layout.\n";
      return 1;
#endif
    } else {
      std::cerr << "Unknown --dataset_layout: " << layout
                << ". Use 'column' or 'row'.\n";
      return 1;
    }
  }
  else if (mode == "tfrecord") {
    LOG(INFO) << "Reading TFRECORD";

    const std::string path = absl::GetFlag(FLAGS_ds_path);
    if (path.empty()) {
      std::cerr << "--ds_path required in tfrecord mode\n";
      return 1;
    }
    LOG(INFO) << "Loading TFRecord dataset from: " << path;
    CHECK_OK(file::GetBinaryProto(path + ".data_spec.pb", &data_spec,
                                  file::Defaults()));
    tf_ds = std::make_unique<dataset::VerticalDataset>();
    CHECK_OK(dataset::LoadVerticalDataset("tfrecord:" + path,
                                          data_spec,
                                          tf_ds.get()));
    ds_ptr = tf_ds.get();
  } else {
  std::cerr << "Unknown input_mode: " << mode
            << ". Use csv, uniform, trunk, or tfrecord.\n";
  return 1;
}

  // 2) Configure learner
  model::proto::TrainingConfig train_config;
  train_config.set_task(model::proto::Task::CLASSIFICATION);
  train_config.set_label(label_col);

  model::proto::DeploymentConfig deploy_config;
  // GPU vs CPU is chosen at compile time via --config=oblique_gpu; the runtime
  // toggle is commented out until hybrid CPU-GPU offloading is added.
  // deploy_config.set_use_gpu(absl::GetFlag(FLAGS_use_gpu));

  /* #region Handle num_threads */
  int num_threads_flag = absl::GetFlag(FLAGS_num_threads);
  if (num_threads_flag > 0) {
    LOG(INFO) << "Running with " << num_threads_flag << " threads, as requested.";
    deploy_config.set_num_threads(num_threads_flag);

  } else if (num_threads_flag == -1) {
    // Automatically detect number of CPUs
    unsigned int cpu_count = std::thread::hardware_concurrency();
    if (cpu_count == 0) {
      cpu_count = 1;  // fallback if detection fails
    }
    LOG(INFO) << "-1 (automatic) threads requested. "
              << cpu_count << " threads set.";
    deploy_config.set_num_threads(cpu_count);

  } else {
    std::cerr << "Invalid value for --num_threads: "
              << num_threads_flag
              << ". Must be >0 for fixed threads or -1 for automatic.\n";
    return 1;
  }
  /* #endregion */

  // Configure ensemble method: Bagging (RF) or Boosting (GBT/MART)
  const std::string ensemble_method = absl::GetFlag(FLAGS_ensemble_method);
  model::decision_tree::proto::DecisionTreeTrainingConfig* dt_config;

  if (ensemble_method == "Bagging") {
    train_config.set_learner("RANDOM_FOREST");
    auto& rf = *train_config.MutableExtension(
        model::random_forest::proto::random_forest_config);
    rf.set_num_trees(absl::GetFlag(FLAGS_num_trees));
    rf.set_bootstrap_training_dataset(absl::GetFlag(FLAGS_bootstrap_training_dataset));
    rf.set_bootstrap_size_ratio(1.0);
    rf.set_winner_take_all_inference(false);
    rf.set_compute_oob_performances(
        absl::GetFlag(FLAGS_compute_oob_performances));
    dt_config = rf.mutable_decision_tree();
  } else if (ensemble_method == "Boosting") {
    train_config.set_learner("GRADIENT_BOOSTED_TREES");
    auto& gbt = *train_config.MutableExtension(
        model::gradient_boosted_trees::proto::gradient_boosted_trees_config);
    gbt.set_num_trees(absl::GetFlag(FLAGS_num_trees));
    gbt.set_shrinkage(absl::GetFlag(FLAGS_shrinkage));
    // Disable the validation split + early stopping so a clean AP A/B isn't
    // diluted by a withheld 10% of training data and per-tree validation
    // inference (num_trees becomes the actual tree count, not a ceiling).
    gbt.set_validation_set_ratio(0);
    gbt.set_early_stopping(
        model::gradient_boosted_trees::proto::
            GradientBoostedTreesTrainingConfig::NONE);
    dt_config = gbt.mutable_decision_tree();
  } else {
    std::cerr << "Unknown ensemble_method: " << ensemble_method
              << ". Use 'Bagging' or 'Boosting'.\n";
    return 1;
  }

  train_config.set_random_seed(absl::GetFlag(FLAGS_seed));

  // Shared decision tree config
  int tree_depth = absl::GetFlag(FLAGS_tree_depth);
  if (ensemble_method == "Boosting" && tree_depth == -1) {
    tree_depth = 6;
  }
  dt_config->set_max_depth(tree_depth);
  dt_config->set_min_examples(ensemble_method == "Boosting" ? 5 : 1);

  const auto growing_strategy = absl::GetFlag(FLAGS_growing_strategy);
  if (growing_strategy == "GlobalBestFirst") {
    dt_config->mutable_growing_strategy_best_first_global()->set_max_num_nodes(-1);
  } else if (growing_strategy != "Local") {
    std::cerr << "Unknown growing_strategy: " << growing_strategy
              << ". Use Local or GlobalBestFirst.\n";
    return 1;
  }

  /* #region Conditional Feature Split Type Configuration */
  const std::string feature_split_type = absl::GetFlag(FLAGS_feature_split_type);

  if (feature_split_type == "Oblique") {
    LOG(INFO) << "Configuring oblique splits";
    auto* sos = dt_config->mutable_sparse_oblique_split();
    sos->set_max_num_projections(
        absl::GetFlag(FLAGS_max_num_projections));
    sos->set_projection_density_factor(
        absl::GetFlag(FLAGS_projection_density_factor));
    sos->set_num_projections_exponent(
        absl::GetFlag(FLAGS_num_projections_exponent));
    sos->set_dynamic_split_threshold(
        absl::GetFlag(FLAGS_dynamic_split_threshold));
  } else if (feature_split_type == "Axis Aligned") {
    LOG(INFO) << "Using axis-aligned splits";

    // Configure sparse_oblique_split parameters to reuse GetNumProjections formula
    // This does NOT enable oblique splits - it just sets params for the calculation
    auto* sos = dt_config->mutable_sparse_oblique_split();
    sos->set_max_num_projections(absl::GetFlag(FLAGS_max_num_projections));
    sos->set_num_projections_exponent(absl::GetFlag(FLAGS_num_projections_exponent));

    int num_numerical_features = 0;
    for (int i = 0; i < data_spec.columns_size(); ++i) {
      if (data_spec.columns(i).type() == dataset::proto::NUMERICAL) {
        num_numerical_features++;
      }
    }

    const int num_candidates = model::decision_tree::GetNumProjections(
        *dt_config, num_numerical_features);
    dt_config->set_num_candidate_attributes(num_candidates);

    // Clear sparse_oblique_split so axis-aligned splits are used, not oblique
    dt_config->clear_sparse_oblique_split();

    // AA sorting strategy — see flag description for the why.
    using DTI = model::decision_tree::proto::DecisionTreeTrainingConfig_Internal;
    const std::string aa_sort = absl::GetFlag(FLAGS_aa_sorting_strategy);
    DTI::SortingStrategy strategy;
    if (aa_sort == "in_node") {
      strategy = DTI::IN_NODE;
    } else if (aa_sort == "presorted") {
      strategy = DTI::PRESORTED;
    } else if (aa_sort == "force_presorted") {
      strategy = DTI::FORCE_PRESORTED;
    } else if (aa_sort == "auto") {
      strategy = DTI::AUTO;
    } else {
      std::cerr << "Unknown aa_sorting_strategy: " << aa_sort
                << ". Use 'in_node', 'presorted', 'force_presorted', or 'auto'.\n";
      return 1;
    }
    dt_config->mutable_internal()->set_sorting_strategy(strategy);
    LOG(INFO) << "AA sorting strategy: " << aa_sort;

    LOG(INFO) << "Num Candidate Attributes: " << num_candidates;
  } else {
    std::cerr << "Unknown feature_split_type: " << feature_split_type
              << ". Use 'Axis Aligned' or 'Oblique'.\n";
    return 1;
  }
  /* #endregion */

  // Configure histogram splitting - Updated to match Yggdrasil implementation
  auto* numerical_split = dt_config->mutable_numerical_split();
  
  const std::string hist_type = absl::GetFlag(FLAGS_numerical_split_type);
  if (hist_type == "Exact") {
    numerical_split->set_type(
        model::decision_tree::proto::NumericalSplit::EXACT);
    LOG(INFO) << "Using exact splitting";
  } else if (hist_type == "Random") {
    numerical_split->set_type(
        model::decision_tree::proto::NumericalSplit::HISTOGRAM_RANDOM);
    numerical_split->set_num_candidates(absl::GetFlag(FLAGS_histogram_num_bins));
    LOG(INFO) << "Using histogram splitting: Random with "
              << absl::GetFlag(FLAGS_histogram_num_bins) << " bins";
  } else if (hist_type == "Equal Width") {
    numerical_split->set_type(
        model::decision_tree::proto::NumericalSplit::HISTOGRAM_EQUAL_WIDTH);
    numerical_split->set_num_candidates(absl::GetFlag(FLAGS_histogram_num_bins));
    LOG(INFO) << "Using histogram splitting: Equal Width with "
              << absl::GetFlag(FLAGS_histogram_num_bins) << " bins";
  } else if (hist_type == "Dynamic Random Histogram") {
    numerical_split->set_type(
    model::decision_tree::proto::NumericalSplit::DYNAMIC_RANDOM_HISTOGRAM);
    numerical_split->set_num_candidates(absl::GetFlag(FLAGS_histogram_num_bins));
    LOG(INFO) << "Using " << hist_type << " with "
              << absl::GetFlag(FLAGS_histogram_num_bins) << " samples";
  } else if (hist_type == "Dynamic Equal Width Histogram") {
    numerical_split->set_type(
        model::decision_tree::proto::NumericalSplit::DYNAMIC_EQUAL_WIDTH_HISTOGRAM);
    numerical_split->set_num_candidates(absl::GetFlag(FLAGS_histogram_num_bins));
    LOG(INFO) << "Using " << hist_type << " with "
              << absl::GetFlag(FLAGS_histogram_num_bins) << " samples";
  }
   else {
    std::cerr << "Unknown histogram type: " << hist_type 
              << ". Use 'Exact', 'Random', 'Equal Width', 'Dynamic Equal Width Histogram' or 'Dynamic Random Histogram'.\n";
    return 1;
  }




  // -----------Done Configuring Model. Start Training-----------

  std::unique_ptr<model::AbstractLearner> learner;
  CHECK_OK(model::GetLearner(train_config, &learner, deploy_config));

  // 3) Train with timing
  // Everything before this point (flag parsing, dataspec inference, synthetic
  // generation, tfrecord load). The csv/column dataset load happens inside
  // TrainWithStatus and is logged separately by abstract_learner.cc.
  {
    const std::chrono::duration<double> init_dur =
        std::chrono::high_resolution_clock::now() - t_main_start;
    LOG(INFO) << "train_oblique_forest wall-time - Loading/Init (pre-train): "
              << init_dur.count() << "s";
  }
  auto start = std::chrono::high_resolution_clock::now();
  absl::StatusOr<std::unique_ptr<model::AbstractModel>> model_or;

  if (mode == "csv") {
#if defined(ROW_MAJOR_DATASET_LAYOUT)
    std::string csv_layout = absl::GetFlag(FLAGS_dataset_layout);
    if (csv_layout == "row") {
      // Row-major from CSV: same scheme as Dynamic_Row_Col_Major but only
      // the row-major store. The VerticalDataset stays the fp32 source for
      // split application and OOB. Reuse the single-pass dataset if present.
      const auto t_load_start = std::chrono::high_resolution_clock::now();
      static std::unique_ptr<dataset::VerticalDataset> csv_ds;
      dataset::VerticalDataset* base_ds = ds_ptr;
      if (base_ds == nullptr) {
        csv_ds = std::make_unique<dataset::VerticalDataset>();
        CHECK_OK(dataset::LoadVerticalDataset("csv:" + csv_path, data_spec,
                                              csv_ds.get()));
        base_ds = csv_ds.get();
      }
      const int64_t nrow = base_ds->nrow();
      const int total_cols = base_ds->data_spec().columns_size();
      static std::unique_ptr<dataset::RowMajorFeatureMatrix> csv_rows;
      csv_rows =
          std::make_unique<dataset::RowMajorFeatureMatrix>(nrow, total_cols);
      const int mirrored = FillMatrixFromDataset(csv_rows.get(), *base_ds);
      dataset::RowMajorFeatureMatrix::SetActive(csv_rows.get());
      const std::chrono::duration<double> load_dur =
          std::chrono::high_resolution_clock::now() - t_load_start;
      LOG(INFO) << "train_oblique_forest Dataset load took: "
                << load_dur.count() << " s";
      LOG(INFO) << "Row-major CSV matrix: " << mirrored
                << " numerical columns, "
                << (csv_rows->bytes() / (1024.0 * 1024.0 * 1024.0)) << " GiB";
      model_or = learner->TrainWithStatus(*base_ds);
    } else if (csv_layout == "column") {
      if (ds_ptr != nullptr) {
        model_or = learner->TrainWithStatus(*ds_ptr);
      } else {
        model_or = learner->TrainWithStatus("csv:" + csv_path, data_spec);
      }
    } else {
      std::cerr << "Unsupported --dataset_layout for csv mode: " << csv_layout
                << ". Use 'column' or 'row'.\n";
      return 1;
    }
#else
    if (absl::GetFlag(FLAGS_dataset_layout) != "column") {
      std::cerr << "--dataset_layout=" << absl::GetFlag(FLAGS_dataset_layout)
                << " in csv mode requires "
                   "--config=row_major_dataset_layout.\n";
      return 1;
    }
    if (ds_ptr != nullptr) {
      model_or = learner->TrainWithStatus(*ds_ptr);
    } else {
      model_or = learner->TrainWithStatus("csv:" + csv_path, data_spec);
    }
#endif
  } else {
    model_or = learner->TrainWithStatus(*ds_ptr);
  }

  if (!model_or.ok()) {
    std::cerr << "Training failed: " << model_or.status().message() << std::endl;
    return 1;
  }
  auto model_ptr = std::move(model_or.value());

  auto end = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double> dur = end - start;
  LOG(INFO) << "train_oblique_forest wall-time - Training + Post-Processing: " << dur.count() << "s";

  // 4) Save model if requested
  const std::string out_dir = absl::GetFlag(FLAGS_model_out_dir);
  if (!out_dir.empty()) {
    auto save_status = model::SaveModel(out_dir, *model_ptr);
    if (!save_status.ok()) {
      std::cerr << "Could not save model: " << save_status.message() << std::endl;
      return 1;
    }
    LOG(INFO) << "Model saved to: " << out_dir;
  }

  return 0;
}
