/*
 * Dense flat feature matrices for dataset-layout experiments.
 *
 * These matrices are intentionally process-local experimental backends for
 * synthetic trunk benchmarks. They let ProjectionEvaluator read numerical
 * features without going through VerticalDataset's per-column vector table.
 */

#ifndef YGGDRASIL_DECISION_FORESTS_DATASET_ROW_MAJOR_FEATURE_MATRIX_H_
#define YGGDRASIL_DECISION_FORESTS_DATASET_ROW_MAJOR_FEATURE_MATRIX_H_

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <memory>

namespace yggdrasil_decision_forests {
namespace dataset {

class RowMajorFeatureMatrix {
 public:
  inline static const RowMajorFeatureMatrix* g_active = nullptr;
  static void SetActive(const RowMajorFeatureMatrix* m) { g_active = m; }
  static const RowMajorFeatureMatrix* Active() { return g_active; }

  RowMajorFeatureMatrix(int64_t num_rows, int num_cols)
      : num_rows_(num_rows),
        num_cols_(num_cols),
        data_(new float[static_cast<size_t>(num_rows) *
                        static_cast<size_t>(num_cols)]) {}

  float Get(int64_t row, int col) const {
    return data_[static_cast<size_t>(row) * static_cast<size_t>(num_cols_) +
                 static_cast<size_t>(col)];
  }

  void Set(int64_t row, int col, float v) {
    data_[static_cast<size_t>(row) * static_cast<size_t>(num_cols_) +
          static_cast<size_t>(col)] = v;
  }

  const float* row_ptr(int64_t row) const {
    return &data_[static_cast<size_t>(row) * static_cast<size_t>(num_cols_)];
  }

  int64_t num_rows() const { return num_rows_; }
  int num_cols() const { return num_cols_; }

  uint64_t bytes() const {
    return static_cast<uint64_t>(num_rows_) *
           static_cast<uint64_t>(num_cols_) * sizeof(float);
  }

 private:
  int64_t num_rows_;
  int num_cols_;
  std::unique_ptr<float[]> data_;
};

class FlatColMajorFeatureMatrix {
 public:
  inline static const FlatColMajorFeatureMatrix* g_active = nullptr;
  static void SetActive(const FlatColMajorFeatureMatrix* m) { g_active = m; }
  static const FlatColMajorFeatureMatrix* Active() { return g_active; }

  FlatColMajorFeatureMatrix(int64_t num_rows, int num_cols)
      : num_rows_(num_rows),
        num_cols_(num_cols),
        data_(new float[static_cast<size_t>(num_rows) *
                        static_cast<size_t>(num_cols)]) {}

  float Get(int64_t row, int col) const {
    return data_[static_cast<size_t>(col) * static_cast<size_t>(num_rows_) +
                 static_cast<size_t>(row)];
  }

  void Set(int64_t row, int col, float v) {
    data_[static_cast<size_t>(col) * static_cast<size_t>(num_rows_) +
          static_cast<size_t>(row)] = v;
  }

  const float* col_ptr(int col) const {
    return &data_[static_cast<size_t>(col) * static_cast<size_t>(num_rows_)];
  }

  int64_t num_rows() const { return num_rows_; }
  int num_cols() const { return num_cols_; }

  uint64_t bytes() const {
    return static_cast<uint64_t>(num_rows_) *
           static_cast<uint64_t>(num_cols_) * sizeof(float);
  }

 private:
  int64_t num_rows_;
  int num_cols_;
  std::unique_ptr<float[]> data_;
};

// bf16 helpers: round-to-nearest-even on store, exponent-preserving widen on
// load. NaN payloads survive (exponent stays all-ones).
inline uint16_t FloatToBf16(float v) {
  uint32_t u;
  std::memcpy(&u, &v, 4);
  if ((u & 0x7FFFFFFFu) > 0x7F800000u) {  // NaN: RNE carry could wrap to Inf
    return 0x7FC0;                        // canonical qNaN
  }
  u += 0x7FFFu + ((u >> 16) & 1u);
  return static_cast<uint16_t>(u >> 16);
}
inline float Bf16ToFloat(uint16_t b) {
  uint32_t u = static_cast<uint32_t>(b) << 16;
  float f;
  std::memcpy(&f, &u, 4);
  return f;
}

// Half-width (bf16) variants of the two experimental layouts. Same
// process-local Active() registration pattern. Storage is uint16; Get/Set
// convert at the boundary so callers keep float semantics.
class Bf16RowMajorFeatureMatrix {
 public:
  inline static const Bf16RowMajorFeatureMatrix* g_active = nullptr;
  static void SetActive(const Bf16RowMajorFeatureMatrix* m) { g_active = m; }
  static const Bf16RowMajorFeatureMatrix* Active() { return g_active; }

  Bf16RowMajorFeatureMatrix(int64_t num_rows, int num_cols)
      : num_rows_(num_rows),
        num_cols_(num_cols),
        data_(new uint16_t[static_cast<size_t>(num_rows) *
                           static_cast<size_t>(num_cols)]) {}

  float Get(int64_t row, int col) const {
    return Bf16ToFloat(
        data_[static_cast<size_t>(row) * static_cast<size_t>(num_cols_) +
              static_cast<size_t>(col)]);
  }

  void Set(int64_t row, int col, float v) {
    data_[static_cast<size_t>(row) * static_cast<size_t>(num_cols_) +
          static_cast<size_t>(col)] = FloatToBf16(v);
  }

  const uint16_t* row_ptr(int64_t row) const {
    return &data_[static_cast<size_t>(row) * static_cast<size_t>(num_cols_)];
  }

  int64_t num_rows() const { return num_rows_; }
  int num_cols() const { return num_cols_; }

  uint64_t bytes() const {
    return static_cast<uint64_t>(num_rows_) *
           static_cast<uint64_t>(num_cols_) * sizeof(uint16_t);
  }

 private:
  int64_t num_rows_;
  int num_cols_;
  std::unique_ptr<uint16_t[]> data_;
};

class Bf16FlatColMajorFeatureMatrix {
 public:
  inline static const Bf16FlatColMajorFeatureMatrix* g_active = nullptr;
  static void SetActive(const Bf16FlatColMajorFeatureMatrix* m) {
    g_active = m;
  }
  static const Bf16FlatColMajorFeatureMatrix* Active() { return g_active; }

  Bf16FlatColMajorFeatureMatrix(int64_t num_rows, int num_cols)
      : num_rows_(num_rows),
        num_cols_(num_cols),
        data_(new uint16_t[static_cast<size_t>(num_rows) *
                           static_cast<size_t>(num_cols)]) {}

  float Get(int64_t row, int col) const {
    return Bf16ToFloat(
        data_[static_cast<size_t>(col) * static_cast<size_t>(num_rows_) +
              static_cast<size_t>(row)]);
  }

  void Set(int64_t row, int col, float v) {
    data_[static_cast<size_t>(col) * static_cast<size_t>(num_rows_) +
          static_cast<size_t>(row)] = FloatToBf16(v);
  }

  const uint16_t* col_ptr(int col) const {
    return &data_[static_cast<size_t>(col) * static_cast<size_t>(num_rows_)];
  }

  int64_t num_rows() const { return num_rows_; }
  int num_cols() const { return num_cols_; }

  uint64_t bytes() const {
    return static_cast<uint64_t>(num_rows_) *
           static_cast<uint64_t>(num_cols_) * sizeof(uint16_t);
  }

 private:
  int64_t num_rows_;
  int num_cols_;
  std::unique_ptr<uint16_t[]> data_;
};

}  // namespace dataset
}  // namespace yggdrasil_decision_forests

#endif  // YGGDRASIL_DECISION_FORESTS_DATASET_ROW_MAJOR_FEATURE_MATRIX_H_
