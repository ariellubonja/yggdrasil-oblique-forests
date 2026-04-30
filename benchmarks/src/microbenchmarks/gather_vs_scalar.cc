// Standalone microbench: scalar M=4 vs AVX2 8-wide gather vs AVX-512 16-wide gather.
//
// Simulates the inner kernel of EvaluateProjectionRowBlocks (oblique_cpu_depthwise_1pass.cc)
// under DRAM-bound conditions matching real training:
//   - 4096 feature columns × 64K floats = 1 GB total (>> 32 MB L3)
//   - 5000 sorted random row indices (depth-10 node shape: 3M / 2^10 ≈ 2930–5000)
//   - density 2 features per projection
//   - 4096 projections, each pair of features offset by 2048 so L3 cannot reuse them
//
// Compile (standalone, no Bazel):
//   g++ -O3 -march=native -mavx512f -std=c++17 -o /tmp/gather_vs_scalar \
//       benchmarks/src/microbenchmarks/gather_vs_scalar.cc
//
// Decision rule: if AVX-512 gather beats scalar M=4 by ≥20% in per-projection
// median time, proceed with 5th formal experiment. Otherwise enter plan mode.

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <numeric>
#include <random>
#include <vector>

// AVX2 + AVX-512 intrinsics
#include <immintrin.h>

// ── Kernel A: scalar M=4 (current V2-rev3 baseline) ─────────────────────────
// Matches EvaluateProjectionRowBlocks in oblique_cpu_depthwise_1pass.cc exactly.
static void kernel_scalar_m4(
    const float* const* __restrict__ cols,
    const int32_t* __restrict__ col_idx,
    const float* __restrict__ weights,
    int num_feats,
    const uint32_t* __restrict__ sel,
    int num_rows,
    float* __restrict__ out) {
  int i = 0;
  for (; i + 4 <= num_rows; i += 4) {
    const uint32_t ex0 = sel[i+0], ex1 = sel[i+1],
                   ex2 = sel[i+2], ex3 = sel[i+3];
    float acc0 = 0.f, acc1 = 0.f, acc2 = 0.f, acc3 = 0.f;
    for (int f = 0; f < num_feats; ++f) {
      const float* col = cols[col_idx[f]];
      const float w = weights[f];
      acc0 += w * col[ex0];
      acc1 += w * col[ex1];
      acc2 += w * col[ex2];
      acc3 += w * col[ex3];
    }
    out[i+0] = acc0; out[i+1] = acc1;
    out[i+2] = acc2; out[i+3] = acc3;
  }
  for (; i < num_rows; ++i) {
    const uint32_t ex = sel[i];
    float acc = 0.f;
    for (int f = 0; f < num_feats; ++f)
      acc += weights[f] * cols[col_idx[f]][ex];
    out[i] = acc;
  }
}

// ── Kernel B: AVX2 8-wide gather ─────────────────────────────────────────────
#if defined(__AVX2__) && defined(__FMA__)
static void kernel_avx2_m8(
    const float* const* __restrict__ cols,
    const int32_t* __restrict__ col_idx,
    const float* __restrict__ weights,
    int num_feats,
    const uint32_t* __restrict__ sel,
    int num_rows,
    float* __restrict__ out) {
  int i = 0;
  for (; i + 8 <= num_rows; i += 8) {
    __m256i vidx = _mm256_loadu_si256((const __m256i*)(sel + i));
    __m256 acc = _mm256_setzero_ps();
    for (int f = 0; f < num_feats; ++f) {
      __m256 vw  = _mm256_set1_ps(weights[f]);
      __m256 vv  = _mm256_i32gather_ps(cols[col_idx[f]], vidx, 4);
      acc = _mm256_fmadd_ps(vw, vv, acc);
    }
    _mm256_storeu_ps(out + i, acc);
  }
  for (; i < num_rows; ++i) {
    const uint32_t ex = sel[i];
    float acc = 0.f;
    for (int f = 0; f < num_feats; ++f)
      acc += weights[f] * cols[col_idx[f]][ex];
    out[i] = acc;
  }
}
#else
static void kernel_avx2_m8(
    const float* const* cols, const int32_t* col_idx, const float* weights,
    int num_feats, const uint32_t* sel, int num_rows, float* out) {
  kernel_scalar_m4(cols, col_idx, weights, num_feats, sel, num_rows, out);
}
#endif

// ── Kernel C: AVX-512 16-wide gather ─────────────────────────────────────────
// Note: _mm512_i32gather_ps signature differs from AVX2: (vindex, base, scale).
#if defined(__AVX512F__)
static void kernel_avx512_m16(
    const float* const* __restrict__ cols,
    const int32_t* __restrict__ col_idx,
    const float* __restrict__ weights,
    int num_feats,
    const uint32_t* __restrict__ sel,
    int num_rows,
    float* __restrict__ out) {
  int i = 0;
  for (; i + 16 <= num_rows; i += 16) {
    __m512i vidx = _mm512_loadu_si512((const void*)(sel + i));
    __m512 acc = _mm512_setzero_ps();
    for (int f = 0; f < num_feats; ++f) {
      __m512 vw = _mm512_set1_ps(weights[f]);
      __m512 vv = _mm512_i32gather_ps(vidx, (const void*)cols[col_idx[f]], 4);
      acc = _mm512_fmadd_ps(vw, vv, acc);
    }
    _mm512_storeu_ps(out + i, acc);
  }
  for (; i < num_rows; ++i) {
    const uint32_t ex = sel[i];
    float acc = 0.f;
    for (int f = 0; f < num_feats; ++f)
      acc += weights[f] * cols[col_idx[f]][ex];
    out[i] = acc;
  }
}
#else
static void kernel_avx512_m16(
    const float* const* cols, const int32_t* col_idx, const float* weights,
    int num_feats, const uint32_t* sel, int num_rows, float* out) {
  kernel_scalar_m4(cols, col_idx, weights, num_feats, sel, num_rows, out);
}
#endif

// ── Kernel D: scalar M=8 (wider scalar unroll, control) ──────────────────────
static void kernel_scalar_m8(
    const float* const* __restrict__ cols,
    const int32_t* __restrict__ col_idx,
    const float* __restrict__ weights,
    int num_feats,
    const uint32_t* __restrict__ sel,
    int num_rows,
    float* __restrict__ out) {
  int i = 0;
  for (; i + 8 <= num_rows; i += 8) {
    const uint32_t ex0=sel[i+0], ex1=sel[i+1], ex2=sel[i+2], ex3=sel[i+3];
    const uint32_t ex4=sel[i+4], ex5=sel[i+5], ex6=sel[i+6], ex7=sel[i+7];
    float a0=0,a1=0,a2=0,a3=0, a4=0,a5=0,a6=0,a7=0;
    for (int f = 0; f < num_feats; ++f) {
      const float* col = cols[col_idx[f]];
      const float w = weights[f];
      a0 += w*col[ex0]; a1 += w*col[ex1]; a2 += w*col[ex2]; a3 += w*col[ex3];
      a4 += w*col[ex4]; a5 += w*col[ex5]; a6 += w*col[ex6]; a7 += w*col[ex7];
    }
    out[i+0]=a0; out[i+1]=a1; out[i+2]=a2; out[i+3]=a3;
    out[i+4]=a4; out[i+5]=a5; out[i+6]=a6; out[i+7]=a7;
  }
  for (; i < num_rows; ++i) {
    const uint32_t ex = sel[i];
    float acc = 0.f;
    for (int f = 0; f < num_feats; ++f)
      acc += weights[f] * cols[col_idx[f]][ex];
    out[i] = acc;
  }
}

// ── Kernel E: scalar M=16 (control, to see if more scalar unroll helps) ──────
static void kernel_scalar_m16(
    const float* const* __restrict__ cols,
    const int32_t* __restrict__ col_idx,
    const float* __restrict__ weights,
    int num_feats,
    const uint32_t* __restrict__ sel,
    int num_rows,
    float* __restrict__ out) {
  int i = 0;
  for (; i + 16 <= num_rows; i += 16) {
    float a[16] = {};
    for (int f = 0; f < num_feats; ++f) {
      const float* col = cols[col_idx[f]];
      const float w = weights[f];
      for (int k = 0; k < 16; ++k) a[k] += w * col[sel[i+k]];
    }
    for (int k = 0; k < 16; ++k) out[i+k] = a[k];
  }
  for (; i < num_rows; ++i) {
    const uint32_t ex = sel[i];
    float acc = 0.f;
    for (int f = 0; f < num_feats; ++f)
      acc += weights[f] * cols[col_idx[f]][ex];
    out[i] = acc;
  }
}

int main() {
  // ── Config ────────────────────────────────────────────────────────────────
  const int num_cols  = 4096;   // total feature columns (matches real dataset)
  const int col_size  = 65536;  // floats per column (256 KB/col, 1 GB total)
  const int num_rows  = 5000;   // examples per node (depth-10 shape)
  const int density   = 2;      // features per projection
  const int num_projs = 4096;   // projection iterations per timed rep
  const int num_reps  = 12;     // timed repetitions (take median)

  printf("gather_vs_scalar microbench\n");
  printf("  cols=%d × %d floats = %.0f MB, rows=%d, density=%d, projs=%d\n\n",
         num_cols, col_size,
         (double)num_cols * col_size * 4 / (1 << 20),
         num_rows, density, num_projs);

  // ── Allocate columns ──────────────────────────────────────────────────────
  printf("Allocating columns... ");
  fflush(stdout);
  std::vector<std::vector<float>> cols_data(num_cols,
      std::vector<float>(col_size));
  std::mt19937 rng(42);
  {
    std::uniform_real_distribution<float> fdist(-1.f, 1.f);
    for (auto& col : cols_data)
      for (auto& v : col) v = fdist(rng);
  }
  std::vector<const float*> col_ptrs(num_cols);
  for (int c = 0; c < num_cols; ++c) col_ptrs[c] = cols_data[c].data();
  printf("done (%.0f MB)\n", (double)num_cols * col_size * 4 / (1 << 20));

  // ── Sorted row indices (simulating a depth-10 bag) ─────────────────────
  std::vector<uint32_t> sel(num_rows);
  {
    std::uniform_int_distribution<uint32_t> idist(0, col_size - 1);
    for (auto& s : sel) s = idist(rng);
    std::sort(sel.begin(), sel.end());
  }

  // ── Projection feature weights ────────────────────────────────────────────
  std::vector<float> weights(density);
  {
    std::uniform_real_distribution<float> wdist(-2.f, 2.f);
    for (auto& w : weights) w = wdist(rng);
  }

  // ── Output buffer ────────────────────────────────────────────────────────
  std::vector<float> out(num_rows);
  volatile float sink = 0.f;  // prevent DCE

  // ── Benchmark harness ────────────────────────────────────────────────────
  struct KernelDef {
    const char* name;
    void (*fn)(const float* const*, const int32_t*, const float*,
               int, const uint32_t*, int, float*);
  };

  const KernelDef kernels[] = {
    {"scalar_m4",        kernel_scalar_m4},
    {"scalar_m8",        kernel_scalar_m8},
    {"scalar_m16",       kernel_scalar_m16},
    {"avx2_gather_m8",   kernel_avx2_m8},
    {"avx512_gather_m16",kernel_avx512_m16},
  };

  double baseline_us = -1.0;
  printf("%-24s  median_total_s  per_proj_us  vs_scalar_m4\n", "kernel");
  printf("%-24s  --------------  -----------  ------------\n", "------");

  for (const auto& kd : kernels) {
    std::vector<double> rep_times;
    rep_times.reserve(num_reps);
    for (int r = 0; r < num_reps; ++r) {
      auto t0 = std::chrono::steady_clock::now();
      for (int p = 0; p < num_projs; ++p) {
        // Two features offset by num_cols/2 so L3 can't reuse them:
        // column j is accessed at p=j and p=j+2048 — 2048 projections apart.
        int32_t ci[2] = {(int32_t)(p % num_cols),
                         (int32_t)((p + num_cols / 2) % num_cols)};
        kd.fn(col_ptrs.data(), ci, weights.data(), density,
              sel.data(), num_rows, out.data());
        sink += out[p % num_rows];  // touch output; vary index to prevent hoisting
      }
      auto t1 = std::chrono::steady_clock::now();
      rep_times.push_back(
          std::chrono::duration<double>(t1 - t0).count());
    }
    std::sort(rep_times.begin(), rep_times.end());
    double med = rep_times[num_reps / 2];
    double per_us = med / num_projs * 1e6;

    if (baseline_us < 0) baseline_us = per_us;
    double speedup = baseline_us / per_us;

    printf("%-24s  %14.3f  %11.2f  %12.2fx\n",
           kd.name, med, per_us, speedup);
    fflush(stdout);
  }

  printf("\n(sink=%f)\n", (float)sink);
  printf("\nDecision threshold: AVX-512 gather must beat scalar_m4 by >= 20%% (speedup >= 1.20x)\n");
  return 0;
}
