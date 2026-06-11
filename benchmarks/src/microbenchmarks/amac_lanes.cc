// Pre-screen for Idea 4 (AMAC lane interleaving across (node, projection)
// tasks). Hot-band shape: many small tasks (R rows each), d=3 items per
// projection, 1 GB column-major dataset >> L3 so every gather is a DRAM miss.
// Compares task-sequential V2-rev3-style m4 kernel vs interleaving L tasks.
//
// g++ -O3 -march=native -o /tmp/amac_lanes amac_lanes.cc
#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <random>
#include <vector>

constexpr int kCols = 4096;
constexpr int kRows = 65536;  // 4096 * 65536 * 4B = 1 GB
constexpr int kD = 3;         // items per projection
constexpr int kR = 128;       // rows per node (hot-band size)
constexpr int kProjsPerNode = 8;
constexpr int kNodes = 12500;  // 100K tasks total
constexpr int kTasks = kNodes * kProjsPerNode;

struct Task {
  const uint32_t* rows;  // kR sorted row indices
  int cols[kD];
  float w[kD];
  float* out;  // kR outputs
};

static inline void RunRowBlock4(const Task& t, int i, const float* data) {
  float a0 = 0, a1 = 0, a2 = 0, a3 = 0;
  for (int k = 0; k < kD; ++k) {
    const float* col = data + (size_t)t.cols[k] * kRows;
    const float w = t.w[k];
    a0 += w * col[t.rows[i + 0]];
    a1 += w * col[t.rows[i + 1]];
    a2 += w * col[t.rows[i + 2]];
    a3 += w * col[t.rows[i + 3]];
  }
  t.out[i + 0] = a0;
  t.out[i + 1] = a1;
  t.out[i + 2] = a2;
  t.out[i + 3] = a3;
}

// Task-sequential, 4-row unroll (V2-rev3 analogue).
double KernelSeq(const std::vector<Task>& tasks, const float* data) {
  auto t0 = std::chrono::steady_clock::now();
  for (const Task& t : tasks)
    for (int i = 0; i < kR; i += 4) RunRowBlock4(t, i, data);
  return std::chrono::duration<double>(std::chrono::steady_clock::now() - t0)
      .count();
}

// L tasks interleaved at row-block granularity.
template <int L>
double KernelLanes(const std::vector<Task>& tasks, const float* data) {
  auto t0 = std::chrono::steady_clock::now();
  size_t n = tasks.size();
  size_t g = 0;
  for (; g + L <= n; g += L)
    for (int i = 0; i < kR; i += 4)
      for (int l = 0; l < L; ++l) RunRowBlock4(tasks[g + l], i, data);
  for (; g < n; ++g)
    for (int i = 0; i < kR; i += 4) RunRowBlock4(tasks[g], i, data);
  return std::chrono::duration<double>(std::chrono::steady_clock::now() - t0)
      .count();
}

int main() {
  std::mt19937 rng(42);
  std::vector<float> data((size_t)kCols * kRows);
  for (auto& v : data) v = (float)(rng() & 0xffff) / 65536.f;

  // Node row sets: kR sorted distinct random rows each.
  std::vector<uint32_t> all_rows((size_t)kNodes * kR);
  std::uniform_int_distribution<uint32_t> row_dist(0, kRows - 1);
  for (int n = 0; n < kNodes; ++n) {
    uint32_t* r = &all_rows[(size_t)n * kR];
    for (int i = 0; i < kR; ++i) r[i] = row_dist(rng);
    std::sort(r, r + kR);
  }
  std::vector<float> out((size_t)kTasks * kR);
  std::uniform_int_distribution<int> col_dist(0, kCols - 1);
  std::vector<Task> tasks(kTasks);
  for (int t = 0; t < kTasks; ++t) {
    tasks[t].rows = &all_rows[(size_t)(t / kProjsPerNode) * kR];
    for (int k = 0; k < kD; ++k) {
      tasks[t].cols[k] = col_dist(rng);
      tasks[t].w[k] = (float)(rng() & 0xff) / 256.f;
    }
    tasks[t].out = &out[(size_t)t * kR];
  }

  // Warm-up pass (page-fault the dataset).
  KernelSeq(tasks, data.data());

  constexpr int kReps = 3;
  auto report = [&](const char* name, double (*fn)(const std::vector<Task>&,
                                                   const float*)) {
    double best = 1e30;
    for (int r = 0; r < kReps; ++r)
      best = std::min(best, fn(tasks, data.data()));
    double checksum = 0;
    for (size_t i = 0; i < out.size(); i += 9973) checksum += out[i];
    printf("%-12s best=%.3fs  (%.1f ns/load, checksum %.3f)\n", name, best,
           best * 1e9 / ((double)kTasks * kR * kD), checksum);
    return best;
  };

  double base = report("seq_m4", KernelSeq);
  double l2 = report("lanes2_m4", KernelLanes<2>);
  double l4 = report("lanes4_m4", KernelLanes<4>);
  double l8 = report("lanes8_m4", KernelLanes<8>);
  printf("speedup vs seq_m4: L2 %.2fx  L4 %.2fx  L8 %.2fx\n", base / l2,
         base / l4, base / l8);
  return 0;
}
