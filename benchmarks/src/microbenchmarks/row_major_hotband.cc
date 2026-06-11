// Pre-screen: row-major shadow layout for hot-band AP (small nodes).
// A node = R sorted random rows; K=64 projections x d=3 items = 192 column
// draws shared by all R rows. Column-major baseline does R*192 isolated DRAM
// gathers. Row-major reads, per row, 192 offsets inside one contiguous 16KB
// row (4 pages, ~65% of lines) -- TLB/prefetch friendly.
// Also tests a THP(2MB-page) variant of the column-major gather.
//
// g++ -O3 -march=native -o /tmp/row_major_hotband row_major_hotband.cc
#include <sys/mman.h>

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <random>
#include <vector>

constexpr int kCols = 4096;
constexpr int kRows = 65536;  // 1 GB
constexpr int kK = 64;        // projections per node
constexpr int kD = 3;         // items per projection
constexpr int kReps = 3;

struct Item {  // flattened (projection, column, weight)
  int col;
  int proj;
  float w;
};

struct Node {
  std::vector<uint32_t> rows;       // R sorted row indices
  std::vector<Item> items;          // K*d items, sorted by col
  std::vector<int> proj_col[kK];    // per-projection columns (CM kernel)
  std::vector<float> proj_w[kK];
};

static float* AllocBuf(size_t bytes, bool thp) {
  void* p = mmap(nullptr, bytes, PROT_READ | PROT_WRITE,
                 MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
  if (thp) madvise(p, bytes, MADV_HUGEPAGE);
  return (float*)p;
}

using Clock = std::chrono::steady_clock;
static double Secs(Clock::time_point t0) {
  return std::chrono::duration<double>(Clock::now() - t0).count();
}

// Column-major: V2-rev3-style, per (proj), 4-row blocks.
double KernelCM(const std::vector<Node>& nodes, const float* cm,
                std::vector<float>& out) {
  auto t0 = Clock::now();
  for (const Node& n : nodes) {
    const int R = n.rows.size();
    for (int p = 0; p < kK; ++p) {
      float* o = &out[(size_t)p * R];
      int i = 0;
      for (; i + 4 <= R; i += 4) {
        float a0 = 0, a1 = 0, a2 = 0, a3 = 0;
        for (size_t k = 0; k < n.proj_col[p].size(); ++k) {
          const float* col = cm + (size_t)n.proj_col[p][k] * kRows;
          const float w = n.proj_w[p][k];
          a0 += w * col[n.rows[i + 0]];
          a1 += w * col[n.rows[i + 1]];
          a2 += w * col[n.rows[i + 2]];
          a3 += w * col[n.rows[i + 3]];
        }
        o[i] = a0; o[i + 1] = a1; o[i + 2] = a2; o[i + 3] = a3;
      }
      for (; i < R; ++i) {
        float a = 0;
        for (size_t k = 0; k < n.proj_col[p].size(); ++k)
          a += n.proj_w[p][k] *
               cm[(size_t)n.proj_col[p][k] * kRows + n.rows[i]];
        o[i] = a;
      }
    }
  }
  return Secs(t0);
}

// Row-major shadow: per row, one pass over the node's K*d items (col-sorted),
// scattering into K accumulators (L1-resident), then store K values.
double KernelRM(const std::vector<Node>& nodes, const float* rm,
                std::vector<float>& out) {
  auto t0 = Clock::now();
  float acc[kK];
  for (const Node& n : nodes) {
    const int R = n.rows.size();
    for (int i = 0; i < R; ++i) {
      const float* row = rm + (size_t)n.rows[i] * kCols;
      std::memset(acc, 0, sizeof(acc));
      for (const Item& it : n.items) acc[it.proj] += it.w * row[it.col];
      for (int p = 0; p < kK; ++p) out[(size_t)p * R + i] = acc[p];
    }
  }
  return Secs(t0);
}

int main() {
  std::mt19937 rng(42);
  const size_t bytes = (size_t)kCols * kRows * 4;
  float* cm = AllocBuf(bytes, false);
  float* cm_thp = AllocBuf(bytes, true);
  float* rm = AllocBuf(bytes, false);
  for (size_t i = 0; i < (size_t)kCols * kRows; ++i)
    cm[i] = (float)(rng() & 0xffff) / 65536.f;
  std::memcpy(cm_thp, cm, bytes);
  for (int c = 0; c < kCols; ++c)  // transpose
    for (int r = 0; r < kRows; ++r)
      rm[(size_t)r * kCols + c] = cm[(size_t)c * kRows + r];

  std::uniform_int_distribution<uint32_t> row_d(0, kRows - 1);
  std::uniform_int_distribution<int> col_d(0, kCols - 1);

  for (int R : {128, 512, 2048}) {
    int n_nodes = 12'000'000 / (R * 24);  // ~constant total loads per config
    std::vector<Node> nodes(n_nodes);
    for (Node& n : nodes) {
      n.rows.resize(R);
      for (auto& r : n.rows) r = row_d(rng);
      std::sort(n.rows.begin(), n.rows.end());
      for (int p = 0; p < kK; ++p)
        for (int k = 0; k < kD; ++k) {
          int c = col_d(rng);
          float w = (float)(rng() & 0xff) / 256.f;
          n.proj_col[p].push_back(c);
          n.proj_w[p].push_back(w);
          n.items.push_back({c, p, w});
        }
      std::sort(n.items.begin(), n.items.end(),
                [](const Item& a, const Item& b) { return a.col < b.col; });
    }
    std::vector<float> out((size_t)kK * R);
    const double total_loads = (double)n_nodes * R * kK * kD;

    auto bench = [&](const char* name, auto&& fn, const float* buf) {
      double best = 1e30;
      fn(nodes, buf, out);  // warm-up
      for (int rep = 0; rep < kReps; ++rep)
        best = std::min(best, fn(nodes, buf, out));
      double cs = 0;
      for (size_t i = 0; i < out.size(); i += 97) cs += out[i];
      printf("  %-10s best=%.3fs  %.2f ns/load  (cs %.2f)\n", name, best,
             best * 1e9 / total_loads, cs);
      return best;
    };
    printf("R=%d (%d nodes, %d items/node):\n", R, n_nodes, kK * kD);
    double cmt = bench("CM", KernelCM, cm);
    double cmthp = bench("CM+THP", KernelCM, cm_thp);
    double rmt = bench("RM", KernelRM, rm);
    printf("  -> RM vs CM: %.2fx   THP vs CM: %.2fx\n\n", cmt / rmt,
           cmt / cmthp);
  }
  return 0;
}
