# Leaderboard — end-to-end training runtime (seconds, lower is better)

**Protocol:** `benchmarks/evaluation/runtime.sh` vectorized section —
Oblique + Dynamic Random Histogram, 64 bins, `dynamic_split_threshold=250`,
`NUM_TREES = nproc×5`, `num_threads=-1`, AVX2 build. Median of N runs
(N in the source CSV). **Update a cell only with a number produced by this
protocol**, and link the CSV.

**Machine: AWS m7i (Xeon Platinum 8488C, 8 vCPU, 61 GiB).** NUM_TREES=40.

**Bold** = row best. `oom` = exceeded 61 GiB (dual layouts keep 2 full
fp32 copies). Blank = not measured.

| Dataset | stock (DFS col) | DFS row | BFS col | BFS row | Dyn RC BFS | Dyn RC DFS | PMC nodewise | DW1 1-pass | 4row+cache |
|---|---|---|---|---|---|---|---|---|---|
| HIGGS 11m×29 | 282 | 283 | 336 | 310 | 415 | 266 | 394 | 387 | **251.6** |
| trunk 3m×4096 | 688 | 690 | 705 | 735 | oom | oom | oom | oom | **576.1** |
| trunk 1.5m×4096 | 331 | 320 | 316 | 341 | 256 | **241** | 271 | 261 | 255.8 |
| trunk 300k×40k | 288 | 293 | 287 | 294 | oom | oom | | | **186.9** |
| trunk 150k×40k | | | | | 105 | 110 | | | **85.6** |
| trunk 30k×400k | 245 | 154 | 223 | 152 | oom | oom | 215 | 215 | **73.4** |
| trunk 10k×400k | 78 | 56 | 79 | 56 | 40 | 47 | | | **22.6** |

## Column recipes (exact build + binary flags)

| Column | EXTRA_BAZEL_CONFIGS | EXTRA_TRAIN_ARGS | Memory |
|---|---|---|---|
| stock (DFS col) | — | — | 1× dataset |
| DFS row | `--config=row_major_dataset_layout` | `--dataset_layout=row` | 1× (row store only, trunk) |
| BFS col | `--config=bfs_only` | — | 1× |
| BFS row | `--config=bfs_only --config=row_major_dataset_layout` | `--dataset_layout=row` | 1× |
| Dyn RC BFS (T=5000) | `--config=projection_matrix_control --config=row_major_dataset_layout` ⁽¹⁾ | `--dataset_layout=dynamic_row_col_major` | **2×** |
| Dyn RC DFS (T=5000) | `--config=row_major_dataset_layout` ⁽¹⁾ (commit 891369ae) | `--dataset_layout=dynamic_row_col_major` | **2×** |
| PMC nodewise control | `--config=projection_matrix_control` ⁽¹⁾ | — | 1× + P·n slab |
| DW1 1-pass | `--config=depthwise_1_pass` ⁽¹⁾ | — | 1× + level slabs |
| 4row+cache | `--config=evaluate_4row --config=cache_projection_evaluator` | — | 1× |

⁽¹⁾ Best-effort reconstruction — these runs predate the provenance header
in `runtime.sh` (the logs did not record `EXTRA_BAZEL_CONFIGS`). Treat
with care when re-running; from now on every log/CSV gets a `.meta`
sidecar with exact configs.

## Source CSVs (m7i)

- 4row+cache: `runtime_3runs_4rowcache_confirm_m7i.csv` (3-run medians),
  `runtime_1runs_4rowcache_fullsize_m7i.csv` (full-size, 1 run),
  `runtime_1runs_4row_cache_m7i.csv` (first 1-run pass)
- Dyn RC: `runtime_bfs_dynamic_row_col_major_fp32_m7i.csv`,
  `3runs_dfs_dynamic_row_col_major_fp32_m7i.log`
- stock / layout columns: `runtime_1runs_dfs_stock_ydf.csv`,
  `runtime_1runs_{bfs_colmajor,bfs_rowmajor,dfs_rowmajor}.csv`
- other columns: user-recorded spreadsheet (2026-06-12), pre-provenance
