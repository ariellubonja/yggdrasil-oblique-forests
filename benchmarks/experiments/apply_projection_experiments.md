# ApplyProjection experiments log

Hardware (unless noted): Intel Core Ultra 9 185H, P-cores only, turbo off, taskset CPU 0.
Workload: 3M×4096 trunk, 1 thread, 1 tree, depth=-1, Oblique+Exact.
Timing: `kProjectionEvaluate` chrono scope. Gate: ≥20% on targeted scope.

---

## Variants

| Tag | Build flag | What it does |
|---|---|---|
| baseline | — | Per-call `ProjectionEvaluator::Evaluate`, one per (node, projection). |
| projection-matrix control (PMC) | `--config=projection_matrix_control` | Per-node fused rows-outer/projs-inner. |
| V2 | — | Single-pass (n,i,p) triplets + flat buffers. 3.85× slower — abandoned. |
| V2-rev1 | — | Stateful (n,i,p) walk, no flat_out. 1.14× V1. |
| V2-rev2 | — | V2-rev1 + prefetch K=16. 1.74× V1 (regression). |
| **V2-rev3** ★ | `--config=depthwise_1_pass` | (node,projection) tasks, 4-row inner unroll, 4 independent accumulators → 4-way ILP. **0.66× V1.** |
| loop_swap | `--config=projeval_loop_swap` | Projection-items-outer/rows-inner Evaluate. No signal. |

---

## Phase A — Baseline diagnosis

### Top-level `perf record cycles:u`

| Function | % cycles |
|---|---|
| `MakeTrunkDataset` | 30.70 % |
| `FindBestConditionSparseObliqueTemplate` (inlined Evaluate) | **16.46 %** |
| `FindSplitLabelClassificationFeatureNumericalCart` | 12.84 % |
| libm logf | 14.70 % |
| hwy AVX2 sort | ~14.3 % |
| `ProjectionEvaluator` ctor | 1.51 % |

### Hot lines inside `Evaluate` (perf annotate)

| Insn | Source | % in-fn cycles |
|---|---|---|
| `addss` (3f836f) | accumulate item-1 | **48.10 %** |
| `addss` (3f84bd) | accumulate item-2 | **19.75 %** |
| `addss` (3f8499) | accumulate item-3 | 7.56 % |
| `movss (%r14,%r13,4)` | gather attr_values[ex] | 5.46 % |

Three serialized `addss` = **75.4 %** of in-function cycles (load-use stall on random DRAM gathers).

### Memory counters (`perf stat`, 1 tree)

| Metric | Value |
|---|---|
| L1 d-cache loads | 517 G |
| L1 miss rate | 1.36 % |
| LLC-load-misses / LLC-loads | **90.2 %** |
| IPC | 2.20 |
| Tree wall | 266.8 s |

---

## Phase B — V2 iteration (3M×4096, 1 thread, 1 tree)

### Run summary

| Variant | tree wall | ΣProjEval | vs baseline |
|---|---|---|---|
| baseline | — | 57.23 s | 1.00× |
| projection-matrix control | 133.8 s | 56.74 s | 0.99× |
| V2 (n,i,p triplet) | 292.5 s | ~218 s | 3.81× ⛔ |
| V2-rev1 (stateful walk) | 138.8 s | ~65 s | 1.14× |
| V2-rev2 (rev1 + prefetch K=16) | 170.9 s | ~100 s | 1.75× ⛔ |
| **V2-rev3** (4-row unroll) | **113.1 s** | **~38.8 s** | **0.68×** |

### V2-rev3 median (5 trees, `parallel_chrono.py`)

| Tree | Baseline | V2-rev3 |
|---|---|---|
| 0 | 57.23 s | 45.30 s |
| 1 | 57.32 s | 46.94 s |
| 2 | 56.62 s | 53.02 s |
| 3 | — | 47.27 s |
| 4 | — | 45.80 s |
| **median** | **57.23 s** | **46.94 s** |

**17.98% — below gate by ~2 pp.** Gap to pinned measurement (34.45%) is CPU migration cost on the 4-DRAM-miss-in-flight kernel.

---

## Phase F — Bandit-pruned projection sampling

Sample 2N candidates, partial AP on R/16 bag, keep top N/2. Theoretical: 0.625 N×R vs 1.0 → 37.5% savings.

### Chrono (5-tree median, 1 thread, 3M×4096)

| Tree | Bandit AP (s) |
|---|---|
| 0–4 | 112.5 / 114.3 / 111.9 / 113.2 / 112.2 |
| **median** | **112.5 s** vs baseline 129.2 s |

**13.0% — FAIL (gate ≥20%).** Bandit fires only at R≥1000 (depths 1–12 ≈ 35% of work); 37.5% × 35% ≈ 13.1% — matches.

### Accuracy (eval_ab_e2e, 30 trees)

| Dataset | Δ acc (pp) |
|---|---|
| PhishingWebsites | +0.151 |
| bank-marketing | −0.069 |
| Internet-Advertisements | −0.237 |
| credit-approval | 0.000 |

Accuracy: PASS. Speed: FAIL. Branch not merged.

---

## Phase G — Column-streaming pregather

Sample N projections, build union U distinct columns, pregather U×R dense buffer, compute N dot products from sequential reads.

### Chrono (5-tree, 1 thread, 3M×4096)

| Tree | Baseline AP+EvalProj | Pregather FP+AP+EvalProj |
|---|---|---|
| 0–4 | 132.4 / 132.0 / 136.5 / 134.5 / 134.5 | 121.2 / 120.1 / 119.9 / 121.8 / 120.2 |
| **median** | **134.5 s** | **120.2 s** |

**10.6% — FAIL.** kProjectionEvaluate drops 63s→3.3s (19×) but kFeaturePregather adds 50s/tree. Root cause: dense_buf (1.14 GB) >> L3 (32 MB) — 3× more DRAM traffic offsets per-byte latency advantage.

### Per-component (tree 0)

| depth | nodes | Baseline AP | Pregather FP | Pregather AP |
|---|---|---|---|---|
| 1 | 1 | 0.263s | 0.188s | 0.172s |
| 5 | 16 | 1.422s | 1.106s | 0.145s |
| 10 | 512 | 2.780s | 2.114s | 0.080s |
| 15 | 12538 | 3.792s | 3.261s | 0.118s |
| 20 | 46932 | 3.604s | 3.012s | 0.185s |

### E2e (eval_ab_e2e, 30 trees)

| Dataset | Δ time | Δ acc (pp) |
|---|---|---|
| trunk 100K×4096 | +3.85% | — |
| epsilon | +8.88% | — |
| PhishingWebsites | +11.30% | +0.050 |
| bank-marketing | +3.00% | −0.162 |
| Internet-Advertisements | +1.38% | −0.305 |
| credit-approval | −6.27% | −0.483 |

Accuracy PASS (max |Δ| 0.483 pp). Speed FAIL. Branch not merged.

---

## Phase H — AVX-512 gather microbench

Setup: 4096 cols × 65536 floats = 1 GB (>> L3), DRAM-bound.

| Kernel | per_proj_us | vs scalar_m4 |
|---|---|---|
| scalar_m4 | 38.48 | 1.00× |
| scalar_m8 | 37.97 | 1.01× |
| scalar_m16 | 37.11 | 1.04× |
| avx2_gather_m8 | 34.26 | 1.12× |
| avx512_gather_m16 | 35.22 | 1.09× |

**Pre-judged FAIL** — both below 1.20×. DRAM latency bottleneck; packing into gather instructions doesn't add MLP beyond OOO.

---

## ★ Phase I — V2-rev3 vs stock YDF (m7i, 5 trees, 1 thread)

| Config | AP total | EP total | Combined |
|---|---|---|---|
| stock YDF | 320.58s | 349.35s | 669.93s |
| **V2-rev3** | **266.89s** | 348.80s | 615.67s |
| **Speedup** | **★ 1.201× (+20.1%)** | 1.001× | 1.088× |

**GATE PASSED.** Mechanism: M=4 accumulators expose 4-way ILP → 4 concurrent DRAM misses/item.

---

## Phase J — THP=always on top of V2-rev3 (m7i, 5 trees, 1 thread)

| Config | AP total | EP total | Combined |
|---|---|---|---|
| stock YDF | 320.58s | 349.35s | 669.93s |
| V2-rev3 (no THP) | 266.89s | 348.80s | 615.67s |
| **V2-rev3 + THP=always** | **258.79s** | 345.21s | 604.00s |
| THP marginal | 1.031× | 1.010× | 1.019× |

**★ 1.239× vs stock. THP alone +3.1%.** Mechanism: 2MB pages reduce TLB entries for 48GB column array from ~262K to ~512. Environmental only — requires `echo always`; madvise path below failed.

---

## Phase K — madvise(MADV_HUGEPAGE) per-column — ABORTED (regression)

Expected ~16 min; aborted after >24 min. AnonHugePages = 0 throughout. Root cause: `khugepaged` scans only 16 MB/10s interval → 8 hours to promote 48 GB. Compaction attempts cause TLB shootdowns → ~50% regression. Path forward requires `mmap(MAP_HUGETLB)` or aligned allocator — out of scope.

---

## Chrono Coverage Audit — BFS-only vs baseline (2026-06-07)

| | baseline | bfs_only |
|---|---|---|
| TreeTrain | 168.3 s | 172.3 s |
| NodeTrain / BfsNodeLoop | 167.9 s (99.77%) | 171.8 s (99.75%) |
| AP | 66.5 s | 67.0 s |
| SampleProjection | 7.0 s | 7.4 s |

BFS-only cost ≈ DFS at this shape.

---

## Phase L — Row-major shadow layout + dual-bf16 hybrid (2026-06-09–10)

**Motivation:** AP is MLP-saturated. 66% of AP is depths 12–23 (70–1500 rows/node). In that band, CM pays R×192 isolated DRAM lines; RM puts all 192 reads for one example inside one 16 KB row. First RM-winning depth: 9.

### Microbench (`row_major_hotband.cc`, 1 GB, taskset CPU 0)

| R (rows/node) | CM ns/load | RM ns/load | RM vs CM |
|---|---|---|---|
| 128 | 8.00 | 4.34 | **1.84×** |
| 512 | 6.89 | 4.38 | **1.57×** |
| 2048 | 3.76 | 4.28 | 0.88× (CM wins) |

AMAC lane interleaving: 0.87×/0.74×/0.66× at lanes 2/4/8 — FAILED (OOO already extracts cross-task MLP).

### Same-state chrono: CM vs RM-everywhere (turbo ON, 3M×4096, 1 tree)

| depth | ctrl AP | RM AP | ratio |
|---|---|---|---|
| 1 | 0.23 | 2.64 | 0.09× |
| 5 | 1.03 | 2.10 | 0.49× |
| 10 | 2.23 | 2.08 | 1.07× |
| 15 | 3.00 | 2.06 | 1.46× |
| 18 | 3.48 | 1.85 | 1.88× |
| 20 | 3.05 | 1.39 | 2.19× |
| 23 | 1.53 | 0.62 | 2.46× |
| **ΣAP** | **50.5** | **44.7** | **1.13×** |

Hybrid prediction min(CM,RM) per depth = 35.1 s → **1.44× ΣAP**.

### Threshold ablation: bf16 CM vs RM vs hybrid (1-run probes, turbo ON)

| depth | ctrl fp32-CM | all-CM-bf16 | all-RM-bf16 | hybrid 20K |
|---|---|---|---|---|
| 1 | 0.23 | 0.33 | 0.33 | 0.34 |
| 5 | 1.03 | 0.67 | 1.61 | 0.69 |
| 15 | 3.00 | 1.99 | 1.61 | 1.58 |
| 18 | 3.48 | 2.17 | 1.46 | 1.41 |
| 23 | 1.53 | 0.87 | 0.47 | 0.50 |
| **ΣAP** | **50.5** | **33.2** | **32.8** | **26.8** |

bf16 alone = 1.52×; layout dispatch adds 1.24× → hybrid 1.88×. Hybrid ≈ per-depth min(CM,RM).

### ★★ 5-tree median — GATE PASSED (turbo ON, E-cores ON)

| | median ΣAP | median TreeTrain |
|---|---|---|
| control fp32-CM | **64.14 s** | **117.3 s** |
| hybrid dual-bf16 (RM≤20K) | **26.57 s** | **77.6 s** |
| **ratio** | **★ 2.41×** | **1.51×** |

### Canonical-state confirmation (2026-06-10, P-cores only / no HT / no turbo)

| metric (5-tree median) | control fp32 CM | hybrid dual-bf16 | ratio |
|---|---|---|---|
| ΣApplyProjection | 79.56 s | 35.03 s | **2.27×** |
| TreeTrain | 186.9 s | 130.1 s | **1.44×** |

### Multithread

| config | threads | wall |
|---|---|---|
| control | 22 (6P+16E) | 417.4 s |
| hybrid | 22 (6P+16E) | 408.2 s (+2.3% — wash) |
| control | 6P only | 640.5 s |
| hybrid | 6P only | **422.1 s (★ 1.52×)** |

22-thread wash is E-core artifact (scalar bf16 widen expensive on E-cores). Xeon (homogeneous P-cores): expect 1.5× to transfer.

### bf16 accuracy gate (CC18, Oblique Exact, 30 trees, 3 seeds)

| task | Δ OOB (pp), seeds 1/2/3 |
|---|---|
| PhishingWebsites | 0 / 0 / 0 |
| bank-marketing | +0.13 / −0.05 / +0.03 |
| Internet-Advertisements | 0 / −0.07 / −0.31 |
| credit-approval | +0.48 / 0 / +0.48 |

Max |Δ| = 0.48 pp — **PASS** (gate ≤5 pp). Epsilon (400K×2000, seed 1): Δ −0.27 pp — PASS.

**Phase L verdict:** AP 2.41× / TreeTrain 1.51× (5-tree medians), accuracy ≤0.5 pp. Engineering remaining: vectorized bf16 widen (depth 1–3 regresses), real-dataset dual-store wiring, threshold auto-tuning.

---

## Phase M — dual_fp32 (precision-neutral dispatch, 2026-06-10)

A/B at 1.5M×4096 (fits 2× fp32 in 61 GB), canonical P-core state:

| metric | control fp32 CM | dual_fp32 (rm20k) | ratio |
|---|---|---|---|
| ΣApplyProjection | 36.14 s | 19.32 s | **★ 1.87×** |
| TreeTrain | 88.5 s | 64.5 s | **1.37×** |

dual_bf16 at same shape: AP 16.47 s (2.19×). Decomposition: dispatch alone 1.87×, bf16 halving adds 1.17× on top. dual_fp32 captures ~85% of bf16-hybrid AP win with zero precision change.

### dual_fp32 threshold ablation at 1.5M×4096 (ΣAP 5-tree medians)

| thr | 0 (pure CM) | 312 | 625 | **1250** | 2500 | 5000 | 10k | 20k | 40k | INF (pure RM) |
|---|---|---|---|---|---|---|---|---|---|---|
| AP | 21.42 | 18.85 | **18.77** | 18.88 | 19.03 | 19.18 | 19.00 | 19.32 | 19.63 | 23.62 |

| thr (750k×4096) | 0 | 156 | 312 | 625 | 1250 | 2500 | 5000 | INF |
|---|---|---|---|---|---|---|---|---|
| AP | 9.10 | **8.43** | 8.49 | 8.55 | 8.66 | 8.80 | 8.84 | 11.00 |

Optimum is a flat plateau ~150–10k rows; both endpoints lose clearly. Recommend default ~625–1000. Most of the dual_fp32 win is the kernel (stock CM 36.14 → dual-kernel pure-CM 21.42); dispatch adds last ~12%.

**Naming:** `--dataset_layout=dynamic_row_col_major` (fp32) / `dynamic_row_col_major_bf16` (bf16).

---

## Phase N — column-centric depthwise_1_pass rewrite (2026-06-10)

Old dw1 was just the nodewise gather loop batched per depth (no column sharing). Rewrite: per depth, counting-sort (node,proj,weight) references by column, sweep each column once over the block's merged bag.

### 3M×4096, 5-tree medians, canonical state

| | old dw1 | column-centric | ratio |
|---|---|---|---|
| ΣApplyProjection | 77.91 s | 49.62 s | **★ 1.57×** |
| TreeTrain | 181.2 s | 153.0 s | 1.18× |

Block-size ablation (ΣAP): 4 MiB → 55.02, 16 MiB → 49.62, 64 MiB → 47.05, 256 MiB → **45.62 s (1.71×)**. Default set to 256 MiB.

### Shape comparison

| shape | new dw1 | nodewise control | ratio |
|---|---|---|---|
| 3M×4096 | 45.62 s | 79.56 s | **1.71×** |
| 30k×400k | 3.71 s | 5.09 s | **1.37×** |
| 11M×28 (HIGGS) | 19.70 s | 25.13 s | **1.28×** |

### O.5 PMC decomposition — column-sharing vs kernel quality (3M×4096)

nodewise 79.6 → **PMC 59.97** → dw1 45.6. ~25% from kernel quality (direct pointers, big/small split — PMC has it all in node order), ~24% from actual cross-node column sharing. Both halves real.

### HIGGS decomposition (11M×28)

| | nodewise | PMC | dw1 |
|---|---|---|---|
| AP | 25.13 | 25.39 | **19.70** |
| TT | **85.63** | 106.32 | 99.32 |

PMC ≈ nodewise AP: kernel order irrelevant at 28 cols. dw1 wins AP (−22%) but loses TT: driver overhead (SampleProjection +5.7s, SplitExamplesInPlace +2.9s, slab/frontier bookkeeping ~+7s) = +19s vs AP win −5.4s. Fix is driver cost reduction, not AP kernel.

---

## Phase O — e2e validation (m7i, 2026-06-10)

### O.1 Dynamic_Row_Col_Major fp32 — m7i e2e (7 runs median)

| shape | control CM (s) | dynamic fp32 (s) | speedup |
|---|---|---|---|
| 1.5M×4096 | 330.93 ±0.53 | 265.65 ±1.83 | **1.246× (−19.7%)** |
| 15k×400k | 117.30 ±0.72 | 64.27 ±0.83 | **1.825× (−45.2%)** |

Cost: 2× dataset RAM.

### O.2 Column-centric dw1 — laptop e2e (3-run medians)

| shape | new dw1 | old dw1 | DFS baseline |
|---|---|---|---|
| HIGGS | 520.1 | 539 | **372** |
| 3M×4096 | **755.8** | 853 | 788 |
| 30k×400k | 307.3 | 307 | **274** |

First depth-batched config to beat DFS at 3M×4096 (−4.1%), but well short of single-thread chrono projection (thread-oversubscription: 6 trees × per-depth ThreadPool(6) ≈ 36 threads on 6 P-cores).

### O.4 YDF_DW1_THREADS=1 on HIGGS — FALSIFIED

Median 550.0 s ±9.3 vs multithreaded dw1 520.1. Serializing is worse — oversubscription was not the issue. Root cause: HIGGS has only 28 columns so column sharing is nil; depth-batch driver is pure overhead there.

---

## Standing conclusions

1. **AP is DRAM-latency-bound.** 90% LLC miss rate; every gather is a random read into a 48 GB column-major array.
2. **MLP (independent accumulators) is the one kernel lever that has paid.** Prefetch, HW gather, loop order: all failed.
3. **Adding memory traffic to "fix" locality loses.** Pregather, subtree gather: failed. Dual store wins only because its transpose is paid once per forest.
4. **Sparse oblique sampling math** (P=⌈F^0.5⌉, density 1.5): ~distinct features per node, ~2% overlap between a node and its descendants → no within-node row reuse and no ancestor-descendant column reuse.
5. **Per-node O(num_features) setup costs dominate wide datasets** — check for these before touching kernels.
6. **dw1 column sharing only pays when D >> columns touched per depth** (fails on HIGGS with D=28).
