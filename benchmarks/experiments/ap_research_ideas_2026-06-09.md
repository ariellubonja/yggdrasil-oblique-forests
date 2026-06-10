# Novel systems-research ideas for `ProjectionEvaluator::Evaluate` (ApplyProjection)

**Date:** 2026-06-09. **Inputs:** `apply_projection_experiments.md` (Phases A–K),
chrono CSVs in `benchmarks/results/per_function_timing/.../Dynamic Random Histogram/trunk_3000000_x_4096/`,
`runtime_full_*.csv`, code in `oblique.cc:1212` (`Evaluate`) and
`oblique_cpu_symmetric_depthwise_ap.cc`.

## The constraint surface (what the data already proves)

1. **AP is DRAM-latency-bound, not compute-bound.** 90% LLC miss rate; perf
   attributes ~75% of in-function cycles to the `addss` load-use stall
   (Phase A). Every gather is `col[bag[i]]` into a 48 GB column-major array.
2. **MLP is the one lever that has paid.** V2-rev3's 4 independent
   accumulators → +20.1% AP (Phase I). Prefetch (rev2), AVX2/512 gather
   (Phase H) did *not* add MLP beyond what OOO already extracts → failed.
3. **Bandwidth tricks fail when they add traffic.** Pregather (Phase G)
   converted random→sequential but tripled DRAM volume (copy-out + re-read of
   a 1.14 GB dense buffer) → only 10.6%.
4. **Depth profile (DRH, 3M×4096, 1 thread, `-1Depth-1Threads.csv`):**
   AP = 65.6 s of 174 s TreeTrain. Depths 1–11 ≈ 19 s; **depths 12–23 ≈ 43 s
   (66% of AP)** with 2k–45k nodes of ~70–1500 rows each; depths ≥24 ≈ 3 s.
   Shallow depths are already near-sequential (sorted dense bag, ~21 GB/s
   effective) — *there is little left to win above depth ~8*. The fight is
   the mid-deep band: tiny per-node row sets with page-sized strides.
5. **Sharing works when it removes re-reads.** symmetric_depthwise_ap
   (~20%, −16/−29% e2e on trunk shapes) wins by making one column sweep serve
   all 2^d nodes of a level — but only under the shared-projection constraint.
6. Each node owns a **contiguous, sorted segment** of the bag
   (`SplitExamplesInPlace`), and trees are statistically independent given
   their RNG streams. Both properties are exploitable and currently unused.

Implication: new ideas must either (a) create *reuse* so a DRAM line serves
many (node, projection) consumers, (b) shrink the *footprint* the gathers
land in so they stop being DRAM misses, or (c) raise MLP further. Anything
that merely reshuffles a single tree's single-depth gathers moves no bytes
fewer and will land in the Phase-G graveyard.

---

## Idea 1 — Tree-lockstep training: depth-synchronous column sweeps shared across all trees ★ highest ceiling

**Mechanism.** Train all T trees simultaneously, level-synchronous (the BFS
frontier scheduler from `origin/1-node-ap-column-reuse` already gives the
per-tree machinery). At each depth, collect every (tree, node, projection,
item) over all T trees, group by **column**, and sweep each distinct column
once, scattering `w * col[i]` into per-(tree,node,proj) accumulator slices.
This is symmetric_depthwise_ap's trick with the sharing axis moved from
"nodes within a level" (requires symmetry) to "trees within a forest"
(requires nothing — trees are independent anyway; RNG streams unchanged,
output bit-identical per tree).

**Why the traffic math works where Phase G failed.** Within one tree, node
row-sets at a depth are disjoint → merging them saves nothing. Across trees,
row-sets *overlap* (independent bags/partitions over the same 3M rows). At
depth 18, one tree's users of column c cover ~120K of 2.6M active rows (≈5%,
stride ~25 lines → pure latency). With T=300 trees, E[trees using c] =
T·K·d/4096 ≈ 14, union coverage ≈ 1−(1−0.05)^14 ≈ 50% → **the sweep of
column c becomes dense → sequential**. Input traffic per depth collapses
from T × (K·d·R_node·nodes) random reads to ≤ 48 GB sequential *for the whole
forest*: at depth 18 that is ~2 s of streaming serving what currently costs
300 trees × 4.5 s. Output (`values`) traffic is unchanged — it is written in
the baseline too — so AP becomes output-bandwidth-bound, the right place to be.
No dense_buf copy exists (accumulate directly into each consumer's `values`),
so Phase G's 3× write amplification does not recur.

**Costs / risks.**
- Memory: T live bags (300 × 3M × 4 B ≈ 3.6 GB) + transient `values` per
  frontier. Feasible at 3M×4096 (dataset is 48 GB anyway); needs a cap
  (train trees in cohorts of T' if RAM-bound — benefit scales with T').
- Benefit scales with T: at T=30 (quick e2e config) union density at depth 18
  is only ~7% — still latency-bound. **This idea targets production-sized
  forests (T ≥ 150).** Must benchmark at realistic T, not the 1-tree chrono
  harness — extend `parallel_chrono.py` accordingly.
- Engineering: largest restructure on this list (training loop becomes
  forest-major). Parallelism unit changes from trees to (depth × column
  groups) — arguably better load balance than today's tree-parallel scheme.

**Experiment plan.** Stage 0: offline simulator — replay the bag partitions of
T trees (cheap, no training) and compute exact union densities and projected
DRAM traffic per depth vs T. Gate: model predicts ≥2× AP reduction at T=300
before writing any kernel. Stage 1: prototype on the BFS branch for depth ≤ a
cutoff, falling back to per-tree V2-rev3 below the density crossover.

## Idea 2 — Split-partitioned gather cache: gathered columns ride along with the bag ★ best fit for the hot band, no statistical change

**Mechanism.** When a node at depth d gathers column c at its rows (one
random-gather pass — unavoidable, it's a miss), *keep* the compacted result:
a dense float array aligned to the node's bag segment. Store it in a
node-local cache keyed by column. On `SplitExamplesInPlace`, partition every
cached column with the same permutation as the bag (sequential, cheap, and
it is exactly the operation the counting-sort work just made fast). Any
descendant that samples column c again reads a **dense sequential slice of
its own segment** instead of re-gathering from DRAM.

**Why this attacks the 66% band.** Today each of ~K·d ≈ 192 column draws per
node re-pays full DRAM latency even though ancestors already touched many of
those columns. With the cache, a draw at L levels below the cache root hits
with p = 1−(1−192/4096)^L ≈ 1−0.954^L: 38% at L=10, 61% at L=20. Weighted by
the AP time distribution below a depth-12 root, expected hit rate ≈ 25–35%,
and every *miss* also fills the cache for both children. Hits cost an L2/L3
sequential read (node segments at depth ≥12 are ≤1500 rows → cached slice
≤6 KB per column). Predicted AP reduction in the band: ~30%, i.e. ~13 s of
65.6 s AP (~20% AP scope) — at the gate by itself, and it **stacks with
V2-rev3** (misses still use the 4-accumulator kernel) and is orthogonal to
Idea 1 (cache serves within-tree reuse; lockstep serves cross-tree reuse).
- Enable only below a row threshold (e.g. R ≤ 4096) so partition traffic
  (C·R·8 B per split, sequential) stays trivial; above it, hit probability
  per byte is too low anyway — this is *by construction* the regime where
  pregather failed, so the guard is principled, not a tuning hack.
- Zero accuracy risk: values are bit-identical, RNG untouched.
- Memory: C columns × R floats per active subtree path; bounded by an LRU cap
  (evict least-recently-sampled columns); DFS order keeps only one path alive.

**Variant (statistical, optional):** bias the projection sampler toward
already-cached columns with a bounded tilt (importance-style correction or
plain random-subspace justification, Ho 1998). Raises hit rate from ~30%
toward ~80%+. Needs the standard accuracy gate (CC18 + epsilon, |Δ| ≤ 5 pp);
keep as a separate flag so the unbiased version stands alone.

## Idea 3 — Per-node shared column pools: relax "same projections" to "same columns"

**Mechanism.** symmetric_depthwise_ap assumes identical projections across a
level. The statistical content of sparse random projections only requires
columns to be drawn exchangeably — so instead sample, per node (or per
level), a pool of U columns (U ≈ 32–64), then build the K projections as
sparse signed combinations *from the pool*. AP becomes: gather each pool
column once at the node's rows (U compacted strips, reusing the V2-rev3
kernel), then form all K dot products from L1-resident strips —
K·d/U ≈ 6–12× fewer DRAM gathers per node.

**Why it's different from the failed bandit/pregather attempts.** It does not
subsample rows (bandit's coverage ceiling) and the compacted strips are
node-sized (≤6 KB/column in the hot band), not bag-sized (pregather's 1.14 GB
buffer) — they live in L1/L2 across all K reuses. It is exactly the
generalization path for symmetric_depthwise_ap: same sharing mechanics, per
node instead of per level, with the symmetry assumption deleted.

**Risk is statistical, not systems:** fewer distinct columns per node reduces
split diversity. Precedent (random subspace forests) suggests mild-to-none at
U=64, but this is the one idea on the list whose gate can fail on accuracy.
Run the accuracy A/B *first* (cheap, 30 trees, CC18+epsilon) before investing
in the kernel; the kernel itself is ~2 days on top of existing scratch
infrastructure.

## Idea 4 — AMAC-style lane interleaving: push MLP from 4 to ~12 (cheap, 1 slot)

V2-rev3 proved the latency-hiding lever with M=4 row-blocks inside one
(node, projection) pair; small nodes in the hot band (70–300 rows) don't give
the OOO window enough independent work per pair. Interleave **across pairs**:
software-pipeline 3–4 (node, projection) lanes × 4-row blocks in one loop
(Asynchronous Memory Access Chaining / group prefetching, the hash-join
literature's answer to exactly this pointer-chase profile — Kocberber et al.,
Chen et al.). Each lane owns its accumulators; every loop iteration issues
~12–16 independent loads. Redwood Cove sustains ~16 outstanding L2 misses, so
the ceiling is ~3–4× the current 4 — even at 50% efficiency this clears the
20% gate on the band where V2-rev3 currently starves. Differs from the failed
rev2 prefetch (same address stream, K-ahead) by adding *architecturally
independent* misses, the mechanism Phase H showed matters. Low effort: it is
a restructuring of the existing `depthwise_1_pass` kernel's outer loop; the
work-list already enumerates (n, p) pairs.

## Idea 5 — Fuse AP with its consumers in the DRH path (engineering, stacks with everything)

In the Dynamic-Random-Histogram pipeline the `values` vector is written by AP,
then re-read by `MinMaxNumerical` (10.3 s), then re-read again by
`AssignSamplesToHist` (21.7 s). Fold min/max into AP's accumulate epilogue
(two registers, free) → kills the 10.3 s pass outright (~6% of TreeTrain).
Second step: speculative single-pass binning — bin during AP into a wider
provisional histogram using a range estimate (parent node's min/max, or a
first-1024-rows estimate), with a fallback re-bin only when the estimate is
violated; collapses the remaining values round-trip for most nodes. Not
"research", but it is the cheapest 6–10% e2e on the table and no other idea
touches it.

## Idea 6 — Half-width shadow dataset (bf16/fp16) as a multiplier

AP tolerates reduced precision (it feeds rank/histogram decisions; the winning
split can be recomputed exactly). A bf16 shadow of the numerical columns
halves the footprint (24 GB), halves every sequential sweep in Ideas 1–3, and
doubles rows per cache line / TLB entry for residual random gathers. Pure
random-latency paths gain little (Phase A physics), so this is a **multiplier
for the sweep-based ideas, not a standalone** — evaluate it after one of
Ideas 1–3 lands. Accuracy gate cheap; epsilon/HIGGS features are far from
bf16's 8-bit-mantissa cliff. Bonus: 2× SIMD width for the dense strips of
Idea 3.

## Bench-notes (hardware/system, known magnitudes)

- Code-level huge pages: explicit `mmap(MAP_HUGETLB)` (not madvise — Phase K
  post-mortem) for column allocation: ~+3% (Phase J), worth bundling into any
  layout change that already touches allocation (Ideas 1, 6).
- Intel DSA gather-offload of next-frontier columns (Xeon SPR only, not the
  185H): research-flavored, pairs naturally with Idea 1's column grouping;
  park until a Xeon campaign.

## 2026-06-09 pre-screen results (microbench + analytic models, ~5 min)

1. **Idea 4 (AMAC lanes): pre-judged FAIL.** Microbench
   `benchmarks/src/microbenchmarks/amac_lanes.cc` (1 GB dataset, R=128
   tasks, taskset to CPU 0): seq_m4 = 10.7 ns/load; lanes 2/4/8 = 0.87×/0.74×/0.66×
   (regressions). Root cause: adjacent tasks are already independent in
   program order, so the OOO core extracts cross-task MLP by itself
   (~7–8 misses in flight at 80 ns DRAM latency). Cross-check against real
   data: depth-18 AP = 4.53 s / (2.58M samples × 192 loads) ≈ **9.1 ns/load
   — the production kernel is already MLP-saturated.** Consequence for all
   ideas: latency-hiding is exhausted; only ideas that **reduce loads or
   shrink the footprint** can win. (Same conclusion family as Phase H.)
2. **Idea 2 (gather cache): model says 15.0% AP at R≤4096 threshold, 19.2%
   at R≤16384** (hit cost = 20% of miss, distinct cols/node ≈ 187).
   Borderline vs the 20% gate standalone — the biased-sampling variant or a
   higher threshold is needed to clear it. Cheap next step unchanged:
   instrument a hit-rate counter before building the cache.
3. **Idea 1 (tree-lockstep): strongly validated.** Union-density model from
   the measured depth profile: per-column touched-row density across the
   entire hot band (depths 9–21) is **≥0.98 at T=100 and ≥0.70 at T=30**.
   Column sweeps go fully dense at production forest sizes — input traffic
   for the whole forest collapses to ~one sequential dataset read per depth.
   Lockstep is now the top-ranked research idea; AP becomes output-bound.

## Recommended order

(Updated after the 2026-06-09 pre-screen.)

| # | Idea | Scope gain (AP) predicted | Accuracy risk | Effort | Decision point |
|---|---|---|---|---|---|
| 1 | Tree-lockstep (Idea 1) | multi-× at T≥100 (density model ≥0.98) | none | major | bag-replay simulator → prototype on BFS branch |
| 2 | Column pools (Idea 3) | 30–60% (load-count reduction — the surviving lever) | moderate | accuracy A/B first | kill fast if Δacc > gate |
| 3 | Gather cache (Idea 2) | 15–19% (model) — needs biased variant to clear gate | none unbiased | 1–2 wks | hit-rate counter first (1 day, no kernel) |
| 4 | AP+minmax fusion (Idea 5) | +6% e2e | none | day | just do it |
| 5 | bf16 shadow (Idea 6) | 2× on sweeps | low | days | after 1–2 |
| — | ~~Lane interleaving (Idea 4)~~ | **FAILED pre-screen** (0.66–0.87×; kernel already MLP-saturated at ~9–11 ns/load) | — | — | closed |

Ideas 2+3 compose (within-tree vs cross-tree reuse); 1+5 compose (sweep
bandwidth). The MLP-saturation finding means every surviving idea wins by
issuing *fewer* DRAM loads, which is the correct single axis to optimize.
