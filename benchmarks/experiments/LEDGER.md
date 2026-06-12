# Experiment Ledger — oblique split performance research

**Purpose:** one row per experiment, append-only. This is the first thing
a fresh agent/session reads (after AGENTS.md). Failed experiments are the
most valuable rows — they prevent re-derivation. Keep rows to one line of
idea + one line of outcome; deep narrative goes in the per-topic .md
linked from the row.

Verdicts: ★ win (≥20 % e2e, confirmed) · ✚ stacked/partial win · ~ neutral
/ below gate · ⛔ failed / regression · ☠ invalidated methodology.

| Date | Idea (one line) | Build / flags | Headline result | V | Details |
|---|---|---|---|---|---|
| 2026-04→05 | Projection-matrix control: per-node fused rows×projs kernel | `--config=projection_matrix_control` | ≈1.00× AP vs baseline (laptop, 3m×4096 Exact) | ~ | [apply_projection_experiments.md](apply_projection_experiments.md) |
| 2026-04→05 | V2 (n,i,p)-triplet single-pass AP | (branch V2) | 3.85× SLOWER | ⛔ | same |
| 2026-04→05 | V2-rev2: software prefetch K=16 ahead | (branch) | 1.74× slower than control — prefetch recompute > latency hidden | ⛔ | same |
| 2026-04→05 | **V2-rev3: 4-row blocks, 4 independent accumulators (MLP)** | `--config=depthwise_1_pass` (kernel inside dw1) | −34 % AP, but only −5 % e2e on laptop Exact | ★ | same; Phase I |
| 2026-04→05 | loop_swap: items-outer/rows-inner Evaluate | `--config=projeval_loop_swap` | −2.7 %…+2.4 % — no signal | ~ | same |
| 2026-04→05 | Pregather: random→sequential copy-out buffer | (Phase G) | only −10.6 % — tripled DRAM traffic | ⛔ | [ap_research_ideas_2026-06-09.md](ap_research_ideas_2026-06-09.md) §3 |
| 2026-04→05 | AVX2/512 hardware gather for AP | (Phase H) | no MLP beyond OOO — no gain | ⛔ | same §2 |
| 2026-04-30 | FP16/bf16 column storage (half bandwidth) | — | rejected: approximation, accuracy-changing | ⛔ | [accuracy-changing-ideas.md](accuracy-changing-ideas.md) |
| 2026-05-30 | Symmetric (CatBoost-style) trees: K projections shared per depth, bagwide sweep | `--config=symmetric_depthwise_ap` (+ controls `symmetric_nodewise_control`, `bfs_only`) | ~20 % AP; −16/−29 % e2e trunk shapes; **changes model semantics** | ✚ | [symmetric_trees_status.md](symmetric_trees_status.md) |
| 2026-06-03→10 | Dataset layouts: row-major / flat-col / dual fp32 with per-node row-col dispatch (T=YDF_RM_MAX_ROWS) | `--config=row_major_dataset_layout` (+`--dataset_layout=…`) | dyn dual = best where it fits (1.5m×4096: 241 vs 331 stock; HIGGS 266 vs 282) but **OOMs every ~46 GiB dataset (2 full copies)** | ✚ | [memory_safe_dynamic_2026-06-12.md](memory_safe_dynamic_2026-06-12.md) §Problem; commits 891369ae, bc0fd0c4 |
| 2026-06-10 | Row-vs-col dispatch threshold sweep (T=100…) | env `YDF_RM_MAX_ROWS` | runtime flat in T → threshold is not the active ingredient | ~ | `benchmarks/results/row-vs-col threshold is unimportant.txt` |
| 2026-06-12 | Subtree gather cache: lazily gathered compact feature columns per ≤T-row subtree (memory-safe dyn replacement) | `--config=subtree_gather` (branch `subtree-gather`) | +43 % at 1.5m×4096 — nodes share ~2 % of features with descendants; gather can never amortize. **Why dual store can't be replicated lazily.** | ⛔ | [memory_safe_dynamic_2026-06-12.md](memory_safe_dynamic_2026-06-12.md) |
| 2026-06-12 | **evaluate_4row: V2-rev3 kernel per (node,projection) in stock DFS Evaluate** | `--config=evaluate_4row` (branch `evaluate-4row`) | −8.2 % @4k feats, −7 % @40k, ~0 % ultra-wide; bit-identical, zero memory; 8-row unroll adds nothing | ✚ | same, §Ablation |
| 2026-06-12 | **cache_projection_evaluator: stop rebuilding O(num_features) evaluator tables per node** | `--config=cache_projection_evaluator` (branch `cache-projection-evaluator`) | −6.8 % @4k, −21.7 % @40k, **−63 % @400k feats**; bit-identical; engineering fix (upstream `// TODO: Cache.`) | ★ | same |
| 2026-06-12 | **Both combined — memory-safe leader** | `EXTRA_BAZEL_CONFIGS="--config=evaluate_4row --config=cache_projection_evaluator"` | New best on 6/7 datasets incl. all former OOM rows (3m×4096: 576 vs 688; 30k×400k: 73 vs 152 prior best); dyn dual keeps 1.5m×4096 (241 vs 256) | ★ | same; [LEADERBOARD.md](LEADERBOARD.md) |

## Standing conclusions (read before proposing new AP ideas)

1. **AP is DRAM-latency-bound; MLP (independent accumulators) is the one
   kernel lever that has paid.** Prefetch, HW gathers, loop order: all failed.
2. **Adding memory traffic to "fix" locality loses** (pregather, subtree
   gather). The dual store wins only because its transpose is prepaid once
   per forest — i.e. paid in RAM, not in traffic.
3. **Sparse oblique sampling math** (P=⌈F^0.5⌉, density 1.5): uses ≈
   distinct features per node, ~2 % feature overlap between a node and its
   descendants → no within-node row reuse and no ancestor-descendant
   column reuse to exploit.
4. **Per-node O(num_features) setup costs dominate wide datasets** —
   check for more of these before touching kernels (this was −63 % on
   10k×400k, hiding in `FindObliqueSetup`).
5. **BFS hurts tall-narrow datasets** (HIGGS: BFS col 336 vs DFS 282);
   depthwise/BFS-fused designs must beat that handicap first.
