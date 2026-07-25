# Method A on HIGGS — DW1 oblique gather per-depth line efficiency

**Question (from HANDOFF_cachegrind.md):** does the ~1.9 useful-floats/64B-line
plateau seen on the toy hold on HIGGS, or sink toward the 1.0 worst case?

**Answer: neither.** The "~1.9 plateau" was a truncation artifact of the toy tree
only reaching L7. On HIGGS the curve keeps declining past L7 and settles into a
broad, slow decline around **~1.5** through the bulk-work depths — well above the
1.0 uniform-random floor.

## Method A
Exact, native, multithreaded distinct-cache-line counter. At the DW1 kernel call
site (`training.cc`, guarded by `#ifdef LINECOUNT_A`), per depth and over every
node processed by `ApplyProjectionsDepthwise1Pass`, accumulate:
- `rows += sel.size()` (gather iterations / useful floats)
- `lines += 1 + #{(sel[i]>>4) != (sel[i-1]>>4)}` (distinct 64B lines; `sel` is
  sorted ascending — DCHECK in SplitExamplesInPlace — so one pass is exact)

`useful/line = rows/lines`. This is the exact geometric metric the cachegrind (B)
and callgrind (C) methods estimate from D1 miss counts. Output: per-depth CSV to
`$LINECOUNT_OUT`, dumped at process exit.

## Run
```
/tmp/binA --input_mode=csv --train_csv=benchmarks/data/HIGGS_with_header.csv \
  --label_col=class --num_trees=16 --tree_depth=-1 --num_threads=48 \
  --feature_split_type=Oblique --max_num_projections=200 --growing_strategy=Local
```
Full HIGGS (11M x 28), 16 bootstrap trees, depth -> 72. Wall 2m38s, peak RSS 21.9 GB.
Build: `--config=depthwise_1_pass --copt=-march=native` + `--per_file_copt=...training.cc@-DLINECOUNT_A`.

## Findings
Kernel level = tree depth - 1 (so HIGGS level k aligns to toy level k).

| level | HIGGS u/l | toy u/l |
|---:|---:|---:|
| 1 | 8.84 | 8.19 |
| 4 | 3.06 | 2.55 |
| 7 | 2.02 | 1.91 |
| 8 | 1.87 | (toy stops at 7) |
| 15 | 1.60 | |
| 24 (d25) | 1.57 | |
| 29 (d30) | 1.55 | |
| 34 (d35) | 1.53 | |
| ~56 (d57) | 1.46 | |

- Shallow depths (L1-7) track the toy closely (validates Method A: L6 = 1.95 toy
  vs 2.24 HIGGS, L7 = 1.91 toy vs 2.02 HIGGS — same shape, HIGGS marginally higher).
- HIGGS crosses below 1.9 at L8 and keeps sinking — there is **no 1.9 plateau**.
- It does **not** collapse to 1.0 either: it stabilizes ~1.5-1.6 across the
  bulk-work band and drifts to ~1.45 in the sparse deep tail (the lone 1.0 at
  depth 72 is 2 nodes / 4 rows of noise).

**Gather-work distribution (work ∝ rows):**

| depth band | % of gather work | work-wtd u/l |
|---|---:|---:|
| 2-7   | 21.1% | 3.52 |
| 8-15  | 28.1% | 1.74 |
| 16-24 | 29.6% | 1.59 |
| 25-35 | 18.6% | 1.56 |
| 36-72 |  2.6% | 1.51 |

Global work-weighted `sum(rows)/sum(lines) = 1.83` (shallow depths dominate the
total). 79% of gather work is at depth <= 24; 97.5% by depth 35.

**Takeaway:** at HIGGS production depths the gather salvages ~1.5 of 16 floats per
line (~9% line efficiency) — ~10x wasted bandwidth, ~50% better than uniform-random.
The bottleneck is real and depth-stable; the access-order fix (row-major layout /
gather-once-reuse) remains motivated.

## Artifacts
- `/tmp/methodA_full.csv` — per-depth raw (depth,nodes,rows,lines,useful_per_line)
- `/tmp/methodA_higgs_curve.png` — curve vs toy
- `/tmp/analyzeA.py` — analysis/plot script
- Binary: `/tmp/binA`; instrumentation in `training.cc` under `LINECOUNT_A`.

## Method C status
Launched on full HIGGS under callgrind (`/tmp/binC`, `/tmp/cgC/`) but the 8 GB CSV
parse runs at ~1.4k rows/s under instrumentation (~2 h just to load), and a single
HIGGS column (44 MB) fits the 109 MB LLC so C cannot surface the DRAM-miss cost
that motivated it. Left running best-effort for a shallow-depth cross-check only;
not on the critical path. PID in `/tmp/cgC/pid`.
