# runs/2026-10-30_dist_accuracy_aa

Measurement, stored once. Floats read it through their `inputs.txt`. Index and rules: `benchmarks/results/kdd28/README.md`.

- **Files:** `results.csv` (1836 rows, 33 datasets), `PROVENANCE.txt`. `build_paper_inputs.py` keeps its rc = 0 distributed AA
  rows. Pre-7c0eebd3, AA is unaffected by that fix (AA is CPU-independent: 765/765 = paper cells)
- **Profile:** `table8.dist_aa` (`benchmarks/src/distributed_spo_gbt/params/kdd28_params.json`)
- **Settings:** effective values (`params.py show <arm>`)
  - `table8.dist_aa`: trees=300, max_depth=6, min_examples=5, split_type=EXACT, force_numerical=false, max_unique_discretized=1, workers=1 4 8 16 32, folds=5 CV folds (chronological sets: 1 holdout), seeds=123456, profile=table8.dist_aa
- **Eval suite:** `kdd28_tabarena33` folds (unstamped CSV: no eval columns)
- **Machine:** EC2 m7i.metal-24xl (Xeon 8488C), in-process. SMT/turbo: unknown, not recorded in the CSV
- **Date:** 2026-10-30
- **Code:** `train_distributed_gbt` 8ddb03f7 (`PROVENANCE.txt`)
- **Raw cells:** this file
- **Read by:** tab:dist-accuracy, fig:dist-accuracy-mean, through `outdated/tab-dist-accuracy/derived/table8_results.csv`
