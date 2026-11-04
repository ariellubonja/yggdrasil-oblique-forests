# tab:in-node-dist-vqsort-vertex

Results behind the KDD28 float(s) below. Rebuild the body with the generator command. Index and rules:
`benchmarks/results/kdd28/README.md`.

- **Labels:** `tab:in-node-dist-vqsort-vertex`
  - `tab:in-node-dist-vqsort-vertex`: "Fixing limited SPO in-node parallelism: training time in seconds of SPO-GBT with $T$ threads."
- **Status:** appendix. The label says vertex but the data is the m7i single machine
- **Profile:** `in_node_threads` (`benchmarks/src/distributed_spo_gbt/params/kdd28_params.json`)
- **Settings:** effective values from the profile (`params.py show <arm>`), checked against the CSV stamp columns
  by the generator where the CSV has them
  - `in_node_threads`: trees=10, max_depth=6, min_examples=5, split_type=EXACT, spo_exponent=0.5, spo_max_projections=1000, spo_density=1.5, spo_min_projections=1, folds=shipped test sets, seeds=1, profile=in_node_threads
- **Eval suite:** none (runtime float, shipped test sets)
- **Machine:** EC2 m7i.metal-24xl, single machine thread sweep. SMT/turbo: unknown, not recorded
- **Date:** 2026-10-22
- **Code:** sequential = rebased-main 23053344, parallel = in-node-parallelism-SPOGBT cee3b4c8 (CSV header says c5dd0196 code)
- **Generator:** run from the repo root, writes into the paper dir (`--out` to redirect)

```
python3 benchmarks/src/distributed_spo_gbt/make_in_node_tables.py
```
- **Inputs:** `inputs.txt` (one repo-relative path per line)
  - `benchmarks/results/kdd28/appendix/tab-in-node-dist-vqsort-vertex/in_node_parallelism_2026-10-22/sm_thread_sweep_old_vs_new.csv`
  - `benchmarks/results/kdd28/appendix/tab-in-node-dist-vqsort-vertex/in_node_parallelism_2026-10-22/susy_and_distributed_old_vs_new.txt`
- **Raw cells:** none kept beyond these files (no run script, reconstructed CSV: profile note)

Settings rest on the CSV header comment only (P = ceil(sqrt F), the regime where in-node parallelism was dropped). Regenerated 2026-11-04: equal to the Overleaf copy in every non-`%` line.
