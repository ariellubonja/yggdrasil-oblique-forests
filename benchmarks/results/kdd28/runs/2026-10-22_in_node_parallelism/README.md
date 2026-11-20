# runs/2026-10-22_in_node_parallelism

Measurement, stored once. Floats read it through their `inputs.txt`. Index and rules: `benchmarks/results/kdd28/README.md`.

- **Files:** `sm_thread_sweep_old_vs_new.csv`, `susy_and_distributed_old_vs_new.txt` (read by the generator),
  `old_build_rerun_summary.txt`, `old_build_thread_scaling_explained.md`
- **Profile:** `in_node_threads` (`benchmarks/src/distributed_spo_gbt/params/kdd28_params.json`)
- **Settings:** trees=10, max_depth=6, min_examples=5, split_type=EXACT, spo_exponent=0.5, spo_max_projections=1000, spo_density=1.5, spo_min_projections=1, folds=shipped test sets, seeds=1, profile=in_node_threads. Settings rest on the CSV header comment only (P = ceil(sqrt F))
- **Eval suite:** none (runtime, shipped test sets)
- **Machine:** EC2 m7i.metal-24xl, single machine thread sweep. SMT/turbo: unknown, not recorded
- **Date:** 2026-10-22
- **Code:** sequential = rebased-main 23053344, parallel = in-node-parallelism-SPOGBT cee3b4c8 (CSV header says c5dd0196 code)
- **Raw cells:** none kept beyond these files (no run script, reconstructed CSV: profile note)
- **Read by:** tab:in-node-dist-vqsort-vertex
