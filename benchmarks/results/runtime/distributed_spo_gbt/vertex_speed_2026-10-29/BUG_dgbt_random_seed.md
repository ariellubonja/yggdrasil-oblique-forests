# Bug: DistributedGradientBoostedTreesLearner drops generic hyperparameters (random_seed, ...)

## Symptom
`ydf.DistributedGradientBoostedTreesLearner(random_seed=S, ...)` (Python API) always trains with the proto
default `TrainingConfig.random_seed = 123456`, whatever S is. Found 2026-10-29: a Vertex sweep with seeds 1-5
gave byte-identical metrics on every one of 374 (dataset, level) cells (run `colrep_full20261029`, cells in
`gs://ariel_ydf_bucket/runs/colrep_full20261029/s{1..5}/`, local copy
`ariel_notes/results/2026-10-28_column_replication/vertex_hm4_seed123456x5/results.csv`).
Not affected: the C++ driver `examples/train_distributed_gbt --seed`, which sets `TrainingConfig.random_seed`
directly instead of going through generic hyperparameters.

## Root cause (fork `main`; present in upstream google/yggdrasil-decision-forests too)
`DistributedGradientBoostedTreesLearner::SetHyperParametersImpl`
(`yggdrasil_decision_forests/learner/distributed_gradient_boosted_trees/distributed_gradient_boosted_trees.cc`,
~line 104) builds a temporary `gradient_boosted_trees::GradientBoostedTreesLearner gbt_learner(training_config_)`,
calls `gbt_learner.SetHyperParametersImpl(...)`, and copies back only the `gradient_boosted_trees_config`
extension. `GradientBoostedTreesLearner::SetHyperParametersImpl` calls `AbstractLearner::SetHyperParametersImpl`
(`learner/abstract_learner.cc` ~758-800), which writes four generic fields into the TEMPORARY learner's
`training_config_`:
- `random_seed`
- `maximum_training_duration_seconds`
- `maximum_model_size_in_memory_in_bytes`
- `pure_serving_model`

All four are lost. The Python API passes every constructor kwarg as a generic hyperparameter
(`port/python/ydf/learner/generic_learner.py _get_learner` -> `learner.cc GetLearner` -> `SetHyperParameters`),
so it hits this path. `extra_training_config` is merged after the hyperparameters (`learner.cc GetLearner`),
which is why setting `random_seed` there works as a workaround (used in `vertex_build/train_cells.py`).

## Replicate
1. C++ unit test, no workers needed: construct `DistributedGradientBoostedTreesLearner`, call
   `SetHyperParameters` with a `GenericHyperParameters` that sets `random_seed = 7`
   (and the other three fields), and assert that `training_config()` carries the values. It fails today.
   Note: upstream's `distributed_gradient_boosted_trees_test` target does not compile with OSS gtest, so a
   small new test target may be needed. Build on the EC2 box (see the memory note `ec2-3-92-30-82-colship-box`
   for the icx + boringssl flag); the Mac has DGBT build gotchas (`macos-build-gotchas-dgbt`).
2. End-to-end (optional): train twice through the Python API with 2 local workers (`ydf.start_worker` on two
   ports, `workers=["localhost:p1", "localhost:p2"]`), `split_axis="SPARSE_OBLIQUE"` (random projections)
   and `random_seed=1` vs `2`. The predictions are identical today and differ after the fix.

## Fix
A candidate already exists: commit 713c70f7 on branch `column-replication` copies the four fields back after
the temporary-learner call. It compiles but is untested.
A structural fix is better, so any future generic field cannot be lost the same way: call
`AbstractLearner::SetHyperParametersImpl` on the distributed learner itself, or copy every non-extension
field back. Check that the consumer tolerates a second `Get` of the same parameter.

## Deliverables
- Failing-then-passing unit test plus the fix, on a branch off the fork's `main`.
- A draft upstream PR for google/yggdrasil-decision-forests. **No Co-Authored-By / Claude trailer** (it breaks the
  CLA). Do not open the PR without the user's OK.
- Work in your own git worktree. Never use the main checkout `/Users/ariellubonja/prog/randals_lab/oblique_forests/yggdrasil-decision-forests`:
  other agents use it.
