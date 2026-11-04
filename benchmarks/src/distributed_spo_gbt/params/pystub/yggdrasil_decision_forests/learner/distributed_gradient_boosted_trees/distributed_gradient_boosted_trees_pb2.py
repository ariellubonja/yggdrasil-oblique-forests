"""Test-only proto stub (wheel defaults: split budget on, local count off)."""
from types import SimpleNamespace

distributed_gradient_boosted_trees_config = "dgbt"


def DistributedGradientBoostedTreesTrainingConfig():
    return SimpleNamespace(in_worker_oblique=SimpleNamespace(split_projections_across_workers=True,
                                                             num_projections_from_local_features=False))
