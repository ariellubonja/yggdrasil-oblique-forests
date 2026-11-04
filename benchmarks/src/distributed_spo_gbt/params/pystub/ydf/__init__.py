"""Test-only stub of ydf: records learner kwargs and the extra training config."""
from types import SimpleNamespace

from argvrec import record

__version__ = "stub"
Semantic = SimpleNamespace(NUMERICAL="NUMERICAL", CATEGORICAL="CATEGORICAL")
util = SimpleNamespace(get_vertex_ai_cluster_spec=lambda: SimpleNamespace(
    is_worker=False, workers=["w0", "w1", "w2", "w3"], port=0))
DEFAULTS = {"min_examples": 5, "max_depth": 6, "num_trees": 300, "random_seed": 123456}


def verbose(*a):
    pass


def start_worker(port):
    raise SystemExit(0)


def create_vertical_dataset(path, columns=None):
    return SimpleNamespace(data_spec=lambda: "spec")


class _Model:
    def __init__(self, n):
        self.n = n

    def num_trees(self):
        return self.n

    def evaluate(self, path, **kw):
        return SimpleNamespace(accuracy=0.5, loss=0.7, characteristics=[SimpleNamespace(roc_auc=0.5)])

    def save(self, p):
        pass

    def describe(self, fmt):
        return ""


class _Learner:
    def __init__(self, **kw):
        self.kw = kw
        tc = kw.get("extra_training_config")
        rec = {k: (len(v) if k == "workers" else v) for k, v in kw.items() if k not in ("extra_training_config", "data_spec")}
        record(type(self).__name__, {"kwargs": rec, "extra_training_config": tc.dump() if tc is not None else None})

    @property
    def hyperparameters(self):
        return {**DEFAULTS, **{k: v for k, v in self.kw.items() if k not in ("workers", "extra_training_config", "data_spec")}}

    def train(self, path, valid=None):
        return _Model(self.kw.get("num_trees", 300))


class DistributedGradientBoostedTreesLearner(_Learner):
    pass


class GradientBoostedTreesLearner(_Learner):
    pass
