"""AA GBT on the kdd28_tabarena33 fold CSVs: ordinal codes as NUMERICAL vs the same codes as native CATEGORICAL.
Only the column semantic differs. ydf 0.16.1 defaults, 300 trees, no validation/early stopping, seed 1, 4 threads."""
import json, os, sys, time
import numpy as np, pandas as pd, ydf

HERE = os.path.dirname(os.path.abspath(__file__))
FOLDS = sys.argv[1]  # kdd28_tabarena33 folds (box ~/spo_vs_gbt/folds), sha256 per MANIFEST/inputs.sha256
CATS = json.load(open(os.path.join(HERE, "cat_columns.json")))
DS = ["credit-g", "in_vehicle_coupon_recommendation", "bank-marketing", "Amazon_employee_access"]


def auc(y, p):  # Mann-Whitney with tie averaging
    r = pd.Series(p).rank().values
    n1 = y.sum(); n0 = len(y) - n1
    return (r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)


rows = []
for d in DS:
    for k in range(5):
        tr = pd.read_csv(f"{FOLDS}/{d}/fold{k}_train.csv"); te = pd.read_csv(f"{FOLDS}/{d}/fold{k}_test.csv")
        feats = [c for c in tr.columns if c != "label"]
        for arm in ("ordinal", "native"):
            a, b = tr.copy(), te.copy()
            if arm == "native":
                for c in CATS[d]:
                    a[c] = a[c].map(lambda v: str(int(v))); b[c] = b[c].map(lambda v: str(int(v)))
            sem = {c: (ydf.Semantic.CATEGORICAL if arm == "native" and c in CATS[d] else ydf.Semantic.NUMERICAL) for c in feats}
            L = ydf.GradientBoostedTreesLearner(
                label="label", task=ydf.Task.CLASSIFICATION, features=[(c, s) for c, s in sem.items()],
                include_all_columns=False, num_trees=300, validation_ratio=0.0, early_stopping="NONE",
                random_seed=1, num_threads=4)
            t = time.time(); m = L.train(a, verbose=0); ts = time.time() - t
            got = {f.name: f.semantic for f in m.input_features()}
            assert got == sem, (d, arm, {c: (got.get(c), s) for c, s in sem.items() if got.get(c) != s})
            p = m.predict(b); y = b["label"].values.astype(int)
            pos = m.label_classes()[-1]
            p1 = p if pos == "1" else 1 - p
            rows.append(dict(dataset=d, fold=k, arm=arm, test_auc=auc(y, p1), test_acc=((p1 > .5) == y).mean(),
                             num_trees=m.num_trees(), train_s=round(ts, 2)))
            print(rows[-1], flush=True)
pd.DataFrame(rows).to_csv(os.path.join(HERE, "results.csv"), index=False)
