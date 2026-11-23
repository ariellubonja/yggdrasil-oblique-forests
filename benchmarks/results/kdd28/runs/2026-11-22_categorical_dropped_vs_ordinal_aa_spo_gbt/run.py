"""Single-machine GBT on the kdd28_tabarena33 folds: categoricals dropped vs ordinal, for AA and SPO.
Extends runs/2026-11-19_ordinal_vs_native (same folds, ydf 0.16.1, 300 trees, no validation/early stopping, seed 1, 4 threads).
SPO = SPARSE_OBLIQUE, BINARY weights, other SPO params at proto defaults. Zero features left -> constant predictor, AUC .5."""
import json, os, sys, time
import numpy as np, pandas as pd, ydf

HERE = os.path.dirname(os.path.abspath(__file__))
FOLDS = sys.argv[1]  # kdd28_tabarena33 folds, sha256 per speedup_map_by_dataset/MANIFEST/inputs.sha256
CATS = json.load(open(os.path.join(HERE, "../2026-11-19_ordinal_vs_native/cat_columns.json")))
DS = ["credit-g", "in_vehicle_coupon_recommendation", "bank-marketing", "Amazon_employee_access"]
ARMS = [("aa", "ordinal"), ("aa", "dropped"), ("spo", "ordinal"), ("spo", "dropped")]


def auc(y, p):  # Mann-Whitney with tie averaging, as in the 2026-11-19 run
    r = pd.Series(p).rank().values
    n1 = y.sum(); n0 = len(y) - n1
    return (r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)


rows = []
for d in DS:
    for k in range(5):
        tr = pd.read_csv(f"{FOLDS}/{d}/fold{k}_train.csv"); te = pd.read_csv(f"{FOLDS}/{d}/fold{k}_test.csv")
        allf = [c for c in tr.columns if c != "label"]
        assert set(CATS[d]) <= set(allf), d
        y = te["label"].values.astype(int)
        for split, cats in ARMS:
            feats = [c for c in allf if not (cats == "dropped" and c in CATS[d])]
            row = dict(dataset=d, fold=k, arm=f"{split}_{cats}", split=split, categoricals=cats, num_features=len(feats))
            if not feats:  # nothing to split on: every test row gets the same score
                rows.append({**row, "test_auc": auc(y, np.zeros(len(y))), "test_acc": float((y == int(tr.label.mean() > .5)).mean()),
                             "num_trees": 0, "train_s": 0.0, "note": "no_features_constant_predictor"})
                print(rows[-1], flush=True); continue
            kw = dict(split_axis="SPARSE_OBLIQUE", sparse_oblique_weights="BINARY") if split == "spo" else {}
            L = ydf.GradientBoostedTreesLearner(
                label="label", task=ydf.Task.CLASSIFICATION, features=[(c, ydf.Semantic.NUMERICAL) for c in feats],
                include_all_columns=False, num_trees=300, validation_ratio=0.0, early_stopping="NONE",
                random_seed=1, num_threads=4, **kw)
            t = time.time(); m = L.train(tr[feats + ["label"]], verbose=0); ts = time.time() - t
            got = {f.name: f.semantic for f in m.input_features()}
            assert got == {c: ydf.Semantic.NUMERICAL for c in feats}, (d, row["arm"], got)
            assert m.num_trees() == 300, (d, row["arm"], m.num_trees())
            p = m.predict(te[feats]); p1 = p if m.label_classes()[-1] == "1" else 1 - p
            rows.append({**row, "test_auc": auc(y, p1), "test_acc": ((p1 > .5) == y).mean(), "num_trees": m.num_trees(),
                         "train_s": round(ts, 2), "note": ""})
            print(rows[-1], flush=True)
pd.DataFrame(rows).to_csv(os.path.join(HERE, "results.csv"), index=False)
