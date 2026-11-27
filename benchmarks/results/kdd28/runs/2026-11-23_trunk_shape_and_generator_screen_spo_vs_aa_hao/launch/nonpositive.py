# Writes seed1_nonpositive.txt: trunks whose seed-1 mean SPO - AA test AUC over 5 folds is <= 0 (rc = 0 cells only).
import csv, collections, statistics as st, sys
B = sys.argv[1]; v = collections.defaultdict(dict)
for prof, arm in (("kdd28_hao", "single:SPO"), ("kdd28_ydf", "single:AA")):
    for r in csv.DictReader(open(f"{B}/cells/{prof}/seed1/distacc_stamped/results.csv")):
        if r["rc"] == "0": v[r["dataset"]].setdefault(r["fold"], {})[arm] = float(r["test_auc"])
e = {d: st.mean(x["single:SPO"] - x["single:AA"] for x in f.values()) for d, f in v.items() if len(f) == 5 and all(len(x) == 2 for x in f.values())}
neg = sorted(d for d, x in e.items() if x <= 0)
open(f"{B}/seed1_nonpositive.txt", "w").write("\n".join(neg) + "\n")
print(len(e), "trunks with full seed 1,", len(neg), "non-positive:", " ".join(neg))
