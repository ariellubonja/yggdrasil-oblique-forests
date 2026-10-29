import csv, os, sys
S, R = sys.argv[1], sys.argv[2]
sys.path.insert(0, S)
import importlib
m = importlib.import_module("run_replication_" + "sweep")
rows = list(csv.DictReader(open(R + "/results_rawparse.csv")))
out = []
for r in rows:
    acc, auc, ll, (nc, no, nm), train_s, held = m.parse(os.path.join(R, "logs", r["log"]))
    r.update(test_acc=acc, test_auc=auc, test_logloss=ll, num_conditions=nc, num_oblique=no, num_multi_col=nm, train_s=train_s, held_cols=held)
    out.append(r)
with open(R + "/results.csv", "w", newline="") as f:
    w = csv.DictWriter(f, m.FIELDS); w.writeheader(); w.writerows(out)
print(len(out), "rows reparsed; train_s sample:", out[0]["train_s"], out[-1]["train_s"], "auc:", out[0]["test_auc"])
