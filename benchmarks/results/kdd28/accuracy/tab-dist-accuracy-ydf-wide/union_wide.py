# Union inputs of sec:accuracy-wide: the F>100 rows of the SPAA paper eval (filter_wide.py outputs) + the wide dataset eval.
# Header = union of both (columns a CSV lacks stay empty). Run from the repo root, writes union/<table>/results.csv.
import csv, os
D = os.path.dirname(os.path.abspath(__file__))
SRC = {"distacc": ("distacc/results.csv", "wide_eval/distacc/results.csv"),
       "distacc_aaspo": ("distacc_aaspo/distacc_aaspo.csv", "wide_eval/distacc_aaspo/results.csv"),
       "distacc_exact300": ("distacc_aaspo/distacc_exact300.csv", "wide_eval/distacc_exact300/results.csv"),
       "colrep_distacc": ("colrep_distacc/results.csv", "wide_eval/colrep_distacc/results.csv")}
for t, (a, b) in SRC.items():
    parts = [list(csv.DictReader(open(os.path.join(D, p), newline=""))) for p in (a, b)]
    heads = [next(csv.reader(open(os.path.join(D, p)))) for p in (a, b)]
    head = heads[1] + [c for c in heads[0] if c not in heads[1]]
    ds = [{r["dataset"] for r in rows} for rows in parts]
    assert not ds[0] & ds[1], (t, ds[0] & ds[1])
    os.makedirs(os.path.join(D, "union", t), exist_ok=True)
    with open(os.path.join(D, "union", t, "results.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=head, restval="")
        w.writeheader()
        for rows in parts:
            w.writerows(rows)
    print(t, len(parts[0]), "+", len(parts[1]), "rows,", len(ds[0] | ds[1]), "datasets")
