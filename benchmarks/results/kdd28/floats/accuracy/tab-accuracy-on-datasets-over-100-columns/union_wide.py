# Union inputs of sec:accuracy-on-datasets-over-100-columns: the F>100 rows of the SPAA paper eval (derived/, cut by filter_wide.py) + the wide dataset eval.
# Header = union of both (columns a CSV lacks stay empty). Run from the repo root, writes derived/union/<table>/results.csv.
import csv, os
D = os.path.dirname(os.path.abspath(__file__))
W = "../../../runs/2026-11-04_wide_eval"
SRC = {"distacc": ("derived/distacc/results.csv", f"{W}/distacc/results.csv"),
       "distacc_aaspo": ("derived/distacc_aaspo/distacc_aaspo.csv", f"{W}/distacc_aaspo/results.csv"),
       "distacc_exact300": ("derived/distacc_aaspo/distacc_exact300.csv", f"{W}/distacc_exact300/results.csv"),
       "colrep_distacc": ("derived/colrep_distacc/results.csv", f"{W}/colrep_distacc/results.csv")}
for t, (a, b) in SRC.items():
    parts = [list(csv.DictReader(open(os.path.join(D, p), newline=""))) for p in (a, b)]
    heads = [next(csv.reader(open(os.path.join(D, p)))) for p in (a, b)]
    head = heads[1] + [c for c in heads[0] if c not in heads[1]]
    ds = [{r["dataset"] for r in rows} for rows in parts]
    assert not ds[0] & ds[1], (t, ds[0] & ds[1])
    os.makedirs(os.path.join(D, "derived", "union", t), exist_ok=True)
    with open(os.path.join(D, "derived", "union", t, "results.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=head, restval="")
        w.writeheader()
        for rows in parts:
            w.writerows(rows)
    print(t, len(parts[0]), "+", len(parts[1]), "rows,", len(ds[0] | ds[1]), "datasets")
