# sec:accuracy-where-spo-outperforms-aa inputs: datasets of the SPAA paper eval + wide dataset eval where single-machine SPO-GBT beats AA-GBT
# by more than EDGE test AUC (fold means). Writes derived/{spo_advantage_datasets.txt,spo_advantage_shapes.csv,<table>/results.csv}. Run from repo root.
import csv, os, statistics as st
EDGE = 0.005
H = os.path.dirname(os.path.abspath(__file__)); A = os.path.dirname(H)
R, W = "../../runs", "../../runs/2026-11-04_wide_eval"
SRC = {"distacc": (f"{R}/2026-11-03_distacc/results.csv", f"{W}/distacc/results.csv"),
       "distacc_aaspo": (f"{R}/2026-11-04_distacc_aaspo/distacc_aaspo.csv", f"{W}/distacc_aaspo/results.csv"),
       "distacc_exact300": (f"{R}/2026-11-04_distacc_aaspo/distacc_exact300.csv", f"{W}/distacc_exact300/results.csv"),
       "colrep_distacc": (f"{R}/2026-11-04_colrep_distacc/results.csv", f"{W}/colrep_distacc/results.csv")}
read = lambda p: list(csv.DictReader(open(os.path.join(A, p), newline="")))
acc = {}  # (dataset, arm) -> {fold: auc}, last row wins as in make_paper_tables
for p in SRC["distacc"]:
    for r in read(p):
        if r["rc"] == "0" and r["arm"] in ("single:SPO", "single:AA"):
            acc.setdefault((r["dataset"], r["arm"]), {})[r["fold"]] = float(r["test_auc"])
names = sorted({d for d, _ in acc}, key=str.lower)
edge = {d: st.mean(acc[d, "single:SPO"].values()) - st.mean(acc[d, "single:AA"].values()) for d in names}
keep = [d for d in names if edge[d] > EDGE]
open(os.path.join(H, "derived", "spo_advantage_datasets.txt"), "w").write("".join(f"{d}\t{edge[d]:+.4f}\n" for d in keep))
print(len(keep), "of", len(names), "datasets:", ", ".join(keep))
for t, parts in SRC.items():
    rows = [read(p) for p in parts]
    heads = [list(r[0].keys()) for r in rows]
    head = heads[1] + [c for c in heads[0] if c not in heads[1]]
    os.makedirs(os.path.join(H, "derived", t), exist_ok=True)
    with open(os.path.join(H, "derived", t, "results.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=head, restval=""); w.writeheader()
        for rs in rows:
            w.writerows(r for r in rs if r["dataset"] in keep)
shapes = {}
for p in ("../../../runtime/speedup_map_by_dataset/analysis/per_dataset_summary.csv", "tab-accuracy-hao-vs-ydf-params-single-machine/derived/wide_shapes.csv"):
    for r in read(p):
        shapes.setdefault(r["dataset"], (int(float(r["rows"])), int(float(r["features"]))))
with open(os.path.join(H, "derived", "spo_advantage_shapes.csv"), "w", newline="") as f:
    w = csv.writer(f); w.writerow(["dataset", "rows", "features"]); w.writerows((d, *shapes[d]) for d in keep)
