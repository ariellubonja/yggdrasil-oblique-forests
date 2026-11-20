# sec:accuracy-spo-advantage inputs: datasets of the SPAA paper eval + wide dataset eval where single-machine SPO-GBT beats AA-GBT
# by more than EDGE test AUC (fold means). Writes spo_advantage_datasets.txt, spo_advantage_shapes.csv and <table>/results.csv. Run from repo root.
import csv, os, statistics as st
EDGE = 0.005
H = os.path.dirname(os.path.abspath(__file__)); A = os.path.dirname(H)
SRC = {"distacc": ("tab-dist-accuracy-ydf/distacc/results.csv", "tab-dist-accuracy-ydf-wide/wide_eval/distacc/results.csv"),
       "distacc_aaspo": ("tab-dist-accuracy-ydf/distacc_aaspo/distacc_aaspo.csv", "tab-dist-accuracy-ydf-wide/wide_eval/distacc_aaspo/results.csv"),
       "distacc_exact300": ("tab-dist-accuracy-ydf/distacc_aaspo/distacc_exact300.csv", "tab-dist-accuracy-ydf-wide/wide_eval/distacc_exact300/results.csv"),
       "colrep_distacc": ("tab-column-replication-ydf/colrep_distacc/results.csv", "tab-dist-accuracy-ydf-wide/wide_eval/colrep_distacc/results.csv")}
read = lambda p: list(csv.DictReader(open(os.path.join(A, p), newline="")))
acc = {}  # (dataset, arm) -> {fold: auc}, last row wins as in make_paper_tables
for p in SRC["distacc"]:
    for r in read(p):
        if r["rc"] == "0" and r["arm"] in ("single:SPO", "single:AA"):
            acc.setdefault((r["dataset"], r["arm"]), {})[r["fold"]] = float(r["test_auc"])
names = sorted({d for d, _ in acc}, key=str.lower)
edge = {d: st.mean(acc[d, "single:SPO"].values()) - st.mean(acc[d, "single:AA"].values()) for d in names}
keep = [d for d in names if edge[d] > EDGE]
open(os.path.join(H, "spo_advantage_datasets.txt"), "w").write("".join(f"{d}\t{edge[d]:+.4f}\n" for d in keep))
print(len(keep), "of", len(names), "datasets:", ", ".join(keep))
for t, parts in SRC.items():
    rows = [read(p) for p in parts]
    heads = [list(r[0].keys()) for r in rows]
    head = heads[1] + [c for c in heads[0] if c not in heads[1]]
    os.makedirs(os.path.join(H, t), exist_ok=True)
    with open(os.path.join(H, t, "results.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=head, restval=""); w.writeheader()
        for rs in rows:
            w.writerows(r for r in rs if r["dataset"] in keep)
shapes = {}
for p in ("../../runtime/speedup_map_by_dataset/analysis/per_dataset_summary.csv", "tab-dist-accuracy-ydf-wide/wide_shapes.csv"):
    for r in read(p):
        shapes.setdefault(r["dataset"], (int(float(r["rows"])), int(float(r["features"]))))
with open(os.path.join(H, "spo_advantage_shapes.csv"), "w", newline="") as f:
    w = csv.writer(f); w.writerow(["dataset", "rows", "features"]); w.writerows((d, *shapes[d]) for d in keep)
