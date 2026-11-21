# dataset,rows,features of the F >= 100 sets: SPAA paper eval shapes (make_paper_tables.SHAPES_CSV) for its 6 wide sets,
# kdd28_wide_datasets_manifest.json (whole dataset: train+test rows, features after drop_columns) for the rest.
import csv, json, os, sys
R = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), *[".."] * 6))
spaa = {"APSFailure", "Bioresponse", "ecom-offers", "homecredit-default", "homesite-insurance", "kddcup09_appetency"}
out, seen = [], set()
for r in csv.DictReader(open(os.path.join(R, "benchmarks/results/runtime/speedup_map_by_dataset/analysis/per_dataset_summary.csv"))):
    if r["dataset"] in spaa and r["dataset"] not in seen:
        seen.add(r["dataset"]); out.append((r["dataset"], int(float(r["rows"])), int(float(r["features"]))))
for e in json.load(open(os.path.join(R, "benchmarks/data/kdd28_wide_datasets_manifest.json")))["datasets"]:
    if e["status"] == "built" and e["name"] not in seen:
        n = e["expected_rows"]
        out.append((e["name"], sum(n.values()) if isinstance(n, dict) else n, e["expected_features"]))
w = csv.writer(open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "derived", "wide_shapes.csv"), "w", newline=""))
w.writerow(["dataset", "rows", "features"]); w.writerows(sorted(out, key=lambda t: t[0].lower()))
print(len(out), "datasets")
