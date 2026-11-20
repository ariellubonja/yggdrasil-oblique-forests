# Keep rows of datasets with more than 100 features (shapes from make_paper_tables.SHAPES_CSV).
import csv, sys
shapes, src, dst = sys.argv[1:4]
wide = {r["dataset"] for r in csv.DictReader(open(shapes)) if int(float(r["features"])) > 100}
rows = list(csv.DictReader(open(src, newline="")))
with open(dst, "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=rows[0].keys()); w.writeheader()
    w.writerows(r for r in rows if r["dataset"] in wide)
