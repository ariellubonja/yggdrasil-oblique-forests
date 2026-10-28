#!/usr/bin/env python3
"""Body-only LaTeX table: per-depth feature choice of AA-GBT on CC18 at the 100th/200th/300th tree.
Reads tree_slices_pooled.csv + per_model_tree_slices.csv (medians over datasets), writes
tables/table_cc18_feature_iter_body.tex. Caption and label live in main.tex."""
import csv, statistics, sys
from decimal import Decimal, ROUND_HALF_UP
from pathlib import Path

root = Path(__file__).resolve().parents[2] / "results" / "misc" / "cc18_gbt_feature_study"
pooled = Path(sys.argv[1]) if len(sys.argv) > 1 else root / "tree_slices_pooled.csv"
per_model = Path(sys.argv[2]) if len(sys.argv) > 2 else root / "per_model_tree_slices.csv"
out = Path(sys.argv[3]) if len(sys.argv) > 3 else Path("tables/table_cc18_feature_iter_body.tex")
TREES = ["100", "200", "300"]

def f(x, nd=2, pct=False):
    if x in ("", None):
        return "--"
    v = Decimal(str(100 * float(x) if pct else float(x)))
    return str(v.quantize(Decimal(1).scaleb(-nd), rounding=ROUND_HALF_UP))  # half-up, not banker's

# Pooled per-depth medians, keyed (tree_k, depth) -> row dict.
pd_rows = {(r["tree_k"], r["depth"]): r for r in csv.DictReader(pooled.open())}
depths = sorted({d for _, d in pd_rows})

# Per-tree medians across datasets.
pm = {k: [] for k in TREES}
for r in csv.DictReader(per_model.open()):
    pm[r["tree_k"]].append(r)
def pmed(k, col):
    return statistics.median(float(r[col]) for r in pm[k])

lines = [r"\begin{tabular}{llrrr}", r"\toprule",
         r" & & \multicolumn{3}{c}{Tree} \\", r"\cmidrule(lr){3-5}",
         r" & Depth & 100 & 200 & 300 \\", r"\midrule"]
groups = [("Top-1 \\%", "top1_share", 0, True, depths),
          ("$\\hat{H}$", "norm_entropy", 2, False, depths),
          ("Reuse \\%", "reuse_rate", 0, True, [d for d in depths if d != "0"]),
          ("Sibl. same \\%", "sibling_same_feature", 0, True, [d for d in depths if d != "0"])]
for gi, (name, col, nd, pct, ds) in enumerate(groups):
    if gi:
        lines.append(r"\midrule")
    for i, d in enumerate(ds):
        lab = name if i == 0 else ""
        lines.append(" & ".join([lab, d] + [f(pd_rows[(k, d)][col], nd, pct) for k in TREES]) + r" \\")
lines.append(r"\midrule")
per_tree = [("Internal nodes / tree", "internal_nodes_per_tree", 1, False),
            ("Distinct feats. / tree", "distinct_features_per_tree", 1, False),
            ("Distinct roots / 10 folds", "n_distinct_roots", 0, False),
            ("Root top-1 \\%", "root_top1_share", 0, True),
            ("Reuse \\% (all depths)", "reuse_rate", 0, True)]
for name, col, nd, pct in per_tree:
    lines.append(" & ".join([r"\multicolumn{2}{l}{" + name + "}"] + [f(pmed(k, col), nd, pct) for k in TREES]) + r" \\")
lines += [r"\bottomrule", r"\end{tabular}"]
out.parent.mkdir(parents=True, exist_ok=True)
out.write_text("\n".join(lines) + "\n")
print(out)
