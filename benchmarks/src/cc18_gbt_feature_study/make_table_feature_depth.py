#!/usr/bin/env python3
"""Body-only LaTeX table: per-depth feature choice of AA-GBT on CC18 (medians over datasets).
Reads per_depth_pooled.csv, writes tables/table_cc18_feature_depth_body.tex. Caption lives in main.tex."""
import csv, sys
from pathlib import Path

src = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).resolve().parents[2] / "results" / "misc" / "cc18_gbt_feature_study" / "per_depth_pooled.csv"
out = Path(sys.argv[2]) if len(sys.argv) > 2 else Path("tables/table_cc18_feature_depth_body.tex")

rows = list(csv.reader(src.open()))
hdr, stat = rows[0], rows[1]
col = {(h, s): i for i, (h, s) in enumerate(zip(hdr, stat))}
med = lambda r, h: r[col[(h, "median")]]

def f(x, nd=2, pct=False):
    if x == "":
        return "--"
    v = float(x)
    return f"{100*v:.{nd}f}" if pct else f"{v:.{nd}f}"

lines = [r"\begin{tabular}{rrrrrrrrr}", r"\toprule",
         r"Depth & Splits & Distinct & Top-1 & Top-3 & $\hat{H}$ & $F_{\mathrm{eff}}$ & Sibl. & Level \\",
         r" & / 300 trees & feats. & \% & \% & & & same \% & same \% \\", r"\midrule"]
for r in rows[3:]:
    d = r[0]
    lines.append(" & ".join([d, f(med(r, "n_nodes"), 0), f(med(r, "n_distinct"), 1),
                             f(med(r, "top1_share"), 0, pct=True), f(med(r, "top3_share"), 0, pct=True),
                             f(med(r, "norm_entropy")), f(med(r, "eff_num_features"), 1),
                             f(med(r, "sibling_same_feature"), 0, pct=True),
                             f(med(r, "within_tree_oblivious"), 1, pct=True)]) + r" \\")
lines += [r"\bottomrule", r"\end{tabular}"]
out.parent.mkdir(parents=True, exist_ok=True)
out.write_text("\n".join(lines) + "\n")
print(out)
