#!/usr/bin/env python3
"""Analyze and plot the oblique-vs-axis GBT inference sweep CSV.

Outputs (next to the CSV, prefix matches CSV stem):
  - <stem>_pareto_<dataset>.png   — accuracy vs inference µs/example
  - <stem>_summary.md             — markdown table summary
"""

import argparse
import sys
from pathlib import Path

import csv
try:
    import matplotlib.pyplot as plt
except Exception as e:
    print(f"[WARN] matplotlib not available: {e}", file=sys.stderr)
    plt = None


def load_rows(csv_path):
    rows = []
    with open(csv_path) as fh:
        for r in csv.DictReader(fh):
            try:
                r["max_depth"] = int(r["max_depth"])
                r["accuracy"] = float(r["accuracy"]) if r["accuracy"] else None
                r["log_loss"] = float(r["log_loss"]) if r["log_loss"] else None
                r["best_us_per_ex"] = (float(r["best_us_per_ex"])
                                        if r["best_us_per_ex"] else None)
                r["generic_us_per_ex"] = (
                    float(r["generic_us_per_ex"])
                    if r["generic_us_per_ex"] else None)
                r["train_seconds"] = float(r["train_seconds"])
                r["num_trees_final"] = (int(r["num_trees_final"])
                                         if r["num_trees_final"] else None)
            except (ValueError, KeyError):
                continue
            rows.append(r)
    return rows


def pareto_front(points):
    """Return indices on Pareto frontier (minimize x, maximize y)."""
    pts = sorted(enumerate(points), key=lambda iy: iy[1][0])  # by x asc
    front = []
    best_y = -float("inf")
    for i, (x, y) in pts:
        if y > best_y:
            front.append(i)
            best_y = y
    return front


def plot_pareto(rows, dataset, out_path, x_field="best_us_per_ex"):
    if plt is None:
        return
    plt.figure(figsize=(8, 5))
    for split, marker, color in [
        ("Axis Aligned", "o", "tab:blue"),
        ("Oblique", "s", "tab:orange"),
    ]:
        sub = [r for r in rows
               if r["dataset"] == dataset and r["split_type"] == split
               and r[x_field] is not None and r["accuracy"] is not None]
        if not sub:
            continue
        sub.sort(key=lambda r: r["max_depth"])
        xs = [r[x_field] for r in sub]
        ys = [r["accuracy"] for r in sub]
        labels = [str(r["max_depth"]) for r in sub]
        plt.plot(xs, ys, marker=marker, color=color, label=split,
                 linestyle="--", markersize=8)
        for x, y, lab in zip(xs, ys, labels):
            plt.annotate(f"d={lab}", (x, y), textcoords="offset points",
                         xytext=(6, -3), fontsize=8, color=color)
    pretty_x = ("Inference µs/example (best engine)"
                if x_field == "best_us_per_ex"
                else "Inference µs/example (Generic engine)")
    plt.xlabel(pretty_x)
    plt.ylabel("Test accuracy")
    plt.title(f"{dataset}: GBT inference latency vs accuracy "
              f"({x_field})")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=130)
    plt.close()
    print(f"  wrote {out_path}")


def summary_md(rows):
    out = []
    out.append("| Dataset | Split | Depth | Trees | Acc | Best µs/ex (engine) | Generic µs/ex |")
    out.append("|---------|-------|-------|-------|-----|---------------------|---------------|")
    for r in sorted(rows, key=lambda r: (r["dataset"],
                                         r["split_type"],
                                         r["max_depth"])):
        eng = r.get("best_engine", "").replace(" [virtual interface]", "")
        out.append(
            f"| {r['dataset']} | {r['split_type']} | {r['max_depth']} "
            f"| {r['num_trees_final'] or '?'} "
            f"| {r['accuracy']:.4f} "
            f"| {r['best_us_per_ex']:.3f} ({eng}) "
            f"| {r['generic_us_per_ex']:.3f} |")
    return "\n".join(out)


def comparison_md(rows, x_field="best_us_per_ex"):
    """For each dataset, find each oblique point and the closest-accuracy
    axis point, report inference time ratio."""
    label = ("best engine for each (deployment-realistic)"
             if x_field == "best_us_per_ex"
             else "Generic engine for both (apples-to-apples)")
    title = ("Best-engine comparison" if x_field == "best_us_per_ex"
             else "Apples-to-apples (Generic engine)")
    out = ["", f"## {title}: oblique vs nearest-accuracy axis-aligned",
           "",
           f"({label}. Lower latency = better. "
           "Ratio > 1.0 means axis-aligned is faster.)",
           "",
           "| Dataset | Oblique (d, acc, µs) | Nearest axis (d, acc, µs) "
           "| Δ acc | Speed ratio (axis/oblique) |",
           "|---------|----------------------|----------------------------"
           "|--------|----------------------------|"]
    by_ds = {}
    for r in rows:
        if r["accuracy"] is None or r[x_field] is None:
            continue
        by_ds.setdefault(r["dataset"], []).append(r)
    for ds, sub in by_ds.items():
        ax = [r for r in sub if r["split_type"] == "Axis Aligned"]
        ob = [r for r in sub if r["split_type"] == "Oblique"]
        for o in sorted(ob, key=lambda r: r["max_depth"]):
            if not ax:
                continue
            nearest = min(ax, key=lambda r: abs(r["accuracy"] - o["accuracy"]))
            d_acc = o["accuracy"] - nearest["accuracy"]
            ratio = nearest[x_field] / o[x_field]
            out.append(
                f"| {ds} | d={o['max_depth']} acc={o['accuracy']:.4f} "
                f"µs={o[x_field]:.3f} "
                f"| d={nearest['max_depth']} "
                f"acc={nearest['accuracy']:.4f} "
                f"µs={nearest[x_field]:.3f} "
                f"| {d_acc:+.4f} "
                f"| {ratio:.2f}x |")
    return "\n".join(out)


def pareto_winners(rows, x_field="best_us_per_ex"):
    """For each dataset, list which (split, depth) lies on the
    accuracy↑ vs latency↓ Pareto frontier."""
    title = ("Pareto frontier per dataset (best engine)"
             if x_field == "best_us_per_ex"
             else "Pareto frontier per dataset (apples-to-apples Generic engine)")
    out = ["", f"## {title}", ""]
    by_ds = {}
    for r in rows:
        if r["accuracy"] is None or r[x_field] is None:
            continue
        by_ds.setdefault(r["dataset"], []).append(r)
    for ds, sub in sorted(by_ds.items()):
        pts = [(r[x_field], r["accuracy"]) for r in sub]
        front = pareto_front(pts)
        out.append(f"### {ds}")
        out.append("")
        out.append("| µs/ex | Acc | Split | Depth | Engine |")
        out.append("|-------|-----|-------|-------|--------|")
        for i in sorted(front, key=lambda j: pts[j][0]):
            r = sub[i]
            eng = r.get("best_engine", "").replace(" [virtual interface]", "")
            out.append(
                f"| {r[x_field]:.3f} | {r['accuracy']:.4f} "
                f"| {r['split_type']} | {r['max_depth']} | {eng} |")
        out.append("")
    return "\n".join(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("csv")
    ap.add_argument("--out_dir", default=None)
    args = ap.parse_args()

    csv_path = Path(args.csv)
    out_dir = Path(args.out_dir) if args.out_dir else csv_path.parent
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = load_rows(csv_path)
    if not rows:
        print("No rows.")
        return

    stem = csv_path.stem
    datasets = sorted({r["dataset"] for r in rows})
    for ds in datasets:
        for x_field in ("best_us_per_ex", "generic_us_per_ex"):
            tag = "best" if x_field == "best_us_per_ex" else "generic"
            out_png = out_dir / f"{stem}_pareto_{ds}_{tag}.png"
            plot_pareto(rows, ds, out_png, x_field=x_field)

    md_path = out_dir / f"{stem}_summary.md"
    text = ["# " + stem, ""]
    text.append("## Raw rows")
    text.append("")
    text.append(summary_md(rows))
    text.append(comparison_md(rows, x_field="generic_us_per_ex"))
    text.append(comparison_md(rows, x_field="best_us_per_ex"))
    text.append(pareto_winners(rows, x_field="generic_us_per_ex"))
    text.append(pareto_winners(rows, x_field="best_us_per_ex"))
    md_path.write_text("\n".join(text) + "\n")
    print(f"Wrote {md_path}")


if __name__ == "__main__":
    main()
