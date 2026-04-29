#!/usr/bin/env python3
"""Aggregate Phase B sweep_v3.csv into per-dataset Pareto with 10-seed
mean±std, plot, and a final verdict table.

Verdict per dataset: oblique CROSSES the 20% gate iff there exists
some (oblique d_o, axis d_a) such that
    mean_acc(oblique, d_o) >= mean_acc(axis, d_a)  AND
    mean_us(oblique, d_o) <= 0.8 * mean_us(axis, d_a)
on the apples-to-apples Generic engine.
"""
import argparse
import csv
import math
from collections import defaultdict
from pathlib import Path

try:
    import matplotlib.pyplot as plt
except Exception:
    plt = None


def load_rows(csv_path: Path):
    rows = []
    with open(csv_path) as fh:
        for r in csv.DictReader(fh):
            try:
                r["max_depth"] = int(r["max_depth"])
                r["seed"] = int(r["seed"])
                r["accuracy"] = float(r["accuracy"]) if r["accuracy"] else None
                r["log_loss"] = (float(r["log_loss"]) if r["log_loss"]
                                  else None)
                r["best_us_per_ex"] = (float(r["best_us_per_ex"])
                                        if r["best_us_per_ex"] else None)
                r["generic_us_per_ex"] = (
                    float(r["generic_us_per_ex"])
                    if r["generic_us_per_ex"] else None)
                r["n_features"] = (int(r["n_features"])
                                    if r.get("n_features") else None)
            except (ValueError, KeyError):
                continue
            rows.append(r)
    return rows


def mean_std(values):
    n = len(values)
    if n == 0:
        return None, None
    m = sum(values) / n
    var = sum((v - m) ** 2 for v in values) / max(n, 1)
    return m, math.sqrt(var)


def aggregate(rows):
    """Returns dict keyed (dataset, split, depth) → dict of stats."""
    grouped = defaultdict(list)
    for r in rows:
        if r["accuracy"] is None or r["generic_us_per_ex"] is None:
            continue
        key = (r["dataset"], r["split_type"], r["max_depth"])
        grouped[key].append(r)
    stats = {}
    for key, sub in grouped.items():
        accs = [r["accuracy"] for r in sub]
        gens = [r["generic_us_per_ex"] for r in sub]
        bests = [r["best_us_per_ex"] for r in sub if r["best_us_per_ex"]]
        stats[key] = {
            "n_seeds": len(sub),
            "acc_mean": mean_std(accs)[0],
            "acc_std": mean_std(accs)[1],
            "gen_mean": mean_std(gens)[0],
            "gen_std": mean_std(gens)[1],
            "best_mean": mean_std(bests)[0] if bests else None,
            "best_std": mean_std(bests)[1] if bests else None,
            "n_features": sub[0].get("n_features"),
        }
    return stats


def per_dataset_verdict(stats, gate_ratio=0.8):
    """For each dataset, find any (oblique d_o, axis d_a) pair satisfying
    the AGENTS.md 20% gate at matched-or-better accuracy, on Generic
    engine."""
    by_ds = defaultdict(lambda: {"axis": [], "oblique": []})
    for (ds, split, depth), s in stats.items():
        bucket = "axis" if split == "Axis Aligned" else "oblique"
        by_ds[ds][bucket].append((depth, s))

    out = []
    for ds in sorted(by_ds):
        ax = sorted(by_ds[ds]["axis"], key=lambda t: t[0])
        ob = sorted(by_ds[ds]["oblique"], key=lambda t: t[0])
        winner = None  # (oblique_d, axis_d, ratio, dacc)
        for d_o, s_o in ob:
            for d_a, s_a in ax:
                if s_o["acc_mean"] >= s_a["acc_mean"] - 0.001:
                    if s_o["gen_mean"] <= gate_ratio * s_a["gen_mean"]:
                        ratio = s_a["gen_mean"] / s_o["gen_mean"]
                        d_acc = s_o["acc_mean"] - s_a["acc_mean"]
                        if not winner or ratio > winner[2]:
                            winner = (d_o, d_a, ratio, d_acc)
        # Also report best oblique vs nearest-acc axis (for context).
        best_obl = max(ob, key=lambda t: t[1]["acc_mean"]) if ob else None
        nearest_ax = (min(ax, key=lambda t: abs(
            t[1]["acc_mean"] - best_obl[1]["acc_mean"])) if best_obl and ax
                       else None)
        out.append((ds, winner, best_obl, nearest_ax,
                    by_ds[ds]["axis"][0][1]["n_features"]
                    if ax else None))
    return out


def plot_dataset(rows_for_ds, ds_name, stats, out_png, x_field="gen_mean"):
    if plt is None:
        return
    plt.figure(figsize=(8, 5))
    for split, marker, color in [
        ("Axis Aligned", "o", "tab:blue"),
        ("Oblique", "s", "tab:orange"),
    ]:
        pts = [(d, s) for (ds, sp, d), s in stats.items()
                if ds == ds_name and sp == split]
        pts.sort(key=lambda t: t[0])
        if not pts:
            continue
        xs = [p[1][x_field] for p in pts]
        ys = [p[1]["acc_mean"] for p in pts]
        xerr = [p[1]["gen_std" if "gen" in x_field else "best_std"]
                 for p in pts]
        yerr = [p[1]["acc_std"] for p in pts]
        plt.errorbar(xs, ys, xerr=xerr, yerr=yerr, marker=marker,
                     color=color, label=split, linestyle="--",
                     markersize=7, capsize=3, alpha=0.8)
        for x, y, depth in zip(xs, ys, [p[0] for p in pts]):
            plt.annotate(f"d={depth}", (x, y), textcoords="offset points",
                         xytext=(5, -3), fontsize=7, color=color)
    plt.xlabel("Inference µs/example (apples-to-apples Generic engine)"
               if "gen" in x_field
               else "Inference µs/example (best engine)")
    plt.ylabel("Test accuracy")
    plt.title(f"{ds_name}: GBT inference vs accuracy "
              f"(10-seed mean ± std)")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_png, dpi=120)
    plt.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("csv")
    ap.add_argument("--out_dir", default=None)
    ap.add_argument("--gate_ratio", type=float, default=0.8,
                    help="Oblique µs <= gate_ratio × axis µs at matched acc.")
    args = ap.parse_args()

    csv_path = Path(args.csv)
    out_dir = Path(args.out_dir) if args.out_dir else csv_path.parent
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = load_rows(csv_path)
    if not rows:
        print("No rows.")
        return

    stats = aggregate(rows)
    stem = csv_path.stem

    # Per-dataset plots.
    datasets = sorted({k[0] for k in stats})
    for ds in datasets:
        plot_dataset(
            rows, ds, stats,
            out_dir / f"{stem}_pareto_{ds}_generic.png",
            x_field="gen_mean")

    # Verdict table.
    verdict = per_dataset_verdict(stats, gate_ratio=args.gate_ratio)
    n_total = len(verdict)
    n_pass = sum(1 for v in verdict if v[1] is not None)

    md = [f"# {stem}: Phase B aggregate (10-seed mean ± std)", ""]
    md.append(f"**Final verdict: oblique passes the 20% gate "
              f"({100*(1-args.gate_ratio):.0f}% latency reduction at "
              f"matched-or-better accuracy) on **{n_pass}/{n_total}** "
              f"datasets.**")
    md.append("")
    md.append(f"Per-dataset summary "
              f"(apples-to-apples = `GradientBoostedTreesGeneric` engine "
              f"for both arms):")
    md.append("")
    md.append(
        "| Dataset | Feat | Pass? | Best obl vs nearest-acc axis "
        "| µs ratio (axis/obl) | Δ acc |")
    md.append(
        "|---------|------|-------|------------------------------|"
        "---------------------|--------|")
    for ds, winner, best_obl, nearest_ax, nf in verdict:
        if winner:
            d_o, d_a, ratio, d_acc = winner
            cell = f"oblique d={d_o} vs axis d={d_a}"
            ratio_cell = f"**{ratio:.2f}×**"
            dacc_cell = f"{d_acc:+.4f}"
            pass_cell = "✓"
        else:
            if best_obl and nearest_ax:
                d_o = best_obl[0]
                s_o = best_obl[1]
                d_a = nearest_ax[0]
                s_a = nearest_ax[1]
                cell = f"d={d_o} {s_o['acc_mean']:.3f} vs d={d_a} {s_a['acc_mean']:.3f}"
                ratio_cell = f"{s_a['gen_mean']/s_o['gen_mean']:.2f}×"
                dacc_cell = f"{s_o['acc_mean'] - s_a['acc_mean']:+.4f}"
            else:
                cell = "—"
                ratio_cell = "—"
                dacc_cell = "—"
            pass_cell = "✗"
        md.append(f"| {ds} | {nf or '?'} | {pass_cell} | {cell} | "
                  f"{ratio_cell} | {dacc_cell} |")

    md.append("")
    md.append("## Per-config means (apples-to-apples Generic engine)")
    md.append("")
    for ds in datasets:
        md.append(f"### {ds}")
        md.append("")
        md.append(
            "| Split | Depth | n_seeds | Acc (mean ± std) | Generic µs (mean ± std) |")
        md.append(
            "|-------|-------|---------|------------------|--------------------------|")
        for split in ("Axis Aligned", "Oblique"):
            cells = [(d, s) for (d_, sp, d), s in stats.items()
                      if d_ == ds and sp == split]
            cells = [(k[2], stats[k]) for k in stats
                      if k[0] == ds and k[1] == split]
            cells.sort(key=lambda t: t[0])
            for depth, s in cells:
                md.append(
                    f"| {split} | {depth} | {s['n_seeds']} "
                    f"| {s['acc_mean']:.4f} ± {s['acc_std']:.4f} "
                    f"| {s['gen_mean']:.3f} ± {s['gen_std']:.3f} |")
        md.append("")

    md_path = out_dir / f"{stem}_summary.md"
    md_path.write_text("\n".join(md) + "\n")
    print(f"Wrote {md_path}")
    print(f"Verdict: {n_pass}/{n_total} datasets pass the "
          f"{100*(1-args.gate_ratio):.0f}% gate")


if __name__ == "__main__":
    main()
