#!/usr/bin/env python3
"""Per-feature split histogram of the k-th tree (--trees, 1-based; first = 1) of each CC18 AA-GBT model, pooled
over the 10 folds, on a FIXED x-axis: the dataset's features in CSV header order (x = 1..F, label excluded,
unused features at 0). Datasets = the k (default 3) most / least Pareto-like by the first tree's normalised
entropy (same rule as make_fig_tree1_pareto.py), kept fixed for the later trees; y range shared per dataset.
Writes under --out: figures/evaluation/cc18_tree_feature_hist/<slot>_t<tree>.dat and --body (default cc18_tree_feature_hist_body.tex,
a second figure* in main.tex, appended below fig:cc18-tree1-pareto). Caption/label stay in main.tex."""
import argparse, collections, csv, io, math, random, re, tarfile
from pathlib import Path

here = Path(__file__).resolve().parents[2] / "results" / "misc" / "cc18_gbt_feature_study"
ap = argparse.ArgumentParser()
ap.add_argument("--tar", default=here / "nodes_all.tar.gz")
ap.add_argument("--per_dataset", default=here / "per_dataset.csv")
ap.add_argument("--data", default=here.parents[2] / "data" / "cc18_binary_csv", help="CC18 CSV dir (headers)")
ap.add_argument("--out", required=True, nargs="+", help="paper dir(s): <out>/figures/evaluation/... is written")
ap.add_argument("--body", default="cc18_tree_feature_hist_body.tex")
ap.add_argument("--k", type=int, default=3, help="datasets per row (most / least)")
ap.add_argument("--trees", default="1,100,200,300", help="1-based tree numbers, one block each")
ap.add_argument("--select", choices=["random", "extremes"], default="random",
                help="random: --n datasets drawn with --seed, ordered by F; extremes: k most / k least Pareto-like by tree 1")
ap.add_argument("--n", type=int, default=6, help="datasets for --select random (filled k per row)")
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--datasets", default="", help="comma list of short names; overrides --select, ordered by F")
ap.add_argument("--no_counts", action="store_true", help="panel title = dataset name only, no (used/F)")
ap.add_argument("--hsep", default="1.0cm", help="horizontal sep between panels")
ap.add_argument("--bar_max_f", type=int, default=60, help="ybar up to this many features, ycomb above")
args = ap.parse_args()
trees = [int(t) for t in args.trees.split(",")]

F = {r["dataset"]: int(float(r["F"])) for r in csv.DictReader(open(args.per_dataset))}
counts = {t: collections.defaultdict(collections.Counter) for t in trees}
with tarfile.open(args.tar) as tf:
    for m in tf.getmembers():
        mm = re.search(r"runs/(.+)__fold\d+/nodes\.csv$", m.name)
        if not mm:
            continue
        ds = mm.group(1)
        for r in csv.DictReader(io.TextIOWrapper(tf.extractfile(m), encoding="utf-8")):
            t = int(r["tree"]) + 1  # dumps are 0-based
            if t in counts:
                counts[t][ds][int(r["attribute"])] += 1

# header order = dataspec column order; the label is the last column in every CC18 CSV (asserted below)
headers = {ds: open(Path(args.data) / ds / "repeat0_fold0_sample0_train.csv").readline().rstrip("\n").split(",")
           for ds in F}
for ds, h in headers.items():
    assert len(h) - 1 == F[ds], (ds, len(h), F[ds])
    used = set().union(*(counts[t][ds].keys() for t in trees))
    assert max(used) < F[ds], (ds, "an attribute index points at the label column")

stats = []  # rank datasets by the first tree's normalised entropy
for ds, c in counts[trees[0]].items():
    n = sum(c.values()); p = [v / n for v in c.values()]
    stats.append((-sum(x * math.log(x) for x in p) / math.log(F[ds]), ds))
stats.sort()
if args.datasets:
    want = args.datasets.split(",")
    chosen = sorted([ds for ds in F if re.sub(r"^task_\d+_", "", ds) in want], key=lambda ds: (F[ds], ds))
    assert len(chosen) == len(want), chosen
    slots = [(f"row{i // args.k + 1}", i % args.k + 1, ds) for i, ds in enumerate(chosen)]
elif args.select == "extremes":
    slots = [("most", i + 1, ds) for i, (_, ds) in enumerate(stats[:args.k])] + \
            [("least", i + 1, ds) for i, (_, ds) in enumerate(stats[::-1][:args.k])]
else:  # uniform sample of the study datasets, displayed in increasing F
    chosen = sorted(random.Random(args.seed).sample(sorted(F), args.n), key=lambda ds: (F[ds], ds))
    slots = [(f"row{i // args.k + 1}", i % args.k + 1, ds) for i, ds in enumerate(chosen)]
kinds = list(dict.fromkeys(k for k, _, _ in slots))
hn_of = {ds: hn for hn, ds in stats}

def pretty(ds):
    return re.sub(r"^task_\d+_", "", ds)

for out in map(Path, args.out):
    d = out / "figures" / "evaluation" / "cc18_tree_feature_hist"; d.mkdir(parents=True, exist_ok=True)
    for old in d.glob("*.dat"):
        old.unlink()
    panels, ymax = [], {}
    for t in trees:
        for kind, i, ds in slots:
            c = counts[t][ds]; n = sum(c.values())
            lines = ["x attribute count share_pct"]
            for a in range(F[ds]):  # every feature, header order, zeros included
                lines.append(f"{a + 1} {headers[ds][a]} {c.get(a, 0)} {100 * c.get(a, 0) / n:.3f}")
            (d / f"{kind}{i}_t{t}.dat").write_text("\n".join(lines) + "\n")
            ymax[ds] = max(ymax.get(ds, 0), 100 * max(c.values()) / n)
            panels.append((t, kind, i, ds, pretty(ds) if args.no_counts else f"{pretty(ds)} ({len(c)}/{F[ds]})"))
    body = [r"\definecolor{paretoblue}{HTML}{3B6EA8}", r"\definecolor{paretoorange}{HTML}{F28E2B}"]  # most = orange, least = blue
    for bi, t in enumerate(trees):
        last = bi == len(trees) - 1
        body += ([r"\par\vspace{0.2ex}"] if bi else []) + [
            f"{{\\footnotesize\\bfseries Tree {t}}}\\par\\nopagebreak" + ("\\vspace{0.4ex}" if args.k > 3 else "\\vspace{-0.6ex}"),
            r"\begin{tikzpicture}[trim axis group left, trim axis group right]",
            f"\\begin{{groupplot}}[group style={{group size={args.k} by {len(kinds)}, horizontal sep={args.hsep}, vertical sep=1.05cm,",
            r"    xlabels at=edge bottom, ylabels at=edge left},",
            f"  width={1 / args.k:.3f}\\textwidth, height=2.7cm, ymin=0,",
            ((r"  xlabel={feature index}, " if args.k > 3 else r"  xlabel={feature index (header order)}, ") if last else "  ") + r"ylabel={share (\%)},",
            r"  tick label style={font=\scriptsize}, label style={font=\scriptsize}, title style={font=\scriptsize, yshift=-1ex},",
            r"  enlarge x limits=false, scaled ticks=false]"]
        for kind in kinds:
            for t2, k2, i, ds, title in panels:
                if t2 != t or k2 != kind:
                    continue
                f = F[ds]; col = "paretoorange" if kind == "most" else "paretoblue"
                xtick = f"{{1,...,{f}}}" if f <= (9 if args.k > 3 else 12) else f"{{1,{f}}}"
                style = (f"ybar, draw=none, fill={col}, /pgf/bar width=0.8" if f <= args.bar_max_f
                         else f"ycomb, {col}, line width=0.35pt, mark=none")
                body += [f"% tree {t} {kind} #{i}: tree-1 normalised entropy {hn_of[ds]:.3f}",
                         f"\\nextgroupplot[title={{{title}}}, xmin=0.5, xmax={f + 0.5}, xtick={xtick}, ymax={1.08 * ymax[ds]:.3f}]",
                         f"\\addplot[{style}] table[x=x, y=share_pct] {{figures/evaluation/cc18_tree_feature_hist/{kind}{i}_t{t}.dat}};"]
        body += [r"\end{groupplot}", r"\end{tikzpicture}"]
    (out / "figures" / "evaluation" / args.body).write_text("\n".join(body) + "\n")
    print(out / "figures" / "evaluation" / args.body)
for t, kind, i, ds, title in panels:
    print(f"tree {t} {kind}{i}: {title}  Hn(tree1)={hn_of[ds]:.3f}")
