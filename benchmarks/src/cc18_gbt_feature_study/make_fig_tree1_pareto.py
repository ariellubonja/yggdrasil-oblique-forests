#!/usr/bin/env python3
"""Pareto of split features in the k-th tree (--trees, 1-based; first = 1) of each CC18 AA-GBT model, pooled
over the 10 folds. Datasets = the k (default 3) most / least concentrated by the FIRST tree (normalised entropy
over F), kept fixed for the later trees. Writes under --out: figures/evaluation/cc18_tree1_pareto/<slot>_t<tree>.dat
+ cc18_tree1_pareto_body.tex (one 3-by-2 block per tree: most row, least row). Caption/label stay in main.tex."""
import argparse, collections, csv, io, math, random, re, tarfile
from pathlib import Path

here = Path(__file__).resolve().parents[2] / "results" / "misc" / "cc18_gbt_feature_study"
ap = argparse.ArgumentParser()
ap.add_argument("--tar", default=here / "nodes_all.tar.gz")
ap.add_argument("--per_dataset", default=here / "per_dataset.csv")
ap.add_argument("--out", required=True, nargs="+", help="paper dir(s): <out>/figures/evaluation/... is written")
ap.add_argument("--k", type=int, default=3, help="datasets per row (most / least)")
ap.add_argument("--trees", default="1,100,200,300", help="1-based tree numbers, one block each")
ap.add_argument("--select", choices=["random", "extremes"], default="random",
                help="random: --n datasets drawn with --seed, ordered by F; extremes: k most / k least Pareto-like by tree 1")
ap.add_argument("--n", type=int, default=6, help="datasets for --select random (filled k per row)")
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--datasets", default="", help="comma list of short names; overrides --select, ordered by F")
ap.add_argument("--hsep", default="1.0cm", help="horizontal sep between panels")
args = ap.parse_args()
trees = [int(t) for t in args.trees.split(",")]

F = {r["dataset"]: int(float(r["F"])) for r in csv.DictReader(open(args.per_dataset))}
counts = {t: collections.defaultdict(collections.Counter) for t in trees}
names = collections.defaultdict(dict)
with tarfile.open(args.tar) as tf:
    for m in tf.getmembers():
        mm = re.search(r"runs/(.+)__fold\d+/nodes\.csv$", m.name)
        if not mm:
            continue
        ds = mm.group(1)
        for r in csv.DictReader(io.TextIOWrapper(tf.extractfile(m), encoding="utf-8")):
            t = int(r["tree"]) + 1  # dumps are 0-based
            if t in counts:
                a = int(r["attribute"]); counts[t][ds][a] += 1; names[ds][a] = r["attribute_name"]

stats = []
for ds, c in counts[trees[0]].items():
    n = sum(c.values()); p = [v / n for v in c.values()]
    hn = -sum(x * math.log(x) for x in p) / math.log(F[ds])
    stats.append((hn, ds))
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

def pretty(ds):
    return re.sub(r"^task_\d+_", "", ds)

for out in map(Path, args.out):
    d = out / "figures" / "evaluation" / "cc18_tree1_pareto"; d.mkdir(parents=True, exist_ok=True)
    for old in d.glob("*.dat"):
        old.unlink()
    panels = []
    hn_of = dict((b, a) for a, b in stats)
    for t in trees:
        for kind, i, ds in slots:
            c = counts[t][ds]; n = sum(c.values()); items = sorted(c.items(), key=lambda kv: (-kv[1], kv[0]))
            lines = ["rank attribute count share_pct cum_pct"]; cum = 0
            for rank, (a, v) in enumerate(items, 1):
                cum += v
                lines.append(f"{rank} {names[ds][a]} {v} {100*v/n:.3f} {100*cum/n:.3f}")
            (d / f"{kind}{i}_t{t}.dat").write_text("\n".join(lines) + "\n")
            panels.append((t, kind, i, f"{pretty(ds)} ({len(items)}/{F[ds]})", len(items), hn_of[ds]))
    body = [r"\definecolor{paretoblue}{HTML}{3B6EA8}", r"\definecolor{paretoorange}{HTML}{F28E2B}"]  # most = orange, least = blue
    for bi, t in enumerate(trees):
        last = bi == len(trees) - 1
        body += ([r"\par\vspace{0.2ex}"] if bi else []) + [
            f"{{\\footnotesize\\bfseries Tree {t}}}\\par\\nopagebreak" + ("\\vspace{0.4ex}" if args.k > 3 else "\\vspace{-0.6ex}"),
            r"\begin{tikzpicture}[trim axis group left, trim axis group right]",
            f"\\begin{{groupplot}}[group style={{group size={args.k} by {len(kinds)}, horizontal sep={args.hsep}, vertical sep=1.05cm,",
            r"    xlabels at=edge bottom, ylabels at=edge left},",
            f"  width={1/args.k:.3f}\\textwidth, height=2.7cm, ymin=0, enlarge y limits={{upper, value=0.08}},",
            (r"  xlabel={feature rank}, " if last else "  ") + r"ylabel={share (\%)}, ybar, /pgf/bar width=0.8,",
            r"  tick label style={font=\scriptsize}, label style={font=\scriptsize}, title style={font=\scriptsize, yshift=-1ex},",
            r"  xmin=0.4, enlarge x limits=false, scaled ticks=false]"]
        for kind in kinds:
            for t2, k2, i, title, npk, hn in panels:
                if t2 != t or k2 != kind:
                    continue
                body += [f"% tree {t} {kind} #{i}: tree-1 normalised entropy {hn:.3f}",
                         f"\\nextgroupplot[title={{{title}}}, xmax={npk + 0.6}, xtick={{1,{npk}}}]" if npk > (9 if args.k > 3 else 12)
                         else f"\\nextgroupplot[title={{{title}}}, xmax={npk + 0.6}, xtick={{1,...,{npk}}}]",
                         f"\\addplot[draw=none, fill={{{'paretoorange' if kind == 'most' else 'paretoblue'}}}] table[x=rank, y=share_pct] {{figures/evaluation/cc18_tree1_pareto/{kind}{i}_t{t}.dat}};"]
        body += [r"\end{groupplot}", r"\end{tikzpicture}"]
    (out / "figures" / "evaluation" / "cc18_tree1_pareto_body.tex").write_text("\n".join(body) + "\n")
    print(out / "figures" / "evaluation" / "cc18_tree1_pareto_body.tex")
for t, kind, i, title, npk, hn in panels:
    print(f"tree {t} {kind}{i}: {title}  Hn(tree1)={hn:.3f} panels={npk}")
