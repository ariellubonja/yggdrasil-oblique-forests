#!/usr/bin/env python3
"""Check benchmarks/results/kdd28 against the floats of a KDD28 main.tex (layout: benchmarks/results/kdd28/README.md).

  python3 check_layout.py [main.tex] [--root DIR]    # exit 1 and one ERROR line per problem, else "OK layout ..."
main.tex default: $KDD28_MAIN_TEX, else the Overleaf MCP clone /private/tmp/overleaf-6a7c57000dd0693cd88faa33/main.tex
when it exists, else paper/distributed_spo_gbt_kdd28/main.tex. Errors: a data float (one that inputs a *_body file) with no
dir, a float dir whose labels are not in main.tex, README fields missing, inputs.txt missing or naming absent files, an
outdated/ or analysis/ entry its README does not name, a kdd28_params.json _floats label that main.tex lacks.
"""
import json
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "../../../.."))
ROOT = os.path.join(REPO, "benchmarks/results/kdd28")
CLONE = "/private/tmp/overleaf-6a7c57000dd0693cd88faa33/main.tex"
SNAPSHOT = os.path.join(REPO, "paper/distributed_spo_gbt_kdd28/main.tex")
FIELDS = ["Labels", "Status", "Profile", "Settings", "Eval suite", "Machine", "Date", "Code", "Generator", "Inputs",
          "Raw cells"]
FLOAT_GROUPS = ("accuracy", "runtime", "appendix", "appendix-earlier-settings")
LISTED = ("outdated", "analysis")  # entries must be named in the folder's README
OTHER = ("cells",)


def default_tex():
    return os.environ.get("KDD28_MAIN_TEX") or (CLONE if os.path.exists(CLONE) else SNAPSHOT)


def tex_floats(text):
    """(all tab:/fig: labels, data labels) of a main.tex; data = a *_body input next to the label."""
    text = "\n".join(re.sub(r"(?<!\\)%.*", "", ln) for ln in text.split("\n"))
    ms = list(re.finditer(r"\\label\{([^}]*)\}", text))
    labels, data = [], []
    for i, m in enumerate(ms):  # figures input the body before their \label, tables after it (caption-above)
        lab = m.group(1)
        if re.match(r"(tab|fig):", lab):
            labels.append(lab)
            if lab.startswith("fig:"):
                lo, hi = ms[i - 1].end() if i else 0, m.start()
            else:
                lo, hi = m.end(), ms[i + 1].start() if i + 1 < len(ms) else len(text)
            if re.search(r"\\input\{[^}]*_body\}", text[lo:hi]):
                data.append(lab)
    return labels, data


def readme_labels(text):
    m = re.search(r"\*\*Labels:\*\*(.*)", text)
    return re.findall(r"`((?:tab|fig):[^`]*)`", m.group(1)) if m else []


def check(tex_path, root=ROOT, params_path=os.path.join(HERE, "kdd28_params.json"), repo=REPO):
    errs = []
    labels, data = tex_floats(open(tex_path).read())
    owner = {}
    for g in FLOAT_GROUPS:
        gd = os.path.join(root, g)
        for d in sorted(os.listdir(gd)) if os.path.isdir(gd) else []:
            fd, rel = os.path.join(gd, d), f"{g}/{d}"
            if not os.path.isdir(fd):
                continue
            rm = os.path.join(fd, "README.md")
            if not os.path.exists(rm):
                errs.append(f"{rel}: README.md missing")
                continue
            txt = open(rm).read()
            errs += [f"{rel}/README.md: field **{f}:** missing" for f in FIELDS if f"**{f}:**" not in txt]
            labs = readme_labels(txt)
            if not labs:
                errs.append(f"{rel}/README.md: no labels")
            for lab in labs:
                if lab not in labels:
                    errs.append(f"{rel}: label {lab} not in {tex_path} (dropped float -> outdated/)")
                if lab in owner:
                    errs.append(f"{rel}: label {lab} also claimed by {owner[lab]}")
                owner[lab] = rel
            if labs and d != labs[0].replace(":", "-"):
                errs.append(f"{rel}: dir name should be {labs[0].replace(':', '-')}")
            inp = os.path.join(fd, "inputs.txt")
            if not os.path.exists(inp):
                errs.append(f"{rel}: inputs.txt missing")
                continue
            ents = [ln.split("#", 1)[0].strip() for ln in open(inp)]
            ents = [e for e in ents if e]
            if not ents and "pending" not in txt.lower():
                errs.append(f"{rel}/inputs.txt: empty")
            errs += [f"{rel}/inputs.txt: {e} does not exist" for e in ents if not os.path.exists(os.path.join(repo, e))]
    errs += [f"data float {lab}: no results dir (README **Labels:** line)" for lab in data if lab not in owner]
    for g in LISTED:
        gd = os.path.join(root, g)
        if not os.path.isdir(gd):
            continue
        rm = os.path.join(gd, "README.md")
        txt = open(rm).read() if os.path.exists(rm) else ""
        for e in sorted(os.listdir(gd)):
            if e in ("README.md", ".DS_Store"):
                continue
            name = e + ("/" if os.path.isdir(os.path.join(gd, e)) else "")
            if f"`{name}`" not in txt:
                errs.append(f"{g}/{name}: not named in {g}/README.md (one-line reason)")
    known = set(FLOAT_GROUPS) | set(LISTED) | set(OTHER) | {"README.md", ".DS_Store"}
    errs += [f"{e}: unknown top-level entry" for e in sorted(os.listdir(root)) if e not in known] if os.path.isdir(root) else []
    if params_path and os.path.exists(params_path):
        def walk(p, name):
            for lab in p.get("_floats", []):
                if lab not in labels:
                    errs.append(f"kdd28_params.json {name}._floats: {lab} not in {tex_path}")
            for a, sub in p.get("arms", {}).items():
                walk(sub, f"{name}.{a}")
        for n, p in json.load(open(params_path))["profiles"].items():
            walk(p, n)
    return errs, len(data), len(owner)


def main(argv):
    args = [a for a in argv[1:] if not a.startswith("--")]
    root = argv[argv.index("--root") + 1] if "--root" in argv else ROOT
    if "--root" in argv:
        args.remove(root)
    tex = args[0] if args else default_tex()
    errs, n_data, n_dirs = check(tex, root)
    for e in errs:
        print("ERROR layout:", e)
    if not errs:
        print(f"OK layout {os.path.relpath(root, REPO)} vs {tex}: {n_data} data floats, all in {n_dirs} labelled dirs")
    sys.exit(1 if errs else 0)


if __name__ == "__main__":
    main(sys.argv)
