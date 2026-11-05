"""KDD28 eval suites (eval_suites.json): datasets, folds, fold-file sha256, seeds, metric code, validation, early
stopping a comparable accuracy table shares. Identity mismatches (dataset, fold, file hash, metric, seed, validation,
early stopping, suite, coverage) exit 2 ("STOP eval-suite:") unless the run is declared off-paper; provenance notes
(unstamped rows) only warn. Stamps: eval_suite, data_sha (<train sha256[:16]>+<test sha256[:16]>), metric_impl.
REPLICABILITY.md section K."""
import collections
import concurrent.futures
import hashlib
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.normpath(os.path.join(HERE, "..", "..", "..", ".."))
DB_PATH = os.path.join(HERE, "eval_suites.json")
COLUMNS = ["eval_suite", "data_sha", "metric_impl"]
PREFIX = "WARN eval-suite:"
STOP_PREFIX = "STOP eval-suite:"
OFF_PAPER = "off-paper:"  # eval_suite stamp of rows run with --off_paper "<reason>"
# binary basename -> metric implementation id (eval_suites.json "metric_impls")
BINARY_METRIC = {"train_oblique_forest": "ydf_eval_v1", "train_distributed_gbt": "ydf_eval_v1"}
NO_VALIDATION = ("", "-", "0", "0.0", "none")
NO_EARLY_STOP = ("", "-", "NONE", "none")


def warn(msg, out=None):
    line = f"{PREFIX} {msg}"
    print(line, file=sys.stderr)
    if out is not None:
        out.append(line)


def load_db():
    with open(DB_PATH) as f:
        return json.load(f)


def _manifest(spec):
    """relative path -> sha256 from the suite's manifest ({} when it has none or the file is missing)."""
    h = spec.get("hashes") or {}
    path = os.path.join(ROOT, h["manifest"]) if h.get("manifest") else ""
    out, strip = {}, h.get("strip", "")
    if path and os.path.exists(path):
        with open(path) as f:
            for ln in f:
                parts = ln.split(None, 1)
                if len(parts) == 2 and parts[1].strip().startswith(strip):
                    out[parts[1].strip()[len(strip):]] = parts[0]
    return out


def load_suite(suite, db=None):
    """Suite spec with "name" and "sha" (rel path -> sha256). `suite` is a name or an already-built dict (selftest)."""
    if isinstance(suite, dict):
        return suite
    db = db or load_db()
    if suite not in db["suites"]:
        raise KeyError(f"unknown eval suite {suite!r}; known: {', '.join(db['suites'])}")
    spec = dict(db["suites"][suite], name=suite)
    spec["sha"] = _manifest(spec)
    return spec


def expected_sha(spec, dataset, fold):
    """data_sha the suite records for (dataset, fold), or None when it records no hash for those files."""
    files = [spec["files"][s].format(dataset=dataset, fold=fold) for s in ("train", "test")]
    shas = [spec.get("sha", {}).get(f) for f in files]
    return None if None in shas else data_sha(*shas)


def data_sha(train_sha, test_sha):
    return f"{(train_sha or '')[:16]}+{(test_sha or '')[:16]}"


def reference_note(suite, who, out=None, db=None):
    """One line when `suite` is not the reference suite."""
    db = db or load_db()
    if suite != db["reference"]:
        warn(f"{who} declares eval suite {suite}, not comparable with {db['reference_float']} "
             f"(reference suite {db['reference']})", out)


# ---------------------------------------------------------------- row check

def _v(r, *keys):
    for k in keys:
        if k in r and r[k] is not None and str(r[k]).strip() not in ("", "nan"):
            return str(r[k]).strip()
    return ""


def _fold(spec, r, ds):
    f = _v(r, "fold")
    if f:
        try:
            return int(float(f))
        except ValueError:
            return f
    folds = spec["datasets"].get(ds, [])
    return folds[0] if len(folds) == 1 else None


def check_rows(rows, suite, label, max_lines=40, out=None, strict=False, coverage=True):
    """Reports per dataset where rows differ from the suite (dataset, fold, data_sha, metric_impl, seed, validation,
    early stopping), coverage gaps, unstamped rows and mixed eval_suite values. Returns the lines.
    strict: identity mismatches print STOP lines and exit 2 (unstamped rows stay a WARN). coverage=False skips the
    missing-dataset/fold check (driver runs restricted with --only)."""
    out = [] if out is None else out
    fatal = []

    def bad(msg, _out=None):  # same call shape as warn(msg, out)
        fatal.append(msg)
        warn(msg, out)
    spec = load_suite(suite)
    name = spec.get("name", "?")
    rows = list(rows)
    if not rows:
        warn(f"{label}: no rows to check against {name}", out)
        return out
    stamped = [r for r in rows if _v(r, "eval_suite")]
    if len(stamped) < len(rows):
        warn(f"{label}: unstamped eval: {len(rows) - len(stamped)} of {len(rows)} rows have no eval_suite/data_sha/"
             "metric_impl, checked on dataset, fold, seed, validation, early stopping only", out)
    seen = collections.Counter(_v(r, "eval_suite") or "unstamped" for r in rows)
    if len(seen) > 1:
        # a row from another suite is an identity mismatch below (eval_suite=...), this line is the overview
        warn(f"{label}: mixes eval suites: " + ", ".join(f"{k} ({n} rows)" for k, n in seen.most_common()), out)
    probs, not_in, folds_seen = collections.defaultdict(collections.Counter), collections.Counter(), collections.defaultdict(set)
    bad_folds = collections.defaultdict(collections.Counter)
    # Legacy CC18 CSVs use bare names ("Bioresponse"), the suite uses "task_<id>_Bioresponse".
    bare = {d.split("_", 2)[2]: d for d in spec["datasets"] if d.startswith("task_") and d.count("_") >= 2}
    for r in rows:
        ds = _v(r, "dataset")
        ds = ds if ds in spec["datasets"] else bare.get(ds, ds)
        if ds not in spec["datasets"]:
            not_in[ds or "<empty>"] += 1
            continue
        p, f = probs[ds], _fold(spec, r, ds)
        if f is None:
            p["fold unknown (no fold column, suite has several)"] += 1
        elif f not in spec["datasets"][ds]:
            bad_folds[ds][f] += 1
        else:
            folds_seen[ds].add(f)
        es = _v(r, "eval_suite")
        if es and es != name:
            p[f"eval_suite={es}"] += 1
        sha = _v(r, "data_sha")
        if sha and f is not None:
            want = expected_sha(spec, ds, f)
            if want is not None and sha != want:
                p[f"fold {f} data_sha={sha} (suite: {want})"] += 1
        mi = _v(r, "metric_impl")
        if es and mi != spec["metric_impl"]:
            p[f"metric_impl={mi or '<empty>'} (suite: {spec['metric_impl']})"] += 1
        seed = _v(r, "eff_seed", "seed")
        if seed and spec.get("seeds") and seed not in {str(s) for s in spec["seeds"]}:
            p[f"seed {seed} not in suite seeds"] += 1
        vr = _v(r, "eff_validation_set_ratio", "validation_set_ratio")
        if spec.get("validation") == "none" and vr not in NO_VALIDATION:
            p[f"validation_set_ratio={vr} (suite: none)"] += 1
        est = _v(r, "eff_early_stopping", "early_stopping")
        if spec.get("early_stopping") == "none" and est not in NO_EARLY_STOP:
            p[f"early_stopping={est} (suite: none)"] += 1
    if not_in:
        names = sorted(not_in)
        bad(f"{label}: {len(names)} datasets not in suite {name} ({sum(not_in.values())} rows): "
             + ", ".join(names[:12]) + (f", ... {len(names) - 12} more" if len(names) > 12 else ""), out)
    for ds, c in bad_folds.items():
        probs[ds][f"folds {sorted(c)} not in suite folds {spec['datasets'][ds]}"] = sum(c.values())
    lines = [f"{label}: {ds}: " + ", ".join(f"{k} ({n} rows)" for k, n in c.most_common())
             for ds, c in sorted(probs.items()) if c]
    missing_ds = sorted(set(spec["datasets"]) - set(probs)) if coverage else []
    if missing_ds:
        bad(f"{label}: {len(missing_ds)} of {len(spec['datasets'])} suite datasets missing: "
             + ", ".join(missing_ds[:12]) + (f", ... {len(missing_ds) - 12} more" if len(missing_ds) > 12 else ""), out)
    lines += [f"{label}: {ds}: missing suite folds {sorted(set(spec['datasets'][ds]) - folds_seen[ds])}"
              for ds in sorted(probs) if coverage and folds_seen[ds] and set(spec["datasets"][ds]) - folds_seen[ds]]
    for ln in lines[:max_lines]:
        bad(ln)
    if len(lines) > max_lines:
        warn(f"{label}: ... {len(lines) - max_lines} more dataset lines", out)
    if fatal and strict:
        print(f"{STOP_PREFIX} {label}: {len(fatal)} identity mismatch(es) with suite {name} (lines above). The number "
              f"would not be the one the float claims. Fix the inputs, or rerun with --off_paper \"<reason>\" for an "
              "off-paper study (no paper generator reads off-paper rows).", file=sys.stderr)
        sys.exit(2)
    if not out:
        print(f"OK eval-suite: {label}: {len(rows)} rows match suite {name} (dataset, fold, data_sha, metric_impl, "
              "seed, validation, early stopping)", file=sys.stderr)
    return out


def backfill_csv(path, suite):
    """Appends the eval columns to a CSV recorded before 2026-11-04: data_sha from the suite manifest (only when the
    rows' files are known to equal it, REPLICABILITY.md section K), metric_impl when the row has a test_auc."""
    import csv
    spec = load_suite(suite)
    with open(path, newline="") as f:
        eol = "\r\n" if f.readline().endswith("\r\n") else "\n"  # keep the file's line ending
        f.seek(0)
        rd = csv.reader(f)
        head, rows = next(rd), list(rd)
    if head[-len(COLUMNS):] == COLUMNS:
        sys.exit(f"{path}: already has the eval columns")
    i_ds, i_fold, i_auc = head.index("dataset"), head.index("fold"), head.index("test_auc")
    out = []
    for r in rows:
        ds, fold = r[i_ds], int(float(r[i_fold]))
        sha = expected_sha(spec, ds, fold) if ds in spec["datasets"] and fold in spec["datasets"][ds] else None
        if sha is None:
            sys.exit(f"{path}: {ds} fold {fold} has no manifest hash in {spec['name']}, nothing written")
        out.append(r + [spec["name"], sha, spec["metric_impl"] if r[i_auc].strip() else ""])
    with open(path + ".part", "w", newline="") as f:
        w = csv.writer(f, lineterminator=eol)
        w.writerow(head + COLUMNS)
        w.writerows(out)
    os.replace(path + ".part", path)
    print(f"{path}: {len(out)} rows stamped {spec['name']} from the manifest", file=sys.stderr)


def check_csv(path, suite, label=None, strict=True):
    import csv
    with open(path, newline="") as f:
        rows = list(csv.DictReader(f))
    rows = [r for r in rows if r.get("rc", "0") in ("0", "")]  # failed cells never reach a table
    return check_rows(rows, suite, label or path, strict=strict)


# ---------------------------------------------------------------- run-time hashing (driver)

def _sha_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 22), b""):
            h.update(b)
    return h.hexdigest()


class ShaCache:
    """sha256 per file, cached in a JSON file keyed by real path + size + mtime_ns. Missing files hash to ''."""

    def __init__(self, path=None):
        self.path, self.db = path, {}
        if path and os.path.exists(path):
            try:
                with open(path) as f:
                    self.db = json.load(f)
            except ValueError:
                self.db = {}

    def _key(self, p):
        st = os.stat(p)
        return os.path.realpath(p), [st.st_size, st.st_mtime_ns]

    def get_many(self, paths, threads=8):
        out, todo = {}, []
        for p in dict.fromkeys(paths):
            try:
                k, sig = self._key(p)
            except OSError:
                out[p] = ""
                continue
            hit = self.db.get(k)
            if hit and hit[:2] == sig:
                out[p] = hit[2]
            else:
                todo.append((p, k, sig))
        with concurrent.futures.ThreadPoolExecutor(threads) as ex:  # hashlib releases the GIL
            for (p, k, sig), h in zip(todo, ex.map(lambda t: _sha_file(t[0]), todo)):
                out[p], self.db[k] = h, sig + [h]
        if todo and self.path:
            with open(self.path + ".part", "w") as f:
                json.dump(self.db, f)
            os.replace(self.path + ".part", self.path)
        return out


def _flag(argv, *names):
    for i, a in enumerate(argv):
        for n in names:
            if a == n and i + 1 < len(argv):
                return argv[i + 1]
            if a.startswith(n + "="):
                return a.split("=", 1)[1]
    return None


def plan_row(table, key, argv, sha):
    """Planned result row of one cell (what check_rows sees), from its key and argv."""
    binary = os.path.basename(argv[0])
    tr, te = _flag(argv, "--train_csv"), _flag(argv, "--test_csv")
    if binary == "train_oblique_forest":  # absent flags = the harness defaults (validation 0.1, early stopping on)
        vr, es = _flag(argv, "--validation_set_ratio") or "default", _flag(argv, "--early_stopping") or "default"
    else:  # train_distributed_gbt has no validation split or early stopping (distributed_gradient_boosted_trees.cc:633)
        vr, es = "", ""
    return {"dataset": str(key.get("dataset", "")), "fold": "" if key.get("fold") is None else str(key["fold"]),
            "seed": _flag(argv, "--seed") or "", "validation_set_ratio": vr, "early_stopping": es,
            "label": _flag(argv, "--label_col", "--label") or "", "train": tr, "test": te,
            "data_sha": data_sha(sha.get(tr), sha.get(te)) if tr and te and sha.get(tr) and sha.get(te) else "",
            "metric_impl": BINARY_METRIC.get(binary, f"unknown:{binary}")}


def driver_preflight(cells, cache_path=None, off_paper=None):
    """cells: [(table, key dict, argv)]. Exits 2 when the plan differs from the table's declared suite in identity; returns {(table, key tuple): [eval_suite, data_sha,
    metric_impl]} for the result rows. off_paper: reason; mismatches only warn and rows are stamped off-paper:<reason>.
    Coverage (datasets a run leaves out, e.g. --only) is not checked here: the generators check it."""
    db = load_db()
    by_table = collections.defaultdict(list)
    for t, key, argv in cells:
        by_table[t].append((key, argv))
    files = [f for _, _, argv in cells for f in (_flag(argv, "--train_csv"), _flag(argv, "--test_csv")) if f]
    sha = ShaCache(cache_path).get_many(files)
    stamps = {}
    for t, items in by_table.items():
        suite = db["_driver_tables"].get(t)
        rows = [plan_row(t, k, argv, sha) for k, argv in items]
        if suite is None:
            warn(f"pre-flight {t}: table declares no eval suite (eval_suites.json _driver_tables), rows unstamped")
            suite_name = ""
        else:
            suite_name = suite
            reference_note(suite, f"pre-flight {t}", db=db)
            spec = load_suite(suite, db)
            nofile = sorted({r["dataset"] for r in rows if not r["data_sha"]})
            if nofile:
                warn(f"pre-flight {t}: {len(nofile)} datasets with unreadable fold files (no data_sha): "
                     + ", ".join(nofile[:12]))
            badlabel = []
            if spec.get("label_col"):
                badlabel = sorted({r["dataset"] for r in rows if r["label"] and r["label"] != spec["label_col"]})
                if badlabel:
                    warn(f"pre-flight {t}: label column != {spec['label_col']} for " + ", ".join(badlabel[:12]))
            strict = not off_paper
            if strict and (nofile or badlabel):
                print(f"{STOP_PREFIX} pre-flight {t}: unreadable fold files or wrong label column (lines above)",
                      file=sys.stderr)
                sys.exit(2)
            check_rows([r | {"eval_suite": suite} for r in rows], spec, f"pre-flight {t}", max_lines=20,
                       strict=strict, coverage=False)
            if off_paper:
                suite_name = OFF_PAPER + off_paper
        for (k, _), r in zip(items, rows):
            stamps[(t,) + tuple(str(v) for v in k.values())] = [suite_name, r["data_sha"], r["metric_impl"]]
    return stamps


def stamp(values, metrics):
    """Final eval stamp of a finished cell: metric_impl only when the binary's metric line was parsed."""
    suite, sha, impl = values
    return [suite, sha, impl if str(metrics.get("test_auc", "")).strip() else ""]
