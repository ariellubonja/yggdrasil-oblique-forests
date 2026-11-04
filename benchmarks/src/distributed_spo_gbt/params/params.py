#!/usr/bin/env python3
"""KDD28 run-parameter profiles (kdd28_params.json): loader, shell export, CSV stamps and the generator drift check.

  eval "$(python3 params.py export <profile> [PREFIX])"   # PREFIX default P_: P_TREES=300 P_MAX_DEPTH=6 ...
  python3 params.py get <profile> <field>                  # one value (lists space-joined, null = empty)
  python3 params.py stamp-header [<csv header>]            # stamp columns to append (eff_<name> where the header has <name>)
  python3 params.py stamp <profile> <log>                  # their values, from the log's effective-params line
  python3 params.py require-stamped <csv>                  # exit 2 if <csv> exists without the stamp columns
  python3 params.py show [profile]                         # profile table (values, floats)
  python3 params.py selftest                               # exit 1 unless the matcher / checks behave (stdlib only)
  python3 params.py check-eval <csv> [--suite <name>]       # WARN eval-suite: lines where rows differ from an eval suite
  python3 params.py check-effective <profile> <log> [F=.. W=..] [<stamp col>=<value> ...]
                                                           # exit 2 + field=got (profile: expected) on drift; col=value
                                                           # = the script's own knob (e.g. trees=$TREES) replaces the profile
Run scripts source drift.sh (shell) or call preflight()/check_cell() (Python): exit 2 at the first drifted cell,
ALLOW_DRIFT=1 warns and stamps the row's profile as <profile>!drift, which check_rows refuses.
Profiles are named <profile> or <profile>.<arm>; an arm overrides its parent's fields. Python 3 stdlib only.
Eval suites (datasets, folds, file hashes, metric code; eval_suites.json, evalsuite.py) only warn: generators declare
one with --eval_suite (add_profile_arg), check_rows then checks it and setup_line names it.
"""
import json
import math
import os
import re
import shlex
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import evalsuite  # noqa: E402
DB_PATH = os.path.join(HERE, "kdd28_params.json")
STAMP = ["profile", "trees", "max_depth", "min_examples", "split_type", "spo_exponent", "spo_max_projections",
         "spo_density", "split_budget", "local_projection_count", "seed", "validation_set_ratio", "early_stopping",
         "num_candidate_attributes"]
# stamp column -> key of the "effective-params:" line printed by train_oblique_forest / train_distributed_gbt
LOG_KEY = {"trees": "num_trees", "max_depth": "max_depth", "min_examples": "min_examples", "split_type": "split",
           "spo_exponent": "spo_exponent", "spo_max_projections": "spo_max", "spo_density": "spo_density",
           "split_budget": "split_budget", "local_projection_count": "local_projection_count", "seed": "seed",
           "validation_set_ratio": "validation_set_ratio", "early_stopping": "early_stopping",
           "num_candidate_attributes": "num_candidate_attributes"}
# Recorded, checked only when the profile defines them (train_distributed_gbt has no early stopping)
RECORD_ONLY = {"validation_set_ratio", "early_stopping", "num_candidate_attributes"}
# Checked against the effective-params line when the profile defines them, never CSV stamp columns
# (profile field -> log key). column_shipping: cross-network arm, branch cross-worker-col-comms (2026-11-04).
CHECK_ONLY = {"column_shipping": "column_shipping"}
# stamp column -> profile field (seed is checked against the profile's seed list)
FIELD = {c: c for c in STAMP[1:]} | {"seed": "seeds"}
SPLIT_FLAG = {"EXACT": "Exact", "DYNAMIC_RANDOM_HISTOGRAM": "Dynamic Random Histogram", "HISTOGRAM_RANDOM": "Random",
              "HISTOGRAM_EQUAL_WIDTH": "Equal Width", "DYNAMIC_EQUAL_WIDTH_HISTOGRAM": "Dynamic Equal Width Histogram"}
INT_MAX = 2147483647


def stamp_columns(existing=()):
    """Stamp column names to append to a CSV whose header is `existing`: clashes get an eff_ prefix."""
    return [f"eff_{c}" if c in existing else c for c in STAMP]


def stamped_header(head):
    """True when a CSV header ends with the stamp columns, optionally followed by the eval-suite columns."""
    if head[-len(evalsuite.COLUMNS):] == evalsuite.COLUMNS:
        head = head[:-len(evalsuite.COLUMNS)]
    return head[-len(STAMP):] == stamp_columns(head[:-len(STAMP)])


def load_db():
    with open(DB_PATH) as f:
        return json.load(f)


def _split(name):
    top, _, arm = name.partition(".")
    return top, arm


def load_profile(name, db=None):
    """Values of a profile or sub-profile (no _ keys), plus "profile": name."""
    db = db or load_db()
    top, arm = _split(name)
    if top not in db["profiles"]:
        raise KeyError(f"unknown profile {top!r}; known: {', '.join(db['profiles'])}")
    p = db["profiles"][top]
    if arm and arm not in p.get("arms", {}):
        raise KeyError(f"unknown arm {arm!r} of profile {top!r}; known: {', '.join(p.get('arms', {}))}")
    out = {k: v for k, v in p.items() if not k.startswith("_") and k != "arms"}
    if arm:
        out.update({k: v for k, v in p["arms"][arm].items() if not k.startswith("_")})
    out["profile"] = name
    return out


def profile_meta(name, db=None):
    """_floats, _source (arm sources merged over the parent's), _note of a profile."""
    db = db or load_db()
    top, arm = _split(name)
    p = db["profiles"][top]
    meta = {"_floats": p.get("_floats", []), "_source": dict(p.get("_source", {})), "_note": p.get("_note", "")}
    if arm:
        a = p["arms"][arm]
        meta["_floats"] = a.get("_floats", meta["_floats"])
        meta["_source"].update(a.get("_source", {}))
        meta["_note"] = a.get("_note", meta["_note"])
        meta["_label"] = a.get("_label", "")
    return meta


def generator_profile(script):
    """Default profile of a generator (kdd28_params.json "_generators")."""
    return load_db()["_generators"][os.path.basename(script)]


def _fmt(v):
    if v is None:
        return ""
    if isinstance(v, bool):
        return "true" if v else "false"
    if isinstance(v, list):
        return " ".join(_fmt(x) for x in v)
    return str(v)


def export_lines(name, prefix="P_"):
    p = load_profile(name)
    if p.get("split_type") in SPLIT_FLAG:
        p["split_flag"] = SPLIT_FLAG[p["split_type"]]
    return "export " + " ".join(f"{prefix}{k.upper()}={shlex.quote(_fmt(v))}" for k, v in p.items())


def parse_effective(log_path):
    """key -> value of the last "effective-params:" line of a log ({} if none)."""
    line = None
    try:
        with open(log_path, errors="replace") as f:
            for ln in f:
                if "effective-params:" in ln:
                    line = ln
    except OSError:
        return {}
    if line is None:
        return {}
    return dict(re.findall(r"(\w+)=(\S+)", line.split("effective-params:", 1)[1]))


def stamp_values(name, log_path):
    """Stamp column values: the profile name, then what the binary reported (empty when it printed nothing)."""
    eff = parse_effective(log_path)
    return [name] + [eff.get(LOG_KEY[c], "") for c in STAMP[1:]]


def stamp_from_effective(name, eff):
    """Stamp dict from an already-parsed effective-params mapping (stamp column -> value)."""
    return {"profile": name} | {c: eff.get(LOG_KEY[c], "") for c in STAMP[1:]}


# ---------------------------------------------------------------- drift check (generators)

def _norm(v):
    if v is None:
        return ""
    if isinstance(v, float) and math.isnan(v):
        return ""
    return str(v).strip()


def _match(expected, got, ctx):
    """True / False, or None when the expectation cannot be evaluated for this row."""
    if isinstance(expected, list):
        res = [_match(e, got, ctx) for e in expected]
        return True if True in res else (None if None in res else False)
    if expected is None:
        return got in ("", "-")
    if got in ("", "-"):
        return False
    if isinstance(expected, bool):
        return got.lower() in (("true", "1") if expected else ("false", "0"))
    if isinstance(expected, (int, float)):
        try:
            g = float(got)
        except ValueError:
            return False
        if isinstance(expected, int):  # seed, trees, depth, counts: exact (rel_tol let 123457 pass for 123456)
            return g == expected
        return math.isclose(g, float(expected), rel_tol=1e-5, abs_tol=1e-9)
    if isinstance(expected, str) and "F" in expected and ("log" in expected or "//" in expected):
        if "F" not in ctx or "W" not in ctx:
            return None
        want = eval(expected, {"__builtins__": {}}, {"log": math.log, "max": max, **ctx})  # noqa: S307 (repo data)
        return _match(want, got, ctx)
    return got == str(expected)


def check_rows(rows, profile, allowed, label, arm_of=None, ctx_of=None, columns=None, row_columns=False):
    """Checks every stamp column present in `rows` (list of dicts) against the profile.

    allowed: arm names of `profile` a row may belong to ("" = the profile itself). A row with a "profile" value must
    name one of them; else arm_of(row) picks the arm; else the row must match at least one allowed arm.
    Prints "WARN <label>: unchecked: <fields>" for stamp columns absent from the CSV; exits 2 on any mismatch.
    row_columns: check each row only on the stamp columns it carries (eval.json rows of mixed vintages)."""
    db = load_db()
    names = [f"{profile}.{a}" if a else profile for a in allowed]
    try:
        exp = {n: load_profile(n, db) for n in names}
    except KeyError as e:
        print(f"ERROR {label}: --profile {profile} does not fit this input: {e.args[0]} "
              f"(needs arms {', '.join(a or '<base>' for a in allowed)})", file=sys.stderr)
        sys.exit(2)
    cols = list(columns if columns is not None else (rows[0].keys() if rows else []))
    _eval_rows(rows, label, exp)
    src = {c: (f"eff_{c}" if f"eff_{c}" in cols else c) for c in STAMP}  # eff_<c> = the stamp where <c> was taken
    present = [c for c in STAMP if src[c] in cols]
    absent = [c for c in STAMP if src[c] not in cols]
    if absent:
        print(f"WARN {label}: unchecked: {', '.join(absent)}", file=sys.stderr)
    if not rows:
        print(f"WARN {label}: no rows to check", file=sys.stderr)
        return
    bad, unchecked = [], set()
    for i, r in enumerate(rows):
        ctx = ctx_of(r) if ctx_of else {}
        named = _norm(r.get(src["profile"])) if "profile" in present and src["profile"] in r else ""
        if named:
            cands = [named] if named in exp else []
            if not cands:
                bad.append((i, r, f"profile {named!r} not in {names}"))
                continue
        elif arm_of is not None:
            a = arm_of(r)
            n = f"{profile}.{a}" if a else profile
            cands = [n] if n in exp else []
            if not cands:
                bad.append((i, r, f"arm {a!r} not in {names}"))
                continue
        else:
            cands = names
        errs_by = {}
        for n in cands:
            errs = []
            for c in present:
                if row_columns and src[c] not in r:
                    continue
                if c == "profile" or (c in RECORD_ONLY and FIELD[c] not in exp[n]):
                    continue
                m = _match(exp[n].get(FIELD[c]), _norm(r.get(src[c])), ctx)
                if m is None:
                    unchecked.add(c)
                elif not m:
                    errs.append(f"{src[c]}={_norm(r.get(src[c])) or '<empty>'} (profile {n}: {_fmt(exp[n].get(FIELD[c])) or 'null'})")
            errs_by[n] = errs
            if not errs:
                break
        else:
            n = min(errs_by, key=lambda k: len(errs_by[k]))
            bad.append((i, r, "; ".join(errs_by[n])))
    if unchecked:
        print(f"WARN {label}: unchecked (formula without F/W in the row): {', '.join(sorted(unchecked))}", file=sys.stderr)
    if bad:
        print(f"ERROR {label}: {len(bad)} of {len(rows)} rows drift from profile {profile} ({', '.join(names)}):",
              file=sys.stderr)
        keys = [k for k in ("dataset", "fold", "arm", "workers", "seed", "replication") if k in cols]
        for i, r, why in bad[:50]:
            print(f"  row {i + 2}: " + " ".join(f"{k}={_norm(r.get(k))}" for k in keys) + f": {why}", file=sys.stderr)
        if len(bad) > 50:
            print(f"  ... {len(bad) - 50} more", file=sys.stderr)
        sys.exit(2)
    if present:
        print(f"OK {label}: {len(rows)} rows match profile {profile} on {', '.join(present)}", file=sys.stderr)


# ---------------------------------------------------------------- eval suite of a generator (warnings only)

_EVAL = {"suite": None, "who": "", "seen": {}}


def _declare_eval(name, script):
    """argparse type of --eval_suite: records the generator's suite and notes once when it is not the reference."""
    if not name or name == "none":
        return None
    try:
        evalsuite.load_suite(name)
    except KeyError as e:
        import argparse
        raise argparse.ArgumentTypeError(e.args[0])
    if _EVAL["suite"] != name:  # parse_intermixed_args converts the default twice
        _EVAL.update(suite=name, who=os.path.basename(script))
        evalsuite.reference_note(name, _EVAL["who"])
    return name


def _eval_rows(rows, label, exp=None):
    """Eval-suite warnings for rows a generator checks (only when it declared a suite), for profile arms with a
    validation split or early stopping the suite has not, and when the inputs of this run mix eval_suite values."""
    if not _EVAL["suite"]:
        return
    spec = evalsuite.load_suite(_EVAL["suite"])
    for n, p in (exp or {}).items():
        vr, es = p.get("validation_set_ratio"), p.get("early_stopping")
        if spec.get("validation") == "none" and vr not in (None, 0):
            evalsuite.warn(f"{label}: profile {n} has validation_set_ratio={_fmt(vr)} (suite: none)")
        if spec.get("early_stopping") == "none" and es is not None and str(es).upper() != "NONE":
            evalsuite.warn(f"{label}: profile {n} has early_stopping={es} (suite: none)")
    evalsuite.check_rows(rows, spec, label, max_lines=10)
    seen = _EVAL["seen"]
    before = len(seen)
    for r in rows:
        k = evalsuite._v(r, "eval_suite") or "unstamped"
        seen[k] = seen.get(k, 0) + 1
    if len(seen) > 1 and len(seen) > before and before:
        evalsuite.warn(f"{_EVAL['who']}: table mixes eval suites across inputs: "
                       + ", ".join(f"{k} ({n} rows)" for k, n in seen.items()))


def check_frame(df, profile, allowed, label, arm_of=None, ctx_of=None):
    """check_rows for a pandas DataFrame."""
    check_rows(df.to_dict("records"), profile, allowed, label, arm_of, ctx_of, columns=list(df.columns))


def eval_stamp(e):
    """Stamp-like row of a Vertex cell's eval.json: effective_params (train.py since 2026-11-03), else trees + kwargs."""
    if e.get("effective_params"):
        row = {k: v for k, v in e["effective_params"].items() if k in STAMP}
    else:
        kw, spo = e.get("learner_kwargs", {}), e.get("split_axis") == "SPARSE_OBLIQUE"
        row = {"trees": e.get("num_trees"),
               "spo_exponent": kw.get("sparse_oblique_num_projections_exponent") if spo else "-",
               "spo_density": kw.get("sparse_oblique_projection_density_factor") if spo else "-"}
    return row | {"arm": e.get("arm"), "cols": e.get("cols"), "workers": e.get("workers")}


def check_evals(evals, profile, label, arms=("spo", "aa")):
    """check_rows over eval.json dicts (arm from the cell, F = cols, W = workers in the exponent formula)."""
    rows = [eval_stamp(e) for e in evals]
    cols = sorted({k for r in rows for k in r} - {"arm", "cols", "workers"})
    legacy = sum(1 for e in evals if not e.get("effective_params"))
    if legacy:
        print(f"WARN {label}: unchecked: {legacy} of {len(evals)} rows have no effective_params (eval.json before "
              "2026-11-03), checked on trees, spo_exponent, spo_density only", file=sys.stderr)
    check_rows(rows, profile, list(arms), label, arm_of=lambda r: r["arm"],
               ctx_of=lambda r: {"F": r["cols"], "W": r["workers"]}, columns=cols, row_columns=True)


def check_csv(path, profile, allowed, label=None, arm_of=None, ctx_of=None):
    import csv
    with open(path, newline="") as f:
        rd = csv.DictReader(f)
        rows = list(rd)
        check_rows(rows, profile, allowed, label or os.path.basename(path), arm_of, ctx_of, columns=rd.fieldnames or [])


# ---------------------------------------------------------------- drift check (run scripts)

DRIFT = "!drift"


def _value(v):
    try:
        return json.loads(v)
    except ValueError:
        return v


def effective_mismatches(profile, eff, ctx=None, expect=None):
    """'field=got (profile: expected)' per stamp field of `eff` (parsed effective-params) that drifts from `profile`.
    expect: stamp column -> value the caller chose (replaces the profile's). Undefined fields and formulas without
    F/W in ctx are skipped."""
    if not eff:
        return ["effective-params=<missing> (profile: the binary must print it)"]
    exp = load_profile(profile) | {FIELD[c]: v for c, v in (expect or {}).items()}
    out = []
    for c in STAMP[1:]:
        if FIELD[c] not in exp:
            continue
        got = _norm(eff.get(LOG_KEY[c]))
        if _match(exp[FIELD[c]], got, ctx or {}) is False:
            out.append(f"{c}={got or '<empty>'} (profile: {_fmt(exp[FIELD[c]]) or 'null'})")
    for field, key in CHECK_ONLY.items():
        if field in exp and _match(exp[field], _norm(eff.get(key)), ctx or {}) is False:
            out.append(f"{field}={_norm(eff.get(key)) or '<empty>'} (profile: {_fmt(exp[field]) or 'null'})")
    return out


def _drift_result(profile, mism, where):
    """'' on match; on drift exit 2, or with ALLOW_DRIFT=1 warn and return the !drift suffix."""
    if not mism:
        return ""
    for m in mism:
        print(f"DRIFT {where}: {m}", file=sys.stderr)
    if os.environ.get("ALLOW_DRIFT") == "1":
        print(f"WARN {where}: drift from profile {profile}, continuing (ALLOW_DRIFT=1), row stamped {profile}{DRIFT}",
              file=sys.stderr)
        return DRIFT
    print(f"STOP {where}: drift from profile {profile} (ALLOW_DRIFT=1 to continue)", file=sys.stderr)
    sys.exit(2)


def check_cell(profile, log_path, ctx=None, expect=None):
    """After a cell: '' or the !drift suffix (ALLOW_DRIFT=1); exits 2 on drift otherwise."""
    return _drift_result(profile, effective_mismatches(profile, parse_effective(log_path), ctx, expect), log_path)


def preflight(cmd, profile, log_path, ctx=None, expect=None):
    """Runs the first cell's argv with --print_params_only into log_path, then check_cell."""
    import subprocess
    with open(log_path, "w") as f:
        rc = subprocess.call(list(cmd) + ["--print_params_only"], stdout=f, stderr=subprocess.STDOUT)
    if rc != 0:
        print(f"STOP pre-flight {log_path}: rc={rc}", file=sys.stderr)
        sys.exit(2)
    return check_cell(profile, log_path, ctx, expect)


def check_kwargs(profile, kwargs, keymap, ctx=None, where="kwargs"):
    """Container scripts: learner kwargs vs the profile. keymap = stamp column -> kwarg name. '' or !drift."""
    exp = load_profile(profile)
    mism = []
    for c, k in keymap.items():
        if FIELD[c] not in exp:
            continue
        got = _norm(kwargs.get(k))
        if _match(exp[FIELD[c]], got, ctx or {}) is False:
            mism.append(f"{c}={got or '<empty>'} (profile: {_fmt(exp[FIELD[c]]) or 'null'})")
    return _drift_result(profile, mism, where)


# ---------------------------------------------------------------- setup line (bodies)

def _folds(f):
    if isinstance(f, list):
        return f"{len(f)} CV folds"
    return f or ""


def _seeds(s):
    if s is None:
        return ""
    if len(s) == 1:
        return f"seed {s[0]}" + (" (YDF default)" if s[0] == 123456 else "")
    if s == list(range(s[0], s[0] + len(s))):
        return f"seeds {s[0]}-{s[-1]}"
    return "seeds " + "/".join(map(str, s))


def _num(x):
    return f"{x:g}" if isinstance(x, (int, float)) and not isinstance(x, bool) else str(x)


def _proj(p):
    e, tot = p.get("spo_exponent"), p.get("spo_p_total")
    if e is None and tot is None:
        return "n/a (AA)"
    if tot is not None:
        return f"{tot} total" + (" (floor(F/W) per worker)" if p.get("budget") == "per_worker" else "")
    base = "F" if e == 1 else ("sqrt(F)" if e == 0.5 else f"F^{_num(e)}")
    m = p.get("spo_max_projections")
    if m is not None and m < INT_MAX:
        base = f"min({base},{m})"
    if p.get("workers") is None:
        return base
    if p.get("local_projection_count"):
        return f"{base.replace('F', 'F/W')} per worker (own columns)"
    return f"{base} total over W" if p.get("split_budget") else f"{base} per worker"


def _es_text(p):
    """Setup-line text for the early-stopping fields (proto enum names in the profile)."""
    es = p.get("early_stopping")
    if es is None:
        return ""
    if str(es).upper() == "NONE":
        return "no early stopping"
    ratio = p.get("validation_set_ratio")
    return f"early stopping ({es}" + (f", {ratio:.0%} validation)" if ratio else ")")


def _parts(p):
    """(key, text) parts of the setup line; empty texts are dropped."""
    es = p.get("early_stopping")
    parts = [("trees", f"{p['trees']} trees"), ("depth", f"depth {p['max_depth']}"), ("split", str(p["split_type"])),
             ("P", f"P={_proj(p)}"),
             ("density", f"density {_num(p['spo_density'])}" if p.get("spo_density") is not None else ""),
             ("aa", "+ axis-aligned splits" if p.get("include_axis_aligned") else ""),
             ("es", _es_text(p)),
             ("W", "W=" + "/".join(map(str, p["workers"])) if p.get("workers") else ""),
             ("folds", _folds(p.get("folds"))), ("seeds", _seeds(p.get("seeds")))]
    return dict(parts)


def setup_line(profile, arms=()):
    """'% setup: <trees> trees, depth <d>, <split>, P=<...>, <folds/seeds> (profile <name>)', then ' | arm: <diff>'."""
    bp = _parts(load_profile(profile))
    line = f"% setup: {', '.join(t for t in bp.values() if t)} (profile {profile})"
    for a in arms:
        if not a:
            continue
        ap = _parts(load_profile(f"{profile}.{a}"))
        diff = [ap[k] or ("single machine" if k == "W" else "") for k in ap if ap[k] != bp[k]]
        diff = [d for d in diff if d]
        line += f" | {a}: {', '.join(diff) if diff else 'same'}"
    if _EVAL["suite"]:
        line += f" | eval suite {_EVAL['suite']}"
    return line


def add_setup(text, line):
    """Inserts the setup comment after the body's leading % comment lines (at the top if there are none)."""
    lines = text.split("\n")
    i = 0
    while i < len(lines) and lines[i].startswith("%"):
        i += 1
    return "\n".join(lines[:i] + [line] + lines[i:])


def add_profile_arg(ap, script):
    ap.add_argument("--profile", default=generator_profile(script),
                    help="kdd28_params.json profile the input CSVs must match (default: %(default)s)")
    ap.add_argument("--eval_suite", default=evalsuite.load_db()["_generators"].get(os.path.basename(script), ""),
                    type=lambda s: _declare_eval(s, script),
                    help="eval_suites.json suite the table declares, warns on rows that differ (default: %(default)s)")


# ---------------------------------------------------------------- CLI

def _show(names):
    db = load_db()
    keys = ["trees", "max_depth", "min_examples", "split_type", "spo_exponent", "spo_max_projections", "spo_density",
            "split_budget", "local_projection_count", "workers", "folds", "seeds"]
    for n in names:
        p, m = load_profile(n, db), profile_meta(n, db)
        print(f"{n}: " + " ".join(f"{k}={_fmt(p.get(k)) or '-'}" for k in keys) + f" | floats: {', '.join(m['_floats']) or '-'}")


def selftest():
    """Matcher and check cases that once passed wrongly (verifier 2026-11-03). Exit 1 on any failure."""
    import contextlib
    import io
    cases = [("seed 123457 vs 123456", _match(123456, "123457", {}), False),
             ("max 2147480000 vs INT_MAX", _match(INT_MAX, "2147480000", {}), False),
             ("seed exact", _match(123456, "123456", {}), True), ("int as 300.0", _match(300, "300.0", {}), True),
             ("float tolerance", _match(1.5, "1.500001", {}), True), ("float off", _match(1.5, "1.6", {}), False),
             ("bool", _match(True, "true", {}), True), ("bool off", _match(False, "true", {}), False),
             ("seed list", _match([1, 2, 3], "2", {}), True), ("seed list off", _match([1, 2, 3], "20", {}), False)]

    def code(fn, *a, **k):
        with contextlib.redirect_stderr(io.StringIO()) as err:
            try:
                fn(*a, **k)
            except SystemExit as e:
                return e.code, err.getvalue()
        return 0, err.getvalue()
    rc, err = code(check_rows, [{"trees": "10"}], "table3", ["dist_split"], "t")
    cases.append(("missing arm exits 2 (no KeyError)", (rc, "dist_split" in err), (2, True)))
    new = {"arm": "spo", "cols": 1152, "workers": 16, "effective_params": {
        "trees": "10", "spo_exponent": "0.605695", "spo_density": "1.5", "spo_max_projections": "6000",
        "split_budget": "false", "local_projection_count": "false", "seed": "123456", "max_depth": "6",
        "min_examples": "5", "split_type": "EXACT"}}
    old = {"arm": "spo", "cols": 1152, "workers": 16, "num_trees": 10, "split_axis": "SPARSE_OBLIQUE",
           "learner_kwargs": {"sparse_oblique_num_projections_exponent": 0.605695,
                              "sparse_oblique_projection_density_factor": 1.5}}
    rc, err = code(check_evals, [new, old], "vertex_grid_pf", "e")
    cases.append(("legacy + new eval.json rows", (rc, "1 of 2 rows have no effective_params" in err), (0, True)))
    rc, _ = code(check_evals, [dict(new, workers=4)], "vertex_grid_pf", "e")
    cases.append(("eval.json exponent drift still caught", rc, 2))
    cases += _eval_selftest(code)
    bad = [(n, got, want) for n, got, want in cases if got != want]
    for n, got, want in cases:
        print(f"{'FAIL' if (n, got, want) in bad else 'PASS'} selftest {n}: {got}" + (f" (want {want})" if got != want else ""))
    sys.exit(1 if bad else 0)


def _eval_selftest(code):
    """Eval-suite cases: every mismatch kind warns (never exits), a matching table stays silent."""
    import hashlib
    sha = {f"{d}/fold{k}_{s}.csv": hashlib.sha256(f"{d}{k}{s}".encode()).hexdigest()
           for d, ks in (("A", (0, 1)), ("B", (0,))) for k in ks for s in ("train", "test")}
    spec = {"name": "t_suite", "datasets": {"A": [0, 1], "B": [0]}, "sha": sha, "seeds": [1], "metric_impl": "ydf_eval_v1",
            "files": {"train": "{dataset}/fold{fold}_train.csv", "test": "{dataset}/fold{fold}_test.csv"},
            "validation": "none", "early_stopping": "none"}
    good = [{"dataset": d, "fold": str(k), "seed": "1", "validation_set_ratio": "0", "early_stopping": "NONE",
             "eval_suite": "t_suite", "metric_impl": "ydf_eval_v1",
             "data_sha": evalsuite.data_sha(sha[f"{d}/fold{k}_train.csv"], sha[f"{d}/fold{k}_test.csv"])}
            for d, k in (("A", 0), ("A", 1), ("B", 0))]

    def warns(rows):
        rc, err = code(evalsuite.check_rows, rows, spec, "t")
        return rc, [ln for ln in err.splitlines() if ln.startswith(evalsuite.PREFIX)]

    def has(rows, text):
        rc, w = warns(rows)
        return rc == 0 and any(text in ln for ln in w)
    other = dict(good[1], data_sha=evalsuite.data_sha("0" * 64, sha["A/fold1_test.csv"]))
    cases = [("eval: matching table is silent", warns(good), (0, [])),
             ("eval: changed dataset warns", has(good[:2] + [dict(good[2], dataset="C")], "not in suite t_suite"), True),
             ("eval: changed fold hash warns", has([good[0], other, good[2]], "fold 1 data_sha="), True),
             ("eval: other metric_impl warns", has(good[:2] + [dict(good[2], metric_impl="sklearn_auc_v1")],
                                                   "metric_impl=sklearn_auc_v1"), True),
             ("eval: mixed-suite table warns", has(good[:2] + [dict(good[2], eval_suite="kdd28_cc18_fold0")],
                                                   "mixes eval suites"), True),
             ("eval: early stopping warns", has(good[:2] + [dict(good[2], early_stopping="LOSS_INCREASE")],
                                                "early_stopping=LOSS_INCREASE"), True),
             ("eval: missing fold warns", has(good[:2], "suite datasets missing: B"), True),
             ("eval: unstamped rows warn, no exit", has([{k: v for k, v in r.items() if k not in evalsuite.COLUMNS}
                                                         for r in good], "unstamped eval"), True),
             ("eval: header with eval columns is stamped", stamped_header(["a"] + STAMP + evalsuite.COLUMNS), True),
             ("eval: header without eval columns is stamped", stamped_header(["a"] + STAMP), True),
             ("eval: header without stamps refused", stamped_header(["a", "b"] + evalsuite.COLUMNS), False),
             ("eval: sha cache = hashlib", evalsuite.ShaCache().get_many([evalsuite.DB_PATH])[evalsuite.DB_PATH],
              evalsuite._sha_file(evalsuite.DB_PATH))]
    saved = dict(_EVAL)
    _EVAL.update(suite=spec, who="gen.py", seen={})
    rc1, _ = code(_eval_rows, good, "in1")
    rc2, err = code(_eval_rows, [{k: v for k, v in r.items() if k not in evalsuite.COLUMNS} for r in good], "in2")
    _EVAL.clear()
    _EVAL.update(saved)
    cases.append(("eval: generator mixing inputs warns", (rc1, rc2, "table mixes eval suites across inputs" in err),
                  (0, 0, True)))
    return cases


def main(argv):
    if len(argv) < 2:
        sys.exit(__doc__)
    cmd = argv[1]
    if cmd == "export":
        print(export_lines(argv[2], argv[3] if len(argv) > 3 else "P_"))
    elif cmd == "get":
        print(_fmt(load_profile(argv[2]).get(argv[3])))
    elif cmd == "stamp-header":
        print(",".join(stamp_columns(argv[2].split(",") if len(argv) > 2 else ())))
    elif cmd == "stamp":
        print(",".join(stamp_values(argv[2], argv[3])))
    elif cmd == "require-stamped":
        if os.path.exists(argv[2]):
            with open(argv[2]) as f:
                head = f.readline().strip().split(",")
            if not stamped_header(head):
                sys.exit(print(f"ERROR {argv[2]}: recorded CSV without the stamp columns ({','.join(STAMP)}); "
                               "write new runs to a new results path", file=sys.stderr) or 2)
    elif cmd == "check-effective":
        kv = [a.split("=", 1) for a in argv[4:]]
        bad = [k for k, _ in kv if k not in ("F", "W") and k not in FIELD]
        if bad:
            sys.exit(f"check-effective: unknown key(s) {bad}; use F=, W= or a stamp column ({', '.join(FIELD)})")
        ctx = {k: float(v) for k, v in kv if k in ("F", "W")}
        expect = {k: _value(v) for k, v in kv if k in FIELD}
        mism = effective_mismatches(argv[2], parse_effective(argv[3]), ctx, expect)
        for m in mism:
            print(m)
        sys.exit(2 if mism else 0)
    elif cmd == "check-eval":
        if len(argv) < 3:
            sys.exit("usage: params.py check-eval <csv> [--suite <name>]")
        db = evalsuite.load_db()
        suite = argv[argv.index("--suite") + 1] if "--suite" in argv else db["reference"]
        label = os.path.basename(os.path.dirname(os.path.abspath(argv[2]))) + "/" + os.path.basename(argv[2])
        evalsuite.reference_note(suite, label, db=db)
        evalsuite.check_csv(argv[2], suite, label)
    elif cmd == "selftest":
        selftest()
    elif cmd == "show":
        db = load_db()
        names = argv[2:] or [t + (f".{a}" if a else "") for t, p in db["profiles"].items() for a in [""] + list(p.get("arms", {}))]
        _show(names)
    else:
        sys.exit(f"unknown command {cmd}\n{__doc__}")


if __name__ == "__main__":
    main(sys.argv)
