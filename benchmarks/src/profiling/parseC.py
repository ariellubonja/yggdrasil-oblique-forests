#!/usr/bin/env python3
"""Method C parser: per-depth DW1 gather cache behaviour from callgrind dumps.

Each CALLGRIND_DUMP_STATS_AT("dw1_depth_N") emitted one dump part bracketing the
kernel call (instrumentation on only there). For the gather source line
(oblique_cpu_depthwise_1pass.cc: `float v = col[sel_ptr[i]]`) we read, via
callgrind_annotate: Dr (data reads), D1mr (L1 read misses), DLmr (last-level /
DRAM read misses).

Geometry: the gather line issues 2 reads/iter (sel_ptr[i], then col[..]) so the
iteration count N = Dr/2. sel_ptr is stride-1 (~N/16 L1 misses); the rest of D1mr
is the scattered col gather => distinct 64B lines touched. So
  useful_per_L1_line  = N / (D1mr - N/16)        # L1 line efficiency (matches Method A)
  useful_per_DRAM_line = N / DLmr                 # how many floats per line actually
                                                  # fetched from DRAM (DLmr>0 => the
                                                  # working set exceeds the LLC)
On HIGGS the kernel gathers across all ~28 projection feature-columns (~1.2GB) per
wide top-level node, far exceeding the ~109MB LLC, so DLmr is large at shallow
depths -- the real DRAM cost that Method A (geometry only) cannot show.
"""
import glob, os, re, subprocess, sys

CGDIR = sys.argv[1] if len(sys.argv) > 1 else os.path.expanduser("~/dw1_cachegrind_C/cgout")
GATHER_RE = re.compile(r"col\[sel_ptr\[i\]\]")
ACC_RE = re.compile(r"o\[i\]\s*\+=")

def trigger_depth(path):
    with open(path, errors="ignore") as f:
        for _ in range(60):
            line = f.readline()
            if not line:
                break
            m = re.search(r"Trigger:.*dw1_depth_(\d+)", line)
            if m:
                return int(m.group(1))
    return None

def annotate(path):
    """Return (events_list, {key: [counts]}) for the gather/acc source lines.

    Annotated source lines render each event column as '.', '<n>' or '<n> (pct%)'.
    Strip the parentheticals, then the first len(events) whitespace tokens are the
    event values ('.' -> 0); the remainder is the source text.
    """
    out = subprocess.run(
        ["callgrind_annotate", "--auto=yes", "--threshold=100", path],
        capture_output=True, text=True).stdout
    events = []
    rows = {}
    for ln in out.splitlines():
        m = re.match(r"Events shown:\s*(.+)$", ln)
        if m:
            events = m.group(1).split()
            continue
        if not events:
            continue
        if GATHER_RE.search(ln):
            key = "gather"
        elif ACC_RE.search(ln):
            key = "acc"
        else:
            continue
        stripped = re.sub(r"\([^)]*\)", "", ln)          # drop "(pct%)"
        vals = []
        for t in stripped.split()[:len(events)]:
            if t == ".":
                vals.append(0)
            elif t.replace(",", "").isdigit():
                vals.append(int(t.replace(",", "")))
            else:
                break                                    # reached source text
        if len(vals) == len(events):
            rows[key] = vals
    return events, rows

def main():
    parts = sorted(glob.glob(os.path.join(CGDIR, "callgrind.out.*")))
    if not parts:
        print(f"no dumps in {CGDIR}", file=sys.stderr)
        return
    recs = []
    for p in parts:
        d = trigger_depth(p)
        if d is None:
            continue
        events, rows = annotate(p)
        if "gather" not in rows or not events:
            continue
        idx = {e: i for i, e in enumerate(events)}
        g = rows["gather"]
        def get(name):
            i = idx.get(name)
            return g[i] if (i is not None and i < len(g)) else 0
        Dr, D1mr, DLmr = get("Dr"), get("D1mr"), get("DLmr")
        N = Dr / 2.0
        l1_lines = D1mr - N / 16.0
        upl_l1 = N / l1_lines if l1_lines > 0 else 0.0
        upl_dram = N / DLmr if DLmr > 0 else 0.0
        recs.append((d, N, D1mr, l1_lines, upl_l1, DLmr, upl_dram))
    recs.sort()
    print("depth,iters,D1mr,l1_lines,useful_per_L1_line,DLmr,useful_per_DRAM_line")
    for d, N, D1mr, l1, upl1, DLmr, upld in recs:
        print(f"{d},{N:.0f},{D1mr:.0f},{l1:.0f},{upl1:.4f},{DLmr:.0f},{upld:.4f}")

if __name__ == "__main__":
    main()
