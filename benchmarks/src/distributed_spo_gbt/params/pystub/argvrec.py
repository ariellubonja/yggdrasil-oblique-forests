"""Test-only recorder: one JSON line per captured call into $ARGV_LOG."""
import json
import os
import re


def record(kind, payload):
    line = json.dumps({"kind": kind, **payload}, sort_keys=True, default=str)
    line = re.sub(r"(work/[^\"]*_)\d{6}\b", r"\1HHMMSS", line)  # run_vertex_barriers work_dir timestamp
    with open(os.environ["ARGV_LOG"], "a") as f:
        f.write(line + "\n")
