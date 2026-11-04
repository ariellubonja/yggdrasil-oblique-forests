"""Test-only (check_argv_equivalence.sh): fast sleeps, /tmp and /proc redirected into $ARGV_SANDBOX_TMP."""
import builtins
import io
import os
import time

_sleep, _open, _makedirs = time.sleep, builtins.open, os.makedirs
TMP = os.environ.get("ARGV_SANDBOX_TMP", "")
time.sleep = lambda s: _sleep(min(s, 0.002))


def _redirect(path):
    if TMP and isinstance(path, str) and path.startswith("/tmp/"):
        return os.path.join(TMP, path[len("/tmp/"):])
    return path


def _fake_open(path, *a, **k):
    if isinstance(path, str) and path == "/proc/cpuinfo":
        return io.StringIO("model name\t: stub\ncache size\t: 1 KB\ncpu MHz\t\t: 1\n")
    if isinstance(path, str) and path == "/proc/stat":
        return io.StringIO("cpu  1 1 1 1 1 0 0 0 0 0\ncpu0 1 1 1 1 1 0 0 0 0 0\nintr 0\n")
    return _open(_redirect(path), *a, **k)


def _fake_makedirs(path, *a, **k):
    return _makedirs(_redirect(path), *a, **k)


if TMP:
    builtins.open = _fake_open
    os.makedirs = _fake_makedirs
