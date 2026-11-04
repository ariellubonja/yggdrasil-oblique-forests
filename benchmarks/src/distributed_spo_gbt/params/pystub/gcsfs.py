"""Test-only stub of gcsfs: one fake dataset, nothing finished, writes recorded."""
from argvrec import record


class GCSFileSystem:
    def __init__(self, *a, **k):
        pass

    def exists(self, p):
        return p.endswith("/train.csv")

    def glob(self, p):
        return []

    def ls(self, p):
        return [p.rstrip("/") + "/argvcheck_ds"]

    def find(self, p):
        return []

    def invalidate_cache(self, p=None):
        pass

    def rm(self, p, recursive=False):
        raise FileNotFoundError(p)

    def get(self, src, dst):
        with open(dst, "w") as f:
            f.write("f0,f1,label\n0.1,0.2,1\n")

    def put(self, src, dst):
        pass

    def cat(self, p):
        return b"{}"

    def pipe(self, p, data):
        record("gcs_pipe", {"path": p, "data": data.decode() if p.endswith("eval.json") else "<omitted>"})
