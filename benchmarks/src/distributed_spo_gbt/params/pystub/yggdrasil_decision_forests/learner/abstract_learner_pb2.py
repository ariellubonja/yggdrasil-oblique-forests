"""Test-only proto stub: records attribute writes."""


class _Rec:
    def __init__(self):
        object.__setattr__(self, "_f", {})

    def __getattr__(self, k):
        if k.startswith("__"):
            raise AttributeError(k)
        return self._f.setdefault(k, _Rec())

    def __setattr__(self, k, v):
        self._f[k] = v

    @property
    def Extensions(self):
        return _Ext(self)

    def dump(self):
        return {k: (v.dump() if isinstance(v, _Rec) else v) for k, v in sorted(self._f.items())}


class _Ext:
    def __init__(self, rec):
        self.rec = rec

    def __getitem__(self, key):
        return self.rec._f.setdefault(f"ext:{key}", _Rec())


TrainingConfig = _Rec
