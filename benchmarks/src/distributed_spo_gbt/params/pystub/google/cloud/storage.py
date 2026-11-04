"""Test-only stub of google.cloud.storage."""
from argvrec import record


class _Blob:
    def __init__(self, name):
        self.name = name

    def exists(self):
        return False

    def upload_from_string(self, data, content_type=None, **kw):
        record("gcs_upload", {"blob": self.name, "data": data if self.name.endswith(".json") else "<omitted>"})

    def delete(self, **kw):
        pass


class _Bucket:
    def blob(self, name):
        return _Blob(name)


class Client:
    def bucket(self, name):
        return _Bucket()
