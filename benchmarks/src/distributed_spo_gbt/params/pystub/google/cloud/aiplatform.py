"""Test-only stub of google.cloud.aiplatform: records every CustomJob's worker pools, jobs finish at once."""
from types import SimpleNamespace

from argvrec import record


def init(**kw):
    pass


class CustomJob:
    def __init__(self, display_name, worker_pool_specs, **kw):
        self.display_name = display_name
        self.resource_name = "projects/p/locations/l/customJobs/0"
        self.state = SimpleNamespace(name="JOB_STATE_SUCCEEDED")
        self.error = None
        record("custom_job", {"display_name": display_name, "pools": worker_pool_specs})

    def submit(self):
        pass

    @staticmethod
    def list(filter=None):
        return []
