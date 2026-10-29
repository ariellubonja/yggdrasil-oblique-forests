import ast, sys, types, time, datetime
SRC = "/Users/ariellubonja/prog/randals_lab/oblique_forests/wt-colrep-speed/benchmarks/src/distributed_spo_gbt/vertex_build/train_cells.py"
tree = ast.parse(open(SRC).read())
cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "CacheLock")
mod = ast.Module(body=[cls], type_ignores=[])
# fake google libs
class PreconditionFailed(Exception): pass
class NotFound(Exception): pass
exc_mod = types.ModuleType("google.api_core.exceptions"); exc_mod.PreconditionFailed = PreconditionFailed; exc_mod.NotFound = NotFound
STORE = {}  # name -> (generation, updated_ts, content)
CLOCK = [0.0]; SLEEPS = []; GEN = [0]
class Blob:
  def __init__(self, bucket, name): self.name = name; self.generation = None; self.updated = None
  def exists(self): return self.name in STORE
  def upload_from_string(self, data, if_generation_match=None):
    if if_generation_match == 0 and self.name in STORE: raise PreconditionFailed()
    GEN[0] += 1; STORE[self.name] = (GEN[0], CLOCK[0], data); self.generation = GEN[0]
  def reload(self):
    if self.generation is not None:  # pinned generation, like the real client
      if self.name not in STORE or STORE[self.name][0] != self.generation: raise NotFound()
    if self.name not in STORE: raise NotFound()
    g, t, _ = STORE[self.name]; self.generation = g
    self.updated = datetime.datetime.fromtimestamp(t, datetime.timezone.utc)
  def delete(self, if_generation_match=None):
    if self.name not in STORE: raise NotFound()
    if if_generation_match is not None and STORE[self.name][0] != if_generation_match: raise PreconditionFailed()
    del STORE[self.name]
class Bucket:
  def blob(self, name): return Blob(self, name)
class Client:
  def bucket(self, name): return Bucket()
storage_mod = types.ModuleType("google.cloud.storage"); storage_mod.Client = Client
google = types.ModuleType("google"); cloud = types.ModuleType("google.cloud"); api_core = types.ModuleType("google.api_core")
sys.modules.update({"google": google, "google.cloud": cloud, "google.cloud.storage": storage_mod, "google.api_core": api_core, "google.api_core.exceptions": exc_mod})
fake_time = types.SimpleNamespace(time=lambda: CLOCK[0], sleep=lambda s: (SLEEPS.append(s), CLOCK.__setitem__(0, CLOCK[0] + s)))
ns = {"time": fake_time, "json": __import__("json"), "os": __import__("os"), "LOCK_STALE_S": 45 * 60}
exec(compile(mod, "cachelock", "exec"), ns)
CacheLock = ns["CacheLock"]
def reset(): STORE.clear(); CLOCK[0] = 1000.0; SLEEPS.clear()

# 1 free -> acquire; release deletes
reset(); L = CacheLock("gs://b/root/ds/key"); assert L.acquire_or_wait() is True and "root/ds/key/builder.lock" in STORE; L.release(); assert "root/ds/key/builder.lock" not in STORE; print("1 ok")
# 2 done exists -> False, no lock written
reset(); STORE["root/ds/key/dataset_cache/done"] = (1, 0, "done"); L = CacheLock("gs://b/root/ds/key"); assert L.acquire_or_wait() is False and "root/ds/key/builder.lock" not in STORE; print("2 ok")
# 3 other holds, finishes after 12 s (writes done, releases) -> waiter returns False, polled ~3 times, no spin
reset(); other = CacheLock("gs://b/root/ds/key"); assert other.acquire_or_wait()
orig_exists = Blob.exists
def exists_side(self):
  if self.name.endswith("/done") and CLOCK[0] >= 1012 and "root/ds/key/dataset_cache/done" not in STORE:
    STORE["root/ds/key/dataset_cache/done"] = (9, CLOCK[0], "done"); other.release()
  return orig_exists(self)
Blob.exists = exists_side
L = CacheLock("gs://b/root/ds/key"); r = L.acquire_or_wait(); Blob.exists = orig_exists
assert r is False and 2 <= len(SLEEPS) <= 4, (r, SLEEPS); print("3 ok", SLEEPS)
# 4 stale lock -> takeover
reset(); STORE["root/ds/key/builder.lock"] = (5, 1000.0 - 50 * 60, "{}"); L = CacheLock("gs://b/root/ds/key"); assert L.acquire_or_wait() is True and STORE["root/ds/key/builder.lock"][0] != 5 and len(SLEEPS) == 1; print("4 ok", SLEEPS)
# 5 lock replaced by another process mid-wait (generation changes twice), then that process finishes: no spin
reset(); STORE["root/ds/key/builder.lock"] = (5, 1000.0, "{}")
calls = [0]
def exists_side2(self):
  calls[0] += 1
  if self.name.endswith("/done"):
    if CLOCK[0] >= 1010 and STORE.get("root/ds/key/builder.lock", (0,))[0] == 5: GEN[0] += 1; STORE["root/ds/key/builder.lock"] = (GEN[0], CLOCK[0], "{}")  # takeover by B
    if CLOCK[0] >= 1020 and STORE.get("root/ds/key/builder.lock", (0,))[0] != 5: GEN[0] += 1; STORE["root/ds/key/builder.lock"] = (GEN[0], CLOCK[0], "{}")  # replaced again by C
    if CLOCK[0] >= 1040: STORE["root/ds/key/dataset_cache/done"] = (9, CLOCK[0], "done")
  return orig_exists(self)
Blob.exists = exists_side2
L = CacheLock("gs://b/root/ds/key"); r = L.acquire_or_wait(); Blob.exists = orig_exists
assert r is False and len(SLEEPS) <= 12 and calls[0] <= 15, (r, SLEEPS, calls); print("5 ok", len(SLEEPS), "sleeps,", calls[0], "exists calls")
# 6 lock vanishes between PreconditionFailed and reload -> retry after 1 s then acquire
reset(); STORE["root/ds/key/builder.lock"] = (5, 1000.0, "{}")
state = {"n": 0}
orig_reload = Blob.reload
def reload_side(self):
  if self.name.endswith("builder.lock") and state["n"] == 0:
    state["n"] = 1; del STORE["root/ds/key/builder.lock"]; raise NotFound()
  return orig_reload(self)
Blob.reload = reload_side
L = CacheLock("gs://b/root/ds/key"); r = L.acquire_or_wait(); Blob.reload = orig_reload
assert r is True and SLEEPS == [1], (r, SLEEPS); print("6 ok")
# 7 timeout after 2*LOCK_STALE_S when a lock is kept fresh forever (no stale takeover possible)
reset()
def exists_side3(self):
  if self.name.endswith("/done"): STORE["root/ds/key/builder.lock"] = (5, CLOCK[0], "{}")  # heartbeat
  return orig_exists(self)
Blob.exists = exists_side3
L = CacheLock("gs://b/root/ds/key")
try:
  L.acquire_or_wait(); print("7 FAIL no timeout")
except TimeoutError: print("7 ok timeout after", CLOCK[0] - 1000, "s")
Blob.exists = orig_exists
# 8 release when our own lock was taken over: must not delete the new holder's lock
reset(); A = CacheLock("gs://b/root/ds/key"); assert A.acquire_or_wait(); gA = STORE["root/ds/key/builder.lock"][0]
GEN[0] += 1; STORE["root/ds/key/builder.lock"] = (GEN[0], CLOCK[0], "{}")  # B took over
# real client: delete() on a Blob with a pinned generation sends ?generation=gA -> 404; our fake raises NotFound only when missing,
# so emulate: pinned generation mismatch -> NotFound
def delete_side(self, if_generation_match=None):
  if self.generation is not None and STORE.get(self.name, (None,))[0] != self.generation: raise NotFound()
  return Blob.__dict__["_orig_delete"](self, if_generation_match)
Blob._orig_delete = Blob.delete; Blob.delete = delete_side
A.release(); assert "root/ds/key/builder.lock" in STORE; print("8 ok")
print("ALL OK")
