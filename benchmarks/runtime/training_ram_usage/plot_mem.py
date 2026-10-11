"""Plot process RSS over time from a memsample.sh CSV.
Usage: python3 plot_mem.py <samples.csv> <run.log> <out.png> [title]"""
import csv, re, sys
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

csv_path, log_path, out_png = sys.argv[1:4]
title = sys.argv[4] if len(sys.argv) > 4 else csv_path
t, rss, sysu = [], [], []
for r in csv.DictReader(open(csv_path)):
    if not r["rss_mb"]:          # last sample can race the process exit
        continue
    t.append(float(r["t_s"])); rss.append(float(r["rss_mb"]) / 1024); sysu.append(float(r["sys_used_mb"]) / 1024)
log = open(log_path).read()
pre = re.search(r"Loading/Init \(pre-train\): ([0-9.]+)s", log)
tb = re.search(r"Training block took: ([0-9.]+) s", log)
peak = max(rss); tpeak = t[rss.index(peak)]

fig, ax = plt.subplots(figsize=(9, 4.5), dpi=150)
ax.plot(t, rss, color="#2a78d6", lw=2, label="process RSS (VmRSS)")
ax.plot(t, sysu, color="#eb6834", lw=2, label="system used (MemTotal - MemAvailable)")
ax.set_xlabel("wall time since launch (s)"); ax.set_ylabel("memory (GiB)")
ax.set_title(title, fontsize=10)
left = tpeak < 0.5 * t[-1]
ax.annotate(f"peak RSS {peak:.0f} GiB @ {tpeak:.0f} s", (tpeak, peak), xytext=(12 if left else -120, -4),
            textcoords="offset points", fontsize=8, va="top")
if left: ax.set_xlim(left=-0.02 * t[-1])
note = []
if pre: note.append(f"pre-train load {float(pre.group(1)):.1f} s")
if tb: note.append(f"training block {float(tb.group(1)):.0f} s")
if note: ax.text(0.97 if left else 0.03, 0.93, "\n".join(note), ha="right" if left else "left", fontsize=8, color="#555", transform=ax.transAxes, va="top")
for s in ("top", "right"): ax.spines[s].set_visible(False)
ax.grid(axis="y", color="#e5e5e5", lw=0.6); ax.set_axisbelow(True)
ax.legend(frameon=False, fontsize=8, loc="lower right"); ax.set_ylim(bottom=0)
fig.tight_layout(); fig.savefig(out_png)
print(f"samples={len(t)} peak_rss_gib={peak:.1f} at {tpeak:.0f}s final_rss_gib={rss[-1]:.1f} duration={t[-1]:.0f}s")
