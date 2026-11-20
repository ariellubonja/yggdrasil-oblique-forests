"""Where the parts of the old fixed_reruns_2026-11-01/ folder live in benchmarks/results/kdd28 (moved 2026-11-04, runs/ + floats/ since 2026-11-20).
at(res, rel): rel is a path inside the old folder. res = None maps it to the new tree, a dir keeps the old layout."""
import os

REPO = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../../../.."))
K = os.path.join(REPO, "benchmarks/results/kdd28")
_A1, _O, _R = "floats/appendix-earlier-settings", "outdated", "runs"
MAP = {  # old relative path -> new path under K (longest match wins; anything else -> analysis/fixed_reruns_2026-11-01/)
    "paper_refs": f"{_O}/paper_refs_pre_7c0eebd3",
    **{s: f"{_O}/fixed_reruns_2026-11-01_stages/{s}" for s in (
        "table2/binfix", "table2/fullfix", "table3/binfix", "table8/binfix", "table8/binfix_numeric",
        "table8/fullfix_numeric", "table8/fullfix_rep")},
    "table2/fullfix_splitbudget": f"{_R}/2026-11-01_colrep_vertex_split_budget",
    "table3/fullfix": f"{_R}/2026-11-01_cross_network/table3_fullfix",
    "table3/fullfix_seeds": f"{_R}/2026-11-01_cross_network/table3_fullfix_seeds",
    "table8/fullfix": f"{_R}/2026-11-01_dist_accuracy_spo",
    "paper_inputs/table2": f"{_O}/fixed_reruns_2026-11-01_duplicates/paper_inputs_table2",
    "paper_inputs/table3_results.csv": f"{_O}/fixed_reruns_2026-11-01_duplicates/table3_results.csv",
    "paper_inputs/table3_results_5seed.csv": "floats/runtime/tab-cross-network/derived/table3_results_5seed.csv",
    "paper_inputs/table8_results.csv": f"{_A1}/tab-dist-accuracy/derived/table8_results.csv",
    "paper_inputs/table2_single10": f"{_R}/2026-11-01_colrep_single_spo_10seeds",
}


def at(res, rel):
    if res is not None:
        return os.path.join(res, rel)
    parts = rel.strip("/").split("/")
    for n in range(len(parts), 0, -1):
        key = "/".join(parts[:n])
        if key in MAP:
            return os.path.join(K, MAP[key], *parts[n:])
    return os.path.join(K, "analysis/fixed_reruns_2026-11-01", rel)
