#!/usr/bin/env python3
"""Download the TabReD benchmark (8 industry-grade datasets) into benchmarks/data/tabred/.

Thin wrapper around the upstream downloader (yandex-research/tabred), which pulls
the preprocessed .tabred archives from the Kaggle dataset `irubachev/tabred`
(and, for the competition-sourced datasets, from the competitions themselves).

PREREQUISITES
  1. uv installed (https://docs.astral.sh/uv) -- provides `uvx`.
  2. Kaggle API token at ~/.kaggle/kaggle.json (kaggle.com/settings ->
     "Create New API Token"), chmod 600.
  3. For the competition-sourced datasets (homesite-insurance, ecom-offers,
     homecredit-default, sberbank-housing) the competition rules must have been
     accepted once in the Kaggle web UI, otherwise the download 403s.

  python3 benchmarks/data/download_tabred_datasets.py [dataset ...]   # default: all

Layout produced per dataset (upstream preprocessed format):
  benchmarks/data/tabred/<dataset>/{info.json,x_num.npy[,x_cat.npy][,x_bin.npy],x_meta.npy,y.npy,splits/}
"""
import os
import subprocess
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
OUT_DIR = os.path.join(REPO_ROOT, "benchmarks", "data", "tabred")

DATASETS = [
    "homesite-insurance", "ecom-offers", "homecredit-default", "sberbank-housing",
    "cooking-time", "delivery-eta", "maps-routing", "weather",
]


def main():
    names = sys.argv[1:] or DATASETS
    os.makedirs(OUT_DIR, exist_ok=True)
    cmd = ["uvx", "tabred", "download", "--output-path", OUT_DIR, *names]
    print(" ".join(cmd), flush=True)
    sys.exit(subprocess.call(cmd))


if __name__ == "__main__":
    main()
