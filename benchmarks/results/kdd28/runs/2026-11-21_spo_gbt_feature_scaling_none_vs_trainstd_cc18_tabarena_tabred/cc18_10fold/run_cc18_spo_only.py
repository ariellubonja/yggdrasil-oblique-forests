#!/usr/bin/env python3
"""run_kdd28_ydf_ec2.py --tables cc18, keeping only the cc18_spo arm (driver unchanged, its cells filtered here).
Usage: run_cc18_spo_only.py <driver.py> <driver args...>"""
import importlib.util, sys

spec = importlib.util.spec_from_file_location("drv", sys.argv[1])
drv = importlib.util.module_from_spec(spec)
spec.loader.exec_module(drv)
full = drv.build_cells
drv.build_cells = lambda a: [c for c in full(a) if c.table == "cc18" and c.key[2] == "cc18_spo"]
sys.argv = [sys.argv[1]] + sys.argv[2:]
drv.main()
