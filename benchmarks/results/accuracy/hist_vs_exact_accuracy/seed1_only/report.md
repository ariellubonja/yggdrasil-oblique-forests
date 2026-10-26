# Split-finder accuracy comparison: Random Histogram vs Exact

Inputs: 54 CSVs, 36 datasets, seeds: [1]. Unit of analysis for all aggregate claims: the dataset (per-dataset mean of paired per-(seed,fold) deltas).

## SPO-GBT -- accuracy

### Dynamic_Random_Histogram_thresh250 vs Exact (positive delta = comparator better)

- Datasets: 36
- Mean per-dataset delta: -0.00059 (median -0.00021, range [-0.02756, +0.03519])
- 95% CI (t): [-0.00387, +0.00269]; 95% CI (bootstrap): [-0.00360, +0.00267]
- Wilcoxon signed-rank p = 0.3884; paired t p = 0.7175
- Win/tie/loss (|delta| <= 0.001 is a tie): 11/12/13 (raw sign: 11 up, 20 down)
- TOST equivalence p at margins {0.001: 0.4004, 0.0025: 0.1225, 0.005: 0.0049, 0.01: 6.6e-07, 0.02: 2.8e-14}
- **Equivalent (TOST) within +-0.005 at alpha=0.05**
- Non-inferiority p at margins {0.001: 0.4004, 0.0025: 0.1225, 0.005: 0.0049, 0.01: 6.6e-07, 0.02: 2.8e-14}
- **Non-inferior (no penalty worse than 0.005) at alpha=0.05**
- Leave-one-out (drop task_14954_cylinder-bands): mean -0.00161, 95% CI [-0.00423, +0.00101], TOST margin 0.005, non-inf margin 0.005
- Per-dataset Wilcoxon (Holm-corrected across datasets): no dataset with p < 0.05

### Random vs Exact (positive delta = comparator better)

- Datasets: 36
- Mean per-dataset delta: +0.00006 (median -0.00000, range [-0.01796, +0.03704])
- 95% CI (t): [-0.00257, +0.00269]; 95% CI (bootstrap): [-0.00218, +0.00284]
- Wilcoxon signed-rank p = 1.0000; paired t p = 0.9616
- Win/tie/loss (|delta| <= 0.001 is a tie): 14/10/12 (raw sign: 16 up, 18 down)
- TOST equivalence p at margins {0.001: 0.2373, 0.0025: 0.0342, 0.005: 0.0003, 0.01: 2.7e-09, 0.02: 1.9e-17}
- **Equivalent (TOST) within +-0.0025 at alpha=0.05**
- Non-inferiority p at margins {0.001: 0.2089, 0.0025: 0.0280, 0.005: 0.0002, 0.01: 2.0e-09, 0.02: 0.0e+00}
- **Non-inferior (no penalty worse than 0.0025) at alpha=0.05**
- Leave-one-out (drop task_14954_cylinder-bands): mean -0.00099, 95% CI [-0.00257, +0.00058], TOST margin 0.0025, non-inf margin 0.0025
- Per-dataset Wilcoxon (Holm-corrected across datasets): no dataset with p < 0.05

### Random_bins128 vs Exact (positive delta = comparator better)

- Datasets: 34
- Mean per-dataset delta: +0.00025 (median +0.00000, range [-0.01406, +0.02778])
- 95% CI (t): [-0.00215, +0.00266]; 95% CI (bootstrap): [-0.00192, +0.00265]
- Wilcoxon signed-rank p = 0.9615; paired t p = 0.8316
- Win/tie/loss (|delta| <= 0.001 is a tie): 12/11/11 (raw sign: 15 up, 16 down)
- TOST equivalence p at margins {0.001: 0.2662, 0.0025: 0.0332, 0.005: 0.0002, 0.01: 8.1e-10, 0.02: 5.9e-18}
- **Equivalent (TOST) within +-0.0025 at alpha=0.05**
- Non-inferiority p at margins {0.001: 0.1485, 0.0025: 0.0131, 0.005: 4.7e-05, 0.01: 2.6e-10, 0.02: 0.0e+00}
- **Non-inferior (no penalty worse than 0.0025) at alpha=0.05**
- Leave-one-out (drop task_14954_cylinder-bands): mean -0.00058, 95% CI [-0.00234, +0.00118], TOST margin 0.0025, non-inf margin 0.0025
- Per-dataset Wilcoxon (Holm-corrected across datasets): no dataset with p < 0.05

### Random_bins16 vs Exact (positive delta = comparator better)

- Datasets: 34
- Mean per-dataset delta: +0.00114 (median +0.00000, range [-0.02635, +0.03519])
- 95% CI (t): [-0.00182, +0.00410]; 95% CI (bootstrap): [-0.00156, +0.00405]
- Wilcoxon signed-rank p = 0.4540; paired t p = 0.4378
- Win/tie/loss (|delta| <= 0.001 is a tie): 12/10/12 (raw sign: 16 up, 16 down)
- TOST equivalence p at margins {0.001: 0.5386, 0.0025: 0.1785, 0.005: 0.0061, 0.01: 3.7e-07, 0.02: 8.3e-15}
- **Equivalent (TOST) within +-0.005 at alpha=0.05**
- Non-inferiority p at margins {0.001: 0.0751, 0.0025: 0.0087, 0.005: 8.8e-05, 0.01: 4.0e-09, 0.02: 3.3e-16}
- **Non-inferior (no penalty worse than 0.0025) at alpha=0.05**
- Leave-one-out (drop task_14954_cylinder-bands): mean +0.00011, 95% CI [-0.00204, +0.00226], TOST margin 0.0025, non-inf margin 0.0025
- Per-dataset Wilcoxon (Holm-corrected across datasets): no dataset with p < 0.05

### Random_bins256 vs Exact (positive delta = comparator better)

- Datasets: 34
- Mean per-dataset delta: +0.00077 (median +0.00000, range [-0.01216, +0.04259])
- 95% CI (t): [-0.00220, +0.00374]; 95% CI (bootstrap): [-0.00161, +0.00391]
- Wilcoxon signed-rank p = 0.8393; paired t p = 0.6018
- Win/tie/loss (|delta| <= 0.001 is a tie): 11/13/10 (raw sign: 15 up, 16 down)
- TOST equivalence p at margins {0.001: 0.4379, 0.0025: 0.1224, 0.005: 0.0033, 0.01: 1.9e-07, 0.02: 5.5e-15}
- **Equivalent (TOST) within +-0.005 at alpha=0.05**
- Non-inferiority p at margins {0.001: 0.1172, 0.0025: 0.0160, 0.005: 0.0002, 0.01: 9.1e-09, 0.02: 6.7e-16}
- **Non-inferior (no penalty worse than 0.0025) at alpha=0.05**
- Leave-one-out (drop task_14954_cylinder-bands): mean -0.00050, 95% CI [-0.00202, +0.00103], TOST margin 0.0025, non-inf margin 0.0025
- Per-dataset Wilcoxon (Holm-corrected across datasets): no dataset with p < 0.05

### Random_bins32 vs Exact (positive delta = comparator better)

- Datasets: 34
- Mean per-dataset delta: -0.00014 (median +0.00000, range [-0.02205, +0.02222])
- 95% CI (t): [-0.00268, +0.00240]; 95% CI (bootstrap): [-0.00259, +0.00223]
- Wilcoxon signed-rank p = 0.7611; paired t p = 0.9128
- Win/tie/loss (|delta| <= 0.001 is a tie): 12/11/11 (raw sign: 15 up, 15 down)
- TOST equivalence p at margins {0.001: 0.2474, 0.0025: 0.0337, 0.005: 0.0002, 0.01: 2.1e-09, 0.02: 1.6e-17}
- **Equivalent (TOST) within +-0.0025 at alpha=0.05**
- Non-inferiority p at margins {0.001: 0.2474, 0.0025: 0.0337, 0.005: 0.0002, 0.01: 2.1e-09, 0.02: 0.0e+00}
- **Non-inferior (no penalty worse than 0.0025) at alpha=0.05**
- Leave-one-out (drop task_14954_cylinder-bands): mean -0.00082, 95% CI [-0.00302, +0.00139], TOST margin 0.005, non-inf margin 0.005
- Per-dataset Wilcoxon (Holm-corrected across datasets): no dataset with p < 0.05

- Friedman over 34 datasets: chi2 = 4.13, p = 0.6587; average ranks (1 = best): Dynamic_Random_Histogram_thresh250: 4.34, Exact: 3.88, Random: 4.09, Random_bins128: 4.32, Random_bins16: 3.68, Random_bins256: 3.59, Random_bins32: 4.10

## SPO-GBT -- auc

### Dynamic_Random_Histogram_thresh250 vs Exact (positive delta = comparator better)

- Datasets: 36
- Mean per-dataset delta: +0.00052 (median -0.00016, range [-0.01036, +0.03842])
- 95% CI (t): [-0.00205, +0.00310]; 95% CI (bootstrap): [-0.00152, +0.00334]
- Wilcoxon signed-rank p = 0.5722; paired t p = 0.6831
- Win/tie/loss (|delta| <= 0.001 is a tie): 9/16/11 (raw sign: 13 up, 20 down)
- TOST equivalence p at margins {0.001: 0.3544, 0.0025: 0.0640, 0.005: 0.0006, 0.01: 4.8e-09, 0.02: 2.0e-17}
- **Equivalent (TOST) within +-0.005 at alpha=0.05**
- Non-inferiority p at margins {0.001: 0.1191, 0.0025: 0.0114, 0.005: 5.6e-05, 0.01: 4.4e-10, 0.02: 0.0e+00}
- **Non-inferior (no penalty worse than 0.0025) at alpha=0.05**
- Leave-one-out (drop task_14954_cylinder-bands): mean -0.00056, 95% CI [-0.00194, +0.00082], TOST margin 0.0025, non-inf margin 0.0025
- Per-dataset Wilcoxon (Holm-corrected across datasets): no dataset with p < 0.05

### Random vs Exact (positive delta = comparator better)

- Datasets: 36
- Mean per-dataset delta: +0.00157 (median +0.00000, range [-0.01054, +0.06006])
- 95% CI (t): [-0.00206, +0.00520]; 95% CI (bootstrap): [-0.00105, +0.00558]
- Wilcoxon signed-rank p = 0.8050; paired t p = 0.3863
- Win/tie/loss (|delta| <= 0.001 is a tie): 11/14/11 (raw sign: 17 up, 16 down)
- TOST equivalence p at margins {0.001: 0.6240, 0.0025: 0.3032, 0.005: 0.0317, 0.01: 1.9e-05, 0.02: 1.9e-12}
- **Equivalent (TOST) within +-0.005 at alpha=0.05**
- Non-inferiority p at margins {0.001: 0.0799, 0.0025: 0.0146, 0.005: 0.0004, 0.01: 9.4e-08, 0.02: 2.6e-14}
- **Non-inferior (no penalty worse than 0.0025) at alpha=0.05**
- Leave-one-out (drop task_14954_cylinder-bands): mean -0.00010, 95% CI [-0.00144, +0.00124], TOST margin 0.0025, non-inf margin 0.0025
- Per-dataset Wilcoxon (Holm-corrected across datasets): no dataset with p < 0.05

### Random_bins128 vs Exact (positive delta = comparator better)

- Datasets: 34
- Mean per-dataset delta: -0.00003 (median +0.00000, range [-0.00949, +0.01449])
- 95% CI (t): [-0.00144, +0.00139]; 95% CI (bootstrap): [-0.00135, +0.00137]
- Wilcoxon signed-rank p = 0.8696; paired t p = 0.9700
- Win/tie/loss (|delta| <= 0.001 is a tie): 10/16/8 (raw sign: 15 up, 16 down)
- TOST equivalence p at margins {0.001: 0.0853, 0.0025: 0.0006, 0.005: 1.7e-08, 0.01: 4.4e-16, 0.02: 2.8e-25}
- **Equivalent (TOST) within +-0.0025 at alpha=0.05**
- Non-inferiority p at margins {0.001: 0.0853, 0.0025: 0.0006, 0.005: 1.7e-08, 0.01: 4.4e-16, 0.02: 0.0e+00}
- **Non-inferior (no penalty worse than 0.0025) at alpha=0.05**
- Leave-one-out (drop task_9971_ilpd): mean -0.00047, 95% CI [-0.00160, +0.00066], TOST margin 0.0025, non-inf margin 0.0025
- Per-dataset Wilcoxon (Holm-corrected across datasets): no dataset with p < 0.05

### Random_bins16 vs Exact (positive delta = comparator better)

- Datasets: 34
- Mean per-dataset delta: +0.00081 (median +0.00019, range [-0.01634, +0.02603])
- 95% CI (t): [-0.00131, +0.00292]; 95% CI (bootstrap): [-0.00114, +0.00291]
- Wilcoxon signed-rank p = 0.0727; paired t p = 0.4419
- Win/tie/loss (|delta| <= 0.001 is a tie): 10/20/4 (raw sign: 18 up, 13 down)
- TOST equivalence p at margins {0.001: 0.4276, 0.0025: 0.0566, 0.005: 0.0002, 0.01: 1.6e-10, 0.02: 2.9e-19}
- **Equivalent (TOST) within +-0.005 at alpha=0.05**
- Non-inferiority p at margins {0.001: 0.0455, 0.0025: 0.0016, 0.005: 1.6e-06, 0.01: 3.0e-12, 0.02: 0.0e+00}
- **Non-inferior (no penalty worse than 0.001) at alpha=0.05**
- Leave-one-out (drop task_9971_ilpd): mean +0.00004, 95% CI [-0.00143, +0.00152], TOST margin 0.0025, non-inf margin 0.0025
- Per-dataset Wilcoxon (Holm-corrected across datasets): no dataset with p < 0.05

### Random_bins256 vs Exact (positive delta = comparator better)

- Datasets: 34
- Mean per-dataset delta: +0.00316 (median +0.00062, range [-0.00681, +0.06774])
- 95% CI (t): [-0.00100, +0.00733]; 95% CI (bootstrap): [+0.00030, +0.00767]
- Wilcoxon signed-rank p = 0.0305; paired t p = 0.1321
- Win/tie/loss (|delta| <= 0.001 is a tie): 14/15/5 (raw sign: 20 up, 11 down)
- TOST equivalence p at margins {0.001: 0.8507, 0.0025: 0.6260, 0.005: 0.1882, 0.01: 0.0011, 0.02: 8.6e-10}
- **Equivalent (TOST) within +-0.01 at alpha=0.05**
- Non-inferiority p at margins {0.001: 0.0251, 0.0025: 0.0046, 0.005: 0.0002, 0.01: 1.4e-07, 0.02: 3.4e-13}
- **Non-inferior (no penalty worse than 0.001) at alpha=0.05**
- Leave-one-out (drop task_14954_cylinder-bands): mean +0.00121, 95% CI [-0.00006, +0.00248], TOST margin 0.0025, non-inf margin 0.001
- Per-dataset Wilcoxon (Holm-corrected across datasets): no dataset with p < 0.05

### Random_bins32 vs Exact (positive delta = comparator better)

- Datasets: 34
- Mean per-dataset delta: +0.00065 (median +0.00002, range [-0.01403, +0.01765])
- 95% CI (t): [-0.00127, +0.00256]; 95% CI (bootstrap): [-0.00119, +0.00250]
- Wilcoxon signed-rank p = 0.2724; paired t p = 0.4966
- Win/tie/loss (|delta| <= 0.001 is a tie): 11/16/7 (raw sign: 18 up, 13 down)
- TOST equivalence p at margins {0.001: 0.3551, 0.0025: 0.0287, 0.005: 2.8e-05, 0.01: 9.5e-12, 0.02: 1.1e-20}
- **Equivalent (TOST) within +-0.0025 at alpha=0.05**
- Non-inferiority p at margins {0.001: 0.0447, 0.0025: 0.0010, 0.005: 4.8e-07, 0.01: 3.4e-13, 0.02: 0.0e+00}
- **Non-inferior (no penalty worse than 0.001) at alpha=0.05**
- Leave-one-out (drop task_14954_cylinder-bands): mean +0.00013, 95% CI [-0.00152, +0.00179], TOST margin 0.0025, non-inf margin 0.0025
- Per-dataset Wilcoxon (Holm-corrected across datasets): no dataset with p < 0.05

- Friedman over 34 datasets: chi2 = 6.87, p = 0.3329; average ranks (1 = best): Dynamic_Random_Histogram_thresh250: 4.41, Exact: 4.24, Random: 4.12, Random_bins128: 4.15, Random_bins16: 3.76, Random_bins256: 3.26, Random_bins32: 4.06

## SPO-GBT -- logloss

### Dynamic_Random_Histogram_thresh250 vs Exact (negative delta = comparator better)

- Datasets: 36
- Mean per-dataset delta: -0.00002 (median +0.00000, range [-0.02891, +0.05840])
- 95% CI (t): [-0.00517, +0.00514]; 95% CI (bootstrap): [-0.00455, +0.00526]
- Wilcoxon signed-rank p = 0.3880; paired t p = 0.9952
- Win/tie/loss (|delta| <= 0.001 is a tie): 10/10/16 (raw sign: 17 up, 17 down)
- TOST equivalence p at margins {0.001: 0.3502, 0.0025: 0.1671, 0.005: 0.0287, 0.01: 0.0002, 0.02: 1.5e-09}
- **Equivalent (TOST) within +-0.005 at alpha=0.05**
- Non-inferiority p at margins {0.001: 0.3457, 0.0025: 0.1642, 0.005: 0.0280, 0.01: 0.0002, 0.02: 1.4e-09}
- **Non-inferior (no penalty worse than 0.005) at alpha=0.05**
- Leave-one-out (drop task_15_breast-w): mean -0.00168, 95% CI [-0.00568, +0.00231], TOST margin 0.01, non-inf margin 0.0025
- Per-dataset Wilcoxon (Holm-corrected across datasets): no dataset with p < 0.05

### Random vs Exact (negative delta = comparator better)

- Datasets: 36
- Mean per-dataset delta: -0.00294 (median -0.00063, range [-0.07674, +0.05920])
- 95% CI (t): [-0.01009, +0.00421]; 95% CI (bootstrap): [-0.00974, +0.00398]
- Wilcoxon signed-rank p = 0.1126; paired t p = 0.4099
- Win/tie/loss (|delta| <= 0.001 is a tie): 9/9/18 (raw sign: 15 up, 19 down)
- TOST equivalence p at margins {0.001: 0.7072, 0.0025: 0.5493, 0.005: 0.2813, 0.01: 0.0265, 0.02: 1.3e-05}
- **Equivalent (TOST) within +-0.01 at alpha=0.05**
- Non-inferiority p at margins {0.001: 0.1356, 0.0025: 0.0659, 0.005: 0.0153, 0.01: 0.0004, 0.02: 8.3e-08}
- **Non-inferior (no penalty worse than 0.005) at alpha=0.05**
- Leave-one-out (drop task_9971_ilpd): mean -0.00083, 95% CI [-0.00674, +0.00507], TOST margin 0.01, non-inf margin 0.005
- Per-dataset Wilcoxon (Holm-corrected across datasets): no dataset with p < 0.05

### Random_bins128 vs Exact (negative delta = comparator better)

- Datasets: 34
- Mean per-dataset delta: +0.00029 (median -0.00172, range [-0.06392, +0.05692])
- 95% CI (t): [-0.00659, +0.00717]; 95% CI (bootstrap): [-0.00619, +0.00704]
- Wilcoxon signed-rank p = 0.1438; paired t p = 0.9327
- Win/tie/loss (|delta| <= 0.001 is a tie): 8/7/19 (raw sign: 12 up, 20 down)
- TOST equivalence p at margins {0.001: 0.4172, 0.0025: 0.2588, 0.005: 0.0864, 0.01: 0.0035, 0.02: 8.0e-07}
- **Equivalent (TOST) within +-0.01 at alpha=0.05**
- Non-inferiority p at margins {0.001: 0.4172, 0.0025: 0.2588, 0.005: 0.0864, 0.01: 0.0035, 0.02: 8.0e-07}
- **Non-inferior (no penalty worse than 0.01) at alpha=0.05**
- Leave-one-out (drop task_9971_ilpd): mean +0.00223, 95% CI [-0.00357, +0.00804], TOST margin 0.01, non-inf margin 0.01
- Per-dataset Wilcoxon (Holm-corrected across datasets): no dataset with p < 0.05

### Random_bins16 vs Exact (negative delta = comparator better)

- Datasets: 34
- Mean per-dataset delta: -0.01088 (median -0.00277, range [-0.15225, +0.05846])
- 95% CI (t): [-0.02407, +0.00232]; 95% CI (bootstrap): [-0.02392, +0.00118]
- Wilcoxon signed-rank p = 0.0253; paired t p = 0.1029
- Win/tie/loss (|delta| <= 0.001 is a tie): 7/6/21 (raw sign: 8 up, 24 down)
- TOST equivalence p at margins {0.001: 0.9314, 0.0025: 0.8973, 0.005: 0.8143, 0.01: 0.5534, 0.02: 0.0844}
- **Not TOST-equivalent at any tested margin**
- Non-inferiority p at margins {0.001: 0.0380, 0.0025: 0.0235, 0.005: 0.0099, 0.01: 0.0014, 0.02: 1.9e-05}
- **Non-inferior (no penalty worse than 0.001) at alpha=0.05**
- Leave-one-out (drop task_9971_ilpd): mean -0.00659, 95% CI [-0.01681, +0.00363], TOST margin 0.02, non-inf margin 0.0025
- Per-dataset Wilcoxon (Holm-corrected across datasets): no dataset with p < 0.05

### Random_bins256 vs Exact (negative delta = comparator better)

- Datasets: 34
- Mean per-dataset delta: -0.00283 (median -0.00123, range [-0.06060, +0.04903])
- 95% CI (t): [-0.00921, +0.00354]; 95% CI (bootstrap): [-0.00890, +0.00320]
- Wilcoxon signed-rank p = 0.1024; paired t p = 0.3728
- Win/tie/loss (|delta| <= 0.001 is a tie): 9/7/18 (raw sign: 11 up, 21 down)
- TOST equivalence p at margins {0.001: 0.7186, 0.0025: 0.5418, 0.005: 0.2469, 0.01: 0.0144, 0.02: 2.3e-06}
- **Equivalent (TOST) within +-0.01 at alpha=0.05**
- Non-inferiority p at margins {0.001: 0.1151, 0.0025: 0.0492, 0.005: 0.0088, 0.01: 0.0001, 0.02: 1.2e-08}
- **Non-inferior (no penalty worse than 0.0025) at alpha=0.05**
- Leave-one-out (drop task_9971_ilpd): mean -0.00108, 95% CI [-0.00654, +0.00438], TOST margin 0.01, non-inf margin 0.005
- Per-dataset Wilcoxon (Holm-corrected across datasets): no dataset with p < 0.05

### Random_bins32 vs Exact (negative delta = comparator better)

- Datasets: 34
- Mean per-dataset delta: -0.00573 (median -0.00280, range [-0.09427, +0.05801])
- 95% CI (t): [-0.01514, +0.00368]; 95% CI (bootstrap): [-0.01462, +0.00324]
- Wilcoxon signed-rank p = 0.0521; paired t p = 0.2242
- Win/tie/loss (|delta| <= 0.001 is a tie): 8/4/22 (raw sign: 10 up, 22 down)
- TOST equivalence p at margins {0.001: 0.8430, 0.0025: 0.7550, 0.005: 0.5621, 0.01: 0.1813, 0.02: 0.0020}
- **Equivalent (TOST) within +-0.02 at alpha=0.05**
- Non-inferiority p at margins {0.001: 0.0776, 0.0025: 0.0422, 0.005: 0.0133, 0.01: 0.0009, 0.02: 1.8e-06}
- **Non-inferior (no penalty worse than 0.0025) at alpha=0.05**
- Leave-one-out (drop task_9971_ilpd): mean -0.00305, 95% CI [-0.01096, +0.00487], TOST margin 0.01, non-inf margin 0.005
- Per-dataset Wilcoxon (Holm-corrected across datasets): no dataset with p < 0.05

- Friedman over 34 datasets: chi2 = 23.40, p = 0.0007; average ranks (1 = best): Dynamic_Random_Histogram_thresh250: 4.65, Exact: 4.79, Random: 4.24, Random_bins128: 4.32, Random_bins16: 2.94, Random_bins256: 3.88, Random_bins32: 3.18

## SPO-RF -- accuracy

### Dynamic_Random_Histogram_thresh250 vs Exact (positive delta = comparator better)

- Datasets: 36
- Mean per-dataset delta: +0.00484 (median +0.00007, range [-0.00700, +0.15000])
- 95% CI (t): [-0.00366, +0.01335]; 95% CI (bootstrap): [-0.00015, +0.01379]
- Wilcoxon signed-rank p = 0.1383; paired t p = 0.2552
- Win/tie/loss (|delta| <= 0.001 is a tie): 13/16/7 (raw sign: 22 up, 12 down)
- TOST equivalence p at margins {0.001: 0.8176, 0.0025: 0.7104, 0.005: 0.4853, 0.01: 0.1133, 0.02: 0.0005}
- **Equivalent (TOST) within +-0.02 at alpha=0.05**
- Non-inferiority p at margins {0.001: 0.0858, 0.0025: 0.0441, 0.005: 0.0122, 0.01: 0.0006, 0.02: 4.7e-07}
- **Non-inferior (no penalty worse than 0.0025) at alpha=0.05**
- Leave-one-out (drop task_14954_cylinder-bands): mean +0.00070, 95% CI [-0.00052, +0.00191], TOST margin 0.0025, non-inf margin 0.001
- Per-dataset Wilcoxon (Holm-corrected across datasets): no dataset with p < 0.05

### Random vs Exact (positive delta = comparator better)

- Datasets: 36
- Mean per-dataset delta: +0.00742 (median +0.00076, range [-0.00922, +0.20926])
- 95% CI (t): [-0.00442, +0.01925]; 95% CI (bootstrap): [+0.00046, +0.01984]
- Wilcoxon signed-rank p = 0.0286; paired t p = 0.2116
- Win/tie/loss (|delta| <= 0.001 is a tie): 16/13/7 (raw sign: 23 up, 11 down)
- TOST equivalence p at margins {0.001: 0.8608, 0.0025: 0.7977, 0.005: 0.6595, 0.01: 0.3301, 0.02: 0.0189}
- **Equivalent (TOST) within +-0.02 at alpha=0.05**
- Non-inferiority p at margins {0.001: 0.0788, 0.0025: 0.0489, 0.005: 0.0201, 0.01: 0.0026, 0.02: 2.0e-05}
- **Non-inferior (no penalty worse than 0.0025) at alpha=0.05**
- Leave-one-out (drop task_14954_cylinder-bands): mean +0.00165, 95% CI [-0.00012, +0.00342], TOST margin 0.005, non-inf margin 0.001
- Per-dataset Wilcoxon (Holm-corrected across datasets): no dataset with p < 0.05

### Random_bins128 vs Exact (positive delta = comparator better)

- Datasets: 34
- Mean per-dataset delta: +0.00640 (median +0.00005, range [-0.00923, +0.20741])
- 95% CI (t): [-0.00606, +0.01886]; 95% CI (bootstrap): [-0.00062, +0.01919]
- Wilcoxon signed-rank p = 0.1252; paired t p = 0.3037
- Win/tie/loss (|delta| <= 0.001 is a tie): 12/16/6 (raw sign: 20 up, 11 down)
- TOST equivalence p at margins {0.001: 0.8078, 0.0025: 0.7356, 0.005: 0.5896, 0.01: 0.2803, 0.02: 0.0167}
- **Equivalent (TOST) within +-0.02 at alpha=0.05**
- Non-inferiority p at margins {0.001: 0.1178, 0.0025: 0.0778, 0.005: 0.0358, 0.01: 0.0057, 0.02: 6.9e-05}
- **Non-inferior (no penalty worse than 0.005) at alpha=0.05**
- Leave-one-out (drop task_14954_cylinder-bands): mean +0.00031, 95% CI [-0.00103, +0.00165], TOST margin 0.0025, non-inf margin 0.001
- Per-dataset Wilcoxon (Holm-corrected across datasets): no dataset with p < 0.05

### Random_bins16 vs Exact (positive delta = comparator better)

- Datasets: 34
- Mean per-dataset delta: +0.00618 (median +0.00007, range [-0.01472, +0.19815])
- 95% CI (t): [-0.00574, +0.01809]; 95% CI (bootstrap): [-0.00067, +0.01847]
- Wilcoxon signed-rank p = 0.2167; paired t p = 0.2993
- Win/tie/loss (|delta| <= 0.001 is a tie): 12/16/6 (raw sign: 20 up, 11 down)
- TOST equivalence p at margins {0.001: 0.8084, 0.0025: 0.7327, 0.005: 0.5790, 0.01: 0.2592, 0.02: 0.0122}
- **Equivalent (TOST) within +-0.02 at alpha=0.05**
- Non-inferiority p at margins {0.001: 0.1146, 0.0025: 0.0740, 0.005: 0.0325, 0.01: 0.0047, 0.02: 4.4e-05}
- **Non-inferior (no penalty worse than 0.005) at alpha=0.05**
- Leave-one-out (drop task_14954_cylinder-bands): mean +0.00036, 95% CI [-0.00107, +0.00178], TOST margin 0.0025, non-inf margin 0.001
- Per-dataset Wilcoxon (Holm-corrected across datasets): no dataset with p < 0.05

### Random_bins256 vs Exact (positive delta = comparator better)

- Datasets: 34
- Mean per-dataset delta: +0.00735 (median +0.00046, range [-0.00616, +0.21296])
- 95% CI (t): [-0.00537, +0.02008]; 95% CI (bootstrap): [+0.00033, +0.02035]
- Wilcoxon signed-rank p = 0.0635; paired t p = 0.2482
- Win/tie/loss (|delta| <= 0.001 is a tie): 12/15/7 (raw sign: 21 up, 10 down)
- TOST equivalence p at margins {0.001: 0.8414, 0.0025: 0.7783, 0.005: 0.6454, 0.01: 0.3374, 0.02: 0.0257}
- **Equivalent (TOST) within +-0.02 at alpha=0.05**
- Non-inferiority p at margins {0.001: 0.0954, 0.0025: 0.0624, 0.005: 0.0283, 0.01: 0.0045, 0.02: 5.8e-05}
- **Non-inferior (no penalty worse than 0.005) at alpha=0.05**
- Leave-one-out (drop task_14954_cylinder-bands): mean +0.00112, 95% CI [-0.00001, +0.00226], TOST margin 0.0025, non-inf margin 0.001
- Per-dataset Wilcoxon (Holm-corrected across datasets): no dataset with p < 0.05

### Random_bins32 vs Exact (positive delta = comparator better)

- Datasets: 34
- Mean per-dataset delta: +0.00729 (median +0.00097, range [-0.01280, +0.21296])
- 95% CI (t): [-0.00547, +0.02004]; 95% CI (bootstrap): [+0.00004, +0.02042]
- Wilcoxon signed-rank p = 0.0289; paired t p = 0.2533
- Win/tie/loss (|delta| <= 0.001 is a tie): 17/11/6 (raw sign: 19 up, 12 down)
- TOST equivalence p at margins {0.001: 0.8385, 0.0025: 0.7748, 0.005: 0.6413, 0.01: 0.3341, 0.02: 0.0254}
- **Equivalent (TOST) within +-0.02 at alpha=0.05**
- Non-inferiority p at margins {0.001: 0.0976, 0.0025: 0.0640, 0.005: 0.0292, 0.01: 0.0047, 0.02: 6.1e-05}
- **Non-inferior (no penalty worse than 0.005) at alpha=0.05**
- Leave-one-out (drop task_14954_cylinder-bands): mean +0.00106, 95% CI [-0.00036, +0.00247], TOST margin 0.0025, non-inf margin 0.001
- Per-dataset Wilcoxon (Holm-corrected across datasets): no dataset with p < 0.05

- Friedman over 34 datasets: chi2 = 9.55, p = 0.1448; average ranks (1 = best): Dynamic_Random_Histogram_thresh250: 3.79, Exact: 4.87, Random: 3.41, Random_bins128: 4.09, Random_bins16: 4.12, Random_bins256: 3.93, Random_bins32: 3.79

## SPO-RF -- auc

### Dynamic_Random_Histogram_thresh250 vs Exact (positive delta = comparator better)

- Datasets: 36
- Mean per-dataset delta: +0.00198 (median -0.00008, range [-0.00458, +0.06453])
- 95% CI (t): [-0.00176, +0.00573]; 95% CI (bootstrap): [-0.00046, +0.00604]
- Wilcoxon signed-rank p = 0.9578; paired t p = 0.2891
- Win/tie/loss (|delta| <= 0.001 is a tie): 10/14/12 (raw sign: 13 up, 20 down)
- TOST equivalence p at margins {0.001: 0.7016, 0.0025: 0.3907, 0.005: 0.0554, 0.01: 5.6e-05, 0.02: 7.7e-12}
- **Equivalent (TOST) within +-0.01 at alpha=0.05**
- Non-inferiority p at margins {0.001: 0.0572, 0.0025: 0.0101, 0.005: 0.0003, 0.01: 8.5e-08, 0.02: 3.5e-14}
- **Non-inferior (no penalty worse than 0.0025) at alpha=0.05**
- Leave-one-out (drop task_14954_cylinder-bands): mean +0.00020, 95% CI [-0.00075, +0.00114], TOST margin 0.001, non-inf margin 0.001
- Per-dataset Wilcoxon (Holm-corrected across datasets): no dataset with p < 0.05

### Random vs Exact (positive delta = comparator better)

- Datasets: 36
- Mean per-dataset delta: +0.00340 (median +0.00029, range [-0.00624, +0.10713])
- 95% CI (t): [-0.00269, +0.00948]; 95% CI (bootstrap): [-0.00022, +0.00984]
- Wilcoxon signed-rank p = 0.1948; paired t p = 0.2650
- Win/tie/loss (|delta| <= 0.001 is a tie): 11/19/6 (raw sign: 20 up, 13 down)
- TOST equivalence p at margins {0.001: 0.7853, 0.0025: 0.6167, 0.005: 0.2982, 0.01: 0.0172, 0.02: 1.6e-06}
- **Equivalent (TOST) within +-0.01 at alpha=0.05**
- Non-inferiority p at margins {0.001: 0.0757, 0.0025: 0.0286, 0.005: 0.0041, 0.01: 4.0e-05, 0.02: 1.8e-09}
- **Non-inferior (no penalty worse than 0.0025) at alpha=0.05**
- Leave-one-out (drop task_14954_cylinder-bands): mean +0.00043, 95% CI [-0.00052, +0.00139], TOST margin 0.0025, non-inf margin 0.001
- Per-dataset Wilcoxon (Holm-corrected across datasets): no dataset with p < 0.05

### Random_bins128 vs Exact (positive delta = comparator better)

- Datasets: 34
- Mean per-dataset delta: +0.00315 (median +0.00000, range [-0.00826, +0.11113])
- 95% CI (t): [-0.00357, +0.00987]; 95% CI (bootstrap): [-0.00080, +0.01014]
- Wilcoxon signed-rank p = 0.8092; paired t p = 0.3478
- Win/tie/loss (|delta| <= 0.001 is a tie): 8/20/6 (raw sign: 15 up, 16 down)
- TOST equivalence p at margins {0.001: 0.7398, 0.0025: 0.5769, 0.005: 0.2891, 0.01: 0.0229, 0.02: 6.8e-06}
- **Equivalent (TOST) within +-0.01 at alpha=0.05**
- Non-inferiority p at margins {0.001: 0.1091, 0.0025: 0.0484, 0.005: 0.0095, 0.01: 0.0002, 0.02: 2.6e-08}
- **Non-inferior (no penalty worse than 0.0025) at alpha=0.05**
- Leave-one-out (drop task_14954_cylinder-bands): mean -0.00013, 95% CI [-0.00107, +0.00081], TOST margin 0.001, non-inf margin 0.001
- Per-dataset Wilcoxon (Holm-corrected across datasets): no dataset with p < 0.05

### Random_bins16 vs Exact (positive delta = comparator better)

- Datasets: 34
- Mean per-dataset delta: +0.00439 (median +0.00019, range [-0.00934, +0.11218])
- 95% CI (t): [-0.00249, +0.01127]; 95% CI (bootstrap): [-0.00017, +0.01181]
- Wilcoxon signed-rank p = 0.1821; paired t p = 0.2035
- Win/tie/loss (|delta| <= 0.001 is a tie): 12/13/9 (raw sign: 18 up, 13 down)
- TOST equivalence p at margins {0.001: 0.8381, 0.0025: 0.7098, 0.005: 0.4288, 0.01: 0.0533, 0.02: 2.8e-05}
- **Equivalent (TOST) within +-0.02 at alpha=0.05**
- Non-inferiority p at margins {0.001: 0.0603, 0.0025: 0.0249, 0.005: 0.0045, 0.01: 8.1e-05, 0.02: 1.4e-08}
- **Non-inferior (no penalty worse than 0.0025) at alpha=0.05**
- Leave-one-out (drop task_14954_cylinder-bands): mean +0.00112, 95% CI [-0.00072, +0.00296], TOST margin 0.005, non-inf margin 0.001
- Per-dataset Wilcoxon (Holm-corrected across datasets): no dataset with p < 0.05

### Random_bins256 vs Exact (positive delta = comparator better)

- Datasets: 34
- Mean per-dataset delta: +0.00379 (median -0.00004, range [-0.00414, +0.11499])
- 95% CI (t): [-0.00313, +0.01072]; 95% CI (bootstrap): [-0.00027, +0.01104]
- Wilcoxon signed-rank p = 0.5043; paired t p = 0.2729
- Win/tie/loss (|delta| <= 0.001 is a tie): 10/17/7 (raw sign: 14 up, 17 down)
- TOST equivalence p at margins {0.001: 0.7913, 0.0025: 0.6469, 0.005: 0.3626, 0.01: 0.0386, 0.02: 1.8e-05}
- **Equivalent (TOST) within +-0.01 at alpha=0.05**
- Non-inferiority p at margins {0.001: 0.0841, 0.0025: 0.0367, 0.005: 0.0072, 0.01: 0.0001, 0.02: 2.7e-08}
- **Non-inferior (no penalty worse than 0.0025) at alpha=0.05**
- Leave-one-out (drop task_14954_cylinder-bands): mean +0.00042, 95% CI [-0.00056, +0.00141], TOST margin 0.0025, non-inf margin 0.001
- Per-dataset Wilcoxon (Holm-corrected across datasets): no dataset with p < 0.05

### Random_bins32 vs Exact (positive delta = comparator better)

- Datasets: 34
- Mean per-dataset delta: +0.00416 (median +0.00055, range [-0.00800, +0.11297])
- 95% CI (t): [-0.00266, +0.01097]; 95% CI (bootstrap): [-0.00000, +0.01136]
- Wilcoxon signed-rank p = 0.0829; paired t p = 0.2235
- Win/tie/loss (|delta| <= 0.001 is a tie): 13/15/6 (raw sign: 19 up, 12 down)
- TOST equivalence p at margins {0.001: 0.8235, 0.0025: 0.6878, 0.005: 0.4013, 0.01: 0.0452, 0.02: 2.0e-05}
- **Equivalent (TOST) within +-0.01 at alpha=0.05**
- Non-inferiority p at margins {0.001: 0.0666, 0.0025: 0.0276, 0.005: 0.0050, 0.01: 8.8e-05, 0.02: 1.4e-08}
- **Non-inferior (no penalty worse than 0.0025) at alpha=0.05**
- Leave-one-out (drop task_14954_cylinder-bands): mean +0.00086, 95% CI [-0.00038, +0.00210], TOST margin 0.0025, non-inf margin 0.001
- Per-dataset Wilcoxon (Holm-corrected across datasets): no dataset with p < 0.05

- Friedman over 34 datasets: chi2 = 8.02, p = 0.2368; average ranks (1 = best): Dynamic_Random_Histogram_thresh250: 4.38, Exact: 4.15, Random: 3.76, Random_bins128: 4.53, Random_bins16: 3.68, Random_bins256: 4.12, Random_bins32: 3.38

## SPO-RF -- logloss

### Dynamic_Random_Histogram_thresh250 vs Exact (negative delta = comparator better)

- Datasets: 36
- Mean per-dataset delta: -0.00059 (median +0.00012, range [-0.10397, +0.05433])
- 95% CI (t): [-0.00890, +0.00773]; 95% CI (bootstrap): [-0.00912, +0.00670]
- Wilcoxon signed-rank p = 0.4776; paired t p = 0.8870
- Win/tie/loss (|delta| <= 0.001 is a tie): 13/12/11 (raw sign: 20 up, 14 down)
- TOST equivalence p at margins {0.001: 0.4601, 0.0025: 0.3217, 0.005: 0.1444, 0.01: 0.0138, 0.02: 1.8e-05}
- **Equivalent (TOST) within +-0.01 at alpha=0.05**
- Non-inferiority p at margins {0.001: 0.3505, 0.0025: 0.2281, 0.005: 0.0907, 0.01: 0.0070, 0.02: 7.4e-06}
- **Non-inferior (no penalty worse than 0.01) at alpha=0.05**
- Leave-one-out (drop task_14954_cylinder-bands): mean +0.00237, 95% CI [-0.00357, +0.00830], TOST margin 0.01, non-inf margin 0.01
- Per-dataset Wilcoxon (Holm-corrected across datasets): no dataset with p < 0.05

### Random vs Exact (negative delta = comparator better)

- Datasets: 36
- Mean per-dataset delta: -0.00494 (median -0.00005, range [-0.17589, +0.02805])
- 95% CI (t): [-0.01543, +0.00556]; 95% CI (bootstrap): [-0.01642, +0.00243]
- Wilcoxon signed-rank p = 0.7230; paired t p = 0.3464
- Win/tie/loss (|delta| <= 0.001 is a tie): 13/12/11 (raw sign: 15 up, 19 down)
- TOST equivalence p at margins {0.001: 0.7742, 0.0025: 0.6797, 0.005: 0.4951, 0.01: 0.1670, 0.02: 0.0031}
- **Equivalent (TOST) within +-0.02 at alpha=0.05**
- Non-inferiority p at margins {0.001: 0.1294, 0.0025: 0.0797, 0.005: 0.0314, 0.01: 0.0033, 0.02: 1.4e-05}
- **Non-inferior (no penalty worse than 0.005) at alpha=0.05**
- Leave-one-out (drop task_14954_cylinder-bands): mean -0.00005, 95% CI [-0.00360, +0.00350], TOST margin 0.005, non-inf margin 0.005
- Per-dataset Wilcoxon (Holm-corrected across datasets): no dataset with p < 0.05

### Random_bins128 vs Exact (negative delta = comparator better)

- Datasets: 34
- Mean per-dataset delta: -0.00348 (median -0.00001, range [-0.17936, +0.02317])
- 95% CI (t): [-0.01455, +0.00760]; 95% CI (bootstrap): [-0.01533, +0.00354]
- Wilcoxon signed-rank p = 0.4319; paired t p = 0.5274
- Win/tie/loss (|delta| <= 0.001 is a tie): 12/13/9 (raw sign: 15 up, 17 down)
- TOST equivalence p at margins {0.001: 0.6740, 0.0025: 0.5706, 0.005: 0.3906, 0.01: 0.1196, 0.02: 0.0023}
- **Equivalent (TOST) within +-0.02 at alpha=0.05**
- Non-inferiority p at margins {0.001: 0.2084, 0.0025: 0.1401, 0.005: 0.0645, 0.01: 0.0093, 0.02: 6.9e-05}
- **Non-inferior (no penalty worse than 0.01) at alpha=0.05**
- Leave-one-out (drop task_14954_cylinder-bands): mean +0.00185, 95% CI [-0.00046, +0.00417], TOST margin 0.005, non-inf margin 0.005
- Per-dataset Wilcoxon (Holm-corrected across datasets): no dataset with p < 0.05

### Random_bins16 vs Exact (negative delta = comparator better)

- Datasets: 34
- Mean per-dataset delta: -0.00090 (median +0.00007, range [-0.17927, +0.10689])
- 95% CI (t): [-0.01415, +0.01234]; 95% CI (bootstrap): [-0.01479, +0.01051]
- Wilcoxon signed-rank p = 0.6511; paired t p = 0.8904
- Win/tie/loss (|delta| <= 0.001 is a tie): 15/7/12 (raw sign: 18 up, 14 down)
- TOST equivalence p at margins {0.001: 0.4942, 0.0025: 0.4040, 0.005: 0.2668, 0.01: 0.0859, 0.02: 0.0030}
- **Equivalent (TOST) within +-0.02 at alpha=0.05**
- Non-inferiority p at margins {0.001: 0.3859, 0.0025: 0.3023, 0.005: 0.1855, 0.01: 0.0517, 0.02: 0.0015}
- **Non-inferior (no penalty worse than 0.02) at alpha=0.05**
- Leave-one-out (drop task_14954_cylinder-bands): mean +0.00450, 95% CI [-0.00312, +0.01212], TOST margin 0.02, non-inf margin 0.02
- Per-dataset Wilcoxon (Holm-corrected across datasets): no dataset with p < 0.05

### Random_bins256 vs Exact (negative delta = comparator better)

- Datasets: 34
- Mean per-dataset delta: -0.00571 (median +0.00003, range [-0.18296, +0.05398])
- 95% CI (t): [-0.01823, +0.00680]; 95% CI (bootstrap): [-0.01919, +0.00399]
- Wilcoxon signed-rank p = 0.8900; paired t p = 0.3596
- Win/tie/loss (|delta| <= 0.001 is a tie): 13/11/10 (raw sign: 17 up, 15 down)
- TOST equivalence p at margins {0.001: 0.7756, 0.0025: 0.6976, 0.005: 0.5459, 0.01: 0.2454, 0.02: 0.0133}
- **Equivalent (TOST) within +-0.02 at alpha=0.05**
- Non-inferiority p at margins {0.001: 0.1414, 0.0025: 0.0954, 0.005: 0.0454, 0.01: 0.0077, 0.02: 0.0001}
- **Non-inferior (no penalty worse than 0.005) at alpha=0.05**
- Leave-one-out (drop task_14954_cylinder-bands): mean -0.00034, 95% CI [-0.00664, +0.00595], TOST margin 0.01, non-inf margin 0.005
- Per-dataset Wilcoxon (Holm-corrected across datasets): no dataset with p < 0.05

### Random_bins32 vs Exact (negative delta = comparator better)

- Datasets: 34
- Mean per-dataset delta: -0.00665 (median +0.00000, range [-0.18307, +0.02885])
- 95% CI (t): [-0.02199, +0.00869]; 95% CI (bootstrap): [-0.02299, +0.00541]
- Wilcoxon signed-rank p = 0.7468; paired t p = 0.3844
- Win/tie/loss (|delta| <= 0.001 is a tie): 11/14/9 (raw sign: 16 up, 16 down)
- TOST equivalence p at margins {0.001: 0.7704, 0.0025: 0.7070, 0.005: 0.5858, 0.01: 0.3298, 0.02: 0.0429}
- **Equivalent (TOST) within +-0.02 at alpha=0.05**
- Non-inferiority p at margins {0.001: 0.1589, 0.0025: 0.1169, 0.005: 0.0660, 0.01: 0.0172, 0.02: 0.0006}
- **Non-inferior (no penalty worse than 0.01) at alpha=0.05**
- Leave-one-out (drop task_14954_cylinder-bands): mean -0.00130, 95% CI [-0.01247, +0.00986], TOST margin 0.02, non-inf margin 0.01
- Per-dataset Wilcoxon (Holm-corrected across datasets): no dataset with p < 0.05

- Friedman over 34 datasets: chi2 = 2.26, p = 0.8940; average ranks (1 = best): Dynamic_Random_Histogram_thresh250: 4.12, Exact: 3.94, Random: 3.76, Random_bins128: 4.35, Random_bins16: 4.18, Random_bins256: 3.85, Random_bins32: 3.79

