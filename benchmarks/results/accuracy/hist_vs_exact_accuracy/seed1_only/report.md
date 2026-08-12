# Split-finder accuracy comparison: Random Histogram vs Exact

Inputs: 51 CSVs, 36 datasets, seeds: [1]. Unit of analysis for all aggregate claims: the dataset (per-dataset mean of paired per-(seed,fold) deltas).

## SPO-GBT -- accuracy

### Dynamic_Random_Histogram_thresh250 vs Exact (positive delta = comparator better)

- Datasets: 36
- Mean per-dataset delta: +0.00178 (median +0.00062, range [-0.01724, +0.04815])
- 95% CI (t): [-0.00125, +0.00480]; 95% CI (bootstrap): [-0.00059, +0.00507]
- Wilcoxon signed-rank p = 0.0864; paired t p = 0.2412
- Win/tie/loss (|delta| <= 0.001 is a tie): 16/12/8 (raw sign: 19 up, 14 down)
- TOST equivalence p at margins {0.001: 0.6971, 0.0025: 0.3150, 0.005: 0.0187, 0.01: 1.7e-06, 0.02: 1.7e-14}
- **Equivalent (TOST) within +-0.005 at alpha=0.05**
- Non-inferiority p at margins {0.001: 0.0354, 0.0025: 0.0035, 0.005: 3.1e-05, 0.01: 1.4e-09, 0.02: 1.1e-16}
- **Non-inferior (no penalty worse than 0.001) at alpha=0.05**
- Leave-one-out (drop task_14954_cylinder-bands): mean +0.00045, 95% CI [-0.00097, +0.00188], TOST margin 0.0025, non-inf margin 0.001
- Per-dataset Wilcoxon (Holm-corrected across datasets): no dataset with p < 0.05

### Random vs Exact (positive delta = comparator better)

- Datasets: 36
- Mean per-dataset delta: +0.00094 (median -0.00000, range [-0.01690, +0.04074])
- 95% CI (t): [-0.00190, +0.00378]; 95% CI (bootstrap): [-0.00138, +0.00390]
- Wilcoxon signed-rank p = 0.8575; paired t p = 0.5064
- Win/tie/loss (|delta| <= 0.001 is a tie): 10/14/12 (raw sign: 16 up, 18 down)
- TOST equivalence p at margins {0.001: 0.4826, 0.0025: 0.1358, 0.005: 0.0032, 0.01: 9.0e-08, 0.02: 7.2e-16}
- **Equivalent (TOST) within +-0.005 at alpha=0.05**
- Non-inferiority p at margins {0.001: 0.0872, 0.0025: 0.0095, 0.005: 7.6e-05, 0.01: 1.7e-09, 0.02: 0.0e+00}
- **Non-inferior (no penalty worse than 0.0025) at alpha=0.05**
- Leave-one-out (drop task_14954_cylinder-bands): mean -0.00020, 95% CI [-0.00190, +0.00150], TOST margin 0.0025, non-inf margin 0.0025
- Per-dataset Wilcoxon (Holm-corrected across datasets): no dataset with p < 0.05

### Random_bins128 vs Exact (positive delta = comparator better)

- Datasets: 34
- Mean per-dataset delta: +0.00124 (median +0.00000, range [-0.01382, +0.03889])
- 95% CI (t): [-0.00141, +0.00389]; 95% CI (bootstrap): [-0.00087, +0.00406]
- Wilcoxon signed-rank p = 0.5361; paired t p = 0.3495
- Win/tie/loss (|delta| <= 0.001 is a tie): 10/16/8 (raw sign: 19 up, 13 down)
- TOST equivalence p at margins {0.001: 0.5715, 0.0025: 0.1698, 0.005: 0.0034, 0.01: 5.8e-08, 0.02: 4.4e-16}
- **Equivalent (TOST) within +-0.005 at alpha=0.05**
- Non-inferiority p at margins {0.001: 0.0477, 0.0025: 0.0036, 0.005: 1.7e-05, 0.01: 2.9e-10, 0.02: 0.0e+00}
- **Non-inferior (no penalty worse than 0.001) at alpha=0.05**
- Leave-one-out (drop task_14954_cylinder-bands): mean +0.00010, 95% CI [-0.00123, +0.00142], TOST margin 0.0025, non-inf margin 0.0025
- Per-dataset Wilcoxon (Holm-corrected across datasets): no dataset with p < 0.05

### Random_bins16 vs Exact (positive delta = comparator better)

- Datasets: 34
- Mean per-dataset delta: +0.00156 (median +0.00066, range [-0.02628, +0.04815])
- 95% CI (t): [-0.00206, +0.00519]; 95% CI (bootstrap): [-0.00164, +0.00527]
- Wilcoxon signed-rank p = 0.2539; paired t p = 0.3871
- Win/tie/loss (|delta| <= 0.001 is a tie): 13/11/10 (raw sign: 20 up, 12 down)
- TOST equivalence p at margins {0.001: 0.6229, 0.0025: 0.3014, 0.005: 0.0313, 0.01: 2.0e-05, 0.02: 3.5e-12}
- **Equivalent (TOST) within +-0.005 at alpha=0.05**
- Non-inferiority p at margins {0.001: 0.0800, 0.0025: 0.0146, 0.005: 0.0004, 0.01: 1.2e-07, 0.02: 5.7e-14}
- **Non-inferior (no penalty worse than 0.0025) at alpha=0.05**
- Leave-one-out (drop task_14954_cylinder-bands): mean +0.00015, 95% CI [-0.00214, +0.00244], TOST margin 0.0025, non-inf margin 0.0025
- Per-dataset Wilcoxon (Holm-corrected across datasets): no dataset with p < 0.05

### Random_bins256 vs Exact (positive delta = comparator better)

- Datasets: 34
- Mean per-dataset delta: +0.00066 (median +0.00000, range [-0.01039, +0.03889])
- 95% CI (t): [-0.00198, +0.00330]; 95% CI (bootstrap): [-0.00140, +0.00349]
- Wilcoxon signed-rank p = 0.8754; paired t p = 0.6130
- Win/tie/loss (|delta| <= 0.001 is a tie): 10/12/12 (raw sign: 16 up, 16 down)
- TOST equivalence p at margins {0.001: 0.3984, 0.0025: 0.0832, 0.005: 0.0010, 0.01: 1.5e-08, 0.02: 1.7e-16}
- **Equivalent (TOST) within +-0.005 at alpha=0.05**
- Non-inferiority p at margins {0.001: 0.1046, 0.0025: 0.0102, 0.005: 6.0e-05, 0.01: 8.8e-10, 0.02: 0.0e+00}
- **Non-inferior (no penalty worse than 0.0025) at alpha=0.05**
- Leave-one-out (drop task_14954_cylinder-bands): mean -0.00050, 95% CI [-0.00173, +0.00074], TOST margin 0.0025, non-inf margin 0.0025
- Per-dataset Wilcoxon (Holm-corrected across datasets): no dataset with p < 0.05

### Random_bins32 vs Exact (positive delta = comparator better)

- Datasets: 34
- Mean per-dataset delta: +0.00079 (median +0.00027, range [-0.02158, +0.02407])
- 95% CI (t): [-0.00143, +0.00301]; 95% CI (bootstrap): [-0.00134, +0.00287]
- Wilcoxon signed-rank p = 0.1574; paired t p = 0.4733
- Win/tie/loss (|delta| <= 0.001 is a tie): 13/14/7 (raw sign: 20 up, 11 down)
- TOST equivalence p at margins {0.001: 0.4249, 0.0025: 0.0635, 0.005: 0.0003, 0.01: 4.7e-10, 0.02: 1.2e-18}
- **Equivalent (TOST) within +-0.005 at alpha=0.05**
- Non-inferiority p at margins {0.001: 0.0550, 0.0025: 0.0024, 0.005: 3.7e-06, 0.01: 1.1e-11, 0.02: 0.0e+00}
- **Non-inferior (no penalty worse than 0.0025) at alpha=0.05**
- Leave-one-out (drop task_14954_cylinder-bands): mean +0.00009, 95% CI [-0.00166, +0.00183], TOST margin 0.0025, non-inf margin 0.0025
- Per-dataset Wilcoxon (Holm-corrected across datasets): no dataset with p < 0.05

- Friedman over 34 datasets: chi2 = 6.14, p = 0.4073; average ranks (1 = best): Dynamic_Random_Histogram_thresh250: 3.38, Exact: 4.44, Random: 4.07, Random_bins128: 4.18, Random_bins16: 3.97, Random_bins256: 4.26, Random_bins32: 3.69

## SPO-GBT -- auc

### Dynamic_Random_Histogram_thresh250 vs Exact (positive delta = comparator better)

- Datasets: 36
- Mean per-dataset delta: +0.00112 (median +0.00000, range [-0.01206, +0.04264])
- 95% CI (t): [-0.00169, +0.00394]; 95% CI (bootstrap): [-0.00107, +0.00417]
- Wilcoxon signed-rank p = 0.8739; paired t p = 0.4234
- Win/tie/loss (|delta| <= 0.001 is a tie): 10/16/10 (raw sign: 16 up, 17 down)
- TOST equivalence p at margins {0.001: 0.5349, 0.0025: 0.1634, 0.005: 0.0041, 0.01: 1.1e-07, 0.02: 7.4e-16}
- **Equivalent (TOST) within +-0.005 at alpha=0.05**
- Non-inferiority p at margins {0.001: 0.0673, 0.0025: 0.0065, 0.005: 4.6e-05, 0.01: 9.5e-10, 0.02: 0.0e+00}
- **Non-inferior (no penalty worse than 0.0025) at alpha=0.05**
- Leave-one-out (drop task_14954_cylinder-bands): mean -0.00006, 95% CI [-0.00156, +0.00143], TOST margin 0.0025, non-inf margin 0.0025
- Per-dataset Wilcoxon (Holm-corrected across datasets): no dataset with p < 0.05

### Random vs Exact (positive delta = comparator better)

- Datasets: 36
- Mean per-dataset delta: +0.00146 (median +0.00012, range [-0.01187, +0.02970])
- 95% CI (t): [-0.00099, +0.00392]; 95% CI (bootstrap): [-0.00068, +0.00400]
- Wilcoxon signed-rank p = 0.3215; paired t p = 0.2348
- Win/tie/loss (|delta| <= 0.001 is a tie): 14/14/8 (raw sign: 18 up, 15 down)
- TOST equivalence p at margins {0.001: 0.6476, 0.0025: 0.1982, 0.005: 0.0030, 0.01: 1.6e-08, 0.02: 2.1e-17}
- **Equivalent (TOST) within +-0.005 at alpha=0.05**
- Non-inferiority p at margins {0.001: 0.0247, 0.0025: 0.0012, 0.005: 2.8e-06, 0.01: 1.7e-11, 0.02: 0.0e+00}
- **Non-inferior (no penalty worse than 0.001) at alpha=0.05**
- Leave-one-out (drop task_14954_cylinder-bands): mean +0.00066, 95% CI [-0.00123, +0.00254], TOST margin 0.0025, non-inf margin 0.001
- Per-dataset Wilcoxon (Holm-corrected across datasets): no dataset with p < 0.05

### Random_bins128 vs Exact (positive delta = comparator better)

- Datasets: 34
- Mean per-dataset delta: +0.00048 (median +0.00007, range [-0.00887, +0.01435])
- 95% CI (t): [-0.00129, +0.00225]; 95% CI (bootstrap): [-0.00120, +0.00219]
- Wilcoxon signed-rank p = 0.6636; paired t p = 0.5858
- Win/tie/loss (|delta| <= 0.001 is a tie): 10/15/9 (raw sign: 17 up, 14 down)
- TOST equivalence p at margins {0.001: 0.2774, 0.0025: 0.0134, 0.005: 5.3e-06, 0.01: 8.5e-13, 0.02: 7.9e-22}
- **Equivalent (TOST) within +-0.0025 at alpha=0.05**
- Non-inferiority p at margins {0.001: 0.0495, 0.0025: 0.0008, 0.005: 2.1e-07, 0.01: 6.7e-14, 0.02: 0.0e+00}
- **Non-inferior (no penalty worse than 0.001) at alpha=0.05**
- Leave-one-out (drop task_9971_ilpd): mean +0.00006, 95% CI [-0.00154, +0.00166], TOST margin 0.0025, non-inf margin 0.0025
- Per-dataset Wilcoxon (Holm-corrected across datasets): no dataset with p < 0.05

### Random_bins16 vs Exact (positive delta = comparator better)

- Datasets: 34
- Mean per-dataset delta: +0.00142 (median +0.00000, range [-0.01991, +0.02629])
- 95% CI (t): [-0.00167, +0.00452]; 95% CI (bootstrap): [-0.00152, +0.00444]
- Wilcoxon signed-rank p = 0.4798; paired t p = 0.3567
- Win/tie/loss (|delta| <= 0.001 is a tie): 14/10/10 (raw sign: 16 up, 15 down)
- TOST equivalence p at margins {0.001: 0.6086, 0.0025: 0.2422, 0.005: 0.0125, 0.01: 1.4e-06, 0.02: 4.5e-14}
- **Equivalent (TOST) within +-0.005 at alpha=0.05**
- Non-inferiority p at margins {0.001: 0.0605, 0.0025: 0.0073, 0.005: 9.0e-05, 0.01: 6.3e-09, 0.02: 8.9e-16}
- **Non-inferior (no penalty worse than 0.0025) at alpha=0.05**
- Leave-one-out (drop task_14954_cylinder-bands): mean +0.00067, 95% CI [-0.00211, +0.00345], TOST margin 0.005, non-inf margin 0.0025
- Per-dataset Wilcoxon (Holm-corrected across datasets): no dataset with p < 0.05

### Random_bins256 vs Exact (positive delta = comparator better)

- Datasets: 34
- Mean per-dataset delta: +0.00118 (median +0.00000, range [-0.00717, +0.03165])
- 95% CI (t): [-0.00112, +0.00348]; 95% CI (bootstrap): [-0.00073, +0.00357]
- Wilcoxon signed-rank p = 0.6357; paired t p = 0.3052
- Win/tie/loss (|delta| <= 0.001 is a tie): 11/13/10 (raw sign: 15 up, 16 down)
- TOST equivalence p at margins {0.001: 0.5620, 0.0025: 0.1254, 0.005: 0.0009, 0.01: 2.7e-09, 0.02: 6.4e-18}
- **Equivalent (TOST) within +-0.005 at alpha=0.05**
- Non-inferiority p at margins {0.001: 0.0314, 0.0025: 0.0013, 0.005: 2.4e-06, 0.01: 1.1e-11, 0.02: 0.0e+00}
- **Non-inferior (no penalty worse than 0.001) at alpha=0.05**
- Leave-one-out (drop task_14954_cylinder-bands): mean +0.00025, 95% CI [-0.00112, +0.00163], TOST margin 0.0025, non-inf margin 0.001
- Per-dataset Wilcoxon (Holm-corrected across datasets): no dataset with p < 0.05

### Random_bins32 vs Exact (positive delta = comparator better)

- Datasets: 34
- Mean per-dataset delta: +0.00126 (median +0.00000, range [-0.01548, +0.02218])
- 95% CI (t): [-0.00129, +0.00381]; 95% CI (bootstrap): [-0.00111, +0.00378]
- Wilcoxon signed-rank p = 0.4919; paired t p = 0.3212
- Win/tie/loss (|delta| <= 0.001 is a tie): 10/16/8 (raw sign: 16 up, 15 down)
- TOST equivalence p at margins {0.001: 0.5824, 0.0025: 0.1655, 0.005: 0.0027, 0.01: 2.9e-08, 0.02: 1.5e-16}
- **Equivalent (TOST) within +-0.005 at alpha=0.05**
- Non-inferiority p at margins {0.001: 0.0401, 0.0025: 0.0025, 0.005: 9.4e-06, 0.01: 1.1e-10, 0.02: 0.0e+00}
- **Non-inferior (no penalty worse than 0.001) at alpha=0.05**
- Leave-one-out (drop task_9971_ilpd): mean +0.00063, 95% CI [-0.00164, +0.00290], TOST margin 0.005, non-inf margin 0.0025
- Per-dataset Wilcoxon (Holm-corrected across datasets): no dataset with p < 0.05

- Friedman over 34 datasets: chi2 = 1.63, p = 0.9503; average ranks (1 = best): Dynamic_Random_Histogram_thresh250: 3.94, Exact: 4.15, Random: 3.62, Random_bins128: 4.06, Random_bins16: 4.00, Random_bins256: 4.09, Random_bins32: 4.15

## SPO-GBT -- logloss

### Dynamic_Random_Histogram_thresh250 vs Exact (negative delta = comparator better)

- Datasets: 36
- Mean per-dataset delta: -0.00242 (median -0.00093, range [-0.05545, +0.06416])
- 95% CI (t): [-0.00880, +0.00395]; 95% CI (bootstrap): [-0.00834, +0.00377]
- Wilcoxon signed-rank p = 0.0752; paired t p = 0.4460
- Win/tie/loss (|delta| <= 0.001 is a tie): 8/10/18 (raw sign: 12 up, 22 down)
- TOST equivalence p at margins {0.001: 0.6731, 0.0025: 0.4900, 0.005: 0.2085, 0.01: 0.0106, 0.02: 1.3e-06}
- **Equivalent (TOST) within +-0.01 at alpha=0.05**
- Non-inferiority p at margins {0.001: 0.1417, 0.0025: 0.0631, 0.005: 0.0119, 0.01: 0.0002, 0.02: 1.3e-08}
- **Non-inferior (no penalty worse than 0.005) at alpha=0.05**
- Leave-one-out (drop task_29_credit-approval): mean -0.00432, 95% CI [-0.00955, +0.00090], TOST margin 0.01, non-inf margin 0.001
- Per-dataset Wilcoxon (Holm-corrected across datasets): no dataset with p < 0.05

### Random vs Exact (negative delta = comparator better)

- Datasets: 36
- Mean per-dataset delta: -0.00565 (median -0.00125, range [-0.06019, +0.06587])
- 95% CI (t): [-0.01294, +0.00164]; 95% CI (bootstrap): [-0.01265, +0.00134]
- Wilcoxon signed-rank p = 0.0187; paired t p = 0.1244
- Win/tie/loss (|delta| <= 0.001 is a tie): 7/11/18 (raw sign: 12 up, 22 down)
- TOST equivalence p at margins {0.001: 0.8983, 0.0025: 0.8071, 0.005: 0.5717, 0.01: 0.1172, 0.02: 0.0002}
- **Equivalent (TOST) within +-0.02 at alpha=0.05**
- Non-inferiority p at margins {0.001: 0.0362, 0.0025: 0.0147, 0.005: 0.0027, 0.01: 5.5e-05, 0.02: 1.2e-08}
- **Non-inferior (no penalty worse than 0.001) at alpha=0.05**
- Leave-one-out (drop task_29_credit-approval): mean -0.00770, 95% CI [-0.01387, -0.00152], TOST margin 0.02, non-inf margin 0.001
- Per-dataset Wilcoxon (Holm-corrected across datasets): no dataset with p < 0.05

### Random_bins128 vs Exact (negative delta = comparator better)

- Datasets: 34
- Mean per-dataset delta: -0.00265 (median -0.00050, range [-0.04135, +0.05203])
- 95% CI (t): [-0.00825, +0.00294]; 95% CI (bootstrap): [-0.00785, +0.00272]
- Wilcoxon signed-rank p = 0.1658; paired t p = 0.3414
- Win/tie/loss (|delta| <= 0.001 is a tie): 9/9/16 (raw sign: 13 up, 19 down)
- TOST equivalence p at margins {0.001: 0.7242, 0.0025: 0.5221, 0.005: 0.1997, 0.01: 0.0058, 0.02: 1.9e-07}
- **Equivalent (TOST) within +-0.01 at alpha=0.05**
- Non-inferiority p at margins {0.001: 0.0965, 0.0025: 0.0348, 0.005: 0.0044, 0.01: 3.0e-05, 0.02: 8.1e-10}
- **Non-inferior (no penalty worse than 0.0025) at alpha=0.05**
- Leave-one-out (drop task_29_credit-approval): mean -0.00431, 95% CI [-0.00892, +0.00029], TOST margin 0.01, non-inf margin 0.001
- Per-dataset Wilcoxon (Holm-corrected across datasets): no dataset with p < 0.05

### Random_bins16 vs Exact (negative delta = comparator better)

- Datasets: 34
- Mean per-dataset delta: -0.01354 (median -0.00251, range [-0.12740, +0.08412])
- 95% CI (t): [-0.02645, -0.00063]; 95% CI (bootstrap): [-0.02602, -0.00154]
- Wilcoxon signed-rank p = 0.0205; paired t p = 0.0404
- Win/tie/loss (|delta| <= 0.001 is a tie): 9/6/19 (raw sign: 11 up, 21 down)
- TOST equivalence p at margins {0.001: 0.9717, 0.0025: 0.9544, 0.005: 0.9062, 0.01: 0.7096, 0.02: 0.1581}
- **Not TOST-equivalent at any tested margin**
- Non-inferiority p at margins {0.001: 0.0142, 0.0025: 0.0082, 0.005: 0.0031, 0.01: 0.0004, 0.02: 4.0e-06}
- **Non-inferior (no penalty worse than 0.001) at alpha=0.05**
- Leave-one-out (drop task_9971_ilpd): mean -0.01009, 95% CI [-0.02128, +0.00110], TOST margin 0.02, non-inf margin 0.001
- Per-dataset Wilcoxon (Holm-corrected across datasets): no dataset with p < 0.05

### Random_bins256 vs Exact (negative delta = comparator better)

- Datasets: 34
- Mean per-dataset delta: -0.00103 (median +0.00025, range [-0.03764, +0.05882])
- 95% CI (t): [-0.00672, +0.00466]; 95% CI (bootstrap): [-0.00618, +0.00447]
- Wilcoxon signed-rank p = 0.9632; paired t p = 0.7148
- Win/tie/loss (|delta| <= 0.001 is a tie): 13/10/11 (raw sign: 18 up, 14 down)
- TOST equivalence p at margins {0.001: 0.5044, 0.0025: 0.3015, 0.005: 0.0826, 0.01: 0.0015, 0.02: 4.9e-08}
- **Equivalent (TOST) within +-0.01 at alpha=0.05**
- Non-inferiority p at margins {0.001: 0.2364, 0.0025: 0.1078, 0.005: 0.0192, 0.01: 0.0002, 0.02: 6.0e-09}
- **Non-inferior (no penalty worse than 0.005) at alpha=0.05**
- Leave-one-out (drop task_29_credit-approval): mean -0.00284, 95% CI [-0.00732, +0.00163], TOST margin 0.01, non-inf margin 0.001
- Per-dataset Wilcoxon (Holm-corrected across datasets): no dataset with p < 0.05

### Random_bins32 vs Exact (negative delta = comparator better)

- Datasets: 34
- Mean per-dataset delta: -0.00951 (median -0.00176, range [-0.11230, +0.06759])
- 95% CI (t): [-0.01940, +0.00038]; 95% CI (bootstrap): [-0.01926, -0.00042]
- Wilcoxon signed-rank p = 0.0228; paired t p = 0.0590
- Win/tie/loss (|delta| <= 0.001 is a tie): 9/8/17 (raw sign: 10 up, 22 down)
- TOST equivalence p at margins {0.001: 0.9553, 0.0025: 0.9206, 0.005: 0.8198, 0.01: 0.4601, 0.02: 0.0192}
- **Equivalent (TOST) within +-0.02 at alpha=0.05**
- Non-inferiority p at margins {0.001: 0.0190, 0.0025: 0.0094, 0.005: 0.0027, 0.01: 0.0002, 0.02: 3.9e-07}
- **Non-inferior (no penalty worse than 0.001) at alpha=0.05**
- Leave-one-out (drop task_9971_ilpd): mean -0.00639, 95% CI [-0.01423, +0.00144], TOST margin 0.02, non-inf margin 0.001
- Per-dataset Wilcoxon (Holm-corrected across datasets): no dataset with p < 0.05

- Friedman over 34 datasets: chi2 = 20.92, p = 0.0019; average ranks (1 = best): Dynamic_Random_Histogram_thresh250: 4.00, Exact: 4.71, Random: 3.68, Random_bins128: 4.24, Random_bins16: 3.09, Random_bins256: 4.91, Random_bins32: 3.38

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

