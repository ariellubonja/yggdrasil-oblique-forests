# Split-finder accuracy comparison: Random Histogram vs Exact

Inputs: 90 CSVs, 36 datasets, seeds: [1, 2, 3]. Unit of analysis for all aggregate claims: the dataset (per-dataset mean of paired per-(seed,fold) deltas).

## SPO-GBT -- accuracy

### Dynamic_Random_Histogram_thresh250 vs Exact (positive delta = comparator better)

- Datasets: 36
- Mean per-dataset delta: +0.00082 (median +0.00000, range [-0.01834, +0.03457])
- 95% CI (t): [-0.00193, +0.00358]; 95% CI (bootstrap): [-0.00163, +0.00364]
- Wilcoxon signed-rank p = 0.5965; paired t p = 0.5487
- Win/tie/loss (|delta| <= 0.001 is a tie): 11/19/6 (raw sign: 16 up, 17 down)
- TOST equivalence p at margins {0.001: 0.4482, 0.0025: 0.1123, 0.005: 0.0020, 0.01: 3.9e-08, 0.02: 2.5e-16}
- **Equivalent (TOST) within +-0.005 at alpha=0.05**
- Non-inferiority p at margins {0.001: 0.0941, 0.0025: 0.0098, 0.005: 6.7e-05, 0.01: 1.1e-09, 0.02: 0.0e+00}
- **Non-inferior (no penalty worse than 0.0025) at alpha=0.05**
- Leave-one-out (drop task_14954_cylinder-bands): mean -0.00014, 95% CI [-0.00214, +0.00186], TOST margin 0.0025, non-inf margin 0.0025
- Per-dataset Wilcoxon (Holm): 2 dataset(s) significant: task_14954_cylinder-bands (delta +0.0346, p=0.0142); task_219_electricity (delta -0.0176, p=6.2e-05)

- Seed yardstick (34 datasets): median |method gap| 0.00098 vs median Exact seed spread 0.00148 (ratio 1.15); method gap within 1 seed-spread on 50% of datasets

### Random vs Exact (positive delta = comparator better)

- Datasets: 36
- Mean per-dataset delta: +0.00066 (median +0.00006, range [-0.01866, +0.03642])
- 95% CI (t): [-0.00182, +0.00315]; 95% CI (bootstrap): [-0.00142, +0.00334]
- Wilcoxon signed-rank p = 0.4776; paired t p = 0.5920
- Win/tie/loss (|delta| <= 0.001 is a tie): 11/17/8 (raw sign: 19 up, 15 down)
- TOST equivalence p at margins {0.001: 0.3921, 0.0025: 0.0711, 0.005: 0.0006, 0.01: 3.0e-09, 0.02: 8.4e-18}
- **Equivalent (TOST) within +-0.005 at alpha=0.05**
- Non-inferiority p at margins {0.001: 0.0916, 0.0025: 0.0071, 0.005: 2.5e-05, 0.01: 1.4e-10, 0.02: 0.0e+00}
- **Non-inferior (no penalty worse than 0.0025) at alpha=0.05**
- Leave-one-out (drop task_14954_cylinder-bands): mean -0.00036, 95% CI [-0.00177, +0.00105], TOST margin 0.0025, non-inf margin 0.0025
- Per-dataset Wilcoxon (Holm): 2 dataset(s) significant: task_14954_cylinder-bands (delta +0.0364, p=0.0105); task_219_electricity (delta -0.0187, p=6.2e-05)

- Seed yardstick (34 datasets): median |method gap| 0.00132 vs median Exact seed spread 0.00148 (ratio 1.16); method gap within 1 seed-spread on 47% of datasets

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

- Seed yardstick (34 datasets): median |method gap| 0.00193 vs median Exact seed spread 0.00148 (ratio 1.61); method gap within 1 seed-spread on 26% of datasets

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

- Seed yardstick (34 datasets): median |method gap| 0.00162 vs median Exact seed spread 0.00148 (ratio 1.19); method gap within 1 seed-spread on 41% of datasets

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

- Seed yardstick (34 datasets): median |method gap| 0.00159 vs median Exact seed spread 0.00148 (ratio 1.56); method gap within 1 seed-spread on 35% of datasets

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

- Seed yardstick (34 datasets): median |method gap| 0.00194 vs median Exact seed spread 0.00148 (ratio 1.95); method gap within 1 seed-spread on 38% of datasets

- Friedman over 34 datasets: chi2 = 4.31, p = 0.6347; average ranks (1 = best): Dynamic_Random_Histogram_thresh250: 3.76, Exact: 4.28, Random: 3.93, Random_bins128: 4.40, Random_bins16: 3.60, Random_bins256: 3.76, Random_bins32: 4.26

## SPO-GBT -- auc

### Dynamic_Random_Histogram_thresh250 vs Exact (positive delta = comparator better)

- Datasets: 36
- Mean per-dataset delta: +0.00091 (median -0.00031, range [-0.01021, +0.05979])
- 95% CI (t): [-0.00265, +0.00446]; 95% CI (bootstrap): [-0.00149, +0.00477]
- Wilcoxon signed-rank p = 0.2011; paired t p = 0.6076
- Win/tie/loss (|delta| <= 0.001 is a tie): 7/16/13 (raw sign: 12 up, 21 down)
- TOST equivalence p at margins {0.001: 0.4791, 0.0025: 0.1848, 0.005: 0.0127, 0.01: 4.5e-06, 0.02: 4.3e-13}
- **Equivalent (TOST) within +-0.005 at alpha=0.05**
- Non-inferiority p at margins {0.001: 0.1418, 0.0025: 0.0299, 0.005: 0.0009, 0.01: 1.9e-07, 0.02: 3.4e-14}
- **Non-inferior (no penalty worse than 0.0025) at alpha=0.05**
- Leave-one-out (drop task_14954_cylinder-bands): mean -0.00077, 95% CI [-0.00180, +0.00025], TOST margin 0.0025, non-inf margin 0.0025
- Per-dataset Wilcoxon (Holm): 5 dataset(s) significant: task_14954_cylinder-bands (delta +0.0598, p=0.0019); task_14965_bank-marketing (delta +0.0010, p=0.0002); task_219_electricity (delta -0.0102, p=6.7e-08); task_3904_jm1 (delta -0.0072, p=0.0062); task_9977_nomao (delta -0.0002, p=0.0062)

- Seed yardstick (34 datasets): median |method gap| 0.00158 vs median Exact seed spread 0.00097 (ratio 1.38); method gap within 1 seed-spread on 38% of datasets

### Random vs Exact (positive delta = comparator better)

- Datasets: 36
- Mean per-dataset delta: +0.00160 (median -0.00010, range [-0.01092, +0.07744])
- 95% CI (t): [-0.00295, +0.00614]; 95% CI (bootstrap): [-0.00140, +0.00652]
- Wilcoxon signed-rank p = 0.3669; paired t p = 0.4804
- Win/tie/loss (|delta| <= 0.001 is a tie): 9/13/14 (raw sign: 12 up, 21 down)
- TOST equivalence p at margins {0.001: 0.6043, 0.0025: 0.3445, 0.005: 0.0687, 0.01: 0.0003, 0.02: 5.5e-10}
- **Equivalent (TOST) within +-0.01 at alpha=0.05**
- Non-inferiority p at margins {0.001: 0.1269, 0.0025: 0.0379, 0.005: 0.0028, 0.01: 4.6e-06, 0.02: 1.1e-11}
- **Non-inferior (no penalty worse than 0.0025) at alpha=0.05**
- Leave-one-out (drop task_14954_cylinder-bands): mean -0.00057, 95% CI [-0.00175, +0.00061], TOST margin 0.0025, non-inf margin 0.0025
- Per-dataset Wilcoxon (Holm): 6 dataset(s) significant: task_14954_cylinder-bands (delta +0.0774, p=0.0011); task_14965_bank-marketing (delta +0.0009, p=0.0015); task_219_electricity (delta -0.0109, p=6.7e-08); task_3904_jm1 (delta -0.0075, p=0.0018); task_7592_adult (delta -0.0006, p=0.0103); task_9977_nomao (delta -0.0002, p=0.0002)

- Seed yardstick (34 datasets): median |method gap| 0.00147 vs median Exact seed spread 0.00097 (ratio 1.45); method gap within 1 seed-spread on 41% of datasets

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

- Seed yardstick (34 datasets): median |method gap| 0.00107 vs median Exact seed spread 0.00097 (ratio 1.09); method gap within 1 seed-spread on 50% of datasets

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

- Seed yardstick (34 datasets): median |method gap| 0.00080 vs median Exact seed spread 0.00097 (ratio 0.67); method gap within 1 seed-spread on 62% of datasets

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

- Seed yardstick (34 datasets): median |method gap| 0.00137 vs median Exact seed spread 0.00097 (ratio 1.66); method gap within 1 seed-spread on 38% of datasets

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

- Seed yardstick (34 datasets): median |method gap| 0.00115 vs median Exact seed spread 0.00097 (ratio 1.18); method gap within 1 seed-spread on 44% of datasets

- Friedman over 34 datasets: chi2 = 7.42, p = 0.2834; average ranks (1 = best): Dynamic_Random_Histogram_thresh250: 4.18, Exact: 3.88, Random: 4.47, Random_bins128: 4.26, Random_bins16: 3.94, Random_bins256: 3.21, Random_bins32: 4.06

## SPO-GBT -- logloss

### Dynamic_Random_Histogram_thresh250 vs Exact (negative delta = comparator better)

- Datasets: 36
- Mean per-dataset delta: -0.00103 (median -0.00087, range [-0.02918, +0.05396])
- 95% CI (t): [-0.00620, +0.00413]; 95% CI (bootstrap): [-0.00559, +0.00416]
- Wilcoxon signed-rank p = 0.1012; paired t p = 0.6870
- Win/tie/loss (|delta| <= 0.001 is a tie): 8/10/18 (raw sign: 14 up, 20 down)
- TOST equivalence p at margins {0.001: 0.5053, 0.0025: 0.2842, 0.005: 0.0641, 0.01: 0.0006, 0.02: 5.0e-09}
- **Equivalent (TOST) within +-0.01 at alpha=0.05**
- Non-inferiority p at margins {0.001: 0.2148, 0.0025: 0.0869, 0.005: 0.0117, 0.01: 5.9e-05, 0.02: 4.8e-10}
- **Non-inferior (no penalty worse than 0.005) at alpha=0.05**
- Leave-one-out (drop task_15_breast-w): mean -0.00261, 95% CI [-0.00679, +0.00158], TOST margin 0.01, non-inf margin 0.001
- Per-dataset Wilcoxon (Holm): 7 dataset(s) significant: task_14965_bank-marketing (delta -0.0013, p=0.0031); task_15_breast-w (delta +0.0540, p=7.2e-06); task_167141_churn (delta -0.0065, p=1.6e-05); task_219_electricity (delta +0.0274, p=6.7e-08); task_29_credit-approval (delta +0.0376, p=0.0009); task_3904_jm1 (delta +0.0036, p=0.0402); task_9977_nomao (delta +0.0015, p=0.0004)

- Seed yardstick (34 datasets): median |method gap| 0.00357 vs median Exact seed spread 0.00324 (ratio 1.40); method gap within 1 seed-spread on 38% of datasets

### Random vs Exact (negative delta = comparator better)

- Datasets: 36
- Mean per-dataset delta: -0.00377 (median -0.00260, range [-0.05856, +0.05828])
- 95% CI (t): [-0.01077, +0.00324]; 95% CI (bootstrap): [-0.01020, +0.00315]
- Wilcoxon signed-rank p = 0.0131; paired t p = 0.2826
- Win/tie/loss (|delta| <= 0.001 is a tie): 7/8/21 (raw sign: 12 up, 22 down)
- TOST equivalence p at margins {0.001: 0.7858, 0.0025: 0.6420, 0.005: 0.3613, 0.01: 0.0397, 0.02: 1.9e-05}
- **Equivalent (TOST) within +-0.01 at alpha=0.05**
- Non-inferiority p at margins {0.001: 0.0880, 0.0025: 0.0390, 0.005: 0.0078, 0.01: 0.0002, 0.02: 2.7e-08}
- **Non-inferior (no penalty worse than 0.0025) at alpha=0.05**
- Leave-one-out (drop task_9971_ilpd): mean -0.00220, 95% CI [-0.00863, +0.00423], TOST margin 0.01, non-inf margin 0.005
- Per-dataset Wilcoxon (Holm): 13 dataset(s) significant: task_146820_wilt (delta -0.0047, p=0.0280); task_14965_bank-marketing (delta -0.0013, p=0.0035); task_15_breast-w (delta +0.0566, p=7.0e-06); task_167141_churn (delta -0.0041, p=0.0294); task_219_electricity (delta +0.0290, p=6.7e-08); task_29_credit-approval (delta +0.0583, p=1.3e-07); task_3902_pc4 (delta -0.0235, p=0.0065); task_3904_jm1 (delta +0.0038, p=0.0308); task_3917_kc1 (delta -0.0111, p=0.0412); task_7592_adult (delta +0.0010, p=0.0223); task_9957_qsar-biodeg (delta -0.0298, p=0.0110); task_9971_ilpd (delta -0.0586, p=0.0065); task_9977_nomao (delta +0.0019, p=6.6e-05)

- Seed yardstick (34 datasets): median |method gap| 0.00463 vs median Exact seed spread 0.00324 (ratio 2.80); method gap within 1 seed-spread on 29% of datasets

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

- Seed yardstick (34 datasets): median |method gap| 0.00378 vs median Exact seed spread 0.00324 (ratio 2.08); method gap within 1 seed-spread on 38% of datasets

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

- Seed yardstick (34 datasets): median |method gap| 0.00919 vs median Exact seed spread 0.00324 (ratio 4.25); method gap within 1 seed-spread on 18% of datasets

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

- Seed yardstick (34 datasets): median |method gap| 0.00438 vs median Exact seed spread 0.00324 (ratio 1.86); method gap within 1 seed-spread on 32% of datasets

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

- Seed yardstick (34 datasets): median |method gap| 0.00916 vs median Exact seed spread 0.00324 (ratio 3.29); method gap within 1 seed-spread on 15% of datasets

- Friedman over 34 datasets: chi2 = 19.16, p = 0.0039; average ranks (1 = best): Dynamic_Random_Histogram_thresh250: 4.41, Exact: 4.97, Random: 3.88, Random_bins128: 4.32, Random_bins16: 3.03, Random_bins256: 4.03, Random_bins32: 3.35

## SPO-RF -- accuracy

### Dynamic_Random_Histogram_thresh250 vs Exact (positive delta = comparator better)

- Datasets: 36
- Mean per-dataset delta: +0.00525 (median +0.00054, range [-0.00454, +0.15926])
- 95% CI (t): [-0.00372, +0.01422]; 95% CI (bootstrap): [+0.00028, +0.01448]
- Wilcoxon signed-rank p = 0.0238; paired t p = 0.2428
- Win/tie/loss (|delta| <= 0.001 is a tie): 13/17/6 (raw sign: 21 up, 13 down)
- TOST equivalence p at margins {0.001: 0.8286, 0.0025: 0.7311, 0.005: 0.5224, 0.01: 0.1449, 0.02: 0.0010}
- **Equivalent (TOST) within +-0.02 at alpha=0.05**
- Non-inferiority p at margins {0.001: 0.0830, 0.0025: 0.0441, 0.005: 0.0132, 0.01: 0.0007, 0.02: 9.2e-07}
- **Non-inferior (no penalty worse than 0.0025) at alpha=0.05**
- Leave-one-out (drop task_14954_cylinder-bands): mean +0.00085, 95% CI [+0.00001, +0.00169], TOST margin 0.0025, non-inf margin 0.001
- Per-dataset Wilcoxon (Holm): 2 dataset(s) significant: task_14954_cylinder-bands (delta +0.1593, p=6.2e-05); task_219_electricity (delta -0.0045, p=6.7e-05)

- Seed yardstick (34 datasets): median |method gap| 0.00122 vs median Exact seed spread 0.00111 (ratio 0.99); method gap within 1 seed-spread on 53% of datasets

### Random vs Exact (positive delta = comparator better)

- Datasets: 36
- Mean per-dataset delta: +0.00649 (median +0.00053, range [-0.01031, +0.21358])
- 95% CI (t): [-0.00557, +0.01854]; 95% CI (bootstrap): [-0.00013, +0.01882]
- Wilcoxon signed-rank p = 0.0249; paired t p = 0.2820
- Win/tie/loss (|delta| <= 0.001 is a tie): 12/19/5 (raw sign: 23 up, 11 down)
- TOST equivalence p at margins {0.001: 0.8191, 0.0025: 0.7468, 0.005: 0.5981, 0.01: 0.2789, 0.02: 0.0145}
- **Equivalent (TOST) within +-0.02 at alpha=0.05**
- Non-inferiority p at margins {0.001: 0.1078, 0.0025: 0.0695, 0.005: 0.0306, 0.01: 0.0044, 0.02: 4.0e-05}
- **Non-inferior (no penalty worse than 0.005) at alpha=0.05**
- Leave-one-out (drop task_14954_cylinder-bands): mean +0.00057, 95% CI [-0.00044, +0.00158], TOST margin 0.0025, non-inf margin 0.001
- Per-dataset Wilcoxon (Holm): 2 dataset(s) significant: task_14954_cylinder-bands (delta +0.2136, p=6.1e-05); task_219_electricity (delta -0.0103, p=6.1e-05)

- Seed yardstick (34 datasets): median |method gap| 0.00098 vs median Exact seed spread 0.00111 (ratio 0.94); method gap within 1 seed-spread on 59% of datasets

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

- Seed yardstick (34 datasets): median |method gap| 0.00135 vs median Exact seed spread 0.00111 (ratio 1.18); method gap within 1 seed-spread on 47% of datasets

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

- Seed yardstick (34 datasets): median |method gap| 0.00137 vs median Exact seed spread 0.00111 (ratio 1.25); method gap within 1 seed-spread on 47% of datasets

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

- Seed yardstick (34 datasets): median |method gap| 0.00128 vs median Exact seed spread 0.00111 (ratio 1.17); method gap within 1 seed-spread on 47% of datasets

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

- Seed yardstick (34 datasets): median |method gap| 0.00163 vs median Exact seed spread 0.00111 (ratio 1.73); method gap within 1 seed-spread on 32% of datasets

- Friedman over 34 datasets: chi2 = 10.01, p = 0.1243; average ranks (1 = best): Dynamic_Random_Histogram_thresh250: 3.72, Exact: 4.87, Random: 3.49, Random_bins128: 4.18, Random_bins16: 4.19, Random_bins256: 3.99, Random_bins32: 3.57

## SPO-RF -- auc

### Dynamic_Random_Histogram_thresh250 vs Exact (positive delta = comparator better)

- Datasets: 36
- Mean per-dataset delta: +0.00164 (median +0.00001, range [-0.00285, +0.05143])
- 95% CI (t): [-0.00128, +0.00456]; 95% CI (bootstrap): [-0.00008, +0.00473]
- Wilcoxon signed-rank p = 0.1768; paired t p = 0.2616
- Win/tie/loss (|delta| <= 0.001 is a tie): 9/24/3 (raw sign: 20 up, 13 down)
- TOST equivalence p at margins {0.001: 0.6706, 0.0025: 0.2769, 0.005: 0.0127, 0.01: 6.8e-07, 0.02: 4.9e-15}
- **Equivalent (TOST) within +-0.005 at alpha=0.05**
- Non-inferiority p at margins {0.001: 0.0374, 0.0025: 0.0034, 0.005: 2.5e-05, 0.01: 7.8e-10, 0.02: 0.0e+00}
- **Non-inferior (no penalty worse than 0.001) at alpha=0.05**
- Leave-one-out (drop task_14954_cylinder-bands): mean +0.00022, 95% CI [-0.00022, +0.00065], TOST margin 0.001, non-inf margin 0.001
- Per-dataset Wilcoxon (Holm): 3 dataset(s) significant: task_14954_cylinder-bands (delta +0.0514, p=8.0e-05); task_219_electricity (delta -0.0028, p=6.7e-08); task_7592_adult (delta -0.0013, p=3.3e-07)

- Seed yardstick (34 datasets): median |method gap| 0.00053 vs median Exact seed spread 0.00056 (ratio 0.68); method gap within 1 seed-spread on 59% of datasets

### Random vs Exact (positive delta = comparator better)

- Datasets: 36
- Mean per-dataset delta: +0.00337 (median +0.00040, range [-0.00627, +0.09808])
- 95% CI (t): [-0.00219, +0.00892]; 95% CI (bootstrap): [+0.00008, +0.00925]
- Wilcoxon signed-rank p = 0.0387; paired t p = 0.2264
- Win/tie/loss (|delta| <= 0.001 is a tie): 12/18/6 (raw sign: 22 up, 11 down)
- TOST equivalence p at margins {0.001: 0.8038, 0.0025: 0.6237, 0.005: 0.2775, 0.01: 0.0103, 0.02: 3.0e-07}
- **Equivalent (TOST) within +-0.01 at alpha=0.05**
- Non-inferiority p at margins {0.001: 0.0596, 0.0025: 0.0195, 0.005: 0.0021, 0.01: 1.1e-05, 0.02: 2.2e-10}
- **Non-inferior (no penalty worse than 0.0025) at alpha=0.05**
- Leave-one-out (drop task_14954_cylinder-bands): mean +0.00066, 95% CI [-0.00018, +0.00151], TOST margin 0.0025, non-inf margin 0.001
- Per-dataset Wilcoxon (Holm): 4 dataset(s) significant: task_10101_blood-transfusion-service-center (delta +0.0073, p=0.0235); task_14954_cylinder-bands (delta +0.0981, p=6.7e-08); task_219_electricity (delta -0.0063, p=6.7e-08); task_7592_adult (delta -0.0007, p=0.0050)

- Seed yardstick (34 datasets): median |method gap| 0.00109 vs median Exact seed spread 0.00056 (ratio 1.44); method gap within 1 seed-spread on 41% of datasets

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

- Seed yardstick (34 datasets): median |method gap| 0.00067 vs median Exact seed spread 0.00056 (ratio 1.48); method gap within 1 seed-spread on 38% of datasets

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

- Seed yardstick (34 datasets): median |method gap| 0.00174 vs median Exact seed spread 0.00056 (ratio 2.86); method gap within 1 seed-spread on 21% of datasets

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

- Seed yardstick (34 datasets): median |method gap| 0.00103 vs median Exact seed spread 0.00056 (ratio 1.44); method gap within 1 seed-spread on 35% of datasets

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

- Seed yardstick (34 datasets): median |method gap| 0.00124 vs median Exact seed spread 0.00056 (ratio 2.44); method gap within 1 seed-spread on 32% of datasets

- Friedman over 34 datasets: chi2 = 12.09, p = 0.0599; average ranks (1 = best): Dynamic_Random_Histogram_thresh250: 4.00, Exact: 4.47, Random: 3.38, Random_bins128: 4.76, Random_bins16: 3.76, Random_bins256: 4.18, Random_bins32: 3.44

## SPO-RF -- logloss

### Dynamic_Random_Histogram_thresh250 vs Exact (negative delta = comparator better)

- Datasets: 36
- Mean per-dataset delta: -0.00258 (median +0.00025, range [-0.10435, +0.03473])
- 95% CI (t): [-0.00924, +0.00408]; 95% CI (bootstrap): [-0.00990, +0.00263]
- Wilcoxon signed-rank p = 0.8926; paired t p = 0.4367
- Win/tie/loss (|delta| <= 0.001 is a tie): 12/13/11 (raw sign: 20 up, 14 down)
- TOST equivalence p at margins {0.001: 0.6836, 0.0025: 0.5098, 0.005: 0.2329, 0.01: 0.0150, 0.02: 3.1e-06}
- **Equivalent (TOST) within +-0.01 at alpha=0.05**
- Non-inferiority p at margins {0.001: 0.1413, 0.0025: 0.0652, 0.005: 0.0134, 0.01: 0.0003, 0.02: 2.7e-08}
- **Non-inferior (no penalty worse than 0.005) at alpha=0.05**
- Leave-one-out (drop task_14954_cylinder-bands): mean +0.00033, 95% CI [-0.00285, +0.00350], TOST margin 0.005, non-inf margin 0.005
- Per-dataset Wilcoxon (Holm): 3 dataset(s) significant: task_10093_banknote-authentication (delta -0.0016, p=9.4e-05); task_14954_cylinder-bands (delta -0.1044, p=6.7e-08); task_219_electricity (delta +0.0108, p=6.7e-08)

- Seed yardstick (34 datasets): median |method gap| 0.00145 vs median Exact seed spread 0.00189 (ratio 0.78); method gap within 1 seed-spread on 59% of datasets

### Random vs Exact (negative delta = comparator better)

- Datasets: 36
- Mean per-dataset delta: -0.00576 (median -0.00032, range [-0.17971, +0.02200])
- 95% CI (t): [-0.01619, +0.00466]; 95% CI (bootstrap): [-0.01708, +0.00109]
- Wilcoxon signed-rank p = 0.2087; paired t p = 0.2695
- Win/tie/loss (|delta| <= 0.001 is a tie): 10/10/16 (raw sign: 14 up, 20 down)
- TOST equivalence p at margins {0.001: 0.8199, 0.0025: 0.7353, 0.005: 0.5585, 0.01: 0.2074, 0.02: 0.0044}
- **Equivalent (TOST) within +-0.02 at alpha=0.05**
- Non-inferiority p at margins {0.001: 0.0982, 0.0025: 0.0583, 0.005: 0.0217, 0.01: 0.0021, 0.02: 7.6e-06}
- **Non-inferior (no penalty worse than 0.005) at alpha=0.05**
- Leave-one-out (drop task_14954_cylinder-bands): mean -0.00079, 95% CI [-0.00349, +0.00191], TOST margin 0.005, non-inf margin 0.0025
- Per-dataset Wilcoxon (Holm): 4 dataset(s) significant: task_10101_blood-transfusion-service-center (delta -0.0122, p=0.0064); task_14954_cylinder-bands (delta -0.1797, p=6.7e-08); task_219_electricity (delta +0.0220, p=6.7e-08); task_43_spambase (delta +0.0076, p=0.0167)

- Seed yardstick (34 datasets): median |method gap| 0.00187 vs median Exact seed spread 0.00189 (ratio 1.31); method gap within 1 seed-spread on 41% of datasets

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

- Seed yardstick (34 datasets): median |method gap| 0.00189 vs median Exact seed spread 0.00189 (ratio 1.21); method gap within 1 seed-spread on 44% of datasets

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

- Seed yardstick (34 datasets): median |method gap| 0.00361 vs median Exact seed spread 0.00189 (ratio 2.31); method gap within 1 seed-spread on 26% of datasets

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

- Seed yardstick (34 datasets): median |method gap| 0.00273 vs median Exact seed spread 0.00189 (ratio 1.45); method gap within 1 seed-spread on 41% of datasets

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

- Seed yardstick (34 datasets): median |method gap| 0.00245 vs median Exact seed spread 0.00189 (ratio 1.29); method gap within 1 seed-spread on 44% of datasets

- Friedman over 34 datasets: chi2 = 2.92, p = 0.8192; average ranks (1 = best): Dynamic_Random_Histogram_thresh250: 4.18, Exact: 4.00, Random: 3.68, Random_bins128: 4.35, Random_bins16: 4.21, Random_bins256: 3.85, Random_bins32: 3.74

