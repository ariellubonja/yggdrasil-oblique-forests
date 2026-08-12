# Split-finder accuracy comparison: Random Histogram vs Exact

Inputs: 87 CSVs, 36 datasets, seeds: [1, 2, 3]. Unit of analysis for all aggregate claims: the dataset (per-dataset mean of paired per-(seed,fold) deltas).

## SPO-GBT -- accuracy

### Dynamic_Random_Histogram_thresh250 vs Exact (positive delta = comparator better)

- Datasets: 36
- Mean per-dataset delta: +0.00103 (median +0.00010, range [-0.01629, +0.02346])
- 95% CI (t): [-0.00099, +0.00305]; 95% CI (bootstrap): [-0.00078, +0.00309]
- Wilcoxon signed-rank p = 0.4882; paired t p = 0.3055
- Win/tie/loss (|delta| <= 0.001 is a tie): 9/20/7 (raw sign: 18 up, 16 down)
- TOST equivalence p at margins {0.001: 0.5139, 0.0025: 0.0749, 0.005: 0.0002, 0.01: 6.0e-11, 0.02: 2.3e-20}
- **Equivalent (TOST) within +-0.005 at alpha=0.05**
- Non-inferiority p at margins {0.001: 0.0242, 0.0025: 0.0006, 0.005: 3.2e-07, 0.01: 2.6e-13, 0.02: 0.0e+00}
- **Non-inferior (no penalty worse than 0.001) at alpha=0.05**
- Leave-one-out (drop task_14954_cylinder-bands): mean +0.00039, 95% CI [-0.00120, +0.00199], TOST margin 0.0025, non-inf margin 0.001
- Per-dataset Wilcoxon (Holm): 2 dataset(s) significant: task_219_electricity (delta -0.0163, p=6.2e-05); task_7592_adult (delta -0.0009, p=0.0497)

- Seed yardstick (34 datasets): median |method gap| 0.00084 vs median Exact seed spread 0.00114 (ratio 1.09); method gap within 1 seed-spread on 50% of datasets

### Random vs Exact (positive delta = comparator better)

- Datasets: 36
- Mean per-dataset delta: +0.00108 (median +0.00036, range [-0.01671, +0.01975])
- 95% CI (t): [-0.00083, +0.00299]; 95% CI (bootstrap): [-0.00070, +0.00294]
- Wilcoxon signed-rank p = 0.1887; paired t p = 0.2575
- Win/tie/loss (|delta| <= 0.001 is a tie): 13/15/8 (raw sign: 20 up, 13 down)
- TOST equivalence p at margins {0.001: 0.5350, 0.0025: 0.0706, 0.005: 9.8e-05, 0.01: 1.7e-11, 0.02: 4.2e-21}
- **Equivalent (TOST) within +-0.005 at alpha=0.05**
- Non-inferiority p at margins {0.001: 0.0167, 0.0025: 0.0003, 0.005: 9.5e-08, 0.01: 5.0e-14, 0.02: 0.0e+00}
- **Non-inferior (no penalty worse than 0.001) at alpha=0.05**
- Leave-one-out (drop task_14954_cylinder-bands): mean +0.00055, 95% CI [-0.00107, +0.00217], TOST margin 0.0025, non-inf margin 0.001
- Per-dataset Wilcoxon (Holm): 2 dataset(s) significant: task_219_electricity (delta -0.0167, p=6.2e-05); task_3904_jm1 (delta -0.0036, p=0.0165)

- Seed yardstick (34 datasets): median |method gap| 0.00174 vs median Exact seed spread 0.00114 (ratio 1.48); method gap within 1 seed-spread on 41% of datasets

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

- Seed yardstick (34 datasets): median |method gap| 0.00112 vs median Exact seed spread 0.00114 (ratio 1.31); method gap within 1 seed-spread on 44% of datasets

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

- Seed yardstick (34 datasets): median |method gap| 0.00176 vs median Exact seed spread 0.00114 (ratio 1.66); method gap within 1 seed-spread on 38% of datasets

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

- Seed yardstick (34 datasets): median |method gap| 0.00159 vs median Exact seed spread 0.00114 (ratio 1.51); method gap within 1 seed-spread on 32% of datasets

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

- Seed yardstick (34 datasets): median |method gap| 0.00128 vs median Exact seed spread 0.00114 (ratio 1.06); method gap within 1 seed-spread on 47% of datasets

- Friedman over 34 datasets: chi2 = 3.47, p = 0.7478; average ranks (1 = best): Dynamic_Random_Histogram_thresh250: 3.99, Exact: 4.46, Random: 3.82, Random_bins128: 4.12, Random_bins16: 3.94, Random_bins256: 4.12, Random_bins32: 3.56

## SPO-GBT -- auc

### Dynamic_Random_Histogram_thresh250 vs Exact (positive delta = comparator better)

- Datasets: 36
- Mean per-dataset delta: +0.00041 (median -0.00004, range [-0.01168, +0.02657])
- 95% CI (t): [-0.00152, +0.00234]; 95% CI (bootstrap): [-0.00121, +0.00245]
- Wilcoxon signed-rank p = 0.6850; paired t p = 0.6685
- Win/tie/loss (|delta| <= 0.001 is a tie): 8/17/11 (raw sign: 14 up, 19 down)
- TOST equivalence p at margins {0.001: 0.2690, 0.0025: 0.0172, 0.005: 1.3e-05, 0.01: 3.2e-12, 0.02: 1.8e-21}
- **Equivalent (TOST) within +-0.0025 at alpha=0.05**
- Non-inferiority p at margins {0.001: 0.0732, 0.0025: 0.0021, 0.005: 9.6e-07, 0.01: 3.6e-13, 0.02: 0.0e+00}
- **Non-inferior (no penalty worse than 0.0025) at alpha=0.05**
- Leave-one-out (drop task_14954_cylinder-bands): mean -0.00034, 95% CI [-0.00156, +0.00089], TOST margin 0.0025, non-inf margin 0.0025
- Per-dataset Wilcoxon (Holm): 4 dataset(s) significant: task_219_electricity (delta -0.0117, p=6.7e-08); task_3904_jm1 (delta -0.0049, p=0.0220); task_7592_adult (delta -0.0008, p=0.0058); task_9977_nomao (delta -0.0002, p=0.0008)

- Seed yardstick (34 datasets): median |method gap| 0.00132 vs median Exact seed spread 0.00113 (ratio 1.08); method gap within 1 seed-spread on 44% of datasets

### Random vs Exact (positive delta = comparator better)

- Datasets: 36
- Mean per-dataset delta: +0.00124 (median +0.00048, range [-0.01195, +0.02318])
- 95% CI (t): [-0.00083, +0.00331]; 95% CI (bootstrap): [-0.00063, +0.00334]
- Wilcoxon signed-rank p = 0.0970; paired t p = 0.2305
- Win/tie/loss (|delta| <= 0.001 is a tie): 12/17/7 (raw sign: 22 up, 11 down)
- TOST equivalence p at margins {0.001: 0.5937, 0.0025: 0.1128, 0.005: 0.0004, 0.01: 1.9e-10, 0.02: 7.0e-20}
- **Equivalent (TOST) within +-0.005 at alpha=0.05**
- Non-inferiority p at margins {0.001: 0.0172, 0.0025: 0.0004, 0.005: 2.6e-07, 0.01: 3.0e-13, 0.02: 0.0e+00}
- **Non-inferior (no penalty worse than 0.001) at alpha=0.05**
- Leave-one-out (drop task_14954_cylinder-bands): mean +0.00062, 95% CI [-0.00106, +0.00230], TOST margin 0.0025, non-inf margin 0.001
- Per-dataset Wilcoxon (Holm): 4 dataset(s) significant: task_219_electricity (delta -0.0119, p=6.7e-08); task_3904_jm1 (delta -0.0076, p=0.0062); task_7592_adult (delta -0.0008, p=0.0026); task_9977_nomao (delta -0.0002, p=0.0046)

- Seed yardstick (34 datasets): median |method gap| 0.00152 vs median Exact seed spread 0.00113 (ratio 1.01); method gap within 1 seed-spread on 53% of datasets

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

- Seed yardstick (34 datasets): median |method gap| 0.00144 vs median Exact seed spread 0.00113 (ratio 1.45); method gap within 1 seed-spread on 41% of datasets

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

- Seed yardstick (34 datasets): median |method gap| 0.00318 vs median Exact seed spread 0.00113 (ratio 2.38); method gap within 1 seed-spread on 24% of datasets

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

- Seed yardstick (34 datasets): median |method gap| 0.00182 vs median Exact seed spread 0.00113 (ratio 1.33); method gap within 1 seed-spread on 38% of datasets

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

- Seed yardstick (34 datasets): median |method gap| 0.00121 vs median Exact seed spread 0.00113 (ratio 1.10); method gap within 1 seed-spread on 50% of datasets

- Friedman over 34 datasets: chi2 = 1.20, p = 0.9769; average ranks (1 = best): Dynamic_Random_Histogram_thresh250: 4.00, Exact: 4.24, Random: 3.71, Random_bins128: 4.06, Random_bins16: 3.91, Random_bins256: 4.03, Random_bins32: 4.06

## SPO-GBT -- logloss

### Dynamic_Random_Histogram_thresh250 vs Exact (negative delta = comparator better)

- Datasets: 36
- Mean per-dataset delta: -0.00226 (median -0.00055, range [-0.03195, +0.06525])
- 95% CI (t): [-0.00812, +0.00360]; 95% CI (bootstrap): [-0.00751, +0.00364]
- Wilcoxon signed-rank p = 0.0941; paired t p = 0.4384
- Win/tie/loss (|delta| <= 0.001 is a tie): 10/10/16 (raw sign: 13 up, 21 down)
- TOST equivalence p at margins {0.001: 0.6677, 0.0025: 0.4675, 0.005: 0.1748, 0.01: 0.0056, 0.02: 2.5e-07}
- **Equivalent (TOST) within +-0.01 at alpha=0.05**
- Non-inferiority p at margins {0.001: 0.1331, 0.0025: 0.0540, 0.005: 0.0083, 0.01: 7.6e-05, 0.02: 2.4e-09}
- **Non-inferior (no penalty worse than 0.005) at alpha=0.05**
- Leave-one-out (drop task_29_credit-approval): mean -0.00419, 95% CI [-0.00868, +0.00030], TOST margin 0.01, non-inf margin 0.001
- Per-dataset Wilcoxon (Holm): 7 dataset(s) significant: task_146819_climate-model-simulation-crashes (delta -0.0320, p=0.0094); task_15_breast-w (delta +0.0269, p=1.2e-06); task_219_electricity (delta +0.0284, p=6.7e-08); task_29_credit-approval (delta +0.0652, p=2.0e-07); task_7592_adult (delta +0.0015, p=0.0058); task_9957_qsar-biodeg (delta -0.0276, p=0.0001); task_9977_nomao (delta +0.0022, p=2.0e-06)

- Seed yardstick (34 datasets): median |method gap| 0.00361 vs median Exact seed spread 0.00155 (ratio 1.72); method gap within 1 seed-spread on 32% of datasets

### Random vs Exact (negative delta = comparator better)

- Datasets: 36
- Mean per-dataset delta: -0.00689 (median -0.00352, range [-0.07695, +0.07672])
- 95% CI (t): [-0.01492, +0.00113]; 95% CI (bootstrap): [-0.01444, +0.00096]
- Wilcoxon signed-rank p = 0.0091; paired t p = 0.0901
- Win/tie/loss (|delta| <= 0.001 is a tie): 8/7/21 (raw sign: 10 up, 24 down)
- TOST equivalence p at margins {0.001: 0.9274, 0.0025: 0.8629, 0.005: 0.6823, 0.01: 0.2184, 0.02: 0.0011}
- **Equivalent (TOST) within +-0.02 at alpha=0.05**
- Non-inferiority p at margins {0.001: 0.0269, 0.0025: 0.0116, 0.005: 0.0024, 0.01: 7.0e-05, 0.02: 3.4e-08}
- **Non-inferior (no penalty worse than 0.001) at alpha=0.05**
- Leave-one-out (drop task_9971_ilpd): mean -0.00489, 95% CI [-0.01202, +0.00224], TOST margin 0.02, non-inf margin 0.0025
- Per-dataset Wilcoxon (Holm): 16 dataset(s) significant: task_10093_banknote-authentication (delta -0.0012, p=0.0296); task_146819_climate-model-simulation-crashes (delta -0.0409, p=0.0260); task_146820_wilt (delta -0.0043, p=0.0395); task_15_breast-w (delta +0.0259, p=3.7e-05); task_219_electricity (delta +0.0289, p=6.7e-08); task_29_credit-approval (delta +0.0767, p=1.3e-07); task_3902_pc4 (delta -0.0223, p=0.0023); task_3903_pc3 (delta -0.0183, p=0.0124); task_3904_jm1 (delta +0.0050, p=0.0065); task_3917_kc1 (delta -0.0097, p=0.0497); task_7592_adult (delta +0.0015, p=0.0002); task_9946_wdbc (delta -0.0149, p=0.0190); task_9952_phoneme (delta -0.0036, p=0.0410); task_9957_qsar-biodeg (delta -0.0403, p=2.7e-05); task_9971_ilpd (delta -0.0770, p=1.6e-05); task_9977_nomao (delta +0.0022, p=3.2e-05)

- Seed yardstick (34 datasets): median |method gap| 0.00950 vs median Exact seed spread 0.00155 (ratio 3.57); method gap within 1 seed-spread on 15% of datasets

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

- Seed yardstick (34 datasets): median |method gap| 0.00424 vs median Exact seed spread 0.00155 (ratio 1.86); method gap within 1 seed-spread on 35% of datasets

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

- Seed yardstick (34 datasets): median |method gap| 0.00709 vs median Exact seed spread 0.00155 (ratio 6.36); method gap within 1 seed-spread on 18% of datasets

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

- Seed yardstick (34 datasets): median |method gap| 0.00344 vs median Exact seed spread 0.00155 (ratio 1.98); method gap within 1 seed-spread on 29% of datasets

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

- Seed yardstick (34 datasets): median |method gap| 0.00763 vs median Exact seed spread 0.00155 (ratio 3.12); method gap within 1 seed-spread on 18% of datasets

- Friedman over 34 datasets: chi2 = 22.59, p = 0.0009; average ranks (1 = best): Dynamic_Random_Histogram_thresh250: 4.38, Exact: 4.91, Random: 3.44, Random_bins128: 4.18, Random_bins16: 3.12, Random_bins256: 4.68, Random_bins32: 3.29

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

