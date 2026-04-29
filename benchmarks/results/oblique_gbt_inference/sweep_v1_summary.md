# sweep_v1

## Raw rows

| Dataset | Split | Depth | Trees | Acc | Best µs/ex (engine) | Generic µs/ex |
|---------|-------|-------|-------|-----|---------------------|---------------|
| HIGGS | Axis Aligned | 3 | 100 | 0.6987 | 0.228 (GradientBoostedTreesQuickScorerExtended) | 1.501 |
| HIGGS | Axis Aligned | 4 | 100 | 0.7141 | 0.396 (GradientBoostedTreesQuickScorerExtended) | 2.319 |
| HIGGS | Axis Aligned | 5 | 100 | 0.7231 | 0.655 (GradientBoostedTreesQuickScorerExtended) | 3.228 |
| HIGGS | Axis Aligned | 6 | 100 | 0.7278 | 1.164 (GradientBoostedTreesQuickScorerExtended) | 4.119 |
| HIGGS | Axis Aligned | 7 | 100 | 0.7332 | 2.060 (GradientBoostedTreesQuickScorerExtended) | 5.066 |
| HIGGS | Axis Aligned | 8 | 100 | 0.7366 | 4.323 (GradientBoostedTreesOptPred) | 6.038 |
| HIGGS | Axis Aligned | 10 | 100 | 0.7406 | 6.030 (GradientBoostedTreesOptPred) | 8.842 |
| HIGGS | Axis Aligned | 12 | 99 | 0.7414 | 7.610 (GradientBoostedTreesOptPred) | 10.666 |
| HIGGS | Oblique | 2 | 100 | 0.6788 | 1.166 (GradientBoostedTreesGeneric) | 1.166 |
| HIGGS | Oblique | 3 | 100 | 0.7048 | 2.135 (GradientBoostedTreesGeneric) | 2.135 |
| HIGGS | Oblique | 4 | 100 | 0.7144 | 3.429 (GradientBoostedTreesGeneric) | 3.429 |
| HIGGS | Oblique | 5 | 100 | 0.7223 | 5.243 (GradientBoostedTreesGeneric) | 5.243 |
| HIGGS | Oblique | 6 | 100 | 0.7276 | 6.854 (GradientBoostedTreesGeneric) | 6.854 |
| HIGGS | Oblique | 8 | 100 | 0.7360 | 11.037 (GradientBoostedTreesGeneric) | 11.037 |
| SUSY | Axis Aligned | 3 | 100 | 0.7962 | 0.258 (GradientBoostedTreesQuickScorerExtended) | 1.508 |
| SUSY | Axis Aligned | 4 | 100 | 0.7994 | 0.370 (GradientBoostedTreesQuickScorerExtended) | 2.262 |
| SUSY | Axis Aligned | 5 | 100 | 0.8016 | 0.613 (GradientBoostedTreesQuickScorerExtended) | 3.231 |
| SUSY | Axis Aligned | 6 | 100 | 0.8028 | 1.077 (GradientBoostedTreesQuickScorerExtended) | 4.139 |
| SUSY | Axis Aligned | 7 | 98 | 0.8027 | 1.914 (GradientBoostedTreesQuickScorerExtended) | 4.847 |
| SUSY | Axis Aligned | 8 | 97 | 0.8034 | 4.018 (GradientBoostedTreesOptPred) | 5.565 |
| SUSY | Axis Aligned | 10 | 87 | 0.8032 | 4.854 (GradientBoostedTreesOptPred) | 7.044 |
| SUSY | Axis Aligned | 12 | 58 | 0.8021 | 4.212 (GradientBoostedTreesOptPred) | 5.918 |
| SUSY | Oblique | 2 | 100 | 0.7929 | 1.006 (GradientBoostedTreesGeneric) | 1.006 |
| SUSY | Oblique | 3 | 100 | 0.7988 | 1.925 (GradientBoostedTreesGeneric) | 1.925 |
| SUSY | Oblique | 4 | 100 | 0.8006 | 3.164 (GradientBoostedTreesGeneric) | 3.164 |
| SUSY | Oblique | 5 | 100 | 0.8024 | 4.932 (GradientBoostedTreesGeneric) | 4.932 |
| SUSY | Oblique | 6 | 100 | 0.8032 | 6.719 (GradientBoostedTreesGeneric) | 6.719 |
| SUSY | Oblique | 8 | 100 | 0.8035 | 10.465 (GradientBoostedTreesGeneric) | 10.465 |

## Apples-to-apples (Generic engine): oblique vs nearest-accuracy axis-aligned

(Generic engine for both (apples-to-apples). Lower latency = better. Ratio > 1.0 means axis-aligned is faster.)

| Dataset | Oblique (d, acc, µs) | Nearest axis (d, acc, µs) | Δ acc | Speed ratio (axis/oblique) |
|---------|----------------------|----------------------------|--------|----------------------------|
| SUSY | d=2 acc=0.7929 µs=1.006 | d=3 acc=0.7962 µs=1.508 | -0.0033 | 1.50x |
| SUSY | d=3 acc=0.7988 µs=1.925 | d=4 acc=0.7994 µs=2.262 | -0.0007 | 1.17x |
| SUSY | d=4 acc=0.8006 µs=3.164 | d=5 acc=0.8016 µs=3.231 | -0.0010 | 1.02x |
| SUSY | d=5 acc=0.8024 µs=4.932 | d=12 acc=0.8021 µs=5.918 | +0.0002 | 1.20x |
| SUSY | d=6 acc=0.8032 µs=6.719 | d=10 acc=0.8032 µs=7.044 | -0.0000 | 1.05x |
| SUSY | d=8 acc=0.8035 µs=10.465 | d=8 acc=0.8034 µs=5.565 | +0.0001 | 0.53x |
| HIGGS | d=2 acc=0.6788 µs=1.166 | d=3 acc=0.6987 µs=1.501 | -0.0199 | 1.29x |
| HIGGS | d=3 acc=0.7048 µs=2.135 | d=3 acc=0.6987 µs=1.501 | +0.0062 | 0.70x |
| HIGGS | d=4 acc=0.7144 µs=3.429 | d=4 acc=0.7141 µs=2.319 | +0.0003 | 0.68x |
| HIGGS | d=5 acc=0.7223 µs=5.243 | d=5 acc=0.7231 µs=3.228 | -0.0008 | 0.62x |
| HIGGS | d=6 acc=0.7276 µs=6.854 | d=6 acc=0.7278 µs=4.119 | -0.0002 | 0.60x |
| HIGGS | d=8 acc=0.7360 µs=11.037 | d=8 acc=0.7366 µs=6.038 | -0.0006 | 0.55x |

## Best-engine comparison: oblique vs nearest-accuracy axis-aligned

(best engine for each (deployment-realistic). Lower latency = better. Ratio > 1.0 means axis-aligned is faster.)

| Dataset | Oblique (d, acc, µs) | Nearest axis (d, acc, µs) | Δ acc | Speed ratio (axis/oblique) |
|---------|----------------------|----------------------------|--------|----------------------------|
| SUSY | d=2 acc=0.7929 µs=1.006 | d=3 acc=0.7962 µs=0.258 | -0.0033 | 0.26x |
| SUSY | d=3 acc=0.7988 µs=1.925 | d=4 acc=0.7994 µs=0.370 | -0.0007 | 0.19x |
| SUSY | d=4 acc=0.8006 µs=3.164 | d=5 acc=0.8016 µs=0.613 | -0.0010 | 0.19x |
| SUSY | d=5 acc=0.8024 µs=4.932 | d=12 acc=0.8021 µs=4.212 | +0.0002 | 0.85x |
| SUSY | d=6 acc=0.8032 µs=6.719 | d=10 acc=0.8032 µs=4.854 | -0.0000 | 0.72x |
| SUSY | d=8 acc=0.8035 µs=10.465 | d=8 acc=0.8034 µs=4.018 | +0.0001 | 0.38x |
| HIGGS | d=2 acc=0.6788 µs=1.166 | d=3 acc=0.6987 µs=0.228 | -0.0199 | 0.20x |
| HIGGS | d=3 acc=0.7048 µs=2.135 | d=3 acc=0.6987 µs=0.228 | +0.0062 | 0.11x |
| HIGGS | d=4 acc=0.7144 µs=3.429 | d=4 acc=0.7141 µs=0.396 | +0.0003 | 0.12x |
| HIGGS | d=5 acc=0.7223 µs=5.243 | d=5 acc=0.7231 µs=0.655 | -0.0008 | 0.13x |
| HIGGS | d=6 acc=0.7276 µs=6.854 | d=6 acc=0.7278 µs=1.164 | -0.0002 | 0.17x |
| HIGGS | d=8 acc=0.7360 µs=11.037 | d=8 acc=0.7366 µs=4.323 | -0.0006 | 0.39x |

## Pareto frontier per dataset (apples-to-apples Generic engine)

### HIGGS

| µs/ex | Acc | Split | Depth | Engine |
|-------|-----|-------|-------|--------|
| 1.166 | 0.6788 | Oblique | 2 | GradientBoostedTreesGeneric |
| 1.501 | 0.6987 | Axis Aligned | 3 | GradientBoostedTreesQuickScorerExtended |
| 2.135 | 0.7048 | Oblique | 3 | GradientBoostedTreesGeneric |
| 2.319 | 0.7141 | Axis Aligned | 4 | GradientBoostedTreesQuickScorerExtended |
| 3.228 | 0.7231 | Axis Aligned | 5 | GradientBoostedTreesQuickScorerExtended |
| 4.119 | 0.7278 | Axis Aligned | 6 | GradientBoostedTreesQuickScorerExtended |
| 5.066 | 0.7332 | Axis Aligned | 7 | GradientBoostedTreesQuickScorerExtended |
| 6.038 | 0.7366 | Axis Aligned | 8 | GradientBoostedTreesOptPred |
| 8.842 | 0.7406 | Axis Aligned | 10 | GradientBoostedTreesOptPred |
| 10.666 | 0.7414 | Axis Aligned | 12 | GradientBoostedTreesOptPred |

### SUSY

| µs/ex | Acc | Split | Depth | Engine |
|-------|-----|-------|-------|--------|
| 1.006 | 0.7929 | Oblique | 2 | GradientBoostedTreesGeneric |
| 1.508 | 0.7962 | Axis Aligned | 3 | GradientBoostedTreesQuickScorerExtended |
| 1.925 | 0.7988 | Oblique | 3 | GradientBoostedTreesGeneric |
| 2.262 | 0.7994 | Axis Aligned | 4 | GradientBoostedTreesQuickScorerExtended |
| 3.164 | 0.8006 | Oblique | 4 | GradientBoostedTreesGeneric |
| 3.231 | 0.8016 | Axis Aligned | 5 | GradientBoostedTreesQuickScorerExtended |
| 4.139 | 0.8028 | Axis Aligned | 6 | GradientBoostedTreesQuickScorerExtended |
| 5.565 | 0.8034 | Axis Aligned | 8 | GradientBoostedTreesOptPred |
| 10.465 | 0.8035 | Oblique | 8 | GradientBoostedTreesGeneric |


## Pareto frontier per dataset (best engine)

### HIGGS

| µs/ex | Acc | Split | Depth | Engine |
|-------|-----|-------|-------|--------|
| 0.228 | 0.6987 | Axis Aligned | 3 | GradientBoostedTreesQuickScorerExtended |
| 0.396 | 0.7141 | Axis Aligned | 4 | GradientBoostedTreesQuickScorerExtended |
| 0.655 | 0.7231 | Axis Aligned | 5 | GradientBoostedTreesQuickScorerExtended |
| 1.164 | 0.7278 | Axis Aligned | 6 | GradientBoostedTreesQuickScorerExtended |
| 2.060 | 0.7332 | Axis Aligned | 7 | GradientBoostedTreesQuickScorerExtended |
| 4.323 | 0.7366 | Axis Aligned | 8 | GradientBoostedTreesOptPred |
| 6.030 | 0.7406 | Axis Aligned | 10 | GradientBoostedTreesOptPred |
| 7.610 | 0.7414 | Axis Aligned | 12 | GradientBoostedTreesOptPred |

### SUSY

| µs/ex | Acc | Split | Depth | Engine |
|-------|-----|-------|-------|--------|
| 0.258 | 0.7962 | Axis Aligned | 3 | GradientBoostedTreesQuickScorerExtended |
| 0.370 | 0.7994 | Axis Aligned | 4 | GradientBoostedTreesQuickScorerExtended |
| 0.613 | 0.8016 | Axis Aligned | 5 | GradientBoostedTreesQuickScorerExtended |
| 1.077 | 0.8028 | Axis Aligned | 6 | GradientBoostedTreesQuickScorerExtended |
| 4.018 | 0.8034 | Axis Aligned | 8 | GradientBoostedTreesOptPred |
| 10.465 | 0.8035 | Oblique | 8 | GradientBoostedTreesGeneric |

