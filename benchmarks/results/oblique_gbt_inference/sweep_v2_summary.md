# sweep_v2

## Raw rows

| Dataset | Split | Depth | Trees | Acc | Best µs/ex (engine) | Generic µs/ex |
|---------|-------|-------|-------|-----|---------------------|---------------|
| bioresponse | Axis Aligned | 3 | 100 | 0.7803 | 0.604 (GradientBoostedTreesQuickScorerExtended) | 1.119 |
| bioresponse | Axis Aligned | 4 | 100 | 0.7803 | 1.479 (GradientBoostedTreesQuickScorerExtended) | 1.674 |
| bioresponse | Axis Aligned | 5 | 89 | 0.7909 | 2.146 (GradientBoostedTreesGeneric) | 2.146 |
| bioresponse | Axis Aligned | 6 | 91 | 0.8003 | 2.954 (GradientBoostedTreesGeneric) | 2.954 |
| bioresponse | Axis Aligned | 7 | 71 | 0.7949 | 2.738 (GradientBoostedTreesGeneric) | 2.738 |
| bioresponse | Axis Aligned | 8 | 60 | 0.7883 | 2.785 (GradientBoostedTreesGeneric) | 2.785 |
| bioresponse | Axis Aligned | 10 | 35 | 0.7949 | 2.275 (GradientBoostedTreesGeneric) | 2.275 |
| bioresponse | Axis Aligned | 12 | 26 | 0.7936 | 1.912 (GradientBoostedTreesGeneric) | 1.912 |
| bioresponse | Oblique | 2 | 98 | 0.7617 | 0.756 (GradientBoostedTreesGeneric) | 0.756 |
| bioresponse | Oblique | 3 | 100 | 0.7696 | 1.642 (GradientBoostedTreesGeneric) | 1.642 |
| bioresponse | Oblique | 4 | 100 | 0.7870 | 2.825 (GradientBoostedTreesGeneric) | 2.825 |
| bioresponse | Oblique | 5 | 100 | 0.7870 | 4.348 (GradientBoostedTreesGeneric) | 4.348 |
| bioresponse | Oblique | 6 | 92 | 0.7936 | 5.818 (GradientBoostedTreesGeneric) | 5.818 |
| bioresponse | Oblique | 8 | 61 | 0.7870 | 5.831 (GradientBoostedTreesGeneric) | 5.831 |
| madelon | Axis Aligned | 3 | 42 | 0.6904 | 0.272 (GradientBoostedTreesQuickScorerExtended) | 0.533 |
| madelon | Axis Aligned | 4 | 32 | 0.7269 | 0.357 (GradientBoostedTreesOptPred) | 0.564 |
| madelon | Axis Aligned | 5 | 100 | 0.7788 | 2.064 (GradientBoostedTreesOptPred) | 3.088 |
| madelon | Axis Aligned | 6 | 69 | 0.8058 | 1.787 (GradientBoostedTreesOptPred) | 2.683 |
| madelon | Axis Aligned | 7 | 93 | 0.8135 | 3.366 (GradientBoostedTreesOptPred) | 4.588 |
| madelon | Axis Aligned | 8 | 48 | 0.8269 | 1.932 (GradientBoostedTreesOptPred) | 2.526 |
| madelon | Axis Aligned | 10 | 47 | 0.8154 | 2.416 (GradientBoostedTreesOptPred) | 3.559 |
| madelon | Axis Aligned | 12 | 45 | 0.8192 | 2.536 (GradientBoostedTreesOptPred) | 3.858 |
| madelon | Oblique | 2 | 100 | 0.6404 | 0.766 (GradientBoostedTreesGeneric) | 0.766 |
| madelon | Oblique | 3 | 97 | 0.7250 | 1.716 (GradientBoostedTreesGeneric) | 1.716 |
| madelon | Oblique | 4 | 97 | 0.7712 | 2.917 (GradientBoostedTreesGeneric) | 2.917 |
| madelon | Oblique | 5 | 94 | 0.7942 | 4.107 (GradientBoostedTreesGeneric) | 4.107 |
| madelon | Oblique | 6 | 83 | 0.8212 | 4.978 (GradientBoostedTreesGeneric) | 4.978 |
| madelon | Oblique | 8 | 90 | 0.7962 | 8.993 (GradientBoostedTreesGeneric) | 8.993 |

## Apples-to-apples (Generic engine): oblique vs nearest-accuracy axis-aligned

(Generic engine for both (apples-to-apples). Lower latency = better. Ratio > 1.0 means axis-aligned is faster.)

| Dataset | Oblique (d, acc, µs) | Nearest axis (d, acc, µs) | Δ acc | Speed ratio (axis/oblique) |
|---------|----------------------|----------------------------|--------|----------------------------|
| madelon | d=2 acc=0.6404 µs=0.766 | d=3 acc=0.6904 µs=0.533 | -0.0500 | 0.70x |
| madelon | d=3 acc=0.7250 µs=1.716 | d=4 acc=0.7269 µs=0.564 | -0.0019 | 0.33x |
| madelon | d=4 acc=0.7712 µs=2.917 | d=5 acc=0.7788 µs=3.088 | -0.0077 | 1.06x |
| madelon | d=5 acc=0.7942 µs=4.107 | d=6 acc=0.8058 µs=2.683 | -0.0115 | 0.65x |
| madelon | d=6 acc=0.8212 µs=4.978 | d=12 acc=0.8192 µs=3.858 | +0.0019 | 0.78x |
| madelon | d=8 acc=0.7962 µs=8.993 | d=6 acc=0.8058 µs=2.683 | -0.0096 | 0.30x |
| bioresponse | d=2 acc=0.7617 µs=0.756 | d=3 acc=0.7803 µs=1.119 | -0.0186 | 1.48x |
| bioresponse | d=3 acc=0.7696 µs=1.642 | d=3 acc=0.7803 µs=1.119 | -0.0107 | 0.68x |
| bioresponse | d=4 acc=0.7870 µs=2.825 | d=8 acc=0.7883 µs=2.785 | -0.0013 | 0.99x |
| bioresponse | d=5 acc=0.7870 µs=4.348 | d=8 acc=0.7883 µs=2.785 | -0.0013 | 0.64x |
| bioresponse | d=6 acc=0.7936 µs=5.818 | d=12 acc=0.7936 µs=1.912 | +0.0000 | 0.33x |
| bioresponse | d=8 acc=0.7870 µs=5.831 | d=8 acc=0.7883 µs=2.785 | -0.0013 | 0.48x |

## Best-engine comparison: oblique vs nearest-accuracy axis-aligned

(best engine for each (deployment-realistic). Lower latency = better. Ratio > 1.0 means axis-aligned is faster.)

| Dataset | Oblique (d, acc, µs) | Nearest axis (d, acc, µs) | Δ acc | Speed ratio (axis/oblique) |
|---------|----------------------|----------------------------|--------|----------------------------|
| madelon | d=2 acc=0.6404 µs=0.766 | d=3 acc=0.6904 µs=0.272 | -0.0500 | 0.36x |
| madelon | d=3 acc=0.7250 µs=1.716 | d=4 acc=0.7269 µs=0.357 | -0.0019 | 0.21x |
| madelon | d=4 acc=0.7712 µs=2.917 | d=5 acc=0.7788 µs=2.064 | -0.0077 | 0.71x |
| madelon | d=5 acc=0.7942 µs=4.107 | d=6 acc=0.8058 µs=1.787 | -0.0115 | 0.44x |
| madelon | d=6 acc=0.8212 µs=4.978 | d=12 acc=0.8192 µs=2.536 | +0.0019 | 0.51x |
| madelon | d=8 acc=0.7962 µs=8.993 | d=6 acc=0.8058 µs=1.787 | -0.0096 | 0.20x |
| bioresponse | d=2 acc=0.7617 µs=0.756 | d=3 acc=0.7803 µs=0.604 | -0.0186 | 0.80x |
| bioresponse | d=3 acc=0.7696 µs=1.642 | d=3 acc=0.7803 µs=0.604 | -0.0107 | 0.37x |
| bioresponse | d=4 acc=0.7870 µs=2.825 | d=8 acc=0.7883 µs=2.785 | -0.0013 | 0.99x |
| bioresponse | d=5 acc=0.7870 µs=4.348 | d=8 acc=0.7883 µs=2.785 | -0.0013 | 0.64x |
| bioresponse | d=6 acc=0.7936 µs=5.818 | d=12 acc=0.7936 µs=1.912 | +0.0000 | 0.33x |
| bioresponse | d=8 acc=0.7870 µs=5.831 | d=8 acc=0.7883 µs=2.785 | -0.0013 | 0.48x |

## Pareto frontier per dataset (apples-to-apples Generic engine)

### bioresponse

| µs/ex | Acc | Split | Depth | Engine |
|-------|-----|-------|-------|--------|
| 0.756 | 0.7617 | Oblique | 2 | GradientBoostedTreesGeneric |
| 1.119 | 0.7803 | Axis Aligned | 3 | GradientBoostedTreesQuickScorerExtended |
| 1.912 | 0.7936 | Axis Aligned | 12 | GradientBoostedTreesGeneric |
| 2.275 | 0.7949 | Axis Aligned | 10 | GradientBoostedTreesGeneric |
| 2.954 | 0.8003 | Axis Aligned | 6 | GradientBoostedTreesGeneric |

### madelon

| µs/ex | Acc | Split | Depth | Engine |
|-------|-----|-------|-------|--------|
| 0.533 | 0.6904 | Axis Aligned | 3 | GradientBoostedTreesQuickScorerExtended |
| 0.564 | 0.7269 | Axis Aligned | 4 | GradientBoostedTreesOptPred |
| 2.526 | 0.8269 | Axis Aligned | 8 | GradientBoostedTreesOptPred |


## Pareto frontier per dataset (best engine)

### bioresponse

| µs/ex | Acc | Split | Depth | Engine |
|-------|-----|-------|-------|--------|
| 0.604 | 0.7803 | Axis Aligned | 3 | GradientBoostedTreesQuickScorerExtended |
| 1.912 | 0.7936 | Axis Aligned | 12 | GradientBoostedTreesGeneric |
| 2.275 | 0.7949 | Axis Aligned | 10 | GradientBoostedTreesGeneric |
| 2.954 | 0.8003 | Axis Aligned | 6 | GradientBoostedTreesGeneric |

### madelon

| µs/ex | Acc | Split | Depth | Engine |
|-------|-----|-------|-------|--------|
| 0.272 | 0.6904 | Axis Aligned | 3 | GradientBoostedTreesQuickScorerExtended |
| 0.357 | 0.7269 | Axis Aligned | 4 | GradientBoostedTreesOptPred |
| 1.787 | 0.8058 | Axis Aligned | 6 | GradientBoostedTreesOptPred |
| 1.932 | 0.8269 | Axis Aligned | 8 | GradientBoostedTreesOptPred |

