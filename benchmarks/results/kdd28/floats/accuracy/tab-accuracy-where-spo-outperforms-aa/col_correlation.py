"""Column correlation per dataset, all rows. Usage: col_correlation.py name=train.csv ... (label = last column). Writes CSV to stdout."""
import sys, time
import numpy as np, pandas as pd
from scipy.stats import rankdata

print("dataset,F,n,mean_max_rho,frac_rho_gt_0p9,frac_rho_gt_0p7,mean_r2,frac_r2_gt_0p9,seconds", flush=True)
for arg in sys.argv[1:]:
    name, path = arg.split("=", 1)
    t = time.time()
    df = pd.read_csv(path, dtype=np.float32)
    X = df.iloc[:, :-1].to_numpy(dtype=np.float64)
    del df
    X = X[:, X.std(axis=0) > 0]
    n, f = X.shape
    # Pearson first (X is still raw), then rank in place for Spearman.
    C = np.corrcoef(X, rowvar=False)
    r2 = np.clip(1 - 1 / np.diag(np.linalg.inv(C + 1e-6 * np.eye(f))), 0, 1)
    del C
    for j in range(f):
        X[:, j] = rankdata(X[:, j])
    S = np.abs(np.corrcoef(X, rowvar=False))
    np.fill_diagonal(S, 0)
    mx = S.max(axis=0)
    print(f"{name},{f},{n},{mx.mean():.6f},{(mx > 0.9).mean():.6f},{(mx > 0.7).mean():.6f},"
          f"{r2.mean():.6f},{(r2 > 0.9).mean():.6f},{time.time() - t:.1f}", flush=True)
