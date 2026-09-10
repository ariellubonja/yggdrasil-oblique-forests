# SPO vs XGBoost/LightGBM/CatBoost — data preprocessing

How the 33 suite datasets (30 TabArena binary + 3 TabReD) were turned into the
fold CSVs every arm read. Code: `benchmarks/data/tabular_suite_prep.py` (encoding,
m7i only) and `benchmarks/src/spo_vs_gbt/run_suite.py::make_folds`
(folds + imputation). Replication details: `../../evaluation/spo_vs_gbt/REPLICABILITY.md`.

## Non-numeric columns

Every column is kept; nothing is dropped.

| source dtype | encoding |
|---|---|
| numeric | float32 as-is |
| bool | 0.0 / 1.0 |
| datetime | epoch seconds (none occur in the suite) |
| categorical / string | ordinal codes: distinct values sorted alphabetically → 0, 1, 2, … |

Ordinal order is arbitrary (alphabetical, unrelated to the label) but deterministic,
so fold CSVs hash identically across pandas versions and row orders. All arms,
including CatBoost/LightGBM/XGBoost, receive the same integer codes; the libraries'
native categorical handling is **not** used.

20 of 33 datasets have categorical columns (categorical / total features):
Amazon_employee_access 9/9, NATICUSdroid 86/86, Diabetes130US 39/47,
in_vehicle_coupon_recommendation 22/24, HR_Analytics 10/12,
customer_satisfaction_in_airline 16/21, credit-g 13/20, Is-this-a-good-customer 8/13,
bank-marketing 8/13, Fitness_Club 3/6, kddcup09_appetency 38/212, Marketing_Campaign 9/25,
online_shoppers_intention 6/17 (+1 bool), qsar-biodeg 5/41, coil2000 5/85, churn 4/19,
Bank_Customer_Churn 4/10, E-CommereShippingData 4/10, seismic-bumps 4/15,
credit_card_clients_default 3/23. The other 13 (incl. all TabReD, HIGGS, SUSY, Epsilon)
are fully numeric.

## Missing values

Imputation happens once, at fold-writing time, so YDF and Python arms read identical bytes:

- `train_nan.csv` keeps the NaNs from encoding (categorical NaN stays NaN, not a code).
- Per fold, each NaN feature cell is replaced by the **training fold's** column mean.
  The test fold reuses that same train-fold mean. An all-NaN training column → 0.0.
- Missing categoricals therefore get a fractional mean code (e.g. 1.3).
- TabReD uses one chronological 80/20 holdout; the mean comes from the first 80 % of rows.
- Huge datasets (HIGGS, SUSY, Epsilon) have no missing values.

Missingness kept as-is (no column or dataset is excluded for it):

| dataset | NaN cells | % of matrix | cols >50 % NaN | all-NaN cols |
|---|---|---|---|---|
| kddcup09_appetency | 6,727,959 | 63.5 | 133 / 212 | 0 |
| homecredit-default | 7,038,577 | 25.3 | 121 / 696 | 20 |
| APSFailure | 1,078,695 | 8.3 | 8 / 170 | 0 |
| Diabetes130US | 137,663 | 4.1 | 1 / 47 | 0 |
| GiveMeSomeCredit | 33,655 | 2.2 | 0 | 0 |
| polish_companies_bankruptcy | 4,666 | 1.2 | 0 | 0 |
| homesite-insurance | 87,139 | 0.7 | 2 / 299 | 0 |
| customer_satisfaction_in_airline, HR_Analytics, jm1, Marketing_Campaign, Fitness_Club | ≤393 | <0.2 | 0 | 0 |

The 20 all-NaN homecredit columns are an artifact of the 40k-row chronological HF
mirror; they impute to a constant 0.0 and cannot be split on. `nan_cells_imputed` in
`suite_results.csv` is the whole-dataset count (identical across folds), not per fold.
