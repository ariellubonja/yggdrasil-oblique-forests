# Distributed GBT on Vertex AI: per-worker memory budget and natural datasets that fit

Recorded 2026-10-22 from the saved in-process logs (`logs/*_W1_*.log`, line
`Features loaded in memory ... for N MB`, written by `dataset_cache_reader.cc`) and the two Vertex
scaling runs. Columns are partitioned across workers, so per-worker RAM is

    features  = rows * cols / W * B_cell        (B_cell measured: 4.7 HIGGS, 8.7 SUSY, 9.2 YouTube-8M, 10.9 Epsilon, 11.4 trunk)
    + SPO copy = rows * cols / W * 4 B           (dense fp32 copy of the owned columns, in-worker oblique finder; measured 95 MB on 100k x 256)
    + rows * ~24 B                               (labels, predictions, gradient+hessian, example_to_node, split bitmaps; every worker)

B_cell depends on cardinality: exact columns (> 16,000 unique values, the cache default
`max_unique_values_for_discretized_numerical`) hold in-order floats + presorted example indices +
unique values, about 11-12 B/cell; discretized columns hold ~2 B/cell. Low-cardinality datasets
(Criteo, Numerai, taxi) are therefore upper bounds below; one 1-tree W=1 run on the m7i prints the
exact number.

Budget: Vertex `n2-highmem-2` (the scaling runs' machine type) has 16 GB. Evidence: AA 1.6M x 512
(9.3 GB by this model) ran; SPO 1.6M x 512 (12.6 GB) was killed "low on memory" in tree 1. Plan
**<= 11 GB per worker**; bold cells below fit. `n2-highmem-4` doubles the budget (32 GB).
Dataset-cache creation (per-column sort on the workers) stays below the training peak.

| dataset | rows x numeric cols | B/cell | W=1 AA / SPO | W=2 AA / SPO | W=4 AA / SPO | W=8 AA / SPO | W=16 AA / SPO | min W (16 GB) AA / SPO |
|---|---|---|---|---|---|---|---|---|
| HIGGS | 10.50M x 28 | 4.7 | **1.6** / **2.8** | **0.9** / **1.5** | **0.6** / **0.9** | **0.4** / **0.6** | **0.3** / **0.4** | 1 / 1 |
| SUSY | 4.50M x 18 | 8.7 | **0.8** / **1.1** | **0.5** / **0.6** | **0.3** / **0.4** | **0.2** / **0.2** | **0.2** / **0.2** | 1 / 1 |
| Epsilon | 0.40M x 2000 | 10.9 | **8.7** / 11.9 | **4.4** / **6.0** | **2.2** / **3.0** | **1.1** / **1.5** | **0.6** / **0.8** | 1 / 2 |
| YouTube-8M | 3.89M x 1152 | 9.2 | 41.2 / 59.1 | 20.6 / 29.6 | **10.4** / 14.8 | **5.2** / **7.5** | **2.7** / **3.8** | 4 / 8 |
| Numerai v5.0 | 2.75M x 2376 | 11.0 | 71.8 / 97.9 | 36.0 / 49.0 | 18.0 / 24.5 | **9.0** / 12.3 | **4.6** / **6.2** | 8 / 16 |
| Criteo day | 195.84M x 39 | 11.0 | 88.7 / 119.3 | 46.7 / 62.0 | 25.7 / 33.3 | 15.2 / 19.0 | **10.0** / 11.9 | 16 / >16 |
| NYC taxi | 98.30M x 18 | 11.0 | 21.8 / 28.9 | 12.1 / 15.6 | **7.2** / **9.0** | **4.8** / **5.7** | **3.6** / **4.0** | 4 / 4 |
| Airline | 37.89M x 43 | 11.0 | 18.8 / 25.4 | **9.9** / 13.1 | **5.4** / **7.0** | **3.1** / **4.0** | **2.0** / **2.4** | 2 / 4 |
| Jane Street | 40.90M x 82 | 11.0 | 37.9 / 51.3 | 19.4 / 26.1 | **10.2** / 13.6 | **5.6** / **7.3** | **3.3** / **4.1** | 4 / 8 |
| ClimSim | 10.09M x 124 | 11.0 | 14.0 / 19.0 | **7.1** / **9.6** | **3.7** / **4.9** | **2.0** / **2.6** | **1.1** / **1.4** | 2 / 2 |
| Shifts weather | 3.13M x 127 | 11.0 | **4.4** / **6.0** | **2.3** / **3.1** | **1.2** / **1.6** | **0.6** / **0.8** | **0.3** / **0.4** | 1 / 1 |
| TabReD homecredit-default | 0.38M x 612 | 11.0 | **2.6** / **3.5** | **1.3** / **1.8** | **0.7** / **0.9** | **0.3** / **0.4** | **0.2** / **0.2** | 1 / 1 |
| TabReD maps-routing (regr.) | 0.34M x 984 | 11.0 | **3.7** / **5.0** | **1.9** / **2.5** | **0.9** / **1.3** | **0.5** / **0.6** | **0.2** / **0.3** | 1 / 1 |
| TabReD homesite-insurance | 0.26M x 253 | 11.0 | **0.7** / **1.0** | **0.4** / **0.5** | **0.2** / **0.3** | **0.1** / **0.1** | **0.1** / **0.1** | 1 / 1 |
| TabArena APSFailure | 0.08M x 170 | 11.0 | **0.1** / **0.2** | **0.1** / **0.1** | **0.0** / **0.1** | **0.0** / **0.0** | **0.0** / **0.0** | 1 / 1 |

Notes
- Already in `gs://ariel_ydf_bucket/big/` (uploaded from the m7i, checked 2026-10-24): EPSILON
  (`epsilon_normalized_train.csv` 15.7 GB + `epsilon_test_100k.csv`), HIGGS (`HIGGS_train_10500k.csv` +
  `HIGGS_test_500k_floatlabel.csv`), SUSY (same layout), YOUTUBE8M (train 56 GB + validate 16 GB, label
  sidecars). `folds/` holds 33 small suite datasets as fold CSVs. Everything else (ClimSim, Numerai,
  Criteo, taxi, Airline, Jane Street, weather) lives only on the m7i; upload as CSV shards (16 like
  `gen_trunk.py`, `vertex_build/shard_csv.sh`) and point `vertex_build/train.py` at `csv:gs://.../shard_*.csv`.
- Width matters for the distributed story: HIGGS / SUSY / taxi / Criteo / Airline give < 3 columns
  per worker at W = 16; Epsilon, YouTube-8M, Numerai, ClimSim, homecredit-default are the wide ones.
- Candidates that *need* >= 2 workers on 16 GB (a real strong-scaling story): Epsilon (SPO),
  YouTube-8M (W >= 8 for SPO), Numerai (W >= 8-16 unless discretized), ClimSim, Jane Street, Airline.
- Regression-target sets (TabReD maps-routing, delivery-eta, weather, cooking-time) need the
  median-binarisation rule; the user has not ruled on regression targets.
