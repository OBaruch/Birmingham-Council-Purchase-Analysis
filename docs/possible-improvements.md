# Possible Improvements

> **None of the items below have been applied.** The original implementation is preserved intentionally to retain the historical context of the project. This list is a review of the original code from today's perspective, kept separate from the implementation.

## Correctness

| # | File | Observation |
|---|---|---|
| 1 | `main.py` | `total_records_fetched` and `total_records_after_removing_duplicates` are both `cleaned_data.shape[0]` (post-deduplication), so the "fetched" count printed by `main.py` is wrong. (`fetch_and_save_data` prints the correct numbers.) |
| 2 | `EDA_and_Tranformations.py` | Only the `Directorate` spelling is kept; `DIRECTORATE`, `Directorates` and `Direcorate` (≈15 % of rows, including recent files) are dropped by the <15 % completeness rule, so their directorate is lost. Merging the variants before filtering would preserve it. |
| 3 | `EDA_and_Tranformations.py` | The rare-category grouping of `TRANS TAX DESC` runs **after** label encoding, so it compares strings against integers and has no effect (the column is dropped right after anyway). |
| 4 | `EDA_and_Tranformations.py` | Keeping only the last 4 digits of `CARD NUMBER` merges different cards that share those digits; on CSV reload leading zeros are lost (`0065` → `65`). |
| 5 | `EDA_and_Tranformations.py` | Missing values are encoded as the sentinel `"123"`, which can collide with real codes and is later re-interpreted as NaN in `TimeSeries.ipynb`. |
| 6 | `EDA_and_Tranformations.py` | Typo in output name `Data/data_celan.pkl` (vs `data_clean`). |
| 7 | `EDA_and_Tranformations.py` | `ORIGINAL CUR` is not label-encoded while the other categoricals are – the treated files mix numeric and text categorical columns. |
| 8 | `TimeSeries.ipynb` | Models are fit on individual transactions (an irregular series with many rows per day) rather than on an aggregated daily/monthly spend series; SARIMA seasonality `12` and Prophet `freq='D'` with `periods=len(test)` assume a regular series. |
| 9 | `TimeSeries.ipynb` | Forecasting uses the winsorized + normalized amount, so forecasts are not in currency units. |
| 10 | Notebook markdown | Some narrative cells describe actions not performed by the code (e.g. mean imputation of numeric columns). |

## Robustness

- No HTTP timeouts, status-code checks or retries in `data_extraction.py`; a non-JSON response raises an exception.
- `limit=999999999` relies on the API allowing one huge page; CKAN pagination (`offset`/`limit`) would be safer.
- `pd.concat` inside a loop is quadratic; collect DataFrames in a list and concatenate once.
- `df[col].replace(..., inplace=True)` on a column triggers pandas chained-assignment warnings and will stop working with Copy-on-Write (pandas 3).
- Scraping HTML for resource IDs is fragile; the CKAN `package_show` API returns the resource list directly.

## Structure and reproducibility

- Hard-coded relative paths require running from the repository root; a small path/config module (or `pathlib` anchored on `__file__`) would remove this constraint.
- `EDA_and_Tranformations.py` is a raw notebook export: duplicated imports, commented-out AutoViz calls, trailing expressions with no effect. It could be split into functions (`clean`, `treat`, `normalize`, `profile`).
- `requirements.txt` misses `statsmodels`, `pmdarima`, `prophet`, `jupyter`; the Python version is only stated in the README. The NumPy binary-incompatibility error in `TimeSeries.ipynb` shows the environment was not reproducible.
- No tests (e.g. schema checks on each output stage).
- No incremental download; every run re-fetches all resources.

## Repository / data management

- Large CSVs (≈165 MB in total; `data.csv` is ≈58 MB, close to GitHub's 100 MB file limit) are committed directly. Git LFS, compressed formats (Parquet) or regenerating them on demand would keep the repository lighter.
- The `.gitignore` uses only `!` negation rules under "Allow specific files" without a preceding ignore-all rule, so those lines have no effect.
- License inconsistency: `LICENSE` is Apache-2.0 while the original README says MIT. The author should choose one.

## Analytical next steps (from the original objectives)

- Implement the planned anomaly detection (e.g. Isolation Forest / LOF on amount, merchant, directorate, time features) in `Anomalies.ipynb`.
- Clustering of transactions or cards to discover spending profiles.
- Forecast aggregated monthly spend per directorate.
