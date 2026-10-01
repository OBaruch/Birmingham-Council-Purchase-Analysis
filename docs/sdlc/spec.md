# Specification

> **Retrospective artifact.** Reverse-engineered from the original code and notebooks (2024). It describes the behavior the implementation actually has, with each requirement traced to its source and marked with its implementation status. Nothing here changes the original code.

Status legend: ✅ Implemented · 🟡 Partial / experimental · ⛔ Not implemented

## 1. Scope

A batch pipeline that acquires Birmingham City Council purchase-card data, prepares it in staged datasets, profiles each stage, and (planned) models it for anomaly detection and forecasting. See [intent.md](intent.md).

## 2. Functional requirements

### FR-1 Resource discovery ✅
- **FR-1.1** The system shall download the dataset page `https://birmingham-city-observatory.datopian.com/dataset/purchase-card-transactions` and extract every resource ID from links containing `/resource/`.
- **FR-1.2** Resource IDs shall be de-duplicated and written one per line to `Data/resource_ids.txt`.
- *Trace:* `data_extraction.scrape_resource_ids`.

### FR-2 Data acquisition ✅
- **FR-2.1** For each resource ID the system shall call the CKAN `datastore_search` API and build a DataFrame from `result.records`, dropping the `_id` column.
- **FR-2.2** If the API reports failure, the system shall print the error and continue with the next resource.
- **FR-2.3** All resources shall be concatenated into a single dataset (union of all columns).
- *Trace:* `data_extraction.fetch_data`, `fetch_and_save_data`.

### FR-3 De-duplication ✅
- **FR-3.1** All rows that are exact duplicates shall be written (every copy) to `Data/duplicates.csv`.
- **FR-3.2** The de-duplicated dataset shall be written to `Data/data.csv` and `Data/data.pkl`.
- **FR-3.3** Counts of fetched, duplicated and remaining records shall be printed.

### FR-4 Cleaning ✅
- **FR-4.1** Drop rows with nulls in any column that is >90 % complete.
- **FR-4.2** Drop columns that are <15 % complete.
- **FR-4.3** Reorder columns: `TRANS DATE`, `ORIGINAL GROSS AMT`, `MERCHANT NAME`, `CARD NUMBER`, `TRANS CAC CODE 1–8`, then the rest by completeness.
- **FR-4.4** Drop redundant amount/tax columns: `TRANS TAX AMT`, `TRANS ORIGINAL NET AMT`, `TRANS TAX RATE`, `BILLING GROSS AMT`.
- **FR-4.5** Cast `TRANS DATE` to datetime, `ORIGINAL GROSS AMT` to float, the remaining fields to categorical; replace categorical nulls with `"NULL"`.
- **FR-4.6** Reduce `CARD NUMBER` to its last 4 characters.
- **FR-4.7** Persist to `Data/data_clean.csv` (+ pickle).
- *Trace:* `EDA_and_Tranformations.py`, cells 4–16.

### FR-5 Treatment & encoding ✅
- **FR-5.1** Replace `"NULL"` by the constant `"123"` in categorical columns.
- **FR-5.2** Cap `ORIGINAL GROSS AMT` to its 1st–99th percentile range.
- **FR-5.3** Label-encode `MERCHANT NAME`, `Directorate`, `TRANS VAT DESC`, `TRANS TAX DESC`, `TRANS CAC CODE 1–8`, and persist the encoders to `Encoders/label_encoders.pkl`.
- **FR-5.4** Drop `TRANS TAX DESC` (highly correlated with `TRANS VAT DESC`).
- **FR-5.5** Collapse `ORIGINAL CUR` into `GBP`, `123`, `OTHERS`.
- **FR-5.6** Persist to `Data/data_clean_treated.csv`.

### FR-6 Normalization ✅
- **FR-6.1** Standardize `ORIGINAL GROSS AMT` with a z-score and persist to `Data/data_clean_treated_normalized.csv`.

### FR-7 Exploratory profiling ✅
- **FR-7.1** Generate AutoViz charts (PNG) for each dataset stage into `EDA/<stage>/AutoViz/`.
- *Note:* in the script only the normalized stage is active; earlier stages were generated from the notebook.

### FR-8 Orchestration ✅
- **FR-8.1** `python main.py` shall run FR-1 → FR-3, then the full transformation script (FR-4 → FR-7).

### FR-9 Forecasting 🟡
- **FR-9.1** Build a time series of `ORIGINAL GROSS AMT` indexed by `TRANS DATE` and plot it. ✅
- **FR-9.2** Split 80/20 chronologically and compare ARIMA (`auto_arima`), SARIMA `(1,1,1)(1,1,1,12)` and Prophet by RMSE. ⛔ (code written, execution failed – no results)
- *Trace:* `Notebooks/TimeSeries.ipynb`.

### FR-10 Anomaly detection & clustering ⛔
- Stated in the original README; `Notebooks/Anomalies.ipynb` is empty and the `MODEL TRAINING` section of `main.py` is empty.

## 3. Data contracts

| Stage | File | Shape (committed data) |
|---|---|---|
| Raw | `Data/data.csv` | 300,412 × 63 |
| Clean | `Data/data_clean.csv` | 295,002 × 19 |
| Treated | `Data/data_clean_treated.csv` | 295,002 × 18 |
| Normalized | `Data/data_clean_treated_normalized.csv` | 295,002 × 18 |

Column details: [data-dictionary.md](../data-dictionary.md).

## 4. Non-functional characteristics (as implemented)

| Aspect | Actual behavior |
|---|---|
| Runtime environment | Python 3.9, Anaconda, Jupyter; dependencies in `requirements.txt` |
| Execution context | Must be run from the repository root (relative paths) |
| Data volume | ~300k rows processed in memory with pandas |
| Idempotency | Every run re-downloads and overwrites all outputs |
| Error handling | API-level failures are logged and skipped; network/HTTP errors are not handled |
| Privacy | Card numbers masked (publisher) and truncated to last 4 digits (pipeline) |
| Licensing | Data under OGL v2 with attribution |

## 5. Acceptance checks (derivable from the repository)

- `Data/resource_ids.txt` exists and is non-empty.
- `Data/data.csv` contains no exact duplicate rows.
- `data_clean*.csv` files have identical row counts and no null `TRANS DATE` / `ORIGINAL GROSS AMT`.
- `CARD NUMBER` values in the clean dataset have at most 4 characters.
- `ORIGINAL GROSS AMT` in the normalized file has mean ≈ 0 and std ≈ 1.
- `EDA/<stage>/AutoViz/` contains the AutoViz PNGs for each stage.

These checks describe expected behavior; no automated tests exist in the original project.

## 6. Open questions (Unknown)

- Whether the `datopian.com` API endpoint is still available.
- Which modeling approach was intended for anomaly detection/clustering.
- The intended code license (Apache-2.0 file vs MIT in the README).
