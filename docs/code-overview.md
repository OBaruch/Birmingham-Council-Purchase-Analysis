# Code Overview

Description of every original code file. **None of these files were modified** during the reorganization; observations about defects are listed in [`possible-improvements.md`](possible-improvements.md).

---

## `main.py` – pipeline entry point

Sections (marked by `#####` banners in the code):

1. **DATA EXTRACTION**
   - `webpage_url = 'https://birmingham-city-observatory.datopian.com/dataset/purchase-card-transactions'`, `output_folder = 'Data'`.
   - Calls `data_extraction.scrape_resource_ids` and `data_extraction.fetch_and_save_data`.
   - Reloads `Data/data.pkl` and `Data/duplicates.csv` and prints record/duplicate counts.
2. **EDA AND TRANSFORMATIONS** – runs `EDA_and_Tranformations.py` with `runpy.run_path`.
3. **MODEL TRAINING** – banner only; no code.

Imports: `os`, `pandas`, `data_extraction`, `runpy`.

---

## `data_extraction.py` – data acquisition module

| Function | Inputs | Output / side effects |
|---|---|---|
| `scrape_resource_ids(url, output_folder)` | Dataset page URL, folder | GETs the page, parses all `<a href>` containing `/resource/`, extracts the ID segment, de-duplicates with `set`, writes `resource_ids.txt`, returns the list |
| `fetch_data(resource_id)` | One resource ID | Calls `/api/3/action/datastore_search?resource_id=…&limit=999999999`; on `success` builds a DataFrame from `result.records` and drops `_id`; otherwise prints the API error and returns an empty DataFrame. Returns `(df, api_url)` |
| `fetch_and_save_data(resource_ids, output_folder)` | ID list, folder | Concatenates every non-empty resource, writes all duplicated rows (`keep=False`) to `duplicates.csv`, drops duplicates, writes `data.csv` and `data.pkl`, prints counts |

Imports: `requests`, `bs4.BeautifulSoup`, `os`, `pandas`. Functions have docstrings (Google-like style).

---

## `EDA_and_Tranformations.py` – cleaning, transformation and EDA

An export of `Notebooks/EDA_and_Tranformations.ipynb` (it keeps `# In[n]:` markers and the notebook's markdown as comments). It runs top to bottom as a script. Steps:

| # | Step | Detail |
|---|---|---|
| 1 | Load | `pd.read_pickle('Data/data.pkl')`; prints `describe()` and null counts |
| 2 | AutoViz on raw data | Call present but **commented out** (output kept in `EDA/data/`) |
| 3 | Completeness profile | % of non-null values per column |
| 4 | Row filter | Drop rows with nulls in any column that is >90% complete (`TRANS DATE`, `ORIGINAL GROSS AMT`, `MERCHANT NAME`, `CARD NUMBER`, `TRANS CAC CODE 1–3` in the original run) |
| 5 | Column filter | Drop columns <15% complete (63 → 23 columns in the original run) |
| 6 | Reorder | Key columns first (date, amount, merchant, card, CAC codes 1–8), rest by completeness |
| 7 | Drop redundant | `TRANS TAX AMT`, `TRANS ORIGINAL NET AMT`, `TRANS TAX RATE`, then `BILLING GROSS AMT` |
| 8 | Types | `TRANS DATE` → datetime; amount → float; categoricals → `category`; **`CARD NUMBER` reduced to its last 4 characters**; categorical nulls → `"NULL"` |
| 9 | Save | `Data/data_clean.csv`, `Data/data_celan.pkl` |
| 10 | AutoViz on clean data | Commented out (output kept in `EDA/data_clean/`) |
| 11 | Treatment | Reload CSV; `"NULL"` → `"123"` in categorical columns; winsorize `ORIGINAL GROSS AMT` to its 1st–99th percentile; `LabelEncoder` on merchant, directorate, VAT/tax desc and CAC codes 1–8; encoders pickled to `Encoders/label_encoders.pkl`; group rare `TRANS TAX DESC` then drop the column (high correlation with `TRANS VAT DESC`); collapse `ORIGINAL CUR` into `GBP` / `123` / `OTHERS` |
| 12 | Save | `Data/data_clean_treated.csv` |
| 13 | AutoViz on treated data | Commented out (output kept in `EDA/data_clean_treated/`) |
| 14 | Normalize | z-score of `ORIGINAL GROSS AMT` → `Data/data_clean_treated_normalized.csv` |
| 15 | AutoViz on normalized data | **Active** → `EDA/data_clean_treated_normalized/` |
| 16 | Inspect | `df.describe()`, `df.dtypes`, `df.head()` (no effect when run as a script) |

Imports: `pandas`, `matplotlib.pyplot`, `seaborn`, `sklearn.preprocessing.LabelEncoder`, `autoviz.AutoViz_Class`, `pickle`.

---

## Notebooks

### `Notebooks/Playground.ipynb`
Prototype of the extraction logic before it was moved into `data_extraction.py`: an inline `scrape_resource_ids`, an inline `fetch_data` + concatenation loop writing `resource_ids.txt`, `data.csv`, `data.pkl`, `duplicates.csv` to the **current directory**, and a later check that imports `data_extraction` and loads `Data/data.pkl`. Stored outputs show some resource IDs returning *"Not found"* from the API.

### `Notebooks/EDA_and_Tranformations.ipynb`
Interactive source of `EDA_and_Tranformations.py`, with all AutoViz calls active and stored outputs (dimension prints and embedded charts). Includes markdown "Observation / Action / Result" notes explaining each cleaning decision.

### `Notebooks/TimeSeries.ipynb`
Forecasting experiment (moved here from the repository root during the reorganization; content unchanged).
- Cell 1: loads `Data/data_clean.csv` (treating `"123"` as NaN), indexes by `TRANS DATE` and plots `ORIGINAL GROSS AMT`. Executed successfully.
- Cell 2: loads the normalized dataset, splits 80/20 chronologically, and defines/evaluates `arima_forecast` (`pmdarima.auto_arima`), `sarima_forecast` (`SARIMAX(1,1,1)(1,1,1,12)`), `prophet_forecast` (Prophet, daily frequency) with RMSE and plots. **Failed** in the stored run with `ValueError: numpy.dtype size changed, may indicate binary incompatibility` (an environment issue), so no forecasting results exist.

### `Notebooks/Anomalies.ipynb`
Empty file (0 bytes) – placeholder for the planned anomaly-detection work.

---

## Dependencies observed

`requirements.txt` (original, pinned): `autoviz==0.1.905`, `beautifulsoup4==4.12.3`, `matplotlib==3.9.0`, `pandas==2.2.2`, `Requests==2.32.3`, `scikit_learn==1.5.1`, `seaborn==0.13.2`.

Used in code but not listed: `numpy` (installed transitively with pandas), `statsmodels`, `pmdarima`, `prophet` (TimeSeries notebook only), `jupyter` / `ipykernel` (installed via conda per the original README).
