# Project Context

This document reconstructs the origin and context of the project from the evidence available in the repository. Each statement is labeled:

- **Confirmed** – directly supported by files, code or git history.
- **Inferred** – reasonably deduced from the repository, but not explicitly stated.
- **Unknown** – cannot be determined from the repository.

## Classification

**Project origin: Personal Project** (data-analysis exploration, likely intended as portfolio work) — *Inferred*.

Evidence considered:

| Evidence | Points to |
|---|---|
| No university, course, professor, assignment or grading references in any file, notebook or commit message | Not coursework |
| No PDF/Word/PowerPoint instructions or reports | Not an assignment with a brief |
| Original README written as a public project page ("Contributions are welcome! Please submit a pull request…") | Personal / public portfolio project |
| Self-defined analytical goals (clustering, anomaly detection, forecasting) | Self-directed exploration |
| Dedicated `data_compliance.txt` documenting the data licence | Awareness of publishing the work publicly |

The repository does not state the motivation explicitly; the classification above is an inference.

## Timeline (Confirmed – git history)

| Date | Commit message (original) | What happened |
|---|---|---|
| 2024-07-28 | Initial commit | Repository created on GitHub (LICENSE) |
| 2024-07-28 | Update README and requirements.txt | Project description and environment setup |
| 2024-07-28 | Added data extraction and processing scripts… | First version of extraction |
| 2024-07-28 | DataExtraction Commit / Notebook 2 Script Data Extraction | Extraction moved from a notebook (`DataExtraction.ipynb`, later deleted) into `data_extraction.py` |
| 2024-07-29 | EDA in Notebook + Clean Data | `Data/data.csv`, `duplicates.csv`, `resource_ids.txt`, `EDA.ipynb`, `Playground.ipynb` |
| 2024-07-29 | Ignore Large Files / Fix .gitignore Large Files | `.gitignore` adjustments for large files |
| 2024-08-04 | EDA and Tranformations | Cleaned / treated / normalized CSVs |
| 2024-08-05 | Automate EDA and Transforms in main.py & Starting to Modeling TS and Anomalies | Notebook exported to `EDA_and_Tranformations.py`, notebooks moved to `Notebooks/`, AutoViz outputs, encoders, `TimeSeries.ipynb`, empty `Anomalies.ipynb` |
| 2024-08-05 | Update requirements.txt / Fix: [Typo] / Updated data from source | Pinned versions, README typo, data refresh |

The last original commit is from 2024-08-05. The modeling phase announced in that commit ("Starting to Modeling TS and Anomalies") was not continued in the repository.

## Objective (Confirmed – original README)

Use the historical purchase-card dataset for:

- **Clustering:** discovering transaction profiles/patterns and detecting unusual transactions (anomaly detection).
- **Forecasting:** predicting future transactional behavior, expenditures and "the next likely purchases".

## Scope actually implemented (Confirmed – code)

1. Automated scraping of the dataset page and download of all monthly resources.
2. Consolidation and de-duplication into one dataset.
3. Data cleaning, type conversion, imputation, outlier capping, label encoding, normalization.
4. Automated EDA (AutoViz) at every data stage.
5. A first forecasting experiment (ARIMA / SARIMA / Prophet) in a notebook – its modeling cell failed with an environment error and has no results.

Clustering and anomaly detection were **not implemented** (empty `Notebooks/Anomalies.ipynb`; empty `MODEL TRAINING` section in `main.py`).

## Development environment (Confirmed)

- Windows machine (a stored notebook warning shows a `C:\Users\…\AppData\Local\Temp\…` path).
- Anaconda environment named `pct` ("Purchasing Card Transactions"), Python 3.9.19.
- Jupyter notebooks were used for exploration; the main EDA notebook was then exported to a `.py` script (the script still contains the `# In[n]:` cell markers and a commented-out `get_ipython()` line produced by `jupyter nbconvert`).

## Language notes (Confirmed)

Code, markdown and documentation are mostly in English; a few comments/headings are in Spanish ("Cambiar el tipo de datos y manejar valores nulos", "Guardar datos limpios"), which suggests the author's working language was Spanish (*Inferred*).

## Data source (Confirmed)

- Publisher: Birmingham City Council, published under the Code of Recommended Practice for Local Authorities on Data Transparency.
- Licence: Open Government Licence v2.
- Portal referenced in docs: `https://www.cityobservatory.birmingham.gov.uk/@birmingham-city-council/purchase-card-transactions`
- Endpoint used by the code: `https://birmingham-city-observatory.datopian.com` (CKAN `datastore_search` API).
- Coverage in the committed `Data/data.csv`: 300,412 unique transactions dated 2014-05-07 → 2024-07-03, from 122 resource IDs.

Whether these endpoints are still online today is **Unknown**.

## Contradictions found in the original repository

| Topic | Source A | Source B |
|---|---|---|
| Code license | `README` (original): MIT | `LICENSE` file: Apache License 2.0 |
| Portal URL | Docs: `cityobservatory.birmingham.gov.uk` | Code: `birmingham-city-observatory.datopian.com` (likely the hosting backend of the same portal – *Inferred*) |
| Notebook narrative vs code | Markdown cell says numeric nulls were "filled with the mean" and categorical with `"empty"` | The code fills categoricals with `"NULL"` (later `"123"`); no mean imputation is performed |
| Notebook narrative vs code | Markdown says redundant columns are "Transaction Tax AMT, Transoriginal Rate AMT, Transtax Rate" | Code drops `TRANS TAX AMT`, `TRANS ORIGINAL NET AMT`, `TRANS TAX RATE` (wording difference only) |

These contradictions are documented, not resolved.
