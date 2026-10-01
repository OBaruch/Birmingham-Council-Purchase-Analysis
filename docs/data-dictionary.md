# Data Dictionary

Describes the datasets committed under `Data/`. Row/column counts were measured on the committed files. Column meanings are **not documented by the publisher inside this repository**; descriptions marked *Inferred* come from column names and values.

## Files

| File | Rows | Columns | Description |
|---|---|---|---|
| `Data/resource_ids.txt` | 122 lines | – | CKAN resource IDs scraped from the dataset page (one per monthly file) |
| `Data/data.csv` | 300,412 | 63 | Raw consolidated transactions after exact-duplicate removal; dates 2014-05-07 → 2024-07-03 |
| `Data/duplicates.csv` | 19,800 | 63 | Every row that had at least one exact duplicate (`keep=False`, so both copies are included) |
| `Data/data_clean.csv` | 295,002 | 19 | Filtered, reordered, typed; card number masked; nulls → `NULL` |
| `Data/data_clean_treated.csv` | 295,002 | 18 | Imputed (`123`), amount winsorized, categoricals label-encoded |
| `Data/data_clean_treated_normalized.csv` | 295,002 | 18 | As above, with `ORIGINAL GROSS AMT` z-score normalized |

Transactions per year in `data.csv`: 2014: 18,431 · 2015: 34,854 · 2016: 32,191 · 2017: 28,942 · 2018: 29,727 · 2019: 31,411 · 2020: 21,950 · 2021: 24,466 · 2022: 44,520 · 2023: 25,280 · 2024 (to July): 7,463.

## Raw schema (`data.csv`)

The 63 columns are the union of all monthly files, whose schemas changed over time. Notable groups:

| Columns | Meaning |
|---|---|
| `TRANS DATE`, `TRANS POST DATE` | Transaction / posting date |
| `ORIGINAL GROSS AMT`, `ORIGINAL CUR` | Amount (incl. tax) and currency of the original transaction |
| `BILLING GROSS AMT`, `BILLING CUR CODE` | Billed amount and currency |
| `TRANS ORIGINAL NET AMT`, `TRANS TAX AMT`, `TRANS TAX RATE`, `TRANS TAX DESC`, `TRANS VAT DESC` | Net amount and tax/VAT information |
| `MERCHANT NAME`, `MCC CODE`, `MERCHANT TAX REG NO` | Merchant information |
| `CARD NUMBER` | Card number, already masked by the publisher (e.g. `************0065`) |
| `TRANS CAC CODE 1…12`, `TRANS CAC DESC 1…12` | Accounting / cost-allocation codes and descriptions (*Inferred*) |
| `DIRECTORATE`, `Directorate`, `Directorates`, `Direcorate` | Council directorate – four spellings of the same field across files (*Inferred*) |
| `TRANS ACCOUNT KEY`, `CLAIM NO`, `TRANS EXPENSE CATEGORY`, … | Sparse fields present only in some files |
| `"Adams,Anthony"`, `Local Services`, `REFERENCE`, `Received_Date`, `LOCATION`, `DEV`, `Accepted`, `APPLICANT`, `AGENT`, `WARD`, `geom` | Very sparse columns that look unrelated to card transactions (planning-application–like fields); probably come from a mis-linked or malformed resource (*Inferred*) |

Completeness of the directorate variants in `data.csv`: `Directorate` 85.2 %, `DIRECTORATE` 10.1 %, `Directorates` 3.9 %, `Direcorate` 0.5 %.

## Clean schema (`data_clean.csv`)

`TRANS DATE`, `ORIGINAL GROSS AMT`, `MERCHANT NAME`, `CARD NUMBER` (last 4 digits only), `TRANS CAC CODE 1`–`8`, `Directorate`, `TRANS CAC DESC 1`, `TRANS CAC DESC 2`, `BILLING CUR CODE`, `ORIGINAL CUR`, `TRANS VAT DESC`, `TRANS TAX DESC`.

Missing categorical values are the literal string `NULL`.

## Treated schemas (`data_clean_treated*.csv`)

Same as the clean schema minus `TRANS TAX DESC`, with:

- `MERCHANT NAME`, `Directorate`, `TRANS VAT DESC`, `TRANS CAC CODE 1`–`8` → integer label codes (encoders in `Encoders/label_encoders.pkl`).
- `TRANS CAC DESC 1/2`, `BILLING CUR CODE` → text, missing values = `123`.
- `ORIGINAL CUR` → `GBP`, `123` (missing) or `OTHERS`.
- `CARD NUMBER` → numeric last-4 digits (leading zeros lost on CSV reload, e.g. `0065` → `65`).
- `ORIGINAL GROSS AMT` → capped at the 1st/99th percentile; in the `_normalized` file also standardized (mean 0, std 1).

## `Encoders/label_encoders.pkl`

A pickled `dict` mapping column name → fitted `sklearn.preprocessing.LabelEncoder` for the 12 encoded columns (`MERCHANT NAME`, `Directorate`, `TRANS VAT DESC`, `TRANS TAX DESC`, `TRANS CAC CODE 1`–`8`). Can be used to decode the label codes back to original values (requires a compatible scikit-learn version).

## Sensitive data note

The publisher masks card numbers; the pipeline further reduces them to the last 4 digits. Merchant names are public business names. No personal data handling beyond this is performed or required by the code.
