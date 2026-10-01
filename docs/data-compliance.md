# Data Licence and Compliance

Markdown version of the original [`original/data_compliance.txt`](original/data_compliance.txt), which is preserved unchanged.

## Licence

The purchase-card transactions dataset is published by **Birmingham City Council** under the **Open Government Licence for public sector information, version 2 (OGL v2)**. It allows free and flexible use and re-use, subject to a few conditions – most importantly, **attribution to Birmingham City Council**.

## Source

- Dataset page: <https://www.cityobservatory.birmingham.gov.uk/@birmingham-city-council/purchase-card-transactions>
- Published under the *Code of Recommended Practice for Local Authorities on Data Transparency*.

## Compliance steps defined by the project

1. Always provide proper attribution to Birmingham City Council.
2. Use the data for legitimate analytical purposes.
3. Ensure data security and privacy when handling the dataset.

## Publisher disclaimer (summary)

- The council is not responsible for loss, damage or inconvenience caused by inaccuracies in the data, and does not endorse external linked sites.
- Where data derives from another licence with restricted reuse, the restriction is stated alongside the dataset.
- Errors or out-of-date information can be reported to the council.

## How this repository complies

- Attribution is given in the main [README](../README.md#acknowledgments).
- Card numbers are already masked by the publisher; the pipeline keeps only the last 4 digits.
- The derived datasets in `Data/` are redistributed under the same OGL v2 terms as the source.
