# Intent

> **Retrospective artifact.** Reconstructed in 2026 from the existing repository (original README, code, notebooks and git history of July–August 2024). It records *what the project intended*, not new requirements. Labels: **Confirmed** / **Inferred** / **Unknown**.

## Problem

Birmingham City Council publishes its purchase-card (P-card) spending as open data, split into many monthly files with inconsistent schemas. In that form it is hard to analyse spending over time, understand who spends on what, or spot unusual transactions. *(Confirmed: data structure; Inferred: framing)*

## Intent

Build a reproducible Python pipeline that turns the fragmented public files into a single, clean, analysis-ready dataset, and use it to:

1. **Discover spending profiles and detect unusual transactions** (clustering / anomaly detection). *(Confirmed – original README)*
2. **Forecast future expenditure and likely purchases.** *(Confirmed – original README)*

## Who it is for

The author, as a personal data-science / portfolio project; secondarily anyone interested in public-spending transparency (the README invites contributions). *(Inferred)*

## Outcomes expected

| Outcome | Status in the repository |
|---|---|
| One consolidated, de-duplicated dataset of all published transactions | Achieved |
| Documented, repeatable cleaning and feature preparation | Achieved (script + notebook narrative) |
| Visual understanding of each data stage | Achieved (AutoViz outputs) |
| Anomaly detection / clustering results | Not achieved |
| Forecasting results | Not achieved (experiment started, failed on environment error) |

## Constraints

- Use only open data, respecting the Open Government Licence v2 (attribution, responsible use). *(Confirmed)*
- Do not expose sensitive card data – card numbers stay masked. *(Confirmed by code)*
- Local execution on a personal machine (Anaconda, Python 3.9, Jupyter). *(Confirmed)*

## Non-goals

No production deployment, web application, API, dashboard or real-time processing appears anywhere in the project. *(Confirmed by absence)*

## Repository-reorganization intent (2026)

Preserve the original implementation unchanged while making the repository understandable as a historical portfolio piece: clear README, documentation in `docs/`, original documents preserved in `docs/original/`, and these SDLC artifacts. See [plan.md](plan.md#phase-6--repository-reorganization-2026).
