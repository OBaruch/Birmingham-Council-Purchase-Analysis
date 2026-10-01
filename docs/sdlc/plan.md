# Plan

> **Retrospective artifact.** Reconstructs the delivery plan from the git history and the state of the code, and records the 2026 repository reorganization. Phases 1–5 describe the original work (2024); Phase 6 is documentation-only work that does not alter the original implementation.

Requirement IDs refer to [spec.md](spec.md).

## Phase 1 – Setup ✅ (2024-07-28)
- Create repository, LICENSE, README with objectives and Conda setup.
- Define the `pct` Conda environment (Python 3.9) and Jupyter kernel.

## Phase 2 – Data acquisition ✅ (2024-07-28 → 07-29)
- Prototype scraping + API download in notebooks (`DataExtraction.ipynb`, later deleted; `Playground.ipynb`).
- Promote the logic to `data_extraction.py` and call it from `main.py`. → FR-1, FR-2, FR-3, FR-8
- Commit `Data/data.csv`, `duplicates.csv`, `resource_ids.txt`; tune `.gitignore` for large files.

## Phase 3 – Cleaning, transformation, EDA ✅ (2024-07-29 → 08-04)
- Iterative EDA in `EDA.ipynb` (later `Notebooks/EDA_and_Tranformations.ipynb`) with Observation/Action/Result notes. → FR-4, FR-5, FR-6, FR-7
- Persist staged datasets `data_clean*.csv`.

## Phase 4 – Automation ✅ (2024-08-05)
- Export the notebook to `EDA_and_Tranformations.py` and run it from `main.py` via `runpy`. → FR-8
- Store AutoViz outputs under `EDA/` and encoders under `Encoders/`.
- Pin dependency versions in `requirements.txt`; refresh data from source.

## Phase 5 – Modeling 🟡 / ⛔ (started 2024-08-05, not continued)
- Forecasting experiment in `TimeSeries.ipynb` (ARIMA / SARIMA / Prophet). → FR-9 (blocked by a NumPy binary-incompatibility error)
- Anomaly detection placeholder `Anomalies.ipynb` (empty). → FR-10
- `MODEL TRAINING` section reserved in `main.py` (empty).

## Phase 6 – Repository reorganization ✅ (2026)

Goal: make the repository navigable and self-explanatory **without modifying any original code**.

| Step | Action | Rationale |
|---|---|---|
| 6.1 | Inventory all files, read code, notebooks (incl. stored outputs), git history, data and charts | Recover context from evidence |
| 6.2 | Keep `main.py`, `data_extraction.py`, `EDA_and_Tranformations.py`, `Data/`, `EDA/`, `Encoders/` in place | Original code uses root-relative paths; moving them would break it |
| 6.3 | Move `TimeSeries.ipynb` → `Notebooks/TimeSeries.ipynb` (content untouched) | Follows the author's own convention for notebooks |
| 6.4 | Move original `README.md` → `docs/original/README.original.md` and `data_compliance.txt` → `docs/original/` (content untouched) | Preserve original documentation verbatim |
| 6.5 | Write new `README.md` and `docs/*.md` (context, architecture, code overview, data dictionary, EDA outputs, compliance, possible improvements) | Professional, navigable documentation |
| 6.6 | Add `docs/sdlc/intent.md`, `spec.md`, `plan.md` | Capture intent, behavior and history as reviewable artifacts |
| 6.7 | Add `AGENTS.md` | Make the preservation rules explicit for any human or AI contributor |
| 6.8 | Verify that every original code file is byte-identical (same git blob hashes) before and after | Guarantee the "no code changes" constraint |

Explicitly **not** done: code fixes, formatting, dependency updates, `.gitignore` changes, CI, Docker, tests, linters, Git LFS migration.

## Possible future phases (not planned, not applied)

Ideas are tracked in [possible-improvements.md](../possible-improvements.md). If the project is ever resumed, a new intent/spec/plan cycle should be opened for it, keeping the 2024 implementation as the historical baseline (e.g. on a tag) rather than rewriting it.
