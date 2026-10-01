# AGENTS.md

Guidance for anyone – human contributors or AI coding agents – working in this repository.

## What this repository is

A **historical** personal data-analysis project (2024) on Birmingham City Council purchase-card transactions. Start with [README.md](README.md), then [docs/sdlc/intent.md](docs/sdlc/intent.md), [spec.md](docs/sdlc/spec.md) and [plan.md](docs/sdlc/plan.md).

## Hard rules

1. **Do not modify original code.** `main.py`, `data_extraction.py`, `EDA_and_Tranformations.py` and everything in `Notebooks/` are preserved as written in 2024 – no fixes, formatting, renames, or dependency upgrades.
2. **Do not move** the root scripts or rename `Data/`, `EDA/`, `Encoders/`: the code relies on paths relative to the repository root.
3. **Do not alter files in `docs/original/`**, the committed datasets in `Data/`, or the charts in `EDA/`. They are historical outputs.
4. **Do not add infrastructure** (CI, Docker, linters, test frameworks, packaging) unless a new, explicit intent/spec/plan cycle asks for it.
5. Improvement ideas go to [docs/possible-improvements.md](docs/possible-improvements.md), not into the code.

## Documentation conventions

- English, Markdown, relative links.
- Mark statements as **Confirmed**, **Inferred** or **Unknown**; never present an inference as fact.
- New work follows the cycle *intent → spec → plan → implementation*, recorded under `docs/sdlc/`.

## Running

From the repository root, in the environment described in the README: `python main.py`. It downloads the full dataset from the public portal and overwrites the files in `Data/` and `EDA/` – avoid running it unless regenerating data is the goal.
