# AutoResearch loop for fast tabular phenotype prediction

## Skills-as-hypotheses (prepend)

Before making any changes, read `meta_summary/skills_md.md`.

Use the skills as starting hypotheses — do not rediscover what is already known.

If a skill applies to this phenotype's baseline profile, apply it in iteration 1.

If a skill fails on this phenotype, note it in `agent_results.md` under that skill number.

## This folder is a template (one phenotype run)

**`base_example/`** is the **canonical template**: copy the whole folder to `phenotype_runs/<your_name>/` and treat that copy as **one optimization workspace** for the phenotype(s) declared in **`config.json`**.

- **Phenotype-specific behavior** (which outcome(s) to predict, which models/feature sets to try, fast protocol, where prepared data live, session label) → edit **`config.json` only** when duplicating.
- **`train.py`** is shared across all copies; the human/agent edits it to improve models, **not** to switch phenotype identity (that stays in `config.json`).
- **`agent_loop.py`** is shared; it contains **no** hardcoded phenotype names.

See **`README.md`** in this folder for the copy-and-run workflow.

## Goal

Improve `agent_score` produced by running **`uv run agent_loop.py`** from **this folder** (so `run.log` and `agent_runs.csv` stay here).

This repository benchmarks tabular models for predicting continuous phenotypes from:
- `raw`
- `embed`
- `raw+embed`

The current phase is a fast concept-stage AutoResearch setup.

## Main objective

Maximize:

- `agent_score`

`agent_score` is defined in `train.py` as the mean validation R² of the champion configuration for each target.

For each target, the champion configuration is the row with the highest `valid_r2` across:
- model
- feature set

The log must explicitly print:
- detailed per-run results
- champion row per target
- `agent_score`

## Why this matters

The goal is not merely to improve an abstract average.
The goal is to identify:
- which model is currently winning
- which feature set is currently winning
- which model path should be improved next

The agent should use the champion table to decide what to modify.

## Files

- `prepare.py` at repo root (`olink_to_phenotypes/prepare.py`) is fixed
- `agent_loop.py` in this folder is fixed (finds repo `pyproject.toml` upward; runs `train.py` from this folder)
- `train.py` in this folder is the main file the **agent** edits for modeling experiments (reads **`config.json`** for targets, search space, paths—no phenotype name hardcoded)
- **`config.json`** — phenotype run settings: `target`, `allowed_models`, `allowed_feature_sets`, `fast_mode`, `max_train_samples`, `session_name`, plus `data` / `outputs` for prepared paths
- `program.md` defines the optimization rules (this file)

## Editable surface

You may edit:
- `train.py` (for methodology / model improvements)

Do not edit:
- `prepare.py`
- `agent_loop.py`
- score parsing behavior

Do **not** hardcode phenotype names, target lists, or data paths in `train.py` — use **`config.json`**.

## Current search space

Configured via `config.json` → `allowed_models`. Default template uses fast linear-style models only:
- elasticnet
- ridge
- pls
- stacking_linear

Avoid slow models for now (unless you add them to `allowed_models` deliberately).

## Optimization behavior

After each change:
1. run `uv run agent_loop.py`
2. inspect the detailed results
3. inspect the champion row per target
4. inspect `agent_score`
5. if `agent_score` improves, keep the change
6. if `agent_score` worsens, revert the change

## Decision policy

Use the champion table to guide edits.

Examples:
- if `elasticnet` is champion, improve the elasticnet path
- if `stacking_linear` is champion, improve stacking
- if `embed` is dominating, consider improving embedding-specific preprocessing
- if `raw+embed` is dominating, consider improving concatenated-feature preprocessing

Do not make many unrelated changes at once.

Prefer:
- one hypothesis
- one edit
- one evaluation
- one decision

## Allowed directions

- tune linear model regularization
- tune PLS components
- improve linear stacking structure
- add simple feature filtering
- add simple preprocessing steps
- improve preprocessing for the champion model path
- make local structural changes within a model path, not just hyperparameter tweaks

## Avoid

Do not:
- add LightGBM
- add deep learning
- change dataset construction in `prepare.py`
- change the definition of `agent_score`
- hardcode a different phenotype or target list in `train.py` instead of updating `config.json`

## Code quality

Maintain:
- working CLI
- stable output
- readable code
- parsable `agent_score:`
- explicit `Champion per target:` output
