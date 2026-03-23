# Phenotype run template (`base_example`)

**Canonical template:** copy this entire folder to start a new phenotype. Do **not** edit `train.py` or `agent_loop.py` for phenotype identity—only **`config.json`** (plus `train.py` when the AutoResearch agent improves models).

## Duplicate

From `phenotype_runs/`:

```bash
cp -r base_example my_phenotype_name
```

Edit **`my_phenotype_name/config.json`** (see `program.md` and the list below). Then:

```bash
cd my_phenotype_name
UV_PROJECT_ENVIRONMENT=/path/to/.venv uv run agent_loop.py
```

`agent_loop.py` finds the repo root by walking up until it sees `pyproject.toml`, so the copy must stay **inside** `olink_to_phenotypes/`.

## Files

| File | Role |
|------|------|
| `config.json` | **Phenotype-specific:** targets, models, feature sets, fast settings, data paths, session label |
| `train.py` | Shared logic; agent edits for modeling experiments |
| `agent_loop.py` | Shared driver; unchanged across copies |
| `program.md` | Instructions for the optimization agent |
