# Phenotype-specific optimization workspaces

Each subfolder under `phenotype_runs/` is a **self-contained run** (its own `train.py`, `agent_loop.py`, logs, and `config.json`).  
**Shared prepared data** stays on disk under `DL_nastya/prepared/` (or whatever `project_dir` / `prepared_subdir` you set in `config.json`).

## Template: `base_example/` (canonical)

**Do not rename the template in place**—keep `base_example/` as the reference copy. Duplicate it:

```bash
cd phenotype_runs
cp -r base_example my_phenotype_run
```

Then edit **`my_phenotype_run/config.json`** only for phenotype-specific settings (see table below). Scripts in the copy are the same as the template; no phenotype names are hardcoded in `train.py` / `agent_loop.py`.

### `config.json` keys

| Key | Purpose |
|-----|--------|
| `target` | Phenotype column (string) or list of columns |
| `allowed_models` | Models to benchmark in `agent_eval` |
| `allowed_feature_sets` | Feature views: `raw`, `embed`, `raw+embed` |
| `fast_mode` | If true, `agent_loop` passes `--fast` and train uses fast subsampling rules |
| `max_train_samples` | Cap on training rows (passed through to `train.py` for fast runs) |
| `session_name` | Label printed at the start of a run (console + `run.log` via `train.py`) |
| `data.project_dir` | Root of the DL project (contains `prepared/`) |
| `data.prepared_subdir` | Usually `prepared` |
| `data.files` | Filenames inside `prepared/` (`X_raw.parquet`, etc.) |
| `outputs.results_csv` | Per-run results file (lives next to `config.json`) |

## How to run

From **`olink_to_phenotypes/`** (where `pyproject.toml` lives), with your venv:

```bash
cd phenotype_runs/base_example
UV_PROJECT_ENVIRONMENT=/path/to/.venv uv run agent_loop.py
```

Or from repo root (delegates to `base_example`):

```bash
cd /path/to/olink_to_phenotypes
UV_PROJECT_ENVIRONMENT=/path/to/.venv uv run agent_loop.py
```

Single experiment (use the same column name as in `config.json` → `target`):

```bash
cd phenotype_runs/my_phenotype_run
UV_PROJECT_ENVIRONMENT=/path/to/.venv uv run train.py --mode single \
  --target "Your column name" --feature_set raw+embed --model elasticnet --fast
```

`prepare.py` remains at **`olink_to_phenotypes/prepare.py`** and is unchanged.
