# Agent optimization log — **BMI**

Phenotype workspace: `phenotype_runs/BMI/` · target: **BMI**

## Baseline (fresh `uv run agent_loop.py`)

- **agent_score:** 0.652729  
- **Champion:** `feature_set=raw+embed`, `model=elasticnet`  
- **Note:** ElasticNet (`alpha=0.15`, `l1_ratio=0.5`, `selection="random"`, `max_iter=5000`) **beats** `stacking_linear` (0.640721) on this target — optimization focused on the **standalone ElasticNet** branch.

---

## Optimization iterations

| Iter | Change in `train.py` | Recorded score | vs session best | Decision |
|------|----------------------|----------------|-----------------|----------|
| 1 | ElasticNet `alpha`: 0.15→0.10 | 0.643959 | ↓ | **Revert** |
| 2 | ElasticNet `alpha`: 0.15→0.20 | 0.652089 | ↓ | **Revert** |
| 3 | ElasticNet `l1_ratio`: 0.5→0.45 | 0.652187 | ↓ | **Revert** |
| 4 | ElasticNet `max_iter`: 5000→8000 | 0.652729 | = | **Revert** (no strict gain) |

**Early stop:** **4 consecutive** iterations without improvement over session best (0.652729).

---

## Phenotype session summary

| Metric | Value |
|--------|--------|
| **Baseline agent_score** | 0.652729 |
| **Best agent_score** | **0.652729** |
| **Δ** | 0 |
| **Final champion** | `raw+embed` · `elasticnet` |
| **Strongest model path** | `make_model("elasticnet", …)` — Pipeline(imputer → scaler → ElasticNet **α=0.15**, **l1=0.5**, **`selection="random"`**, **max_iter=5000**) |

### Kept changes

- None (baseline configuration already optimal under tested local edits).

### Reverted / unsuccessful

- Lower/higher `alpha`, lower `l1_ratio`, and higher `max_iter` did not beat the baseline champion.
