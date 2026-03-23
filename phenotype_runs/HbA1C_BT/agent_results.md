# Agent optimization log — **HbA1C (BT)**

Phenotype workspace: `phenotype_runs/HbA1C_BT/` · target: **HbA1C (BT)**

---

## Earlier session (hyperparameter-focused)

See git history / prior notes: baseline **0.286831** → best **0.314526** via stacking ElasticNet / Ridge / meta-Ridge tuning (`agent_runs.csv` earlier rows).

---

## Structural optimization session (current)

**Rules:** Structural edits only (pipelines, feature branches, CV/stack shape). No deliberate tuning of `alpha`, `l1_ratio`, Ridge `alpha`, or PLS `n_components` beyond existing values.

### Structural baseline

- Command: `uv run agent_loop.py` (with pre-session `train.py` from repo: tuned stacking, **no** `raw+embed` ColumnTransformer).
- **agent_score:** **0.314526**
- **Champion:** `raw+embed` · `stacking_linear`
- **session_best** initialized at **0.314526**

---

## Iteration 1

**Hypothesis:** Separate preprocessing for raw vs embed before concatenation (VarianceThreshold + StandardScaler on raw block; StandardScaler on embed block only).

**Change:** `ColumnTransformer` on `raw+embed` with raw branch `Imputer → VarianceThreshold(0) → StandardScaler`, embed branch `Imputer → StandardScaler`; wired through `make_model(..., feature_set, n_raw_features)` and cloned per stacking base.

**agent_score before:** 0.314526 **after:** 0.314526  

**Decision:** Revert  

**Reason:** No strict improvement vs session best (tie); dropped in favor of simpler baseline before next structural test.

---

## Iteration 2

**Hypothesis:** High-dimensional raw Olink features benefit from **univariate screening** on the raw block only (`SelectKBest` f_regression, k=300), embed block unchanged (impute + scale).

**Change:** `_raw_embed_select_k_best_prep`; same `feature_set` / `n_raw_features` plumbing; LGBM `raw+embed` branch uses skb on raw only.

**agent_score before:** 0.314526 **after:** 0.332248  

**Decision:** Keep  

**Reason:** Higher `agent_score`; **champion temporarily became `pls`** on `raw+embed` (stacking second).

---

## Iteration 3

**Hypothesis:** Replace raw-block `SelectKBest` with **PCA** (linear compression of raw before concat).

**Change:** `_raw_embed_pca_raw_prep` (PCA `n_components=min(200, n_raw-1)`, randomized SVD); swapped into all `raw+embed` model paths.

**agent_score before:** 0.332248 **after:** 0.309808  

**Decision:** Revert  

**Reason:** Clear drop vs session best; restored **SelectKBest** prep and **session_best = 0.332248**.

---

## Iteration 4

**Hypothesis:** For **stacking_linear** on `raw+embed`, specialization helps: **ElasticNet** sees full skb-preprocessed concat; **Ridge** sees **embed only**; **PLS** sees **raw-only skb** path.

**Change:** `_embed_only_scaled_prep`, `_raw_only_skb_prep`; stacking uses three different `ColumnTransformer` heads + cloned pipelines.

**agent_score before:** 0.332248 **after:** 0.346210  

**Decision:** Keep  

**Reason:** Gain vs best; **champion back to `stacking_linear`**.

---

## Iteration 5

**Hypothesis:** Add **VarianceThreshold(0)** on raw **before** `SelectKBest` in the shared raw+embed prep (remove near-constant raw features earlier).

**Change:** Insert `VarianceThreshold` in `_raw_embed_select_k_best_prep`, `_raw_only_skb_prep`, and LGBM raw branch.

**agent_score before:** 0.346210 **after:** 0.346210  

**Decision:** Revert  

**Reason:** Tie — no strict improvement.

---

## Iteration 6

**Hypothesis:** **Target scaling** for the raw+embed stacking path stabilizes meta-learner fitting.

**Change:** Wrap `stacking_linear` (when `raw+embed` plumbing active) in `TransformedTargetRegressor(regressor=stack, transformer=StandardScaler())`.

**agent_score before:** 0.346210 **after:** 0.354333  

**Decision:** Keep  

**Reason:** Clear improvement; **session_best = 0.354333**.

---

## Iteration 7

**Hypothesis:** Swap meta-learner from Ridge to **ElasticNet** (same α / l1_ratio / `selection` as base ENET — structural change of estimator class, not a new tuning sweep).

**Change:** `final_estimator=ElasticNet(...)` for `stacking_linear` when `use_re` (then corrected to only affect intended path — experiment still run).

**agent_score before:** 0.354333 **after:** 0.332248  

**Decision:** Revert  

**Reason:** Large drop; restored **Ridge(0.1)** meta-learner.

---

## Iteration 8

**Hypothesis:** Increase raw `SelectKBest` **k** from 300 → 500 (structural capacity, not α/l1 tuning).

**Change:** Default `k_raw=500` in skb helpers + LGBM branch.

**agent_score before:** 0.354333 **after:** 0.334183  

**Decision:** Revert  

**Reason:** Worse than best; restored **k=300**.

---

## Iteration 9

**Hypothesis:** **RobustScaler** on embed-only branches (outlier-resistant scaling) while raw skb path stays `StandardScaler`.

**Change:** `RobustScaler` in embed slices of `_raw_embed_select_k_best_prep` and `_embed_only_scaled_prep`.

**agent_score before:** 0.354333 **after:** 0.354202  

**Decision:** Revert  

**Reason:** Slightly worse than best; restored `StandardScaler` on embed.

---

## Iteration 10

**Hypothesis:** Add a **fourth** stacking base learner: **Ridge on the full** skb-preprocessed `raw+embed` (same Ridge `alpha` as embed-only ridge base — structural redundancy / diversity).

**Change:** Estimator `ridge_full`: `clone(prep_enet)` + `Ridge(0.5)`.

**agent_score before:** 0.354333 **after:** 0.354801  

**Decision:** Keep  

**Reason:** Small but strict improvement → **session_best = 0.354801**.

---

## Structural session summary

| Item | Value |
|------|--------|
| **Structural baseline agent_score** | 0.314526 |
| **Best agent_score (end)** | **0.354801** |
| **Δ** | +0.040275 |
| **Final champion** | `raw+embed` · `stacking_linear` |
| **Iterations run (this session)** | 10 |
| **Kept** | 4 (iter 2, 4, 6, 10) |
| **Reverted** | 6 (iter 1, 3, 5, 7, 8, 9) |
| **Early stop** | Not triggered (improvement on iter 10) |
| **Champion changed mid-session?** | Yes (stacking → PLS → stacking across logged runs) |

### Structural elements retained in `train.py`

- `make_model(..., feature_set=..., n_raw_features=...)` from `run_single_experiment`.
- **`raw+embed`:** `ColumnTransformer` with raw `SelectKBest(f_regression, k=min(300,n_raw))` + scale; embed impute + scale.
- **Stacking (raw+embed):** ENET on full prep; Ridge on **embed-only**; PLS on **raw-only skb**; **fourth** Ridge on full prep; **Ridge(0.1)** meta; **`TransformedTargetRegressor`** + target `StandardScaler`.
- Non–`raw+embed` feature sets / models: unchanged from pre-structural pipeline layout (no skb).

### Notes

- `config.json` has **`fast_mode: true`** → stacking CV stays at **KFold(n_splits=3)**; trying `n_splits=10` would only apply when `fast=False`.
- `prepare.py`, `agent_loop.py`, and `config.json` were **not** modified per instructions.
