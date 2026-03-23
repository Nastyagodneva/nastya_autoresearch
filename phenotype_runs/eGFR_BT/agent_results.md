# Agent results — eGFR_BT

## Iteration 1
Hypothesis: Apply Skill 1 + Skill 2 — separate raw vs embed preprocessing (SelectKBest on raw) and split-view `stacking_linear` (ENET full prep, Ridge embed-only, PLS raw-only, plus Ridge full).
Skill used: Skill 1 + Skill 2
Change: In `train.py`, added raw+embed `ColumnTransformer` preprocessing (raw: impute→SelectKBest(f_regression,k=min(300,n_raw))→StandardScaler; embed: impute→StandardScaler) and changed `stacking_linear` for `raw+embed` to split-view base learners with `final_estimator=Ridge(alpha=0.1)`.
Score before: 0.303790  →  Score after: 0.297276
Decision: Revert
Notes: No improvement vs session_best; split-view preprocessing did not transfer to eGFR_BT.

## Iteration 2
Hypothesis: Improve the `stacking_linear` raw-only path by removing low-signal/noisy raw features: insert SelectKBest on raw features inside the `stacking_linear` base learners when `feature_set=raw`.
Skill used: Skill 7
Change: In `train.py`, for `model_name="stacking_linear"` and `feature_set="raw"`, added `SelectKBest(f_regression, k=300)` after imputation and before scaling for each base learner pipeline (ENET / Ridge / PLS). Other feature sets stayed baseline.
Score before: 0.303790  →  Score after: 0.303790
Decision: Revert
Notes: Tie with session_best; per policy revert when not strictly improved.

## Iteration 3
Hypothesis: eGFR is weak/likely skewed, so apply Skill 3: wrap the champion-ish `stacking_linear` model in `TransformedTargetRegressor(StandardScaler())` to stabilize target scaling.
Skill used: Skill 3
Change: In `train.py`, wrapped `stacking_linear` (any feature_set) with `TransformedTargetRegressor(regressor=stack, transformer=StandardScaler())`.
Score before: 0.303790  →  Score after: 0.239180
Decision: Revert
Notes: Skill 3 harmed performance on eGFR_BT; reverted to baseline.

## Iteration 4
Hypothesis: Improve the last-run champion (`elasticnet` on `embed`) by applying Skill 3 to the standalone `elasticnet` model (target scaling).
Skill used: Skill 3
Change: In `train.py`, wrapped the `elasticnet` pipeline in `TransformedTargetRegressor(regressor=pipeline, transformer=StandardScaler())`.
Score before: 0.303790  →  Score after: 0.303790
Decision: Revert
Notes: Tie with session_best; 4 consecutive non-improvements → stop early per protocol.

## Session summary
Baseline: 0.303790
Best score: 0.303790
Kept changes: None (all iterations reverted)
Skills that transferred: None
Skills that failed: Skill 1+2, Skill 7 (raw SelectKBest in stacking), Skill 3 (stacking + elasticnet wrappings)
New patterns observed: Target transformation helped stacking negatively; even when elasticnet was the transient champion, applying the same wrapper did not improve valid R².

---

## Relaxed optimization session

**Baseline (fresh):** `agent_score` **0.303790** (`stacking_linear` + `raw+embed`).

| Step | Hypothesis / change | Score | Outcome |
|------|---------------------|-------|---------|
| R1 | `raw+embed` stacking: `ColumnTransformer` (raw: impute→SelectKBest k=min(400,n_raw)→scale; embed: impute→scale) + `TransformedTargetRegressor` on stack + meta `Ridge(0.2)` + `KFold(5)` | 0.297276 | Reverted (worse) |
| R2 | `TransformedTargetRegressor` on `stacking_linear` only (no column split) | 0.239180 | Reverted (worse) |
| R3 | Meta `ElasticNet(0.15, l1=0.5)` + `KFold(5)` for `stacking_linear` | **0.306244** | **Kept — new best** |
| R4 | Meta EN `0.12/0.4`, stack Ridge `0.6`, PLS 12 (fast) | 0.305900 | Reverted vs best |
| R5 | `RobustScaler` instead of `StandardScaler` in stacking base pipelines | 0.288693 | Reverted |
| R6 | Meta `Ridge(0.2)` vs EN meta | 0.306199 | Reverted vs best |

**Session summary:** Baseline 0.303790 → **Best 0.306244** (+0.002454). **Final code:** `stacking_linear` with `StandardScaler` bases, `ElasticNet(alpha=0.15,l1_ratio=0.5)` meta, `KFold(5)`. **Helped:** EN meta + more CV folds. **Failed:** heavy raw/embed preprocessing + target scaling on stack; `RobustScaler` bases; marginal meta/base tweaks.
