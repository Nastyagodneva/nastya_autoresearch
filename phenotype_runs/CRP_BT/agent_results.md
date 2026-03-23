# Agent results — CRP_BT

## Iteration 1
Hypothesis: CRP is extremely skewed, so apply Skill 3 — wrap the `elasticnet` (raw champion path) with `TransformedTargetRegressor(StandardScaler())`.
Skill used: Skill 3
Change: In `train.py`, wrapped the `elasticnet` pipeline in `TransformedTargetRegressor(regressor=pipeline, transformer=StandardScaler())`.
Score before: 0.095956  →  Score after: 0.094978
Decision: Revert
Notes: Skill 3 did not improve CRP_BT; per keep/improve policy reverted.

## Iteration 2
Hypothesis: Skill 3 didn’t help, so try light feature filtering on the raw champion path: add VarianceThreshold on raw features for `elasticnet` when `feature_set=raw`.
Skill used: None / None (exploratory VarianceThreshold)
Change: In `train.py`, inserted `VarianceThreshold(threshold=0.0)` after imputation in the `elasticnet` pipeline, conditioned on `feature_set == "raw"`.
Score before: 0.095956  →  Score after: 0.095956
Decision: Revert
Notes: Tie with session_best; per policy revert when not strictly improved.

## Iteration 3
Hypothesis: Skill 3 failed, so try lighter feature filtering — apply VarianceThreshold(1e-6) inside the `elasticnet` preprocessing pipeline.
Skill used: None / None (exploratory VarianceThreshold)
Change: In `train.py`, inserted `VarianceThreshold(threshold=1e-6)` after imputation and before scaling in the `elasticnet` pipeline.
Score before: 0.095956  →  Score after: 0.095956
Decision: Revert
Notes: Tie with session_best; no strict gain.

## Iteration 4
Hypothesis: Increase the aggressiveness slightly — try VarianceThreshold(1e-4) inside the `elasticnet` preprocessing pipeline.
Skill used: None / None (exploratory VarianceThreshold)
Change: In `train.py`, changed the `VarianceThreshold` threshold from `1e-6` to `1e-4` after imputation and before scaling in the `elasticnet` pipeline.
Score before: 0.095956  →  Score after: 0.095956
Decision: Revert
Notes: 4 consecutive non-improvements → stopped early per protocol.

## Session summary
Baseline: 0.095956
Best score: 0.095956
Kept changes: None (all iterations reverted)
Skills that transferred: None
Skills that failed: Skill 3, VarianceThreshold (thresholds 1e-6 and 1e-4) on CRP_BT
New patterns observed: CRP_BT seems to be best handled by the simple `elasticnet + raw` pipeline; adding target scaling or light variance filtering did not improve valid R², and embedding/stacking remained worse in the baseline sweep.

---

## Relaxed optimization session (≤5 iters; ENET regularization + target transform only)

**Baseline (fresh):** `agent_score` **0.095956** (`elasticnet` + `raw`).

| Step | Change | Score | Notes |
|------|--------|-------|-------|
| R1 | `TransformedTargetRegressor(StandardScaler)` on `elasticnet` **and** `alpha` 0.15→**0.10** | **0.108992** | **Best** — large gain vs baseline |
| R2 | `alpha` **0.08** (keep TTR) | 0.102167 | Reverted |
| R3 | `l1_ratio` **0.35** (keep TTR, α=0.10) | 0.096482 | Reverted |
| R4 | `alpha` **0.12** (keep TTR) | 0.103892 | Reverted |
| R5 | `alpha` **0.11** (keep TTR) | 0.106096 | Reverted |

**Summary:** Baseline 0.095956 → **Best 0.108992** (+0.013036). **Final `train.py`:** `elasticnet` pipeline wrapped in `TransformedTargetRegressor(..., StandardScaler())`, `ElasticNet(alpha=0.10, l1_ratio=0.5, ...)`.

**Contrast with prior session:** Earlier, TTR alone *without* lowering `alpha` slightly **hurt** (`0.094978`); here **TTR + slightly lower `alpha` together** helped — suggests target scaling needed a less aggressive coefficient penalty on this setup.
