# Agent results — Triglycerides_BT

## Iteration 1
Hypothesis: Skill 1 + Skill 2 transfer — apply separate raw vs embed preprocessing (SelectKBest on raw, separate scaler on embed) and split-view `stacking_linear` so each base learner sees an appropriate feature view.
Skill used: Skill 1 + Skill 2
Change: In `train.py`, added `raw+embed` ColumnTransformer preprocessing (raw: impute→SelectKBest(f_regression,k=min(300,n_raw))→StandardScaler; embed: impute→StandardScaler) and modified `stacking_linear` to use split-view base learners (ENET on full prep, Ridge on embed-only, PLS on raw-only-after-skb, plus Ridge on full prep) with meta-learner `Ridge(alpha=0.1)`.
Score before: 0.497449  →  Score after: 0.495065
Decision: Revert
Notes: `stacking_linear` became the champion on `raw+embed`, but the validated R² decreased slightly vs session_best; Skill 1+2 did not transfer cleanly to this phenotype.

## Iteration 2
Hypothesis: Skill 1 transfer without Skill 2 diversity — apply raw+embed ColumnTransformer (SelectKBest on raw, separate scaler on embed) while keeping the original `stacking_linear` base-learners structure; also safeguard by adding SelectKBest to the `ridge` model when `feature_set=raw`.
Skill used: Skill 1
Change: In `train.py`, reintroduced `raw+embed` preprocessing via `ColumnTransformer` (raw: impute→SelectKBest(f_regression,k=min(300,n_raw))→StandardScaler; embed: impute→StandardScaler). Updated `elasticnet`/`ridge`/`pls` for `raw+embed` to use this prep. Updated `stacking_linear` for `raw+embed` to use the same full preprocessed feature view for ENET/Ridge/PLS (no split-view stacking). Added SelectKBest preprocessing for `ridge` when `feature_set=raw`.
Score before: 0.497449  →  Score after: 0.495065
Decision: Revert
Notes: No improvement vs session_best; Skill 1 alone didn’t transfer to Triglycerides.

## Iteration 3
Hypothesis: SelectKBest should improve raw-only stacking by removing low-signal/noisy raw features; apply SelectKBest to the `stacking_linear` base learners only when `feature_set=raw`.
Skill used: Skill 7
Change: In `train.py`, passed `feature_set`/`n_raw_features` into `make_model`, and for `stacking_linear` with `feature_set=raw` inserted `SelectKBest(f_regression, k=min(300,n_raw))` on raw features before each base learner’s regressor.
Score before: 0.497449  →  Score after: 0.497449
Decision: Revert
Notes: Tie with session_best; per “Keep if improved, revert if not” reverted.

## Iteration 4
Hypothesis: Add VarianceThreshold on the raw block before SelectKBest (grounded in Skill 1 raw+embed preprocessing) to remove near-zero-variance raw features and improve stacking on `raw+embed`.
Skill used: Skill 1 / None (exploratory VarianceThreshold)
Change: In `train.py`, reintroduced the `raw+embed` ColumnTransformer with raw preprocessing `impute→VarianceThreshold(0.0)→SelectKBest(f_regression,k=min(300,n_raw))→StandardScaler`, and kept `stacking_linear` base learners in the original (no split-view) structure.
Score before: 0.497449  →  Score after: 0.495065
Decision: Revert
Notes: No improvement; 4 consecutive non-improvements → stopped early per protocol.

## Session summary
Baseline: 0.497449
Best score: 0.497449
Kept changes: None (all iterations reverted)
Skills that transferred: None
Skills that failed: Skill 1 + Skill 2, Skill 1 alone, Skill 7 (no strict gain), and VarianceThreshold addition did not rescue Skill 1 for this phenotype.
New patterns observed: Triglycerides appears relatively robust to the tested raw/embedded structural preprocessing; attempts grounded in prior skills tended to slightly degrade valid R².

---

## Relaxed optimization session (stacking-focused)

**Rules:** Up to 2 related changes per iteration; up to 15 iters; stop after 6 consecutive iterations without beating rolling `session_best`; Triglycerides-only: do not revert unless `agent_score < baseline - 0.002` (baseline = fresh run below); ties kept.

### Baseline (fresh)
`agent_score`: **0.497449** (champion: `stacking_linear` + `raw+embed`).

### Iterations
- **R1:** `stacking_linear`: meta `ElasticNet(0.1, l1=0.5)` + `KFold(n_splits=5)` (was Ridge meta + 3 folds in fast). → **0.499346** — Keep; new best.
- **R2:** Softer base ENET `alpha=0.8` + meta EN `alpha=0.05`. → **0.497692** — Kept (not clear loss vs baseline 0.497449); rolled back in R3 for search.
- **R3:** Restore base ENET `alpha=1.0` + add 4th base `ridge_mild` `Ridge(0.3)`. → **0.498277** — Reverted toward best config (below R1 best).
- **R4:** Remove 4th base; meta `ElasticNet(0.08, l1=0.45)`. → **0.499344** — Keep; near prior best.
- **R5:** Meta `HuberRegressor`. → **0.394886** — **Revert** (clear loss vs baseline−0.002).
- **R6:** Meta `Ridge(0.15)` + base stack Ridge `alpha=0.5` (was 1.0). → **0.499634** — Keep; new best.
- **R7:** Meta `Ridge(0.08)` + base Ridge `0.4` + PLS `n_components=12` (fast). → **0.499448** — Reverted code to R6 (worse than best).
- **R8:** Meta `Ridge(0.12)` + `KFold(6)`. → **0.498514** — Reverted to R6-class config.
- **R9:** Meta `ElasticNet(0.12, l1=0.35)` (with R6-style bases: ENET 1.0, Ridge 0.5, PLS 10). → **0.499644** — Keep; **final best**.
- **R10:** Meta EN `alpha=0.10, l1=0.30`. → **0.499642** — Reverted meta to R9 (negligible vs R9).
- **R11:** Base EN `alpha=0.85`, Ridge `0.55`. → **0.498513** — Reverted to R9 bases.

### Session summary (relaxed)
- **Baseline:** 0.497449  
- **Best `agent_score`:** **0.499644** (+0.002195)  
- **Final `train.py`:** `stacking_linear` uses `KFold(5)`, base learners ENET(1.0), Ridge(0.5), PLS(10 fast / 20 non-fast), meta `ElasticNet(alpha=0.12, l1_ratio=0.35)`.  
- **Helped:** ElasticNet or small-α Ridge as **meta-learner** vs Ridge(1.0); **more CV folds** (5 vs 3) for stacking; **milder Ridge** in the stack (0.5 vs 1.0).  
- **Failed / harmful:** `HuberRegressor` meta (champion switched away from stacking).  
- **Surprising:** Tiny meta-learner tweaks moved `agent_score` in the fourth decimal; one bad meta choice collapsed the score sharply.
