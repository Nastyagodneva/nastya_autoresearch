# AutoResearch loop — phenotype-specific optimization

## Before you start — read skills first

Before making any changes, read `../../meta_summary/skills.md`.

Use the skills as **starting hypotheses**, not suggestions to consider later.
- Do not rediscover what is already validated
- Apply the most relevant skill in **iteration 1**, not iteration 5
- If a skill clearly applies based on the baseline profile, use it immediately
- If a skill fails on this phenotype, note it in `agent_results.md` under that skill number

## Goal

Improve `agent_score` produced by:

```
uv run agent_loop.py
```

`agent_score` = mean of best validation R² per target across the eval panel.
The champion table printed after each run shows which model/feature_set is currently winning.

## Files

- `train.py` — the only file you may edit (model code, preprocessing, structure)
- `agent_loop.py` — fixed harness, do not edit
- `config.json` — phenotype settings, do not edit
- `program.md` — this file
- `agent_results.md` — session log, update after every iteration
- `../../meta_summary/skills.md` — accumulated knowledge from past sessions

## Session protocol

1. Run a fresh baseline:
   `uv run agent_loop.py`
2. Record `agent_score` and champion model/feature_set as `session_best`
3. Check baseline R² and champion against skills.md to pick iteration 1 hypothesis
4. For up to 10 iterations:
   - Propose ONE hypothesis (prefer skills-guided over exploratory)
   - Make ONE change in `train.py`
   - Run `uv run agent_loop.py`
   - Compare to `session_best`
   - Keep if improved, revert if not
   - Document in `agent_results.md` using the format below
5. Stop early if 4 consecutive iterations show no improvement

## agent_results.md format

Each iteration must be documented as:

```
## Iteration N
Hypothesis: ...
Skill used: Skill X / None (exploratory)
Change: ...
Score before: X  →  Score after: Y
Decision: Keep / Revert
Notes: ...
```

After the session, add a summary block:

```
## Session summary
Baseline: X
Best score: Y
Kept changes: ...
Skills that transferred: ...
Skills that failed: ...
New patterns observed: ...
```

## Allowed changes

- Preprocessing structure (ColumnTransformer, scaling, filtering)
- Feature selection (SelectKBest, VarianceThreshold)
- Model hyperparameters
- Stacking structure (base learners, meta-learner, CV folds)
- TransformedTargetRegressor wrapping
- Dimensionality reduction (prefer SelectKBest over PCA based on past results)

## Do not change

- `prepare.py`, `agent_loop.py`, `config.json`
- The target, feature sets, or scoring logic
- Output format (agent_score must remain parseable)

## Constraints

- Fast mode is on — keep changes runtime-efficient
- One change at a time — small and interpretable
- Do not ask for manual command execution

## Priority order for iteration 1

Use this decision tree based on the fresh baseline:

1. If baseline R² < 0.35 AND raw+embed wins → apply Skill 1 + Skill 2 (separate preprocessing + split-view stacking)
2. If baseline R² < 0.35 AND raw wins → apply Skill 4 (stacking) with raw only
3. If baseline R² > 0.6 → run max 4 iterations, stop early (Skill 6)
4. If target is likely skewed (metabolic, inflammatory) → try Skill 3 (TransformedTargetRegressor)
5. If uncertain → run baseline sweep first, then match against skills