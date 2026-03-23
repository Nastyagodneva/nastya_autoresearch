# Discovered modeling skills

*Extracted from phenotype optimization sessions. Each skill has a confidence level
(how many phenotypes it was tested on) and a recommended starting condition.*

---

## Skill 1: Separate preprocessing for raw vs embed
**Condition:** feature_set = raw+embed  
**Confidence:** High (validated on HbA1C, +0.040 on full data)  
**Action:** Use a ColumnTransformer that applies different preprocessing to raw and embed columns:
- raw: impute (median) → SelectKBest(f_regression, k=min(300, n_raw)) → StandardScaler
- embed: impute (median) → StandardScaler  

**Why:** raw (~1028 features) is noisy/redundant; embed (~128) is already compressed. Treating them identically wastes the structural difference.  
**Caution:** k=500 was worse than k=300 for HbA1C. Start conservative with k=300.

---

## Skill 2: Split-view stacking
**Condition:** Champion is stacking_linear + raw+embed  
**Confidence:** Medium (validated on HbA1C, +0.040)  
**Action:** Give base learners different views of the feature space:
- ENET on full preprocessed features
- Ridge on embed-only
- PLS on raw-only (after SelectKBest)
- Ridge on full preprocessed features  

Meta-learner: Ridge(alpha=0.1)  
**Why:** Different learners exploit different information in raw vs embed. Forces diversity in the ensemble.  
**Caution:** Not yet tested on Liver or BMI. May not help if one feature set dominates.

---

## Skill 3: TransformedTargetRegressor for skewed targets
**Condition:** Weak baseline phenotype (R² < 0.35), stacking or ENET champion  
**Confidence:** Medium (validated on HbA1C)  
**Action:** Wrap the model in TransformedTargetRegressor(transformer=StandardScaler())  
**Why:** Helps when target distribution has scale/skew issues. HbA1C is right-skewed.  
**Caution:** Only tested with raw+embed. Not yet validated on other phenotypes.

---

## Skill 4: Stacking beats standalone for weak/noisy phenotypes
**Condition:** Baseline R² < 0.35  
**Confidence:** High (HbA1C: stacking won; Liver: stacking won)  
**Action:** Prioritize stacking_linear as starting champion for weak phenotypes.
Start directly with stacking rather than spending iterations on standalone models.  
**Caution:** BMI (strong baseline) preferred standalone ElasticNet. Do not force stacking on strong phenotypes.

---

## Skill 5: Embed sometimes hurts — check raw-only early
**Condition:** Any new phenotype  
**Confidence:** High (Liver: raw-only beat raw+embed; HbA1C and BMI: raw+embed won)  
**Action:** Always include raw-only in the initial agent_eval sweep.
Do not assume raw+embed is better.  
**Why:** UKBB-trained embeddings may not generalise well to all phenotype types.

---

## Skill 6: Strong phenotypes saturate quickly — stop early
**Condition:** Baseline R² > 0.6  
**Confidence:** High (BMI: no improvement after 4 iterations)  
**Action:** Run a short session (4–5 iterations max). If no gain, stop and move resources elsewhere.  
**Why:** Linear models already capture most available signal. LGBM may help later but is expensive.

---

## Skill 7: SelectKBest > PCA for Olink features
**Condition:** raw or raw+embed, linear model family  
**Confidence:** Medium (HbA1C: SelectKBest helped, PCA hurt)  
**Action:** Prefer SelectKBest(f_regression, k=300) over PCA for raw Olink features.  
**Why:** Signal is sparse (few features matter), not low-rank (PCA assumption).
Removing irrelevant features is better than projecting all features into components.  
**Caution:** Only one phenotype tested. Worth verifying on new phenotypes.

---

## How to use these skills

### As a human
Before starting a new phenotype session, scan the skill list and ask:
1. What is the baseline R²? → picks Skill 4 or 6
2. Does raw+embed win in the baseline sweep? → picks Skill 1 + 2 or Skill 5
3. Is the target distribution skewed? → picks Skill 3

### As an agent
Add this block to the top of `program.md` for each new phenotype:

```
Before making any changes, read meta_summary/skills.md.
Use the skills as starting hypotheses — do not rediscover what is already known.
If a skill applies to this phenotype's baseline profile, apply it in iteration 1.
If a skill fails on this phenotype, note it in agent_results.md under that skill number.
```

### Updating skills
- When a skill **transfers** to a new phenotype: increase its confidence level
- When a skill **fails** on a new phenotype: add a note under Caution
- When a **new pattern** emerges from a session: add it as a new skill
- Skills should be updated after every phenotype session completes
