# Cross-phenotype insights

*Sources: `phenotype_runs/*/agent_results.md`, `config.json`, `agent_runs.csv` (as of meta-summary build).*

## Modeling takeaways

1. **Champion architecture differs by phenotype**  
   - **HbA1c** and **Liver**: best configs centered on **`stacking_linear`**.  
   - **BMI**: best config was **standalone `elasticnet`** on `raw+embed` (stacking was second).  
   → A single global “best model” does not transfer; the search space must stay **phenotype-local**.

2. **Feature representation is target-dependent**  
   - **Liver**: **`raw` alone** outperformed `raw+embed` (embed hurt the champion setting).  
   - **HbA1c** and **BMI**: **`raw+embed`** won.  
   → Agents should **read the baseline champion table** before editing; assumptions from one phenotype mislead the next.

3. **Same edit, opposite effect**  
   - Softening the **stacking ElasticNet** base (toward standalone ENET settings) **helped HbA1c** but **hurt Liver**, where a **strong ENET base (α=1.0)** was retained.  
   → **Hyperparameters are interaction effects** across base learners and meta-learner; reuse of “what worked last time” is risky.

4. **Marginal gains vs ceiling**  
   - **HbA1c**: substantial **Δ ≈ +0.028** validation R² (agent_score).  
   - **Liver**: **Δ ≈ +0.001** after heavy iteration—suggests **near-plateau** for the current feature set and model family.  
   - **BMI**: **no improvement** under tested ENET tweaks—baseline already strong for the benchmark.  
   → The workflow correctly surfaces **when to stop** vs **when more search is warranted**.

## Agentic / workflow takeaways

1. **Isolated workspaces worked**  
   Each phenotype folder carried its own `train.py` edits, `agent_results.md` log, and `agent_runs.csv` trail—supporting **clean attribution** and **no cross-folder contamination** (as required by the run protocol).

2. **Explicit keep / revert discipline**  
   Logs record **Keep / Revert** per iteration and tie-handling—useful for **auditing agent decisions** and teaching “small change → evaluate → compare to session best.”

3. **Stopping rules matter**  
   **BMI** used **early stop after 4 non-improving iterations**; **HbA1c / Liver** ran to the **10-iteration cap**.  
   → Meta-analysis should track **stop reason** (cap vs plateau) alongside final score.

4. **Gaps for future automation**  
   - **`agent_used`** is not written by `agent_loop.py`—fill manually if you need provenance (tool + model + human operator).  
   - **`dominant_edit_type`** is interpretive; refine with tagging in `agent_results.md` if you want machine aggregation.

## Reuse

- Append new phenotype rows to the CSVs when you add folders under `phenotype_runs/`.  
- Re-run or extend this meta-summary when new sessions complete.
