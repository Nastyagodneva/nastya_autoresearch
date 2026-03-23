# Agent optimization log — **Liver sound speed (US)**

Phenotype workspace: `phenotype_runs/Liver_sound_speed_US/` · target: **Liver sound speed (US)**

## Baseline (fresh `uv run agent_loop.py`)

- **agent_score:** 0.179515  
- **Champion:** `feature_set=raw`, `model=stacking_linear`  
- Note: **`raw` beats `raw+embed`** for this target (different from HbA1c).

---

## Optimization iterations

| Iter | Change in `train.py` | Recorded score | vs session best | Decision |
|------|----------------------|----------------|-----------------|----------|
| 1 | Stacking base ENET: α 1.0→0.15 + `selection="random"` | 0.148615 | ↓ | **Revert** |
| 2 | Final meta Ridge: 1.0→0.5 | 0.179515 | = | **Revert** |
| 3 | Stacking base Ridge: 1.0→**0.5** | 0.180052 | ↑ | **Keep** |
| 4 | Final Ridge: 1.0→0.5 (on top of iter 3) | 0.180052 | = | **Revert** final |
| 5 | Stacking PLS branch: 20→25 comps | 0.180052 | = | **Revert** |
| 6 | Stacking base ENET: α 1.0→0.5 | 0.170743 | ↓ | **Revert** |
| 7 | Stacking base Ridge: 0.5→**0.3** | 0.180353 | ↑ | **Keep** |
| 8 | Final Ridge: 1.0→0.5 | 0.180353 | = | **Revert** final |
| 9 | Stacking base Ridge: 0.3→**0.25** | 0.180440 | ↑ | **Keep** |
| 10 | Final Ridge: 1.0→0.5 | 0.180440 | = | **Revert** final |

Stopped: **10 iterations** (cap).

---

## Phenotype session summary

| Metric | Value |
|--------|--------|
| **Baseline agent_score** | 0.179515 |
| **Best agent_score** | **0.180440** |
| **Δ** | +0.000925 |
| **Final champion** | `raw` · `stacking_linear` |
| **Strongest model path** | `stacking_linear`: base ENET α=**1.0** (unchanged), base Ridge α=**0.25**, PLS 20 comps, meta Ridge α=**1.0**, KFold(5) |

### Kept changes

- Softer **base Ridge** in the stack (1.0 → **0.25**), tuned via 0.5 → 0.3 → 0.25.

### Reverted / unsuccessful

- Softer stacking ElasticNet (0.15 or 0.5) **hurt** vs strong ENET — this phenotype wants a **stronger** ENET base than HbA1c.
- Meta Ridge 0.5, PLS 25, and final/meta tweaks were **ties or no gain** on top of the tuned base Ridge.
