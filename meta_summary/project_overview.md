# Project overview — Olink → phenotypes (AutoResearch testbed)

## Primary goal

The **main purpose of this project is to improve *agentic* ways of working on ML codebases**: structured loops where an agent (or human–agent pair) reads local context, makes **small, testable edits**, runs a **fixed evaluation harness**, and **documents keep/revert decisions** with a clear paper trail.

**Continuous phenotype prediction** (Olink / tabular features → continuous outcomes) is the **testbed**: it supplies real optimization pressure, heterogeneous targets, and comparable scores—without being the fundamental research product in itself.

## What the codebase provides

| Piece | Role |
|--------|------|
| **`prepare.py`** (repo root) | Fixed data preparation; not part of the agent edit surface for phenotype sessions. |
| **`phenotype_runs/<name>/`** | **One workspace per phenotype**: `config.json` (targets, paths, search space), `train.py` (model code the agent may edit), `agent_loop.py` + `program.md` (how to score and log). |
| **`agent_loop.py` / `train.py` (agent_eval)** | Standard benchmark: `agent_score` = champion validation R² over allowed models × feature sets. |
| **`agent_results.md`** | Human/agent-readable session log (baseline, iterations, summary). |
| **`agent_runs.csv`** | Append-only score trail (timestamp, score, git hash). |
| **`meta_summary/`** (this folder) | **Aggregated view** across phenotypes for reporting and methodology—not a substitute for per-folder logs. |

## How to read success

1. **ML metric**: higher **`agent_score`** on the held-out validation definition in `train.py` (champion row).  
2. **Process metric**: traceable **iterations**, **honest reverts**, **early stops** when plateauing, and **no forbidden edits** (e.g. other phenotype folders, `prepare.py`).

## Audience

- **Methodology / tooling**: stakeholders who care about **repeatable agent workflows**, auditability, and when to stop.  
- **Modeling**: stakeholders who care about **which phenotype** improved, **how much**, and **which model path** won—see `phenotype_summary.csv` and per-folder `agent_results.md`.

## Related files

- Per-phenotype details: `../phenotype_runs/<Phenotype>/agent_results.md`  
- Cross-phenotype synthesis: `cross_phenotype_insights.md`  
- Tabular roll-up: `phenotype_summary.csv`, `agentic_summary.csv`

---

*This document emphasizes intent: phenotype benchmarks serve **agentic ML engineering** research and practice.*
