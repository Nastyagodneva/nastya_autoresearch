"""
Backward-compatible entry point: runs the default phenotype workspace
(`phenotype_runs/base_example/`).

Preferred: `cd phenotype_runs/base_example && uv run agent_loop.py`
(from that directory, with `UV_PROJECT_ENVIRONMENT` set if you use a shared venv).
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


def main() -> None:
    root = Path(__file__).resolve().parent
    target = root / "phenotype_runs" / "base_example" / "agent_loop.py"
    if not target.is_file():
        print(f"error: missing {target}", file=sys.stderr)
        sys.exit(1)
    spec = importlib.util.spec_from_file_location("phenotype_agent_loop", target)
    if spec is None or spec.loader is None:
        print("error: could not load agent_loop spec", file=sys.stderr)
        sys.exit(1)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["phenotype_agent_loop"] = mod
    spec.loader.exec_module(mod)
    mod.main()


if __name__ == "__main__":
    main()
