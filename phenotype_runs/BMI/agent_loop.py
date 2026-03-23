import csv
import json
import re
import subprocess
from datetime import datetime
from pathlib import Path

_RUN_DIR = Path(__file__).resolve().parent
_CONFIG_PATH = _RUN_DIR / "config.json"


def _project_root() -> Path:
    """Directory that contains pyproject.toml (olink_to_phenotypes), for `uv run --project`."""
    for d in [_RUN_DIR, *_RUN_DIR.parents]:
        if (d / "pyproject.toml").is_file():
            return d
    raise RuntimeError(
        "Could not find pyproject.toml in any parent of this folder. "
        "Keep this phenotype run inside the olink_to_phenotypes repository."
    )


_PROJECT_ROOT = _project_root()

LOG_FILE = _RUN_DIR / "run.log"
CSV_FILE = _RUN_DIR / "agent_runs.csv"


def _load_config():
    with open(_CONFIG_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def _build_cmd():
    cfg = _load_config()
    train_script = _RUN_DIR / "train.py"
    cmd = [
        "uv",
        "run",
        "--project",
        str(_PROJECT_ROOT),
        str(train_script),
        "--mode",
        "agent_eval",
    ]
    if cfg.get("fast_mode", False):
        cmd.append("--fast")
    mts = cfg.get("max_train_samples")
    if mts is not None:
        cmd.extend(["--max_train_samples", str(mts)])
    return cmd


def run_eval():
    with open(LOG_FILE, "w", encoding="utf-8") as f:
        subprocess.run(
            _build_cmd(),
            stdout=f,
            stderr=subprocess.STDOUT,
            check=False,
            cwd=str(_RUN_DIR),
        )


def parse_score():
    with open(LOG_FILE, "r", encoding="utf-8") as f:
        text = f.read()

    matches = re.findall(r"agent_score:\s*([0-9.]+)", text)
    if not matches:
        raise ValueError("No agent_score found in run.log")

    return float(matches[-1])


def get_git_hash():
    try:
        return (
            subprocess.check_output(
                ["git", "rev-parse", "HEAD"],
                cwd=str(_PROJECT_ROOT),
                stderr=subprocess.DEVNULL,
            )
            .decode()
            .strip()
        )
    except Exception:
        return "no_git"


def log_result(score):
    exists = CSV_FILE.exists()

    with open(CSV_FILE, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if not exists:
            writer.writerow(["timestamp", "score", "git_hash"])
        writer.writerow([datetime.now().isoformat(), score, get_git_hash()])


def main():
    cfg = _load_config()
    sn = cfg.get("session_name")
    if sn:
        print(f"session_name: {sn}")

    run_eval()
    score = parse_score()
    log_result(score)
    print(f"Recorded score: {score:.6f}")


if __name__ == "__main__":
    main()
