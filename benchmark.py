import argparse
import json
import subprocess
import sys
from pathlib import Path
from itertools import product

import pandas as pd
import os

# ---------------- CONFIG ----------------
BASE_DIR = Path("/home/godnean/PycharmProjects/nastyaPapers/autoresearch/olink_to_phenotypes")
PROJECT_DIR = Path("/net/mraid20/ifs/wisdom/segal_lab/jafar/Nastya/DL_nastya")
# Default phenotype workspace (template)
PHENOTYPE_RUN_DIR = BASE_DIR / "phenotype_runs" / "base_example"
# ---------------- CONFIG ----------------
PREP_DIR = PROJECT_DIR / "prepared"
RESULTS_PATH = PHENOTYPE_RUN_DIR / "results.csv"
TRAIN_SCRIPT = PHENOTYPE_RUN_DIR / "train.py"

DEFAULT_FEATURE_SETS = ["raw", "embed", "raw+embed"]
DEFAULT_MODELS = ["elasticnet", "lgbm", "stacking"]
# ----------------------------------------


def load_targets():
    metadata_path = PREP_DIR / "metadata.json"
    with open(metadata_path, "r") as f:
        metadata = json.load(f)
    return metadata["targets"]


def normalize_list_arg(x):
    if x is None:
        return None
    if isinstance(x, list):
        return x
    return [i.strip() for i in x.split(",") if i.strip()]


def load_completed_runs():
    if not RESULTS_PATH.exists():
        return set()

    df = pd.read_csv(RESULTS_PATH)
    required = {"target", "feature_set", "model"}
    if not required.issubset(df.columns):
        return set()

    completed = set(zip(df["target"], df["feature_set"], df["model"]))
    return completed


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--targets", type=str, default=None,
                        help='Comma-separated target names. Default: all targets from metadata.json')
    parser.add_argument("--feature_sets", type=str, default=None,
                        help='Comma-separated feature sets. Default: raw,embed,raw+embed')
    parser.add_argument("--models", type=str, default=None,
                        help='Comma-separated models. Default: elasticnet,lgbm,stacking')
    parser.add_argument("--random_state", type=int, default=42)
    parser.add_argument("--skip_existing", action="store_true",
                        help="Skip runs already present in results.csv")
    parser.add_argument("--max_runs", type=int, default=None,
                        help="Optional cap on number of runs for testing")
    args = parser.parse_args()

    targets = normalize_list_arg(args.targets) or load_targets()
    feature_sets = normalize_list_arg(args.feature_sets) or DEFAULT_FEATURE_SETS
    models = normalize_list_arg(args.models) or DEFAULT_MODELS

    completed = load_completed_runs() if args.skip_existing else set()

    combinations = list(product(targets, feature_sets, models))

    if args.skip_existing:
        combinations = [
            (target, feature_set, model)
            for (target, feature_set, model) in combinations
            if (target, feature_set, model) not in completed
        ]

    if args.max_runs is not None:
        combinations = combinations[:args.max_runs]

    print(f"Total runs to execute: {len(combinations)}")

    n_ok = 0
    n_fail = 0

    for i, (target, feature_set, model) in enumerate(combinations, start=1):
        cmd = [
            sys.executable,
            str(TRAIN_SCRIPT),
            "--target", target,
            "--feature_set", feature_set,
            "--model", model,
            "--random_state", str(args.random_state),
        ]

        print("=" * 80)
        print(f"[{i}/{len(combinations)}] Running:")
        print(" ".join([f'"{c}"' if " " in c else c for c in cmd]))

        try:
            subprocess.run(cmd, check=True)
            n_ok += 1
        except subprocess.CalledProcessError as e:
            print(f"Run failed: target={target}, feature_set={feature_set}, model={model}")
            print(f"Exit code: {e.returncode}")
            n_fail += 1

    print("=" * 80)
    print("Benchmark finished.")
    print(f"Successful runs: {n_ok}")
    print(f"Failed runs: {n_fail}")


if __name__ == "__main__":
    main()