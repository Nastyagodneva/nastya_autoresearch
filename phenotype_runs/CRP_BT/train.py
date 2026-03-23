import argparse
import json
from pathlib import Path

import pandas as pd

from sklearn.compose import TransformedTargetRegressor
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_absolute_error, root_mean_squared_error
from sklearn.linear_model import ElasticNet
from sklearn.ensemble import StackingRegressor
from sklearn.model_selection import KFold
from sklearn.linear_model import Ridge
from sklearn.cross_decomposition import PLSRegression
from lightgbm import LGBMRegressor

# ---------------------------------------------------------------------------
# One phenotype optimization workspace per folder. Duplicate the template folder,
# edit only config.json for phenotype-specific settings (see program.md).
# Prepared data paths and filenames come from config.json next to this file.
# ---------------------------------------------------------------------------
_RUN_DIR = Path(__file__).resolve().parent
_CONFIG_PATH = _RUN_DIR / "config.json"

with open(_CONFIG_PATH, "r", encoding="utf-8") as _f:
    _CONFIG = json.load(_f)


def _targets_from_config():
    t = _CONFIG["target"]
    if isinstance(t, str):
        return [t]
    return list(t)


_data = _CONFIG["data"]
PREP_DIR = Path(_data["project_dir"]) / _data["prepared_subdir"]
_files = _data["files"]
RESULTS_PATH = _RUN_DIR / _CONFIG["outputs"]["results_csv"]

AGENT_EVAL_TARGETS = _targets_from_config()
AGENT_EVAL_FEATURE_SETS = list(_CONFIG["allowed_feature_sets"])
AGENT_EVAL_MODELS = list(_CONFIG["allowed_models"])

_FEATURE_SET_CHOICES = tuple(AGENT_EVAL_FEATURE_SETS)
_MODEL_CHOICES = tuple(AGENT_EVAL_MODELS)

# Defaults for agent_eval when not overridden on CLI (e.g. direct `uv run train.py --mode agent_eval`)
_CONFIG_FAST_MODE = bool(_CONFIG.get("fast_mode", False))
_CONFIG_MAX_TRAIN_SAMPLES = _CONFIG.get("max_train_samples")


def maybe_subsample(X, y, max_samples=None, random_state=42):
    if max_samples is None or len(y) <= max_samples:
        return X, y

    sampled_idx = y.sample(n=max_samples, random_state=random_state).index
    return X.loc[sampled_idx], y.loc[sampled_idx]


def run_single_experiment(
    target,
    feature_set,
    model_name,
    random_state=42,
    max_train_samples=None,
    fast=False,
):
    X_raw, X_embed, y_df, splits = load_data()

    if target not in y_df.columns:
        raise ValueError(f"Target '{target}' not found in y_df")

    X = get_X(feature_set, X_raw, X_embed)
    y = y_df[target]

    train_ids = splits["train_ids"]
    valid_ids = splits["valid_ids"]
    test_ids = splits["test_ids"]

    train_valid_ids = train_ids + valid_ids

    y_train_valid = y.loc[train_valid_ids].dropna()
    y_test = y.loc[test_ids].dropna()

    X_train_valid = X.loc[y_train_valid.index]
    X_test = X.loc[y_test.index]

    y_train = y.loc[train_ids].dropna()
    y_valid = y.loc[valid_ids].dropna()

    X_train = X.loc[y_train.index]
    X_valid = X.loc[y_valid.index]
    X_train, y_train = maybe_subsample(
        X_train, y_train, max_samples=max_train_samples, random_state=random_state
    )

    model = make_model(model_name, random_state=random_state, fast=fast)
    X_train_valid, y_train_valid = maybe_subsample(
        X_train_valid,
        y_train_valid,
        max_samples=max_train_samples if fast else None,
        random_state=random_state,
    )
    model.fit(X_train, y_train)
    valid_pred = model.predict(X_valid)
    valid_metrics = evaluate(y_valid, valid_pred)

    model = make_model(model_name, random_state=random_state)
    model.fit(X_train_valid, y_train_valid)
    test_pred = model.predict(X_test)
    test_metrics = evaluate(y_test, test_pred)

    result = {
        "target": target,
        "feature_set": feature_set,
        "model": model_name,
        "random_state": random_state,
        "n_train": int(len(y_train)),
        "n_valid": int(len(y_valid)),
        "n_test": int(len(y_test)),
        "valid_r2": valid_metrics["r2"],
        "valid_rmse": valid_metrics["rmse"],
        "valid_mae": valid_metrics["mae"],
        "test_r2": test_metrics["r2"],
        "test_rmse": test_metrics["rmse"],
        "test_mae": test_metrics["mae"],
    }

    return result


def run_agent_eval(random_state=42, max_train_samples=None, fast=False):
    sn = _CONFIG.get("session_name")
    if sn:
        print(f"session_name: {sn}")

    rows = []

    total = len(AGENT_EVAL_TARGETS) * len(AGENT_EVAL_FEATURE_SETS) * len(AGENT_EVAL_MODELS)
    k = 0

    for target in AGENT_EVAL_TARGETS:
        for feature_set in AGENT_EVAL_FEATURE_SETS:
            for model_name in AGENT_EVAL_MODELS:
                k += 1
                print(
                    f"[agent_eval {k}/{total}] target={target} | feature_set={feature_set} | model={model_name}"
                )

                try:
                    result = run_single_experiment(
                        target=target,
                        feature_set=feature_set,
                        model_name=model_name,
                        random_state=random_state,
                        max_train_samples=max_train_samples,
                        fast=fast,
                    )
                    rows.append(result)
                except Exception as e:
                    print(
                        f"agent_eval failed for target={target}, feature_set={feature_set}, model={model_name}: {e}"
                    )

    if len(rows) == 0:
        raise RuntimeError("agent_eval produced no successful runs.")

    df = pd.DataFrame(rows)

    best_idx = df.groupby("target")["valid_r2"].idxmax()
    champions = (
        df.loc[best_idx, ["target", "feature_set", "model", "valid_r2"]]
        .sort_values("target")
        .reset_index(drop=True)
    )

    agent_score = float(champions["valid_r2"].mean())

    print("\nDetailed agent evaluation results:")
    print(
        df[["target", "feature_set", "model", "valid_r2"]].sort_values(
            ["target", "valid_r2"], ascending=[True, False]
        )
    )

    print("\nChampion per target:")
    print(champions.to_string(index=False))

    print(f"\nchampion_models: {','.join(champions['model'].astype(str).tolist())}")
    print(f"agent_score: {agent_score:.6f}")

    print("\nAgent evaluation summary:")
    print(df.groupby(["feature_set", "model"])["valid_r2"].mean().sort_values(ascending=False))

    return df, agent_score


def load_data():
    X_raw = pd.read_parquet(PREP_DIR / _files["x_raw"])
    X_embed = pd.read_parquet(PREP_DIR / _files["x_embed"])
    y_df = pd.read_parquet(PREP_DIR / _files["y"])

    with open(PREP_DIR / _files["splits"], "r", encoding="utf-8") as f:
        splits = json.load(f)

    return X_raw, X_embed, y_df, splits


def get_X(feature_set, X_raw, X_embed):
    if feature_set == "raw":
        return X_raw.copy()
    elif feature_set == "embed":
        return X_embed.copy()
    elif feature_set == "raw+embed":
        return pd.concat([X_raw, X_embed], axis=1)
    else:
        raise ValueError(f"Unknown feature_set: {feature_set}")


def make_model(model_name, random_state=42, fast=False):
    if model_name == "elasticnet":
        pipe = Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
                (
                    "regressor",
                    ElasticNet(
                        alpha=0.10,
                        l1_ratio=0.5,
                        max_iter=5000,
                        random_state=random_state,
                        selection="random",
                    ),
                ),
            ]
        )
        return TransformedTargetRegressor(
            regressor=pipe,
            transformer=StandardScaler(),
        )

    elif model_name == "lgbm":
        model = Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                (
                    "regressor",
                    LGBMRegressor(
                        n_estimators=100 if fast else 300,
                        learning_rate=0.05,
                        num_leaves=31,
                        subsample=0.8,
                        colsample_bytree=0.8,
                        random_state=random_state,
                        n_jobs=-1,
                    ),
                ),
            ]
        )
        return model

    elif model_name == "ridge":
        return Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
                ("regressor", Ridge(alpha=1.0, random_state=random_state)),
            ]
        )

    elif model_name == "pls":
        n_components = 10 if fast else 20
        return Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
                ("regressor", PLSRegression(n_components=n_components)),
            ]
        )

    elif model_name == "stacking_linear":
        estimators = [
            (
                "enet",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="median")),
                        ("scaler", StandardScaler()),
                        (
                            "regressor",
                            ElasticNet(
                                alpha=1.0,
                                l1_ratio=0.5,
                                max_iter=5000,
                                random_state=random_state,
                            ),
                        ),
                    ]
                ),
            ),
            (
                "ridge",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="median")),
                        ("scaler", StandardScaler()),
                        ("regressor", Ridge(alpha=1.0, random_state=random_state)),
                    ]
                ),
            ),
            (
                "pls",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="median")),
                        ("scaler", StandardScaler()),
                        (
                            "regressor",
                            PLSRegression(n_components=10 if fast else 20),
                        ),
                    ]
                ),
            ),
        ]

        return StackingRegressor(
            estimators=estimators,
            final_estimator=Ridge(alpha=1.0, random_state=random_state),
            cv=KFold(n_splits=3 if fast else 5, shuffle=True, random_state=random_state),
            n_jobs=-1,
        )

    elif model_name == "stacking":
        estimators = [
            (
                "enet",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="median")),
                        ("scaler", StandardScaler()),
                        (
                            "regressor",
                            ElasticNet(
                                alpha=1.0,
                                l1_ratio=0.5,
                                max_iter=5000,
                                random_state=random_state,
                            ),
                        ),
                    ]
                ),
            ),
            (
                "lgbm",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="median")),
                        (
                            "regressor",
                            LGBMRegressor(
                                n_estimators=300,
                                learning_rate=0.05,
                                num_leaves=31,
                                subsample=0.8,
                                colsample_bytree=0.8,
                                random_state=random_state,
                                n_jobs=-1,
                            ),
                        ),
                    ]
                ),
            ),
        ]

        final_estimator = ElasticNet(
            alpha=0.1, l1_ratio=0.5, max_iter=5000, random_state=random_state
        )

        model = StackingRegressor(
            estimators=estimators,
            final_estimator=final_estimator,
            cv=KFold(n_splits=3 if fast else 5, shuffle=True, random_state=random_state),
            n_jobs=-1,
        )
        return model

    else:
        raise ValueError(f"Unknown model_name: {model_name}")


def evaluate(y_true, y_pred):
    return {
        "r2": float(r2_score(y_true, y_pred)),
        "rmse": float(root_mean_squared_error(y_true, y_pred)),
        "mae": float(mean_absolute_error(y_true, y_pred)),
    }


def append_results(row_dict):
    row_df = pd.DataFrame([row_dict])
    if RESULTS_PATH.exists():
        old = pd.read_csv(RESULTS_PATH)
        out = pd.concat([old, row_df], ignore_index=True)
    else:
        out = row_df
    out.to_csv(RESULTS_PATH, index=False)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--target", type=str, default=None)
    parser.add_argument(
        "--feature_set",
        type=str,
        choices=_FEATURE_SET_CHOICES,
        default=None,
    )
    parser.add_argument(
        "--model",
        type=str,
        choices=_MODEL_CHOICES,
        default=None,
    )
    parser.add_argument(
        "--mode", type=str, choices=["single", "agent_eval"], default="single"
    )
    parser.add_argument("--random_state", type=int, default=42)
    parser.add_argument("--max_train_samples", type=int, default=None)
    parser.add_argument("--fast", action="store_true")
    args = parser.parse_args()
    if args.mode == "agent_eval":
        max_samples = (
            args.max_train_samples
            if args.max_train_samples is not None
            else _CONFIG_MAX_TRAIN_SAMPLES
        )
        fast_eff = args.fast or _CONFIG_FAST_MODE
        run_agent_eval(
            random_state=args.random_state,
            max_train_samples=max_samples,
            fast=fast_eff,
        )
        return

    if args.mode == "single":
        if args.target is None or args.feature_set is None or args.model is None:
            raise ValueError(
                "In single mode, --target, --feature_set, and --model are required."
            )

        print(
            f"Running target={args.target}, feature_set={args.feature_set}, model={args.model}"
        )

        result = run_single_experiment(
            target=args.target,
            feature_set=args.feature_set,
            model_name=args.model,
            random_state=args.random_state,
            max_train_samples=args.max_train_samples,
            fast=args.fast,
        )

        print(json.dumps(result, indent=2))
        append_results(result)
        print(f"Saved results to {RESULTS_PATH}")


if __name__ == "__main__":
    main()
