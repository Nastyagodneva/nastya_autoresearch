import argparse
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_absolute_error, root_mean_squared_error
from sklearn.linear_model import ElasticNet
from sklearn.ensemble import StackingRegressor
from sklearn.model_selection import KFold

from lightgbm import LGBMRegressor


# ---------------- CONFIG ----------------
BASE_DIR = Path("/net/mraid20/ifs/wisdom/segal_lab/jafar/Nastya/DL_nastya")
PREP_DIR = BASE_DIR / "prepared"
RESULTS_PATH = BASE_DIR / "results.csv"
# ----------------------------------------


def load_data():
    X_raw = pd.read_parquet(PREP_DIR / "X_raw.parquet")
    X_embed = pd.read_parquet(PREP_DIR / "X_embed.parquet")
    y_df = pd.read_parquet(PREP_DIR / "y.parquet")

    with open(PREP_DIR / "splits.json", "r") as f:
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


def make_model(model_name, random_state=42):
    if model_name == "elasticnet":
        model = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("regressor", ElasticNet(alpha=1.0, l1_ratio=0.5, max_iter=5000, random_state=random_state))
        ])
        return model

    elif model_name == "lgbm":
        model = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("regressor", LGBMRegressor(
                n_estimators=300,
                learning_rate=0.05,
                num_leaves=31,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=random_state,
                n_jobs=-1
            ))
        ])
        return model

    elif model_name == "stacking":
        estimators = [
            ("enet", Pipeline([
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
                ("regressor", ElasticNet(alpha=1.0, l1_ratio=0.5, max_iter=5000, random_state=random_state))
            ])),
            ("lgbm", Pipeline([
                ("imputer", SimpleImputer(strategy="median")),
                ("regressor", LGBMRegressor(
                    n_estimators=300,
                    learning_rate=0.05,
                    num_leaves=31,
                    subsample=0.8,
                    colsample_bytree=0.8,
                    random_state=random_state,
                    n_jobs=-1
                ))
            ])),
        ]

        final_estimator = ElasticNet(alpha=0.1, l1_ratio=0.5, max_iter=5000, random_state=random_state)

        model = StackingRegressor(
            estimators=estimators,
            final_estimator=final_estimator,
            cv=KFold(n_splits=5, shuffle=True, random_state=random_state),
            n_jobs=-1
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
    parser.add_argument("--target", type=str, required=True)
    parser.add_argument("--feature_set", type=str, choices=["raw", "embed", "raw+embed"], required=True)
    parser.add_argument("--model", type=str, choices=["elasticnet", "lgbm", "stacking"], required=True)
    parser.add_argument("--random_state", type=int, default=42)
    args = parser.parse_args()

    print(f"Running target={args.target}, feature_set={args.feature_set}, model={args.model}")

    X_raw, X_embed, y_df, splits = load_data()

    if args.target not in y_df.columns:
        raise ValueError(f"Target '{args.target}' not found in y_df")

    X = get_X(args.feature_set, X_raw, X_embed)
    y = y_df[args.target]

    train_ids = splits["train_ids"]
    valid_ids = splits["valid_ids"]
    test_ids = splits["test_ids"]

    train_valid_ids = train_ids + valid_ids

    y_train_valid = y.loc[train_valid_ids].dropna()
    y_test = y.loc[test_ids].dropna()

    X_train_valid = X.loc[y_train_valid.index]
    X_test = X.loc[y_test.index]

    # final split for reporting validation separately
    y_train = y.loc[train_ids].dropna()
    y_valid = y.loc[valid_ids].dropna()

    X_train = X.loc[y_train.index]
    X_valid = X.loc[y_valid.index]

    model = make_model(args.model, random_state=args.random_state)

    # fit on train only -> validation score
    model.fit(X_train, y_train)
    valid_pred = model.predict(X_valid)
    valid_metrics = evaluate(y_valid, valid_pred)

    # refit on train+valid -> test score
    model = make_model(args.model, random_state=args.random_state)
    model.fit(X_train_valid, y_train_valid)
    test_pred = model.predict(X_test)
    test_metrics = evaluate(y_test, test_pred)

    result = {
        "target": args.target,
        "feature_set": args.feature_set,
        "model": args.model,
        "random_state": args.random_state,
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

    print(json.dumps(result, indent=2))
    append_results(result)
    print(f"Saved results to {RESULTS_PATH}")


if __name__ == "__main__":
    main()