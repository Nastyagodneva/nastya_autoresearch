import json
from pathlib import Path

import numpy as np
import pandas as pd
import os
from sklearn.model_selection import train_test_split
PROJECT_DIR='/net/mraid20/ifs/wisdom/segal_lab/jafar/Nastya/DL_nastya'
# ---------------- CONFIG ----------------
DATA_DIR = os.path.join(PROJECT_DIR, "data")
OUT_DIR = os.path.join(PROJECT_DIR,  "prepared")
os.makedirs(OUT_DIR, exist_ok=True)

RAW_PATH = DATA_DIR+ "/features.parquet"
EMBED_PATH = PROJECT_DIR+ "/ukbb_supervised_embedding/hpp_supervised_embeddings.parquet"
Y_PATH = DATA_DIR + "/features_y.parquet"

TEST_SIZE = 0.20
VALID_SIZE_WITHIN_TRAIN = 0.20
RANDOM_STATE = 42
MIN_NON_MISSING_TARGET = 200
# ----------------------------------------


def clean_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [str(c).strip() for c in df.columns]
    return df


def main():
    print("Loading input tables...")
    raw_df = pd.read_parquet(RAW_PATH)
    embed_df = pd.read_parquet(EMBED_PATH)
    y_df = pd.read_parquet(Y_PATH)

    raw_df = clean_columns(raw_df)
    embed_df = clean_columns(embed_df)
    y_df = clean_columns(y_df)

    # Ensure index is string sample ID
    raw_df.index = raw_df.index.astype(str)
    embed_df.index = embed_df.index.astype(str)
    y_df.index = y_df.index.astype(str)

    common_ids = raw_df.index.intersection(embed_df.index).intersection(y_df.index)
    common_ids = common_ids.sort_values()

    if len(common_ids) == 0:
        raise ValueError("No overlapping sample IDs across raw_df, embed_df, and y_df.")

    raw_df = raw_df.loc[common_ids].copy()
    embed_df = embed_df.loc[common_ids].copy()
    y_df = y_df.loc[common_ids].copy()

    # Keep numeric targets only
    y_df = y_df.select_dtypes(include=[np.number]).copy()

    # Keep targets with enough observations and some variance
    valid_targets = []
    for col in y_df.columns:
        s = y_df[col]
        n_non_missing = s.notna().sum()
        std = s.std(skipna=True)
        if n_non_missing >= MIN_NON_MISSING_TARGET and pd.notna(std) and std > 0:
            valid_targets.append(col)

    if len(valid_targets) == 0:
        raise ValueError("No valid continuous targets found after filtering.")

    y_df = y_df[valid_targets].copy()

    # Save aligned tables
    raw_df.to_parquet(OUT_DIR + "/X_raw.parquet")
    embed_df.to_parquet(OUT_DIR + "/X_embed.parquet")
    y_df.to_parquet(OUT_DIR + "/y.parquet")

    # One fixed split on sample IDs
    all_ids = np.array(common_ids)

    train_ids, test_ids = train_test_split(
        all_ids, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )

    train_ids, valid_ids = train_test_split(
        train_ids,
        test_size=VALID_SIZE_WITHIN_TRAIN,
        random_state=RANDOM_STATE,
    )

    splits = {
        "train_ids": train_ids.tolist(),
        "valid_ids": valid_ids.tolist(),
        "test_ids": test_ids.tolist(),
    }

    with open(OUT_DIR + "/splits.json", "w") as f:
        json.dump(splits, f, indent=2)

    metadata = {
        "n_samples": int(len(common_ids)),
        "n_raw_features": int(raw_df.shape[1]),
        "n_embed_features": int(embed_df.shape[1]),
        "n_targets": int(y_df.shape[1]),
        "targets": list(y_df.columns),
        "random_state": RANDOM_STATE,
    }

    with open(OUT_DIR + "/metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    print("Done.")
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()