from __future__ import absolute_import, division, print_function

import argparse
import os
import json
import numpy as np
import pandas as pd
from sklearn import preprocessing
from tqdm import tqdm

import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)

tqdm.pandas()  # enable df.progress_apply

# ----------------------------
# Defaults
# ----------------------------
FILTER_MIN = 5
SAMPLE_NUM = 100
SAMPLE_POP = True
RANDOM_SEED = 42


# -------------------------------------------------
# Loader
# -------------------------------------------------
def load_interactions(path: str) -> pd.DataFrame:
    """
    Reads a plain CSV with NO header in the order:
        item_id, user_id, rating, timestamp
    Returns: user_id, item_id, timestamp
    """
    df = pd.read_csv(
        path,
        header=None,
        names=["item_id", "user_id", "rating", "timestamp"],
        usecols=[0, 1, 2, 3],
        dtype={0: "string", 1: "string", 2: "float32", 3: "int64"},
        low_memory=False,
    )
    return df[["user_id", "item_id", "timestamp"]].copy()


# -------------------------------------------------
# Preprocessing functions
# -------------------------------------------------
def basic_cleanup(df: pd.DataFrame, filter_min: int) -> pd.DataFrame:
    print("Sorting by timestamp...")
    df = df.sort_values(by=["timestamp"], kind="mergesort", ascending=True)

    print("Dropping duplicates...")
    df = df.drop_duplicates(subset=["user_id", "item_id"], keep="first")

    print("Filtering cold-start items...")
    item_counts = df.groupby("item_id").user_id.count()
    keep_items = item_counts[item_counts >= filter_min].index
    df = df[df["item_id"].isin(keep_items)]

    print("Filtering cold-start users...")
    user_counts = df.groupby("user_id").item_id.count()
    keep_users = user_counts[user_counts >= filter_min].index
    df = df[df["user_id"].isin(keep_users)]

    return df


def print_whole_stats(title: str, df: pd.DataFrame) -> None:
    print(f"\n==== {title} ====")
    n = df.user_id.nunique()
    m = df.item_id.nunique()
    p = len(df)
    density = (p / (n * m)) if (n > 0 and m > 0) else 0.0
    print(f"#users: {n}")
    print(f"#items: {m}")
    print(f"#actions: {p}")
    print(f"density: {density:.6f}")
    cnt = df.groupby("user_id").item_id.count()
    if len(cnt) > 0:
        print(f"min #actions per user: {cnt.min():.2f}")
        print(f"max #actions per user: {cnt.max():.2f}")
        print(f"ave #actions per user: {cnt.mean():.2f}")


def leave_one_out_split(df: pd.DataFrame):
    print("Splitting into train/valid/test with leave-one-out...")
    df_test = df.groupby("user_id").tail(1)
    df_train_valid = df.drop(df_test.index, axis="index")

    df_valid = df_train_valid.groupby("user_id").tail(1)
    df_train = df_train_valid.drop(df_valid.index, axis="index")

    df_valid = df_valid[df_valid.item_id.isin(df_train.item_id)]
    df_test = df_test[
        df_test.user_id.isin(df_valid.user_id)
        & (df_test.item_id.isin(df_train.item_id) | df_test.item_id.isin(df_valid.item_id))
    ]
    return df_train, df_valid, df_test


def sample_negatives(df_train, df_valid, df_test, sample_num, by_popularity, rng):
    print("Sampling negatives...")
    df_concat = pd.concat([df_train, df_valid, df_test], axis="index", ignore_index=True)
    sr_user2items = df_concat.groupby("user_id").item_id.unique()
    df_negative = pd.DataFrame({"user_id": df_concat.user_id.unique()})

    if by_popularity:
        sr_item2pop = df_concat.item_id.value_counts(sort=True, ascending=False)
        arr_item = sr_item2pop.index.values
        arr_pop = sr_item2pop.values.astype(float)
        arr_pop = arr_pop / arr_pop.sum()

        def get_negative_sample(pos):
            neg_mask = ~np.in1d(arr_item, pos)
            neg_items = arr_item[neg_mask]
            neg_probs = arr_pop[neg_mask]
            neg_probs = neg_probs / neg_probs.sum()
            return rng.choice(neg_items, size=sample_num, replace=False, p=neg_probs)

    else:
        arr_item = df_concat.item_id.unique()

        def get_negative_sample(pos):
            candidate = arr_item[~np.in1d(arr_item, pos)]
            return rng.choice(candidate, size=sample_num, replace=False)

    arr_sample = tqdm(df_negative.user_id, desc="Negative sampling").progress_apply(
        lambda u: get_negative_sample(sr_user2items[u])
    ).values
    df_negative = pd.concat([df_negative, pd.DataFrame(list(arr_sample))], axis="columns")
    return df_negative


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def save_splits(prefix, train, valid, test, negative):
    ensure_dir(os.path.dirname(prefix))
    train.to_csv(prefix + "train.csv", header=False, index=False)
    valid.to_csv(prefix + "valid.csv", header=False, index=False)
    test.to_csv(prefix + "test.csv", header=False, index=False)
    negative.to_csv(prefix + "negative.csv", header=False, index=False)


# -------------------------------------------------
# Main
# -------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Preprocess two Amazon domains for MGCL-style experiments.")
    parser.add_argument("--x_path", type=str, default="amazon/Clothing_Shoes_and_Jewelry.csv")
    parser.add_argument("--y_path", type=str, default="amazon/Sports_and_Outdoors.csv")
    parser.add_argument("--x_name", type=str, default="Clothing_Shoes_and_Jewelry")
    parser.add_argument("--y_name", type=str, default="Sports_and_Outdoors")
    parser.add_argument("--out_dir", type=str, default="processed_data_all")
    parser.add_argument("--filter_min", type=int, default=FILTER_MIN)
    parser.add_argument("--sample_num", type=int, default=SAMPLE_NUM)
    parser.add_argument("--sample_pop", action="store_true", default=SAMPLE_POP)
    parser.add_argument("--seed", type=int, default=RANDOM_SEED)
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)

    # Load both domains
    print("Loading domain X:", args.x_path)
    dfx_raw = load_interactions(args.x_path)
    print("Loading domain Y:", args.y_path)
    dfy_raw = load_interactions(args.y_path)

    print("\n==== statistic of raw data (X) ====")
    print(f"#users: {dfx_raw.user_id.nunique()}")
    print(f"#items: {dfx_raw.item_id.nunique()}")
    print(f"#actions: {len(dfx_raw)}")

    print("\n==== statistic of raw data (Y) ====")
    print(f"#users: {dfy_raw.user_id.nunique()}")
    print(f"#items: {dfy_raw.item_id.nunique()}")
    print(f"#actions: {len(dfy_raw)}")

    # Basic cleanup
    dfx = basic_cleanup(dfx_raw, args.filter_min)
    dfy = basic_cleanup(dfy_raw, args.filter_min)

    print_whole_stats("processed data (X)", dfx)
    print_whole_stats("processed data (Y)", dfy)

    # Keep only users in BOTH domains
    common_users = sorted(set(dfx.user_id.unique()).intersection(set(dfy.user_id.unique())))
    print(f"\nCommon users across X and Y: {len(common_users)}")
    dfx = dfx[dfx.user_id.isin(common_users)].copy()
    dfy = dfy[dfy.user_id.isin(common_users)].copy()

    # Global re-numbering of IDs
    df_both = pd.concat([dfx.assign(domain="x"), dfy.assign(domain="y")], ignore_index=True)
    user_le = preprocessing.LabelEncoder()
    item_le = preprocessing.LabelEncoder()
    df_both["user_id"] = user_le.fit_transform(df_both["user_id"]) + 1
    df_both["item_id"] = item_le.fit_transform(df_both["item_id"]) + 1

    # Split back by domain
    dfx = df_both[df_both["domain"] == "x"][["user_id", "item_id", "timestamp"]].copy()
    dfy = df_both[df_both["domain"] == "y"][["user_id", "item_id", "timestamp"]].copy()

    dfx = dfx.sort_values(by=["user_id", "timestamp"], kind="mergesort")
    dfy = dfy.sort_values(by=["user_id", "timestamp"], kind="mergesort")

    print_whole_stats("after user intersection (X)", dfx)
    print_whole_stats("after user intersection (Y)", dfy)

    # Leave-one-out splits
    x_train, x_valid, x_test = leave_one_out_split(dfx)
    y_train, y_valid, y_test = leave_one_out_split(dfy)

    # Negatives
    x_neg = sample_negatives(x_train, x_valid, x_test, args.sample_num, args.sample_pop, rng)
    y_neg = sample_negatives(y_train, y_valid, y_test, args.sample_num, args.sample_pop, rng)

    # Save
    x_prefix = os.path.join(args.out_dir, f"{args.x_name}_")
    y_prefix = os.path.join(args.out_dir, f"{args.y_name}_")
    save_splits(x_prefix, x_train, x_valid, x_test, x_neg)
    save_splits(y_prefix, y_train, y_valid, y_test, y_neg)

    print("\n==== split stats (X) ====")
    print(f"#train_users: {x_train.user_id.nunique()}")
    print(f"#train_items: {x_train.item_id.nunique()}")
    print(f"#valid_users: {x_valid.user_id.nunique()}")
    print(f"#test_users:  {x_test.user_id.nunique()}")

    print("\n==== split stats (Y) ====")
    print(f"#train_users: {y_train.user_id.nunique()}")
    print(f"#train_items: {y_train.item_id.nunique()}")
    print(f"#valid_users: {y_valid.user_id.nunique()}")
    print(f"#test_users:  {y_test.user_id.nunique()}")

    ensure_dir(args.out_dir)
    with open(os.path.join(args.out_dir, "id_map.json"), "w") as f:
        json.dump(
            {
                "num_users": int(df_both.user_id.nunique()),
                "num_items_global": int(df_both.item_id.nunique()),
                "domains": [args.x_name, args.y_name],
                "notes": "IDs are label-encoded globally across both domains.",
            },
            f,
            indent=2,
        )

    print("\nDone. Outputs written under:", args.out_dir)


if __name__ == "__main__":
    main()
