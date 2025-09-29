from __future__ import absolute_import, division, print_function

import argparse
import numpy as np
import pandas as pd
from sklearn import preprocessing
import os

raw_sep = ","
filter_min = 5
sample_num = 100
sample_pop = True


def read_raw(path):
    df = pd.read_csv(path, sep=raw_sep, header=None,
                     names=["user_id", "item_id", "rating", "timestamp"])
    return df[["user_id", "item_id", "timestamp"]]


def show_stats(df, title):
    print("\n=== {} ===".format(title))
    n = df["user_id"].nunique()
    m = df["item_id"].nunique()
    p = len(df)
    print("Users:", n)
    print("Items:", m)
    print("Actions:", p)
    if n > 0 and m > 0:
        print("Density: {:.4f}".format(p / n / m))
    count_u = df.groupby("user_id")["item_id"].count()
    if len(count_u):
        print("Min actions/user:", int(count_u.min()))
        print("Max actions/user:", int(count_u.max()))
        print("Avg actions/user: {:.2f}".format(count_u.mean()))


def process_domain(domain, path):
    print("\n" + "=" * 60)
    print("Processing Domain:", domain)
    print("=" * 60)

    df = read_raw(path)
    show_stats(df, f"{domain} raw")

    # sort & dedupe (keep earliest interaction for a given (u,i))
    df = df.sort_values(by="timestamp", kind="mergesort")
    df = df.drop_duplicates(subset=["user_id", "item_id"], keep="first")

    # filter by min user/item frequency
    count_i = df.groupby("item_id")["user_id"].count()
    item_keep = count_i[count_i >= filter_min].index
    df = df[df["item_id"].isin(item_keep)]

    count_u = df.groupby("user_id")["item_id"].count()
    user_keep = count_u[count_u >= filter_min].index
    df = df[df["user_id"].isin(user_keep)]

    show_stats(df, f"{domain} processed")
    return df


def split_and_save(df, prefix, domain):
    os.makedirs(os.path.dirname(prefix), exist_ok=True)

    # chronological leave-one-out split per user: last->test, second last->valid
    df_test = df.groupby("user_id").tail(1)
    df_tr = df.drop(df_test.index)

    df_valid = df_tr.groupby("user_id").tail(1)
    df_train = df_tr.drop(df_valid.index)

    # guard: ensure valid/test items appear in train universe
    df_valid = df_valid[df_valid["item_id"].isin(df_train["item_id"])]
    df_test = df_test[
        df_test["user_id"].isin(df_valid["user_id"])
        & (df_test["item_id"].isin(df_train["item_id"]) | df_test["item_id"].isin(df_valid["item_id"]))
    ]

    df_train.to_csv(prefix + "train.csv", header=False, index=False)
    df_valid.to_csv(prefix + "valid.csv", header=False, index=False)
    df_test.to_csv(prefix + "test.csv", header=False, index=False)

    print("\n--- {} split ---".format(domain))
    print("Train users:", df_train["user_id"].nunique())
    print("Train items:", df_train["item_id"].nunique())
    print("Valid users:", df_valid["user_id"].nunique())
    print("Test users:", df_test["user_id"].nunique())

    # negatives
    df_concat = pd.concat([df_train, df_valid, df_test], ignore_index=True)
    sr_user2items = df_concat.groupby("user_id")["item_id"].unique()
    df_negative = pd.DataFrame({"user_id": df_concat["user_id"].unique()})

    if sample_pop:
        sr_item2pop = df_concat["item_id"].value_counts()
        arr_item = sr_item2pop.index.values
        arr_pop = sr_item2pop.values

        def get_negative(pos):
            neg_idx = ~np.isin(arr_item, pos)
            neg_item = arr_item[neg_idx]
            neg_pop = arr_pop[neg_idx]
            neg_pop = neg_pop / neg_pop.sum()
            return np.random.choice(neg_item, size=sample_num, replace=False, p=neg_pop)

        arr_sample = df_negative["user_id"].apply(lambda x: get_negative(sr_user2items[x])).values
    else:
        arr_item = df_concat["item_id"].unique()
        arr_sample = df_negative["user_id"].apply(
            lambda x: np.random.choice(
                arr_item[~np.isin(arr_item, sr_user2items[x])],
                size=sample_num, replace=False
            )
        ).values

    df_negative = pd.concat([df_negative, pd.DataFrame(list(arr_sample))], axis=1)
    df_negative.to_csv(prefix + "negative.csv", header=False, index=False)


def save_global_maps(le_user, le_item, out_dir):
    os.makedirs(out_dir, exist_ok=True)

    user_map = pd.DataFrame({
        "raw_user_id": le_user.classes_,
        "user_id": np.arange(1, len(le_user.classes_) + 1, dtype=int)
    })
    item_map = pd.DataFrame({
        "raw_item_id": le_item.classes_,
        "item_id": np.arange(1, len(le_item.classes_) + 1, dtype=int)
    })

    user_map.to_csv(os.path.join(out_dir, "_global_user_id_map.csv"), index=False)
    item_map.to_csv(os.path.join(out_dir, "_global_item_id_map.csv"), index=False)
    print(f"\nSaved global maps to {out_dir}")


def save_domain_maps(df_domain_reindexed, domain, out_dir):
    """
    df_domain_reindexed: a domain's interactions AFTER reindexing (columns: user_id, item_id, timestamp)
    This saves the local (domain-filtered) slices of the global maps for convenience.
    """
    os.makedirs(out_dir, exist_ok=True)

    uids = np.sort(df_domain_reindexed["user_id"].unique())
    iids = np.sort(df_domain_reindexed["item_id"].unique())
    # Note: These are already new IDs; we don't have raw here. We'll merge later from global maps if needed.
    # For convenience, we still export the new IDs sets for the domain.
    pd.DataFrame({"user_id": uids}).to_csv(
        os.path.join(out_dir, f"{domain}_user_id_map.csv"), index=False
    )
    pd.DataFrame({"item_id": iids}).to_csv(
        os.path.join(out_dir, f"{domain}_item_id_map.csv"), index=False
    )
    print(f"Saved domain ID lists for {domain} to {out_dir}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-x", default="Books")
    parser.add_argument("-y", default="Movies_and_TV")
    parser.add_argument("-z", default="CDs_and_Vinyl")
    args = parser.parse_args()

    out_dir = "processed_data_all"
    px = f"{out_dir}/{args.x}_"
    py = f"{out_dir}/{args.y}_"
    pz = f"{out_dir}/{args.z}_"

    # 1) Per-domain filtering on raw IDs
    df_x_raw = process_domain(args.x, f"amazon/ratings_{args.x}.csv")
    df_y_raw = process_domain(args.y, f"amazon/ratings_{args.y}.csv")
    df_z_raw = process_domain(args.z, f"amazon/ratings_{args.z}.csv")

    # 2) Keep only users common to ALL three (classic cross-domain setup)
    common_users = set(df_x_raw["user_id"]).intersection(df_y_raw["user_id"]).intersection(df_z_raw["user_id"])
    print("\n" + "=" * 60)
    print("Common across domains")
    print("=" * 60)
    print("Users overlap:", len(common_users))

    df_x_raw = df_x_raw[df_x_raw["user_id"].isin(common_users)]
    df_y_raw = df_y_raw[df_y_raw["user_id"].isin(common_users)]
    df_z_raw = df_z_raw[df_z_raw["user_id"].isin(common_users)]

    # 3) Fit global encoders on the union (ensures consistent IDs across domains)
    df_all_raw = pd.concat(
        [df_x_raw.assign(__domain="x"),
         df_y_raw.assign(__domain="y"),
         df_z_raw.assign(__domain="z")],
        ignore_index=True
    )

    le_user = preprocessing.LabelEncoder()
    le_user.fit(df_all_raw["user_id"])

    le_item = preprocessing.LabelEncoder()
    le_item.fit(df_all_raw["item_id"])

    # Save global maps (raw ➜ new)
    save_global_maps(le_user, le_item, out_dir)

    # 4) Apply reindexing (1-based)
    def reindex(df):
        df = df.copy()
        df["user_id"] = le_user.transform(df["user_id"]) + 1
        df["item_id"] = le_item.transform(df["item_id"]) + 1
        return df

    df_x = reindex(df_x_raw)
    df_y = reindex(df_y_raw)
    df_z = reindex(df_z_raw)

    # 5) Show combined stats (after reindexing)
    df_all = pd.concat([df_x, df_y, df_z], ignore_index=True)
    show_stats(df_all, "All domains processed (reindexed)")

    # 6) Split + save per domain
    split_and_save(df_x, px, args.x)
    split_and_save(df_y, py, args.y)
    split_and_save(df_z, pz, args.z)

    # 7) Save domain-filtered ID lists (new IDs) for convenience
    save_domain_maps(df_x, args.x, out_dir)
    save_domain_maps(df_y, args.y, out_dir)
    save_domain_maps(df_z, args.z, out_dir)

    print("\n=== Done ===")


if __name__ == "__main__":
    main()
