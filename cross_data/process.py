from __future__ import absolute_import, division, print_function

import argparse
import numpy as np
import pandas as pd
from sklearn import preprocessing

raw_sep = ","
filter_min = 5
sample_num = 100
sample_pop = True


def read_raw(path):
    df = pd.read_csv(path, sep=raw_sep, header=None, names=["user_id", "item_id", "rating", "timestamp"])
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
        print("Min actions/user:", count_u.min())
        print("Max actions/user:", count_u.max())
        print("Avg actions/user: {:.2f}".format(count_u.mean()))


def process_domain(domain, path, prefix):
    print("\n" + "=" * 60)
    print("Processing Domain:", domain)
    print("=" * 60)

    df = read_raw(path)
    show_stats(df, f"{domain} raw")

    # sort & dedupe
    df = df.sort_values(by="timestamp", kind="mergesort")
    df = df.drop_duplicates(subset=["user_id", "item_id"], keep="first")

    # filter cold-start
    count_i = df.groupby("item_id")["user_id"].count()
    item_keep = count_i[count_i >= filter_min].index
    df = df[df["item_id"].isin(item_keep)]

    count_u = df.groupby("user_id")["item_id"].count()
    user_keep = count_u[count_u >= filter_min].index
    df = df[df["user_id"].isin(user_keep)]

    show_stats(df, f"{domain} processed")

    return df


def split_and_save(df, prefix, domain):
    df_test = df.groupby("user_id").tail(1)
    df = df.drop(df_test.index)

    df_valid = df.groupby("user_id").tail(1)
    df = df.drop(df_valid.index)

    df_valid = df_valid[df_valid["item_id"].isin(df["item_id"])]
    df_test = df_test[
        df_test["user_id"].isin(df_valid["user_id"])
        & (df_test["item_id"].isin(df["item_id"]) | df_test["item_id"].isin(df_valid["item_id"]))
    ]

    df.to_csv(prefix + "train.csv", header=False, index=False)
    df_valid.to_csv(prefix + "valid.csv", header=False, index=False)
    df_test.to_csv(prefix + "test.csv", header=False, index=False)

    print("\n--- {} split ---".format(domain))
    print("Train users:", df["user_id"].nunique())
    print("Train items:", df["item_id"].nunique())
    print("Valid users:", df_valid["user_id"].nunique())
    print("Test users:", df_test["user_id"].nunique())

    # negatives
    df_concat = pd.concat([df, df_valid, df_test])
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
            lambda x: np.random.choice(arr_item[~np.isin(arr_item, sr_user2items[x])], size=sample_num, replace=False)
        ).values

    df_negative = pd.concat([df_negative, pd.DataFrame(list(arr_sample))], axis=1)
    df_negative.to_csv(prefix + "negative.csv", header=False, index=False)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-x", default="Books")
    parser.add_argument("-y", default="Movies_and_TV")
    parser.add_argument("-z", default="CDs_and_Vinyl")
    args = parser.parse_args()

    px = f"processed_data_all/{args.x}_"
    py = f"processed_data_all/{args.y}_"
    pz = f"processed_data_all/{args.z}_"

    df_x = process_domain(args.x, f"amazon/ratings_{args.x}.csv", px)
    df_y = process_domain(args.y, f"amazon/ratings_{args.y}.csv", py)
    df_z = process_domain(args.z, f"amazon/ratings_{args.z}.csv", pz)

    # common users/items
    common_users = set(df_x["user_id"]).intersection(df_y["user_id"]).intersection(df_z["user_id"])
    common_items = set(df_x["item_id"]).intersection(df_y["item_id"]).intersection(df_z["item_id"])
    print("\n" + "=" * 60)
    print("Common across domains")
    print("=" * 60)
    print("Users overlap:", len(common_users))
    print("Items overlap:", len(common_items))

    df_all = pd.concat([df_x, df_y, df_z], keys=["x", "y", "z"])
    df_all = df_all[df_all["user_id"].isin(common_users)]

    le = preprocessing.LabelEncoder()
    df_all["user_id"] = le.fit_transform(df_all["user_id"]) + 1
    df_all["item_id"] = le.fit_transform(df_all["item_id"]) + 1

    show_stats(df_all, "All domains processed")

    split_and_save(df_all.loc["x"], px, args.x)
    split_and_save(df_all.loc["y"], py, args.y)
    split_and_save(df_all.loc["z"], pz, args.z)

    print("\n=== Done ===")


if __name__ == "__main__":
    main()
