# -*- coding: utf-8 -*-

import copy
import torch
import random
import numpy as np
from collections import defaultdict
from multiprocessing import Process, Queue

# ============================================================
# Sampler utilities
# ============================================================
def random_neq(l, r, s):
    """Random int in [l, r) not in set s."""
    t = np.random.randint(l, r)
    while t in s:
        t = np.random.randint(l, r)
    return t


def _sample_once(user_train, user_train2, time1, time2, usernum, itemnum, maxlen):
    """One sample for a random user."""
    user = np.random.randint(1, usernum + 1)
    while len(user_train.get(user, [])) <= 1:
        user = np.random.randint(1, usernum + 1)

    seq  = np.zeros([maxlen], dtype=np.int32)
    seq2 = np.zeros([maxlen], dtype=np.int32)
    pos  = np.zeros([maxlen], dtype=np.int32)
    neg  = np.zeros([maxlen], dtype=np.int32)
    t1   = np.zeros([maxlen], dtype=np.int32)
    t2   = np.zeros([maxlen], dtype=np.int32)

    # Domain 1 (target)
    nxt = user_train[user][-1]
    idx = maxlen - 1
    ts = set(user_train[user])

    for i, ts_i in reversed(list(zip(user_train[user][:-1], time1.get(user, [])[:-1]))):
        seq[idx] = i
        t1[idx]  = ts_i
        pos[idx] = nxt
        if nxt != 0:
            neg[idx] = random_neq(1, itemnum + 1, ts)
        nxt = i
        idx -= 1
        if idx == -1:
            break

    # Domain 2 (source)
    idx = maxlen - 1
    for i, ts_i in reversed(list(zip(user_train2.get(user, [])[:-1], time2.get(user, [])[:-1]))):
        seq2[idx] = i
        t2[idx]   = ts_i
        idx -= 1
        if idx == -1:
            break

    # Mask: for each step in seq (target), how many source events happened before
    mask = np.zeros([maxlen], dtype=np.int32)
    j = 0
    for k in range(maxlen):
        while j < maxlen and t1[k] >= t2[j]:
            j += 1
        mask[k] = j

    return user, seq, pos, neg, seq2, mask


def _sample_function(user_train, user_train2, time1, time2, usernum, itemnum, batch_size, maxlen, result_queue, seed):
    np.random.seed(seed)
    while True:
        batch = [_sample_once(user_train, user_train2, time1, time2, usernum, itemnum, maxlen)
                 for _ in range(batch_size)]
        # Materialize tuple-of-arrays for robustness
        u, seq, pos, neg, seq2, mask = list(zip(*batch))
        result_queue.put((u, seq, pos, neg, seq2, mask))


class WarpSampler(object):
    """Background multiprocess sampler that yields tuple of arrays."""
    def __init__(self, User, User2, time1, time2, usernum, itemnum, batch_size=64, maxlen=10, n_workers=1):
        self.result_queue = Queue(maxsize=n_workers * 10)
        self.processors = []
        for _ in range(n_workers):
            p = Process(
                target=_sample_function,
                args=(User, User2, time1, time2, usernum, itemnum, batch_size, maxlen, self.result_queue, np.random.randint(2e9))
            )
            p.daemon = True
            p.start()
            self.processors.append(p)

    def next_batch(self):
        return self.result_queue.get()

    def close(self):
        for p in self.processors:
            p.terminate()
            p.join()


# ============================================================
# Data partition & evaluation
# ============================================================
def data_partition(fname, fname2):
    """
    Loads cross-domain splits and aligns users & items.
    Expects files (train/valid/test/negative) in:
      cross_data/processed_data_all/{<domain>_*.csv}
    Returns:
      [user_train1, user_valid1, user_test1, usernum, itemnum1, user_neg1,
       user_train2, user_valid2, user_test2, itemnum2, Time1, Time2,
       item_text_embeddings, item_id_to_idx, reindexed_to_original]
    """
    usernum = 0
    itemnum1 = 0
    User = defaultdict(list)
    User1 = defaultdict(list)
    User2 = defaultdict(list)

    user_train1 = {}
    user_valid1 = {}
    user_test1 = {}
    neglist1 = defaultdict(list)
    user_neg1 = {}

    itemnum2 = 0
    user_train2 = {}
    user_valid2 = {}
    user_test2 = {}

    user_map = {}
    item_map = {}  # original item id -> reindexed id
    user_ids = []
    item_ids1 = []
    item_ids2 = []
    Time = defaultdict(list)
    Time1 = {}
    Time2 = {}

    # -------- Domain 1 (target) --------
    for split in ["train", "valid", "test"]:
        with open(f"cross_data/processed_data_all/{fname}_{split}.csv", "r") as f:
            for line in f:
                u, i, t = map(int, line.rstrip().split(","))
                user_ids.append(u)
                item_ids1.append(i)
                User[u].append(i)
                Time[u].append(t)

    unique_users = sorted(set(user_ids))
    user_map = {u: i + 1 for i, u in enumerate(unique_users)}
    usernum = len(user_map)

    unique_items1 = sorted(set(item_ids1))
    item_map = {i: j + 1 for j, i in enumerate(unique_items1)}
    itemnum1 = len(item_map)

    for user, items in User.items():
        if user in user_map:
            u = user_map[user]
            User1[u] = [item_map[i] for i in items if i in item_map]
            Time1[u] = Time[user]

    # reset temp
    User.clear(); Time.clear()

    # -------- Domain 2 (source) --------
    for split in ["train", "valid", "test"]:
        with open(f"cross_data/processed_data_all/{fname2}_{split}.csv", "r") as f:
            for line in f:
                u, i, t = map(int, line.rstrip().split(","))
                if u in user_map:
                    item_ids2.append(i)
                    User[u].append(i)
                    Time[u].append(t)

    unique_items2 = sorted(set(item_ids2))
    # extend item_map with new items in domain 2
    for i in unique_items2:
        if i not in item_map:
            item_map[i] = len(item_map) + 1
    itemnum2 = len(item_map) - itemnum1  # number of new items in domain2

    for user, items in User.items():
        if user in user_map:
            u = user_map[user]
            User2[u] = [item_map[i] for i in items if i in item_map]
            Time2[u] = Time[user]

    # -------- Negatives for domain 1 --------
    with open(f"cross_data/processed_data_all/{fname}_negative.csv", "r") as f:
        for line in f:
            l = line.rstrip().split(",")
            u_raw = int(l[0])
            if u_raw in user_map:
                u = user_map[u_raw]
                neglist1[u] = [item_map[int(j)] for j in l[1:] if int(j) in item_map]

    # -------- Train/valid/test split dicts --------
    for u in range(1, usernum + 1):
        if u in User1:
            n1 = len(User1[u])
            if n1 < 3:
                user_train1[u] = User1.get(u, [])
                user_valid1[u] = []
                user_test1[u] = []
            else:
                user_train1[u] = User1[u][:-2]
                user_valid1[u] = [User1[u][-2]]
                user_test1[u]  = [User1[u][-1]]
            user_neg1[u] = neglist1.get(u, [])

        if u in User2:
            n2 = len(User2.get(u, []))
            if n2 < 3:
                user_train2[u] = User2.get(u, [])
                user_valid2[u] = []
                user_test2[u] = []
            else:
                user_train2[u] = User2[u][:-2]
                user_valid2[u] = [User2[u][-2]]
                user_test2[u]  = [User2[u][-1]]

    # -------- Text embeddings (precomputed) --------
    try:
        # The .npy must be a dict with keys:
        #   'embeddings' -> np.ndarray [num_text_items, dim]
        #   'item_id_to_idx' -> dict mapping ORIGINAL item_id -> row index in embeddings
        data = np.load("cross_data/processed_data_all/item_text_embeddings.npy", allow_pickle=True).item()
        item_text_embeddings = torch.tensor(data["embeddings"], dtype=torch.float)
        item_id_to_idx = dict(data["item_id_to_idx"])
        print("Successfully loaded pre-computed text embeddings.")
    except FileNotFoundError:
        print("Warning: item_text_embeddings.npy not found. Text features will not be used.")
        item_text_embeddings = None
        item_id_to_idx = None

    # Build reverse map: reindexed -> original
    reindexed_to_original = {v: k for k, v in item_map.items()}

    return [
        user_train1, user_valid1, user_test1, usernum, itemnum1, user_neg1,
        user_train2, user_valid2, user_test2, itemnum2, Time1, Time2,
        item_text_embeddings, item_id_to_idx, reindexed_to_original
    ]


# ============================================================
# Evaluation
# ============================================================
def _make_user_inputs(u, train, valid, test, user_train2, time1, time2, maxlen, for_valid):
    seq  = np.zeros([maxlen], dtype=np.int32)
    seq2 = np.zeros([maxlen], dtype=np.int32)
    t1   = np.zeros([maxlen], dtype=np.int32)
    t2   = np.zeros([maxlen], dtype=np.int32)

    tgt_items = valid.get(u, []) if for_valid else test.get(u, [])
    tgt = tgt_items[0] if tgt_items else 0
    item_idx = [tgt]

    idx = maxlen - 1
    if not for_valid and valid.get(u):
        seq[idx] = valid[u][0]
        idx -= 1

    for i, t in reversed(list(zip(train.get(u, []), time1.get(u, [])))):
        if idx < 0: break
        seq[idx] = i; t1[idx] = t; idx -= 1

    idx = maxlen - 1
    for i, t in reversed(list(zip(user_train2.get(u, []), time2.get(u, [])))):
        if idx < 0: break
        seq2[idx] = i; t2[idx] = t; idx -= 1

    mask = np.zeros([maxlen], dtype=np.int32)
    j = 0
    for k in range(len(seq)):
        while j < maxlen and t1[k] >= t2[j]:
            j += 1
        mask[k] = j

    return seq, seq2, mask, item_idx


def _evaluate_batched(model, dataset, args, for_valid=False, batch_users=512, quiet=True):
    (
        train, valid, test, usernum, itemnum1, neg, user_train2, _, _,
        _, time1, time2, item_text_embeddings, item_id_to_idx, reindexed_to_original
    ) = copy.deepcopy(dataset)

    NDCG = 0.0
    HT = 0.0

    users = [u for u in range(1, usernum + 1) if train.get(u) and (valid.get(u) if for_valid else test.get(u))]
    valid_user_count = len(users)

    for s in range(0, len(users), batch_users):
        batch_u = users[s:s + batch_users]
        if not batch_u:
            continue

        seq_b, seq2_b, mask_b, items_b = [], [], [], []
        for u in batch_u:
            seq, seq2, mask, item_idx = _make_user_inputs(
                u, train, valid, test, user_train2, time1, time2, args.maxlen, for_valid
            )
            item_idx.extend(neg.get(u, []))  # candidate set: [tgt] + negatives
            seq_b.append(seq); seq2_b.append(seq2); mask_b.append(mask); items_b.append(item_idx)

        # model returns scores where higher = better; we compute ranks via argsort of negative scores
        scores = -model.predict_batch(
            np.array(batch_u), np.array(seq_b), np.array(seq2_b), np.array(items_b), np.array(mask_b)
        ).detach().cpu().numpy()

        ranks = scores.argsort(axis=1).argsort(axis=1)[:, 0]  # position of the target (first column)
        NDCG += (1.0 / np.log2(ranks + 2.0))[ranks < 10].sum()
        HT += (ranks < 10).sum()

        if not quiet and (s % (batch_users * 10) == 0):
            print(".", end="", flush=True)

    denom = max(1, valid_user_count)
    return NDCG / denom, HT / denom


def evaluate(model, dataset, args):
    return _evaluate_batched(model, dataset, args, for_valid=False)


def evaluate_valid(model, dataset, args):
    return _evaluate_batched(model, dataset, args, for_valid=True)


# ============================================================
# Graph inputs for GNN (fast version)
# ============================================================
def get_slice(inputs, max_n_node=None):
    """
    Build per-session graph slices for HeteroGNN-like step.
    Returns numpy arrays for speed:
      alias_inputs: (B, L)
      A:            (B, 2*max_n_node, max_n_node)   2 blocks: in/out
      items:        (B, max_n_node)
    """
    B = len(inputs)
    L = len(inputs[0]) if B > 0 else 0

    if max_n_node is None:
        max_n_node = L

    alias_inputs = np.zeros((B, L), dtype=np.int64)
    A = np.zeros((B, 2 * max_n_node, max_n_node), dtype=np.float32)
    items = np.zeros((B, max_n_node), dtype=np.int64)

    for b, u_input in enumerate(inputs):
        node = np.unique(u_input)
        node = node[node != 0]  # exclude pad
        node = node[:max_n_node]
        n = len(node)
        if n > 0:
            items[b, :n] = node

        u_A = np.zeros((max_n_node, max_n_node), dtype=np.float32)
        if len(u_input) > 1 and n > 1:
            node_map = {val: i for i, val in enumerate(node)}
            for i in range(len(u_input) - 1):
                a, c = u_input[i], u_input[i + 1]
                if a in node_map and c in node_map:
                    u_A[node_map[a], node_map[c]] = 1.0

        # normalize
        u_sum_in = u_A.sum(0); u_sum_in[u_sum_in == 0] = 1.0
        u_A_in = u_A / u_sum_in
        u_sum_out = u_A.sum(1); u_sum_out[u_sum_out == 0] = 1.0
        u_A_out = (u_A.T / u_sum_out[:, np.newaxis]).T

        A[b, :max_n_node, :] = u_A_in
        A[b, max_n_node:, :] = u_A_out.T  # match original concat shape

        alias_map = {val: i for i, val in enumerate(node)}
        # fill only up to L entries of this sequence
        fill_len = min(L, len(u_input))
        alias_inputs[b, :fill_len] = [alias_map.get(i, 0) for i in u_input[:fill_len]]

    return alias_inputs, A, items
