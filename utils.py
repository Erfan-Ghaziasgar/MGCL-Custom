import sys
import copy
import torch
import random
import numpy as np
import pandas as pd
from collections import defaultdict
from multiprocessing import Process, Queue


def random_neq(l, r, s):
    t = np.random.randint(l, r)
    while t in s:
        t = np.random.randint(l, r)
    return t


def sample_function(user_train, user_train2, time1, time2, usernum, itemnum, batch_size, maxlen, result_queue, SEED):
    def sample():
        user = np.random.randint(1, usernum + 1)
        while len(user_train.get(user, [])) <= 1:
            user = np.random.randint(1, usernum + 1)

        seq = np.zeros([maxlen], dtype=np.int32)
        seq2 = np.zeros([maxlen], dtype=np.int32)
        pos = np.zeros([maxlen], dtype=np.int32)
        neg = np.zeros([maxlen], dtype=np.int32)
        t1 = np.zeros([maxlen], dtype=np.int32)
        t2 = np.zeros([maxlen], dtype=np.int32)

        nxt = user_train[user][-1]
        idx = maxlen - 1

        ts = set(user_train[user])
        for i, t in reversed(list(zip(user_train[user][:-1], time1.get(user, [])[:-1]))):
            seq[idx] = i
            t1[idx] = t
            pos[idx] = nxt
            if nxt != 0:
                neg[idx] = random_neq(1, itemnum + 1, ts)
            nxt = i
            idx -= 1
            if idx == -1:
                break

        idx = maxlen - 1
        for i, t in reversed(list(zip(user_train2.get(user, [])[:-1], time2.get(user, [])[:-1]))):
            seq2[idx] = i
            t2[idx] = t
            idx -= 1
            if idx == -1:
                break

        mask = np.zeros([maxlen], dtype=np.int32)
        idx2 = 0
        for idx in range(len(seq)):
            while idx2 < maxlen and t1[idx] >= t2[idx2]:
                idx2 += 1
            mask[idx] = idx2

        return (user, seq, pos, neg, seq2, mask)

    np.random.seed(SEED)
    while True:
        one_batch = []
        for _ in range(batch_size):
            one_batch.append(sample())
        result_queue.put(zip(*one_batch))


class WarpSampler(object):
    def __init__(self, User, User2, time1, time2, usernum, itemnum, batch_size=64, maxlen=10, n_workers=1):
        self.result_queue = Queue(maxsize=n_workers * 10)
        self.processors = []
        for _ in range(n_workers):
            p = Process(
                target=sample_function,
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


def data_partition(fname, fname2):
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
    item_map = {}
    user_ids = []
    item_ids1 = []
    item_ids2 = []
    Time = defaultdict(list)
    Time1 = {}
    Time2 = {}

    for split in ['train', 'valid', 'test']:
        with open(f'cross_data/processed_data_all/{fname}_{split}.csv', 'r') as f:
            for line in f:
                u, i, t = map(int, line.rstrip().split(','))
                user_ids.append(u)
                item_ids1.append(i)
                User[u].append(i)
                Time[u].append(t)
    
    unique_users = sorted(list(set(user_ids)))
    user_map = {u: i + 1 for i, u in enumerate(unique_users)}
    usernum = len(user_map)

    unique_items1 = sorted(list(set(item_ids1)))
    item_map = {i: j + 1 for j, i in enumerate(unique_items1)}
    itemnum1 = len(item_map)

    for user, items in User.items():
        if user in user_map:
            u = user_map[user]
            User1[u] = [item_map[i] for i in items if i in item_map]
            Time1[u] = Time[user]

    User = defaultdict(list)
    Time = defaultdict(list)

    for split in ['train', 'valid', 'test']:
        with open(f'cross_data/processed_data_all/{fname2}_{split}.csv', 'r') as f:
            for line in f:
                u, i, t = map(int, line.rstrip().split(','))
                if u in user_map:
                    item_ids2.append(i)
                    User[u].append(i)
                    Time[u].append(t)

    unique_items2 = sorted(list(set(item_ids2)))
    for i in unique_items2:
        if i not in item_map:
            item_map[i] = len(item_map) + 1
    itemnum2 = len(item_map) - itemnum1

    for user, items in User.items():
        if user in user_map:
            u = user_map[user]
            User2[u] = [item_map[i] for i in items if i in item_map]
            Time2[u] = Time[user]

    with open(f'cross_data/processed_data_all/{fname}_negative.csv', 'r') as f:
        for line in f:
            l = line.rstrip().split(',')
            u_raw = int(l[0])
            if u_raw in user_map:
                u = user_map[u_raw]
                neglist1[u] = [item_map[int(j)] for j in l[1:] if int(j) in item_map]

    for user in range(1, usernum + 1):
        if user in User1:
            nfeedback = len(User1[user])
            if nfeedback < 3:
                user_train1[user] = User1.get(user, [])
                user_valid1[user] = []
                user_test1[user] = []
            else:
                user_train1[user] = User1[user][:-2]
                user_valid1[user] = [User1[user][-2]]
                user_test1[user] = [User1[user][-1]]
            user_neg1[user] = neglist1.get(user, [])

        if user in User2:
            nfeedback = len(User2.get(user, []))
            if nfeedback < 3:
                user_train2[user] = User2.get(user, [])
                user_valid2[user] = []
                user_test2[user] = []
            else:
                user_train2[user] = User2[user][:-2]
                user_valid2[user] = [User2[user][-2]]
                user_test2[user] = [User2[user][-1]]

    # Load pre-computed embeddings
    try:
        embedding_data = np.load('cross_data/processed_data_all/item_text_embeddings.npy', allow_pickle=True).item()
        item_text_embeddings = torch.tensor(embedding_data['embeddings'], dtype=torch.float)
        item_id_to_idx = embedding_data['item_id_to_idx']
        print("Successfully loaded pre-computed text embeddings.")
    except FileNotFoundError:
        print("Warning: item_text_embeddings.npy not found. Text features will not be used.")
        item_text_embeddings = None
        item_id_to_idx = None

    return [user_train1, user_valid1, user_test1, usernum, itemnum1, user_neg1,
            user_train2, user_valid2, user_test2, itemnum2, Time1, Time2,
            item_text_embeddings, item_id_to_idx]


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
    idx2 = 0
    for k in range(len(seq)):
        while idx2 < maxlen and t1[k] >= t2[idx2]:
            idx2 += 1
        mask[k] = idx2

    return seq, seq2, mask, item_idx


def _evaluate_batched(model, dataset, args, for_valid=False, batch_users=512, quiet=True):
    [train, valid, test, usernum, itemnum1, neg, user_train2, _, _,
     _, time1, time2, item_text_embeddings, item_id_to_idx] = copy.deepcopy(dataset)

    NDCG = 0.0
    HT = 0.0
    
    users = [u for u in range(1, usernum + 1) if train.get(u) and (valid.get(u) if for_valid else test.get(u))]
    valid_user_count = len(users)

    for s in range(0, len(users), batch_users):
        batch_u = users[s:s + batch_users]
        if not batch_u: continue

        seq_b, seq2_b, mask_b, items_b = [], [], [], []
        for u in batch_u:
            seq, seq2, mask, item_idx = _make_user_inputs(u, train, valid, test, user_train2, time1, time2, args.maxlen, for_valid)
            item_idx.extend(neg.get(u, []))
            seq_b.append(seq); seq2_b.append(seq2); mask_b.append(mask); items_b.append(item_idx)

        scores = -model.predict_batch(np.array(batch_u), np.array(seq_b), np.array(seq2_b), np.array(items_b), np.array(mask_b)).detach().cpu().numpy()
        ranks = scores.argsort(axis=1).argsort(axis=1)[:, 0]
        
        NDCG += (1.0 / np.log2(ranks + 2.0))[ranks < 10].sum()
        HT += (ranks < 10).sum()

        if not quiet and (s % (batch_users * 10) == 0):
            print('.', end='', flush=True)
            
    return NDCG / max(1, valid_user_count), HT / max(1, valid_user_count)


def evaluate(model, dataset, args):
    return _evaluate_batched(model, dataset, args, for_valid=False)


def evaluate_valid(model, dataset, args):
    return _evaluate_batched(model, dataset, args, for_valid=True)


def get_slice(inputs):
    items, A, alias_inputs = [], [], []
    max_n_node = 100
    for u_input in inputs:
        node = np.unique(u_input)
        node = node[node != 0] # Exclude padding
        items.append(np.pad(node, (0, max_n_node - len(node)), 'constant'))
        
        u_A = np.zeros((max_n_node, max_n_node))
        if len(node) > 1:
            node_map = {val: i for i, val in enumerate(node)}
            for i in range(len(u_input) - 1):
                if u_input[i] in node_map and u_input[i+1] in node_map:
                    u_A[node_map[u_input[i]], node_map[u_input[i+1]]] = 1
        
        u_sum_in = u_A.sum(0); u_sum_in[u_sum_in == 0] = 1
        u_A_in = u_A / u_sum_in
        
        u_sum_out = u_A.sum(1); u_sum_out[u_sum_out == 0] = 1
        u_A_out = (u_A.T / u_sum_out[:, np.newaxis]).T
        
        A.append(np.concatenate([u_A_in, u_A_out.T]).T)
        
        alias_map = {val: i for i, val in enumerate(node)}
        alias_inputs.append([alias_map.get(i, 0) for i in u_input])
        
    return alias_inputs, np.array(A), items