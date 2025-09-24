import sys
import copy
import torch
import random
import numpy as np
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
        while len(user_train[user]) <= 1:
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
        for i, t in reversed(list(zip(user_train[user][:-1], time1[user][:-1]))):
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
        for i, t in reversed(list(zip(user_train2[user][:-1], time2[user][:-1]))):
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
                args=(
                    User, User2, time1, time2, usernum, itemnum,
                    batch_size, maxlen, self.result_queue, np.random.randint(2e9)
                )
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
    neglist2 = defaultdict(list)
    user_neg2 = {}

    user_map = dict()
    item_map = dict()

    user_ids = list()
    item_ids1 = list()
    item_ids2 = list()

    Time = defaultdict(list)
    Time1 = {}
    Time2 = {}

    f = open('cross_data/processed_data_all/%s_train.csv' % fname, 'r')
    for line in f:
        u, i, t = line.rstrip().split(',')
        u = int(u); i = int(i); t = int(t)
        user_ids.append(u); item_ids1.append(i)
        User[u].append(i); Time[u].append(t)

    f = open('cross_data/processed_data_all/%s_valid.csv' % fname, 'r')
    for line in f:
        u, i, t = line.rstrip().split(',')
        u = int(u); i = int(i); t = int(t)
        user_ids.append(u); item_ids1.append(i)
        User[u].append(i); Time[u].append(t)

    f = open('cross_data/processed_data_all/%s_test.csv' % fname, 'r')
    for line in f:
        u, i, t = line.rstrip().split(',')
        u = int(u); i = int(i); t = int(t)
        user_ids.append(u); item_ids1.append(i)
        User[u].append(i); Time[u].append(t)

    for u in user_ids:
        if u not in user_map:
            user_map[u] = usernum + 1
            usernum += 1
    for i in item_ids1:
        if i not in item_map:
            item_map[i] = itemnum1 + 1
            itemnum1 += 1

    for user in User:
        u = user_map[user]
        for item in User[user]:
            i = item_map[item]
            User1[u].append(i)
        Time1[u] = Time[user]

    User = defaultdict(list); Time = defaultdict(list)

    f = open('cross_data/processed_data_all/%s_train.csv' % fname2, 'r')
    for line in f:
        u, i, t = line.rstrip().split(',')
        u = int(u); i = int(i); t = int(t)
        user_ids.append(u); item_ids2.append(i)
        User[u].append(i); Time[u].append(t)

    f = open('cross_data/processed_data_all/%s_valid.csv' % fname2, 'r')
    for line in f:
        u, i, t = line.rstrip().split(',')
        u = int(u); i = int(i); t = int(t)
        user_ids.append(u); item_ids2.append(i)
        User[u].append(i); Time[u].append(t)

    f = open('cross_data/processed_data_all/%s_test.csv' % fname2, 'r')
    for line in f:
        u, i, t = line.rstrip().split(',')
        u = int(u); i = int(i); t = int(t)
        user_ids.append(u); item_ids2.append(i)
        User[u].append(i); Time[u].append(t)

    for i in item_ids2:
        if i not in item_map:
            item_map[i] = itemnum1 + itemnum2 + 1
            itemnum2 += 1

    for user in User:
        u = user_map[user]
        for item in User[user]:
            i = item_map[item]
            User2[u].append(i)
        Time2[u] = Time[user]

    f = open('cross_data/processed_data_all/%s_negative.csv' % fname, 'r')
    for line in f:
        l = line.rstrip().split(',')
        u = user_map[int(l[0])]
        for j in range(1, 101):
            i = item_map[int(l[j])]
            neglist1[u].append(i)

    for user in User1:
        nfeedback = len(User1[user])
        if nfeedback < 3:
            user_train1[user] = User1[user]; user_valid1[user] = []; user_test1[user] = []
        else:
            user_train1[user] = User1[user][:-2]
            user_valid1[user] = [User1[user][-2]]
            user_test1[user]  = [User1[user][-1]]
        user_neg1[user] = neglist1[user]

    for user in User2:
        nfeedback = len(User2[user])
        if nfeedback < 3:
            user_train2[user] = User2[user]; user_valid2[user] = []; user_test2[user] = []
        else:
            user_train2[user] = User2[user][:-2]
            user_valid2[user] = [User2[user][-2]]
            user_test2[user]  = [User2[user][-1]]

    return [user_train1, user_valid1, user_test1, usernum, itemnum1, user_neg1,
            user_train2, user_valid2, user_test2, itemnum2, Time1, Time2]


# ---------- Batched evaluation (fast) ----------
def _make_user_inputs(u, train, valid, test, user_train2, time1, time2, maxlen, for_valid):
    """Build (seq, seq2, mask, item_idx[101]) for one user."""
    seq  = np.zeros([maxlen], dtype=np.int32)
    seq2 = np.zeros([maxlen], dtype=np.int32)
    t1   = np.zeros([maxlen], dtype=np.int32)
    t2   = np.zeros([maxlen], dtype=np.int32)

    # target item
    tgt = valid[u][0] if for_valid else test[u][0]
    item_idx = [tgt]  # + 100 negatives later

    # domain A history (train)
    idx = maxlen - 1
    if for_valid:
        # when validating, the last interaction to predict is valid[u][0]
        # so we only push train[u]
        pass
    else:
        # for test, code previously did: seq[idx] = valid[u][0]; idx -= 1
        # keep identical behavior for fairness
        if len(valid[u]) > 0:
            seq[idx] = valid[u][0]
            idx -= 1
    for i, t in reversed(list(zip(train[u], time1[u]))):
        if idx < 0: break
        seq[idx] = i; t1[idx] = t; idx -= 1

    # domain B history
    idx = maxlen - 1
    for i, t in reversed(list(zip(user_train2[u], time2[u]))):
        if idx < 0: break
        seq2[idx] = i; t2[idx] = t; idx -= 1

    # cross-timeline mask
    mask = np.zeros([maxlen], dtype=np.int32)
    idx2 = 0
    for k in range(len(seq)):
        while idx2 < maxlen and t1[k] >= t2[idx2]:
            idx2 += 1
        mask[k] = idx2

    return seq, seq2, mask, item_idx


def _evaluate_batched(model, dataset, args, for_valid=False, batch_users=512, quiet=True):
    [train, valid, test, usernum, itemnum1, neg, user_train2, user_valid2, user_test2,
     itemnum2, time1, time2] = copy.deepcopy(dataset)

    NDCG = 0.0
    HT = 0.0
    valid_user = 0

    # gather all candidate users
    users = []
    for u in range(1, usernum + 1):
        if len(train[u]) < 1:
            continue
        if for_valid and len(valid[u]) < 1:
            continue
        if (not for_valid) and len(test[u]) < 1:
            continue
        users.append(u)

    # mini-batch over users
    B = batch_users
    for s in range(0, len(users), B):
        batch_u = users[s:s + B]
        if not batch_u:
            continue

        seq_b, seq2_b, mask_b, items_b = [], [], [], []
        for u in batch_u:
            seq, seq2, mask, item_idx = _make_user_inputs(
                u, train, valid, test, user_train2, time1, time2, args.maxlen, for_valid
            )
            # append 100 negatives (fixed length)
            for i in neg[u]:
                item_idx.append(i)
            seq_b.append(seq); seq2_b.append(seq2); mask_b.append(mask); items_b.append(item_idx)

        # predict in batch (prefer predict_batch if present)
        if hasattr(model, "predict_batch"):
            logits = model.predict_batch(
                np.array(batch_u), np.array(seq_b), np.array(seq2_b),
                np.array(items_b), np.array(mask_b)
            ).detach().cpu().numpy()  # [B, K]
            # original code uses negative logits for ranking
            scores = -logits
        else:
            # fallback: loop but still amortize collation
            scores = []
            for i, u in enumerate(batch_u):
                sc = -model.predict(
                    *[np.array(l) for l in [[u], [seq_b[i]], [seq2_b[i]], items_b[i], [mask_b[i]]]]
                ).detach().cpu().numpy()[0]
                scores.append(sc)
            scores = np.stack(scores, axis=0)

        # compute ranks: positive is at index 0 in each row
        # argsort twice to get ranks, identical to your original approach
        order = scores.argsort(axis=1).argsort(axis=1)  # [B, K]
        ranks = order[:, 0]  # pos at col 0

        # accumulate metrics
        for r in ranks:
            valid_user += 1
            if r < 10:
                NDCG += 1.0 / np.log2(r + 2.0)
                HT += 1.0

        if (not quiet) and (valid_user % 5000 == 0):
            print('.', end='')
            sys.stdout.flush()

    return NDCG / max(1, valid_user), HT / max(1, valid_user)


def evaluate(model, dataset, args):
    # test set eval
    return _evaluate_batched(model, dataset, args, for_valid=False, batch_users=512, quiet=True)


def evaluate_valid(model, dataset, args):
    # valid set eval
    return _evaluate_batched(model, dataset, args, for_valid=True, batch_users=512, quiet=True)


def get_slice(inputs):
    items, n_node, A, alias_inputs = [], [], [], []
    for u_input in inputs:
        n_node.append(len(np.unique(u_input)))
    # max number of nodes per session-graph; original fixed to 100
    max_n_node = 100

    for u_input in inputs:
        node = np.unique(u_input)
        items.append(node.tolist() + (max_n_node - len(node)) * [0])
        u_A = np.zeros((max_n_node, max_n_node))
        for i in np.arange(len(u_input) - 1):
            if u_input[i + 1] == 0:
                continue
            u = np.where(node == u_input[i])[0][0]
            v = np.where(node == u_input[i + 1])[0][0]
            u_A[u][v] = 1
        u_sum_in = np.sum(u_A, 0); u_sum_in[np.where(u_sum_in == 0)] = 1
        u_A_in = np.divide(u_A, u_sum_in)
        u_sum_out = np.sum(u_A, 1); u_sum_out[np.where(u_sum_out == 0)] = 1
        u_A_out = np.divide(u_A.transpose(), u_sum_out)
        u_A = np.concatenate([u_A_in, u_A_out]).transpose()
        A.append(u_A)
        alias_inputs.append([np.where(node == i)[0][0] for i in u_input])
    return alias_inputs, A, items
