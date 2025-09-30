import numpy as np
import torch
from torch.nn import Module, Parameter
import torch.nn.functional as F
from utils import get_slice
import scipy.sparse as sp

class PointWiseFeedForward(torch.nn.Module):
    def __init__(self, hidden_units, dropout_rate):
        super(PointWiseFeedForward, self).__init__()
        self.conv1 = torch.nn.Conv1d(hidden_units, hidden_units, kernel_size=1)
        self.dropout1 = torch.nn.Dropout(p=dropout_rate)
        self.relu = torch.nn.ReLU()
        self.conv2 = torch.nn.Conv1d(hidden_units, hidden_units, kernel_size=1)
        self.dropout2 = torch.nn.Dropout(p=dropout_rate)

    def forward(self, inputs):
        outputs = self.dropout2(self.conv2(self.relu(self.dropout1(self.conv1(inputs.transpose(-1, -2))))))
        return (outputs.transpose(-1, -2) + inputs)

class SASRec(torch.nn.Module):
    def __init__(self, user_train1, user_train2, user_num, item_num, 
                 item_text_embeddings, item_id_to_idx, args):
        super(SASRec, self).__init__()
        self.user_num, self.item_num, self.dev = user_num, item_num, args.device
        self.train_data = self.merge_data(user_train1, user_train2)

        # Standard Embeddings
        self.item_emb = torch.nn.Embedding(self.item_num + 1, args.hidden_units, padding_idx=0)
        self.user_emb = torch.nn.Embedding(self.user_num + 1, args.hidden_units, padding_idx=0)
        self.pos_emb = torch.nn.Embedding(args.maxlen, args.hidden_units)
        self.emb_dropout = torch.nn.Dropout(p=args.dropout_rate)

        # Text Embedding Setup
        if item_text_embeddings is not None:
            self.item_text_emb = torch.nn.Embedding.from_pretrained(item_text_embeddings, freeze=True)
            self.text_projection = torch.nn.Linear(item_text_embeddings.shape[1], args.hidden_units)
            self.text_contrastive_projection = torch.nn.Linear(item_text_embeddings.shape[1], args.hidden_units)
            self.item_id_to_idx = item_id_to_idx
        else:
            self.item_text_emb = None

        # Model Architecture
        self.gnn = GNN(args.hidden_units, step=1)
        self.gnn2 = GNN(args.hidden_units, step=1)
        self.GCN = LightGCN(args, user_num, item_num, self.user_emb, self.item_emb, self.train_data)
        
        self.attention_layernorms = torch.nn.ModuleList([torch.nn.LayerNorm(args.hidden_units, eps=1e-8) for _ in range(args.num_blocks)])
        self.attention_layers = torch.nn.ModuleList([torch.nn.MultiheadAttention(args.hidden_units, args.num_heads, args.dropout_rate) for _ in range(args.num_blocks)])
        self.cross_attention_layernorms = torch.nn.ModuleList([torch.nn.LayerNorm(args.hidden_units, eps=1e-8) for _ in range(args.num_blocks)])
        self.cross_attention_layers = torch.nn.ModuleList([torch.nn.MultiheadAttention(args.hidden_units, args.num_heads, args.dropout_rate) for _ in range(args.num_blocks)])
        self.forward_layernorms = torch.nn.ModuleList([torch.nn.LayerNorm(args.hidden_units, eps=1e-8) for _ in range(args.num_blocks)])
        self.forward_layers = torch.nn.ModuleList([PointWiseFeedForward(args.hidden_units, args.dropout_rate) for _ in range(args.num_blocks)])
        self.cross_forward_layernorms = torch.nn.ModuleList([torch.nn.LayerNorm(args.hidden_units, eps=1e-8) for _ in range(args.num_blocks)])
        self.cross_forward_layers = torch.nn.ModuleList([PointWiseFeedForward(args.hidden_units, args.dropout_rate) for _ in range(args.num_blocks)])

        self.last_layernorm = torch.nn.LayerNorm(args.hidden_units, eps=1e-8)
        self.last_cross_layernorm = torch.nn.LayerNorm(2 * args.hidden_units, eps=1e-8)
        self.dropout2 = torch.nn.Dropout(p=args.dropout_rate)
        self.dropout3 = torch.nn.Dropout(p=args.dropout_rate)
        self.gating3 = torch.nn.Linear(2 * args.hidden_units, args.hidden_units)
        self.gating2 = torch.nn.Linear(1 * args.hidden_units, args.hidden_units)
        self.w1gating = torch.nn.Linear(1 * args.hidden_units, args.hidden_units)
        self.w2gating = torch.nn.Linear(1 * args.hidden_units, args.hidden_units)
        self.w4gating = torch.nn.Linear(1 * args.hidden_units, args.hidden_units)
        self.dropout4 = torch.nn.Dropout(p=args.dropout_rate)
        self.dropout5 = torch.nn.Dropout(p=args.dropout_rate)
        self.dropout6 = torch.nn.Dropout(p=args.dropout_rate)
        self.dropout7 = torch.nn.Dropout(p=args.dropout_rate)
        self.seq_layernorm1 = torch.nn.LayerNorm(args.hidden_units, eps=1e-8)
        self.seq_layernorm2 = torch.nn.LayerNorm(args.hidden_units, eps=1e-8)
        self.gate1 = torch.nn.Linear(2 * args.hidden_units, 2)
        self.gate2 = torch.nn.Linear(2 * args.hidden_units, 2)

    def merge_data(self, user_train1, user_train2):
        for k, v in user_train2.items(): user_train1.setdefault(k, []).extend(v)
        return user_train1

    def get_text_embedding_lookup(self, item_indices, for_contrastive=False):
        if self.item_text_emb is None:
            return torch.zeros(*item_indices.shape, self.item_emb.embedding_dim, device=self.dev)
        
        lookup_indices = torch.tensor([self.item_id_to_idx.get(i.item(), 0) for i in item_indices.flatten()], dtype=torch.long, device=self.dev)
        text_embs = self.item_text_emb(lookup_indices)
        projector = self.text_contrastive_projection if for_contrastive else self.text_projection
        return projector(text_embs).view(*item_indices.shape, -1)

    def log2feats(self, user_ids, log_seqs1, log_seqs2, mask, isTrain=True):
        # Domain 1
        _, gcn_hidden1 = self.GCN.get_embedding(torch.as_tensor(user_ids, dtype=torch.long, device=self.dev), torch.as_tensor(log_seqs1, dtype=torch.long, device=self.dev), isTrain)
        alias_inputs, A, items = get_slice(log_seqs1)
        gnn_nodes = self.gnn(torch.from_numpy(A).float().to(self.dev), self.item_emb(torch.as_tensor(items, dtype=torch.long, device=self.dev)))
        alias = torch.as_tensor(alias_inputs, dtype=torch.long, device=self.dev)
        gnn_hidden = torch.gather(gnn_nodes, 1, alias.unsqueeze(-1).expand(-1, -1, gnn_nodes.size(-1)))
        
        seqs1_cat = torch.cat((gcn_hidden1, gnn_hidden), dim=2)
        z = F.softmax(self.gate1(seqs1_cat), dim=2)
        seqs1 = self.seq_layernorm1(z[:, :, 0:1] * gcn_hidden1 + z[:, :, 1:2] * gnn_hidden)
        
        seqs1 += self.get_text_embedding_lookup(torch.as_tensor(log_seqs1, device=self.dev))
        seqs1 *= self.item_emb.embedding_dim ** 0.5
        seqs1 += self.pos_emb(torch.arange(log_seqs1.shape[1], device=self.dev).unsqueeze(0).repeat(log_seqs1.shape[0], 1))
        seqs1 = self.emb_dropout(seqs1)
        timeline_mask1 = torch.as_tensor(log_seqs1 == 0, dtype=torch.bool, device=self.dev)
        seqs1 *= ~timeline_mask1.unsqueeze(-1)

        # Domain 2
        _, gcn_hidden2 = self.GCN.get_embedding(torch.as_tensor(user_ids, dtype=torch.long, device=self.dev), torch.as_tensor(log_seqs2, dtype=torch.long, device=self.dev), isTrain)
        alias_inputs2, A2, items2 = get_slice(log_seqs2)
        gnn_nodes2 = self.gnn2(torch.from_numpy(A2).float().to(self.dev), self.item_emb(torch.as_tensor(items2, dtype=torch.long, device=self.dev)))
        alias2 = torch.as_tensor(alias_inputs2, dtype=torch.long, device=self.dev)
        gnn_hidden2 = torch.gather(gnn_nodes2, 1, alias2.unsqueeze(-1).expand(-1, -1, gnn_nodes2.size(-1)))
        
        seqs2_cat = torch.cat((gcn_hidden2, gnn_hidden2), dim=2)
        zz = F.softmax(self.gate2(seqs2_cat), dim=2)
        seqs2 = self.seq_layernorm2(zz[:, :, 0:1] * gcn_hidden2 + zz[:, :, 1:2] * gnn_hidden2)
        
        seqs2 += self.get_text_embedding_lookup(torch.as_tensor(log_seqs2, device=self.dev))
        seqs2 *= self.item_emb.embedding_dim ** 0.5
        seqs2 += self.pos_emb(torch.arange(log_seqs2.shape[1], device=self.dev).unsqueeze(0).repeat(log_seqs2.shape[0], 1))
        seqs2 = self.emb_dropout(seqs2)
        timeline_mask2 = torch.as_tensor(log_seqs2 == 0, dtype=torch.bool, device=self.dev)
        seqs2 *= ~timeline_mask2.unsqueeze(-1)

        # Attention blocks
        att_seq1 = seqs1
        mask_t = torch.as_tensor(mask, dtype=torch.long, device=self.dev)
        src_idx = (mask_t.clamp(min=1) - 1)
        att_seq2 = torch.gather(seqs2, 1, src_idx.unsqueeze(-1).expand(-1, -1, seqs2.size(-1)))
        
        for i in range(args.num_blocks):
            att_seq1_T = att_seq1.transpose(0,1)
            Q1 = self.attention_layernorms[i](att_seq1_T)
            mha_out1, _ = self.attention_layers[i](Q1, att_seq1_T, att_seq1_T, attn_mask=~torch.tril(torch.ones((seqs1.shape[1], seqs1.shape[1]), dtype=torch.bool, device=self.dev)))
            att_seq1 = self.forward_layers[i](self.forward_layernorms[i]((Q1 + mha_out1).transpose(0,1))) * ~timeline_mask1.unsqueeze(-1)

            att_seq2_T = att_seq2.transpose(0,1)
            Q2 = self.cross_attention_layernorms[i](att_seq2_T)
            attention_mask2 = ~(torch.arange(seqs2.shape[1], device=self.dev)[None, None, :] < mask_t.unsqueeze(-1))
            attention_mask2[:, :, 0] = False
            mha_out2, _ = self.cross_attention_layers[i](Q2, seqs2.transpose(0,1), seqs2.transpose(0,1), attn_mask=attention_mask2)
            att_seq2 = self.cross_forward_layers[i](self.cross_forward_layernorms[i]((Q2 + mha_out2).transpose(0,1))) * ~timeline_mask2.unsqueeze(-1)

        log_feats = self.last_layernorm(self.dropout3(self.gating3(self.last_cross_layernorm(torch.cat((att_seq1, att_seq2), dim=2)))))
        
        return log_feats, gcn_hidden1, gnn_hidden, gcn_hidden2, gnn_hidden2, att_seq1, att_seq2

    def forward(self, user_ids, log_seqs, log_seqs2, pos_seqs, neg_seqs, mask):
        log_feats, gcn1, gnn1, gcn2, gnn2, att1, att2 = self.log2feats(user_ids, log_seqs, log_seqs2, mask)
        
        pos_embs = self.item_emb(torch.LongTensor(pos_seqs).to(self.dev))
        neg_embs = self.item_emb(torch.LongTensor(neg_seqs).to(self.dev))
        pos_logits = (log_feats * pos_embs).sum(dim=-1)
        neg_logits = (log_feats * neg_embs).sum(dim=-1)

        con_loss  = SSL(att1, self.dropout2(self.gating2(att2)))
        con_loss2 = SSL(self.dropout4(self.w1gating(gcn1)), self.dropout5(self.w2gating(gnn1)))
        con_loss3 = SSL(self.dropout6(self.w1gating(gcn2)), self.dropout7(self.w4gating(gnn2)))
        
        text_embs1 = self.get_text_embedding_lookup(torch.as_tensor(log_seqs, device=self.dev), for_contrastive=True)
        text_embs2 = self.get_text_embedding_lookup(torch.as_tensor(log_seqs2, device=self.dev), for_contrastive=True)
        semantic_con_loss = SSL_semantic(gnn1, text_embs1) + SSL_semantic(gnn2, text_embs2)

        return pos_logits, neg_logits, con_loss, con_loss2, con_loss3, semantic_con_loss

    def predict(self, user_ids, log_seqs, log_seqs2, item_indices, mask):
        log_feats, *_ = self.log2feats(user_ids, log_seqs, log_seqs2, mask, isTrain=False)
        final_feat = log_feats[:, -1, :]
        item_embs = self.item_emb(torch.LongTensor(item_indices).to(self.dev))
        item_embs += self.get_text_embedding_lookup(torch.as_tensor(item_indices, device=self.dev)).squeeze()
        return item_embs.matmul(final_feat.unsqueeze(-1)).squeeze(-1)

    def predict_batch(self, user_ids, log_seqs, log_seqs2, item_indices, mask):
        log_feats, *_ = self.log2feats(user_ids, log_seqs, log_seqs2, mask, isTrain=False)
        final_feat = log_feats[:, -1, :]
        item_embs = self.item_emb(torch.as_tensor(item_indices, device=self.dev))
        item_embs += self.get_text_embedding_lookup(torch.as_tensor(item_indices, device=self.dev))
        return (item_embs * final_feat.unsqueeze(1)).sum(dim=-1)

class GNN(Module):
    def __init__(self, hidden_size, step=1):
        super(GNN, self).__init__()
        self.step = step
        self.hidden_size = hidden_size
        self.input_size = hidden_size * 2
        self.gate_size = 3 * hidden_size
        self.w_ih = Parameter(torch.Tensor(self.gate_size, self.input_size))
        self.w_hh = Parameter(torch.Tensor(self.gate_size, self.hidden_size))
        self.b_ih = Parameter(torch.Tensor(self.gate_size))
        self.b_hh = Parameter(torch.Tensor(self.gate_size))
        self.b_iah = Parameter(torch.Tensor(self.hidden_size))
        self.b_oah = Parameter(torch.Tensor(self.hidden_size))
        self.linear_edge_in = torch.nn.Linear(self.hidden_size, self.hidden_size, bias=True)
        self.linear_edge_out = torch.nn.Linear(self.hidden_size, self.hidden_size, bias=True)

    def GNNCell(self, A, hidden):
        input_in = torch.matmul(A[:, :, :A.shape[1]], self.linear_edge_in(hidden)) + self.b_iah
        input_out = torch.matmul(A[:, :, A.shape[1]:], self.linear_edge_out(hidden)) + self.b_oah
        inputs = torch.cat([input_in, input_out], 2)
        gi = F.linear(inputs, self.w_ih, self.b_ih)
        gh = F.linear(hidden, self.w_hh, self.b_hh)
        i_r, i_i, i_n = gi.chunk(3, 2)
        h_r, h_i, h_n = gh.chunk(3, 2)
        resetgate = torch.sigmoid(i_r + h_r)
        inputgate = torch.sigmoid(i_i + h_i)
        newgate = torch.tanh(i_n + resetgate * h_n)
        return hidden - inputgate * (hidden - newgate)

    def forward(self, A, hidden):
        for _ in range(self.step):
            hidden = self.GNNCell(A, hidden)
        return hidden

def SSL(sess_emb_hgnn, sess_emb_lgcn):
    def row_column_shuffle(embedding):
        return embedding[torch.randperm(embedding.size(0))][:, torch.randperm(embedding.size(1))]
    pos = torch.sum(sess_emb_hgnn * sess_emb_lgcn, 2)
    neg = torch.sum(sess_emb_hgnn * row_column_shuffle(sess_emb_lgcn), 2)
    return torch.sum(-torch.log(1e-8 + torch.sigmoid(pos)) - torch.log(1e-8 + (1 - torch.sigmoid(neg))))

def SSL_semantic(behavioral_emb, semantic_emb):
    pos = torch.sum(behavioral_emb * semantic_emb, 2)
    neg = torch.sum(behavioral_emb * semantic_emb[torch.randperm(semantic_emb.size(0))], 2)
    return torch.sum(-torch.log(1e-8 + torch.sigmoid(pos)) - torch.log(1e-8 + (1 - torch.sigmoid(neg))))

class LightGCN(torch.nn.Module):
    def __init__(self, args, user_num, item_num, user_emb, item_emb, data_list):
        super(LightGCN, self).__init__()
        self.device = args.device
        self.user_count, self.item_count = user_num + 1, item_num + 1
        self.n_layers = 3
        self.user_embedding, self.item_embedding = user_emb, item_emb
        self.data_list = data_list
        self.A_adj_matrix = self._get_a_adj_matrix()
        self.user_all_embedding, self.item_all_embedding = self.forward()

    def _get_a_adj_matrix(self):
        rows, cols = [], []
        for user, items in self.data_list.items():
            rows.extend([user] * len(items))
            cols.extend(items)
        R = sp.coo_matrix((np.ones(len(rows)), (rows, cols)), shape=(self.user_count, self.item_count))
        R_t = R.transpose()
        N = self.user_count + self.item_count
        
        A = sp.coo_matrix((np.ones(R.nnz + R_t.nnz), (np.concatenate([R.row, R_t.row + self.user_count]), 
                                                      np.concatenate([R.col + self.user_count, R_t.col]))), shape=(N, N))
        
        deg = np.array(A.sum(axis=1)).flatten() + 1e-7
        D_inv_sqrt = sp.diags(np.power(deg, -0.5))
        A_adj = (D_inv_sqrt @ A @ D_inv_sqrt).tocoo()
        
        indices = torch.from_numpy(np.vstack([A_adj.row, A_adj.col])).long()
        values = torch.from_numpy(A_adj.data).float()
        return torch.sparse.FloatTensor(indices, values, A_adj.shape).to(self.device)

    def forward(self):
        all_embeddings = torch.cat([self.user_embedding.weight, self.item_embedding.weight], dim=0).to(self.device)
        embeddings_list = [all_embeddings]
        for _ in range(self.n_layers):
            all_embeddings = torch.sparse.mm(self.A_adj_matrix, all_embeddings)
            embeddings_list.append(all_embeddings)
        return torch.split(torch.mean(torch.stack(embeddings_list, dim=1), dim=1), [self.user_count, self.item_count])

    def get_embedding(self, user_ids, log_seqs, isTrain):
        user_emb, item_emb = self.user_all_embedding, self.item_all_embedding
        if isTrain:
            user_emb, item_emb = self.forward()
        return user_emb[user_ids], item_emb[log_seqs]