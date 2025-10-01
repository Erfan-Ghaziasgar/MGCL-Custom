# -*- coding: utf-8 -*-

import numpy as np
import torch
from torch.nn import Module, Parameter
import torch.nn.functional as F
import scipy.sparse as sp

from utils import get_slice

# -----------------------------------------------------------
# Layers
# -----------------------------------------------------------
class PointWiseFeedForward(torch.nn.Module):
    def __init__(self, hidden_units, dropout_rate):
        super().__init__()
        self.conv1 = torch.nn.Conv1d(hidden_units, hidden_units, kernel_size=1)
        self.dropout1 = torch.nn.Dropout(p=dropout_rate)
        self.relu = torch.nn.ReLU()
        self.conv2 = torch.nn.Conv1d(hidden_units, hidden_units, kernel_size=1)
        self.dropout2 = torch.nn.Dropout(p=dropout_rate)

    def forward(self, x):
        y = self.dropout1(self.conv1(x.transpose(-1, -2)))
        y = self.relu(y)
        y = self.dropout2(self.conv2(y))
        return (y.transpose(-1, -2) + x)


# -----------------------------------------------------------
# Main Model
# -----------------------------------------------------------
class SASRec(torch.nn.Module):
    def __init__(self, user_train1, user_train2, user_num, item_num,
                 item_text_embeddings, item_id_to_idx, reindexed_to_original, args):
        super().__init__()
        self.user_num, self.item_num, self.dev = user_num, item_num, args.device
        self.args = args
        self.train_data = self._merge_data(user_train1, user_train2)

        # Embeddings
        self.item_emb = torch.nn.Embedding(self.item_num + 1, args.hidden_units, padding_idx=0)
        self.user_emb = torch.nn.Embedding(self.user_num + 1, args.hidden_units, padding_idx=0)
        self.pos_emb  = torch.nn.Embedding(args.maxlen, args.hidden_units)
        self.emb_dropout = torch.nn.Dropout(p=args.dropout_rate)

        # Text features
        self.use_text_features = item_text_embeddings is not None and item_id_to_idx is not None
        if self.use_text_features:
            print(f"Initializing text embeddings: shape={tuple(item_text_embeddings.shape)}")
            # store as buffer (not a trainable parameter)
            self.register_buffer("item_text_table", item_text_embeddings)  # [N_text, text_dim]
            text_dim = int(item_text_embeddings.shape[1])

            self.text_projection = torch.nn.Sequential(
                torch.nn.Linear(text_dim, args.hidden_units),
                torch.nn.LayerNorm(args.hidden_units),
                torch.nn.Dropout(args.dropout_rate)
            )
            self.text_contrastive_projection = torch.nn.Sequential(
                torch.nn.Linear(text_dim, args.hidden_units),
                torch.nn.ReLU(),
                torch.nn.Linear(args.hidden_units, args.hidden_units),
                torch.nn.LayerNorm(args.hidden_units)
            )
            self.fusion_gate = torch.nn.Sequential(
                torch.nn.Linear(args.hidden_units * 2, args.hidden_units),
                torch.nn.Sigmoid()
            )
            self.contextual_fusion_gate = torch.nn.Sequential(
                torch.nn.Linear(args.hidden_units * 3, args.hidden_units),
                torch.nn.Sigmoid()
            )
            self.item_id_to_idx = item_id_to_idx               # ORIGINAL item_id -> row in text table
            self.reindexed_to_original = reindexed_to_original # REINDEX -> ORIGINAL item_id
            print("Text embeddings initialized successfully. Using fusion mechanism.")
        else:
            self.register_buffer("item_text_table", None)
            self.item_id_to_idx = None
            self.reindexed_to_original = None
            print("Warning: No text embeddings provided. Running without semantic features.")

        # Architectures
        self.gnn  = SessionGNN(args.hidden_units, step=1)
        self.gnn2 = SessionGNN(args.hidden_units, step=1)
        self.GCN = LightGCN(args, user_num, item_num, self.user_emb, self.item_emb, self.train_data)

        # Transformer blocks (batch_first=False)
        self.attention_layernorms = torch.nn.ModuleList([
            torch.nn.LayerNorm(args.hidden_units, eps=1e-8) for _ in range(args.num_blocks)
        ])
        self.attention_layers = torch.nn.ModuleList([
            torch.nn.MultiheadAttention(args.hidden_units, args.num_heads, args.dropout_rate, batch_first=False)
            for _ in range(args.num_blocks)
        ])
        self.cross_attention_layernorms = torch.nn.ModuleList([
            torch.nn.LayerNorm(args.hidden_units, eps=1e-8) for _ in range(args.num_blocks)
        ])
        self.cross_attention_layers = torch.nn.ModuleList([
            torch.nn.MultiheadAttention(args.hidden_units, args.num_heads, args.dropout_rate, batch_first=False)
            for _ in range(args.num_blocks)
        ])
        self.forward_layernorms = torch.nn.ModuleList([
            torch.nn.LayerNorm(args.hidden_units, eps=1e-8) for _ in range(args.num_blocks)
        ])
        self.forward_layers = torch.nn.ModuleList([
            PointWiseFeedForward(args.hidden_units, args.dropout_rate) for _ in range(args.num_blocks)
        ])
        self.cross_forward_layernorms = torch.nn.ModuleList([
            torch.nn.LayerNorm(args.hidden_units, eps=1e-8) for _ in range(args.num_blocks)
        ])
        self.cross_forward_layers = torch.nn.ModuleList([
            PointWiseFeedForward(args.hidden_units, args.dropout_rate) for _ in range(args.num_blocks)
        ])

        self.last_layernorm = torch.nn.LayerNorm(args.hidden_units, eps=1e-8)
        self.last_cross_layernorm = torch.nn.LayerNorm(2 * args.hidden_units, eps=1e-8)
        self.dropout2 = torch.nn.Dropout(p=args.dropout_rate)
        self.dropout3 = torch.nn.Dropout(p=args.dropout_rate)

        # gating for SSL
        self.gating3 = torch.nn.Linear(2 * args.hidden_units, args.hidden_units)
        self.gating2 = torch.nn.Linear(args.hidden_units, args.hidden_units)
        self.w1gating = torch.nn.Linear(args.hidden_units, args.hidden_units)
        self.w2gating = torch.nn.Linear(args.hidden_units, args.hidden_units)
        self.w4gating = torch.nn.Linear(args.hidden_units, args.hidden_units)

        self.dropout4 = torch.nn.Dropout(p=args.dropout_rate)
        self.dropout5 = torch.nn.Dropout(p=args.dropout_rate)
        self.dropout6 = torch.nn.Dropout(p=args.dropout_rate)
        self.dropout7 = torch.nn.Dropout(p=args.dropout_rate)

        self.seq_layernorm1 = torch.nn.LayerNorm(args.hidden_units, eps=1e-8)
        self.seq_layernorm2 = torch.nn.LayerNorm(args.hidden_units, eps=1e-8)
        self.gate1 = torch.nn.Linear(2 * args.hidden_units, 2)
        self.gate2 = torch.nn.Linear(2 * args.hidden_units, 2)

    # -----------------
    # Helpers
    # -----------------
    @staticmethod
    def _merge_data(user_train1, user_train2):
        merged = {k: v[:] for k, v in user_train1.items()}
        for k, v in user_train2.items():
            merged.setdefault(k, []).extend(v)
        return merged

    def _text_lookup_indices(self, item_indices_flat: torch.Tensor) -> torch.Tensor:
        """Map reindexed item ids -> original ids -> text table row; default 0 if missing."""
        if not self.use_text_features:
            return torch.zeros_like(item_indices_flat, dtype=torch.long, device=self.dev)
        to_orig = [self.reindexed_to_original.get(int(x), None) for x in item_indices_flat.tolist()]
        to_row = [self.item_id_to_idx.get(orig, 0) if orig is not None else 0 for orig in to_orig]
        return torch.tensor(to_row, dtype=torch.long, device=self.dev)

    def _text_embed_project(self, row_idx: torch.Tensor, for_contrastive: bool):
        # row_idx: (N,)
        text_raw = torch.nn.functional.embedding(row_idx, self.item_text_table)  # (N, text_dim)
        proj = self.text_contrastive_projection if for_contrastive else self.text_projection
        return proj(text_raw)

    def get_text_embedding_lookup(self, item_indices, for_contrastive=False):
        """Return projected text embeddings aligned to item_indices shape."""
        if not self.use_text_features:
            h = self.item_emb.embedding_dim
            return torch.zeros(*item_indices.shape, h, device=self.dev)
        flat = item_indices.flatten()
        row_idx = self._text_lookup_indices(flat)
        proj_text = self._text_embed_project(row_idx, for_contrastive)  # (N, H)
        return proj_text.view(*item_indices.shape, -1)

    def fuse_id_text_embeddings(self, id_emb, text_emb, context=None):
        if not self.use_text_features:
            return id_emb
        if context is not None:
            if context.dim() < id_emb.dim():
                for _ in range(id_emb.dim() - context.dim()):
                    context = context.unsqueeze(1)
            context = context.expand(id_emb.shape)
            gate_input = torch.cat([id_emb, text_emb, context], dim=-1)
            gate = self.contextual_fusion_gate(gate_input)
        else:
            gate = self.fusion_gate(torch.cat([id_emb, text_emb], dim=-1))
        return gate * id_emb + (1.0 - gate) * text_emb

    # -----------------
    # Forward pieces
    # -----------------
    def log2feats(self, user_ids, log_seqs1, log_seqs2, mask, isTrain=True):
        user_ids_t = torch.as_tensor(user_ids, dtype=torch.long, device=self.dev)
        log_seqs1_t = torch.as_tensor(log_seqs1, dtype=torch.long, device=self.dev)
        log_seqs2_t = torch.as_tensor(log_seqs2, dtype=torch.long, device=self.dev)

        # LightGCN embeddings (cache if isTrain=False)
        refresh_cache = bool(isTrain)
        _, gcn_hidden1 = self.GCN.get_embedding(user_ids_t, log_seqs1_t, refresh_cache)
        second_is_train = False if refresh_cache else bool(isTrain)
        _, gcn_hidden2 = self.GCN.get_embedding(user_ids_t, log_seqs2_t, second_is_train)

        # GNN over session graphs (domain 1)
        seq_width1 = log_seqs1.shape[1] if hasattr(log_seqs1, "shape") and len(log_seqs1.shape) > 1 else (
            len(log_seqs1[0]) if len(log_seqs1) > 0 else 0
        )
        max_nodes1 = getattr(self.args, "maxlen", None) or seq_width1
        alias_inputs, A, items = get_slice(log_seqs1, max_n_node=max_nodes1)
        A_t = torch.from_numpy(A).to(self.dev)              # (B, 2N, N)
        items_t = torch.from_numpy(items).to(self.dev)      # (B, N)
        gnn_nodes = self.gnn(A_t, self.item_emb(items_t))   # (B, N, H)
        alias = torch.as_tensor(alias_inputs, dtype=torch.long, device=self.dev)  # (B, L)
        gnn_hidden = torch.gather(gnn_nodes, 1, alias.unsqueeze(-1).expand(-1, -1, gnn_nodes.size(-1)))

        seqs1_cat = torch.cat((gcn_hidden1, gnn_hidden), dim=2)
        z = torch.softmax(self.gate1(seqs1_cat), dim=2)
        seqs1 = self.seq_layernorm1(z[:, :, 0:1] * gcn_hidden1 + z[:, :, 1:2] * gnn_hidden)

        if self.use_text_features:
            text_emb1 = self.get_text_embedding_lookup(log_seqs1_t)
            user_ctx = self.user_emb(user_ids_t).unsqueeze(1).expand_as(seqs1)
            seqs1 = self.fuse_id_text_embeddings(seqs1, text_emb1, context=user_ctx)

        # position + dropout + mask
        H = self.item_emb.embedding_dim
        seqs1 = seqs1 * (H ** 0.5)
        positions = torch.arange(log_seqs1.shape[1], device=self.dev).unsqueeze(0).repeat(log_seqs1.shape[0], 1)
        seqs1 = self.emb_dropout(seqs1 + self.pos_emb(positions))
        timeline_mask1 = (log_seqs1_t == 0)
        seqs1 = seqs1 * (~timeline_mask1).unsqueeze(-1)

        # GNN (domain 2)
        seq_width2 = log_seqs2.shape[1] if hasattr(log_seqs2, "shape") and len(log_seqs2.shape) > 1 else (
            len(log_seqs2[0]) if len(log_seqs2) > 0 else 0
        )
        max_nodes2 = getattr(self.args, "maxlen", None) or seq_width2
        alias_inputs2, A2, items2 = get_slice(log_seqs2, max_n_node=max_nodes2)
        A2_t = torch.from_numpy(A2).to(self.dev)
        items2_t = torch.from_numpy(items2).to(self.dev)
        gnn_nodes2 = self.gnn2(A2_t, self.item_emb(items2_t))
        alias2 = torch.as_tensor(alias_inputs2, dtype=torch.long, device=self.dev)
        gnn_hidden2 = torch.gather(gnn_nodes2, 1, alias2.unsqueeze(-1).expand(-1, -1, gnn_nodes2.size(-1)))

        seqs2_cat = torch.cat((gcn_hidden2, gnn_hidden2), dim=2)
        zz = torch.softmax(self.gate2(seqs2_cat), dim=2)
        seqs2 = self.seq_layernorm2(zz[:, :, 0:1] * gcn_hidden2 + zz[:, :, 1:2] * gnn_hidden2)

        if self.use_text_features:
            text_emb2 = self.get_text_embedding_lookup(log_seqs2_t)
            user_ctx2 = self.user_emb(user_ids_t).unsqueeze(1).expand_as(seqs2)
            seqs2 = self.fuse_id_text_embeddings(seqs2, text_emb2, context=user_ctx2)

        seqs2 = seqs2 * (H ** 0.5)
        seqs2 = self.emb_dropout(seqs2 + self.pos_emb(positions))
        timeline_mask2 = (log_seqs2_t == 0)
        seqs2 = seqs2 * (~timeline_mask2).unsqueeze(-1)

        # Cross-domain: build att_seq2 based on temporal alignment mask
        att_seq1 = seqs1
        mask_t = torch.as_tensor(mask, dtype=torch.long, device=self.dev)  # (B, L1)
        src_idx = (mask_t.clamp(min=1) - 1)
        att_seq2 = torch.gather(seqs2, 1, src_idx.unsqueeze(-1).expand(-1, -1, seqs2.size(-1)))  # (B, L1, H)

        # Multihead blocks (self on domain1, cross on domain2) with robust masks
        L1, L2 = seqs1.shape[1], seqs2.shape[1]
        nh = self.attention_layers[0].num_heads

        for i in range(len(self.attention_layers)):
            # Self-attn on domain1
            Q1 = self.attention_layernorms[i](att_seq1).transpose(0, 1)  # (L1,B,H)
            K1 = Q1; V1 = Q1
            causal = ~torch.tril(torch.ones((L1, L1), dtype=torch.bool, device=self.dev))  # (L1,L1)
            out1, _ = self.attention_layers[i](Q1, K1, V1, attn_mask=causal)
            att_seq1 = self.forward_layers[i](self.forward_layernorms[i]((Q1 + out1).transpose(0, 1)))
            att_seq1 = att_seq1 * (~timeline_mask1).unsqueeze(-1)

            # Cross-attn: query is aligned sequence (L1,B,H), keys are full seq2 (L2,B,H)
            Q2 = self.cross_attention_layernorms[i](att_seq2).transpose(0, 1)  # (L1,B,H)
            K2 = seqs2.transpose(0, 1)  # (L2,B,H)
            V2 = K2

            # Build (B, L1, L2) mask that includes both "future" and padding
            base_future = ~(torch.arange(L2, device=self.dev)[None, None, :] < mask_t.unsqueeze(-1))  # (B,L1,L2)
            pad_mask = timeline_mask2.unsqueeze(1).expand(-1, L1, -1)  # (B,L1,L2)
            combined = base_future | pad_mask
            # ensure at least one unmasked key (column 0) per query
            combined[:, :, 0] = False

            attn_mask2 = combined.repeat_interleave(nh, dim=0)  # (B*nh, L1, L2)

            # pass only attn_mask (no key_padding_mask) to avoid all-masked rows → NaN
            out2, _ = self.cross_attention_layers[i](Q2, K2, V2, attn_mask=attn_mask2)
            att_seq2 = self.cross_forward_layers[i](self.cross_forward_layernorms[i]((Q2 + out2).transpose(0, 1)))
            att_seq2 = att_seq2 * (~timeline_mask2).unsqueeze(-1)

        # Final fusion
        log_feats = self.last_layernorm(
            self.dropout3(self.gating3(self.last_cross_layernorm(torch.cat((att_seq1, att_seq2), dim=2))))
        )
        return log_feats, gcn_hidden1, gnn_hidden, gcn_hidden2, gnn_hidden2, att_seq1, att_seq2

    def forward(self, user_ids, log_seqs, log_seqs2, pos_seqs, neg_seqs, mask):
        log_feats, gcn1, gnn1, gcn2, gnn2, att1, att2 = self.log2feats(user_ids, log_seqs, log_seqs2, mask)

        pos_embs = self.item_emb(torch.as_tensor(pos_seqs, dtype=torch.long, device=self.dev))
        neg_embs = self.item_emb(torch.as_tensor(neg_seqs, dtype=torch.long, device=self.dev))
        if self.use_text_features:
            pos_text = self.get_text_embedding_lookup(torch.as_tensor(pos_seqs, device=self.dev))
            neg_text = self.get_text_embedding_lookup(torch.as_tensor(neg_seqs, device=self.dev))
            pos_embs = self.fuse_id_text_embeddings(pos_embs, pos_text, context=log_feats)
            neg_embs = self.fuse_id_text_embeddings(neg_embs, neg_text, context=log_feats)

        pos_logits = (log_feats * pos_embs).sum(dim=-1)
        neg_logits = (log_feats * neg_embs).sum(dim=-1)

        # SSL losses
        con_loss  = SSL(att1, self.dropout2(self.gating2(att2)))
        con_loss2 = SSL(self.dropout4(self.w1gating(gcn1)), self.dropout5(self.w2gating(gnn1)))
        con_loss3 = SSL(self.dropout6(self.w1gating(gcn2)), self.dropout7(self.w4gating(gnn2)))

        semantic_con_loss = torch.tensor(0.0, device=self.dev)
        if self.use_text_features:
            t1 = self.get_text_embedding_lookup(torch.as_tensor(log_seqs, device=self.dev), for_contrastive=True)
            t2 = self.get_text_embedding_lookup(torch.as_tensor(log_seqs2, device=self.dev), for_contrastive=True)
            semantic_con_loss = SSL_semantic(gnn1, t1, temp=self.args.temp) + \
                                SSL_semantic(gnn2, t2, temp=self.args.temp)

        return pos_logits, neg_logits, con_loss, con_loss2, con_loss3, semantic_con_loss

    # Inference
    def predict(self, user_ids, log_seqs, log_seqs2, item_indices, mask):
        log_feats, *_ = self.log2feats(user_ids, log_seqs, log_seqs2, mask, isTrain=False)
        final_feat = log_feats[:, -1, :]
        item_embs = self.item_emb(torch.as_tensor(item_indices, dtype=torch.long, device=self.dev))
        if self.use_text_features:
            text_lookup = self.get_text_embedding_lookup(torch.as_tensor(item_indices, device=self.dev))
            text_embs = text_lookup.squeeze() if text_lookup.dim() != item_embs.dim() else text_lookup
            item_ctx = final_feat
            if item_embs.dim() == 3:
                item_ctx = final_feat.unsqueeze(1).expand_as(item_embs)
            elif item_embs.dim() == 2:
                if final_feat.dim() == 2 and final_feat.size(0) == item_embs.size(0):
                    item_ctx = final_feat
                else:
                    pooled = final_feat.mean(dim=0, keepdim=True)
                    item_ctx = pooled.expand(item_embs.size(0), -1)
            item_embs = self.fuse_id_text_embeddings(item_embs, text_embs, context=item_ctx)
        return item_embs.matmul(final_feat.unsqueeze(-1)).squeeze(-1)

    def predict_batch(self, user_ids, log_seqs, log_seqs2, item_indices, mask):
        log_feats, *_ = self.log2feats(user_ids, log_seqs, log_seqs2, mask, isTrain=False)
        final_feat = log_feats[:, -1, :]
        item_embs = self.item_emb(torch.as_tensor(item_indices, dtype=torch.long, device=self.dev))
        if self.use_text_features:
            text_embs = self.get_text_embedding_lookup(torch.as_tensor(item_indices, device=self.dev))
            item_ctx = final_feat.unsqueeze(1)
            if item_ctx.shape != item_embs.shape:
                item_ctx = item_ctx.expand_as(item_embs)
            item_embs = self.fuse_id_text_embeddings(item_embs, text_embs, context=item_ctx)
        return (item_embs * final_feat.unsqueeze(1)).sum(dim=-1)


# -----------------------------------------------------------
# Session GNN / SSL / LightGCN
# -----------------------------------------------------------
class SessionGNN(Module):
    """Session-level GNN with explicit forward."""
    def __init__(self, hidden_size, step=1):
        super().__init__()
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

        self.linear_edge_in  = torch.nn.Linear(self.hidden_size, self.hidden_size, bias=True)
        self.linear_edge_out = torch.nn.Linear(self.hidden_size, self.hidden_size, bias=True)

        torch.nn.init.xavier_uniform_(self.w_ih)
        torch.nn.init.xavier_uniform_(self.w_hh)
        torch.nn.init.zeros_(self.b_ih)
        torch.nn.init.zeros_(self.b_hh)
        torch.nn.init.zeros_(self.b_iah)
        torch.nn.init.zeros_(self.b_oah)

    def _cell(self, A, hidden):
        """
        A:      (B, 2N, N)   [0:N]=in, [N:2N]=out
        hidden: (B, N, H)
        """
        B, twoN, N = A.shape
        Nhalf = twoN // 2

        A_in  = A[:, :Nhalf, :]   # (B, N, N)
        A_out = A[:, Nhalf:, :]   # (B, N, N)

        h_in  = self.linear_edge_in(hidden)   # (B, N, H)
        h_out = self.linear_edge_out(hidden)  # (B, N, H)

        # (B,N,N) @ (B,N,H) -> (B,N,H)
        input_in  = torch.matmul(A_in,  h_in)  + self.b_iah
        input_out = torch.matmul(A_out, h_out) + self.b_oah

        inputs = torch.cat([input_in, input_out], dim=2)  # (B, N, 2H)

        gi = F.linear(inputs, self.w_ih, self.b_ih)
        gh = F.linear(hidden, self.w_hh, self.b_hh)

        i_r, i_i, i_n = gi.chunk(3, dim=2)
        h_r, h_i, h_n = gh.chunk(3, dim=2)

        resetgate = torch.sigmoid(i_r + h_r)
        inputgate = torch.sigmoid(i_i + h_i)
        newgate   = torch.tanh(i_n + resetgate * h_n)

        return hidden - inputgate * (hidden - newgate)

    def forward(self, A, hidden):
        """
        A:      (B, 2N, N)
        hidden: (B, N, H)
        returns: (B, N, H)
        """
        h = hidden
        for _ in range(self.step):
            h = self._cell(A, h)
        return h


def SSL(sess_emb_hgnn, sess_emb_lgcn):
    """Row/column shuffling negative sampling."""
    def row_column_shuffle(emb):
        return emb[torch.randperm(emb.size(0))][:, torch.randperm(emb.size(1))]
    pos = torch.sum(sess_emb_hgnn * sess_emb_lgcn, dim=2)
    neg = torch.sum(sess_emb_hgnn * row_column_shuffle(sess_emb_lgcn), dim=2)
    return torch.sum(-torch.log(1e-8 + torch.sigmoid(pos)) - torch.log(1e-8 + (1 - torch.sigmoid(neg))))


def SSL_semantic(behavioral_emb, semantic_emb, temp=0.1):
    """Temperature-scaled semantic alignment."""
    behavioral_emb = F.normalize(behavioral_emb, p=2, dim=-1)
    semantic_emb   = F.normalize(semantic_emb, p=2, dim=-1)
    pos = torch.sum(behavioral_emb * semantic_emb, dim=2) / temp
    neg = torch.sum(behavioral_emb * semantic_emb[torch.randperm(semantic_emb.size(0))], dim=2) / temp
    return torch.sum(-torch.log(1e-8 + torch.sigmoid(pos)) - torch.log(1e-8 + (1 - torch.sigmoid(neg))))


class LightGCN(torch.nn.Module):
    def __init__(self, args, user_num, item_num, user_emb, item_emb, data_list):
        super().__init__()
        self.device = args.device
        self.user_count, self.item_count = user_num + 1, item_num + 1
        self.n_layers = 3
        self.user_embedding, self.item_embedding = user_emb, item_emb
        self.data_list = data_list
        self.A_adj_matrix = self._get_a_adj_matrix()

        # Gradient-carrying tensors from the most recent forward() call.
        self.user_all_embedding = None
        self.item_all_embedding = None

        # Detached cache for reuse when gradients are disabled (e.g., eval).
        self._cached_user_embedding = None
        self._cached_item_embedding = None

        # Populate the caches once so inference can proceed immediately.
        self._refresh_embedding_cache()

    def _get_a_adj_matrix(self):
        rows, cols = [], []
        for user, items in self.data_list.items():
            rows.extend([user] * len(items))
            cols.extend(items)
        R   = sp.coo_matrix((np.ones(len(rows)), (rows, cols)), shape=(self.user_count, self.item_count))
        Rt  = R.transpose()
        N   = self.user_count + self.item_count

        A = sp.coo_matrix(
            (np.ones(R.nnz + Rt.nnz),
             (np.concatenate([R.row, Rt.row + self.user_count]),
              np.concatenate([R.col + self.user_count, Rt.col]))),
            shape=(N, N)
        )
        deg = np.array(A.sum(axis=1)).flatten() + 1e-7
        D_inv_sqrt = sp.diags(np.power(deg, -0.5))
        A_adj = (D_inv_sqrt @ A @ D_inv_sqrt).tocoo()

        indices = torch.from_numpy(np.vstack([A_adj.row, A_adj.col])).long()
        values  = torch.from_numpy(A_adj.data).float()
        return torch.sparse_coo_tensor(indices, values, A_adj.shape, device=self.device)

    def forward(self):
        all_emb = torch.cat([self.user_embedding.weight, self.item_embedding.weight], dim=0).to(self.device)
        embs = [all_emb]
        for _ in range(self.n_layers):
            all_emb = torch.sparse.mm(self.A_adj_matrix, all_emb)
            embs.append(all_emb)
        out = torch.mean(torch.stack(embs, dim=1), dim=1)
        return torch.split(out, [self.user_count, self.item_count])

    def get_embedding(self, user_ids, log_seqs, isTrain):
        if isTrain:
            user_emb, item_emb = self._refresh_embedding_cache()
        else:
            # If no gradient path is needed, prefer the detached cache.
            if not torch.is_grad_enabled():
                if self._cached_user_embedding is None or self._cached_item_embedding is None:
                    self._refresh_embedding_cache()
                user_emb, item_emb = self._cached_user_embedding, self._cached_item_embedding
            else:
                if self.user_all_embedding is None or self.item_all_embedding is None:
                    user_emb, item_emb = self._refresh_embedding_cache()
                else:
                    user_emb, item_emb = self.user_all_embedding, self.item_all_embedding

        return user_emb[user_ids], item_emb[log_seqs]

    def _refresh_embedding_cache(self):
        user_emb, item_emb = self.forward()
        self.user_all_embedding, self.item_all_embedding = user_emb, item_emb
        self._cached_user_embedding = user_emb.detach()
        self._cached_item_embedding = item_emb.detach()
        return user_emb, item_emb
