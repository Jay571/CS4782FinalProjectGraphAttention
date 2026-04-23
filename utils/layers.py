import torch
import torch.nn as nn
import torch.nn.functional as F


class AttentionHead(nn.Module):
    """Dense graph attention head. Equivalent to attn_head() in the original TF implementation."""

    def __init__(self, in_features, out_sz, in_drop=0.0, coef_drop=0.0, residual=False, activation=F.elu):
        super().__init__()
        self.in_drop = in_drop
        self.coef_drop = coef_drop
        self.residual = residual
        self.activation = activation
        
        
        self.W = nn.Linear(in_features, out_sz, bias=False)
        # Attention score projections: f_1 and f_2
        self.a_1 = nn.Linear(out_sz, 1)
        self.a_2 = nn.Linear(out_sz, 1)
        
        self.bias = nn.Parameter(torch.zeros(out_sz))

        # Residual projection when dimensions differ
        if residual and in_features != out_sz:
            self.res_proj = nn.Linear(in_features, out_sz, bias=False)
        else:
            self.res_proj = None

    def forward(self, seq, bias_mat):
        # seq:      [batch, nb_nodes, in_features]
        # bias_mat: [batch, nb_nodes, nb_nodes] or broadcastable
        #           contains 0 for edges, -1e9 for non-edges (same role as TF bias_mat)

        if self.in_drop != 0.0 and self.training:
            seq = F.dropout(seq, p=self.in_drop)

        seq_fts = self.W(seq)                          # [batch, nb_nodes, out_sz]

        f_1 = self.a_1(seq_fts)                        # [batch, nb_nodes, 1]
        f_2 = self.a_2(seq_fts)                        # [batch, nb_nodes, 1]
        # Outer sum: logits[b, i, j] = f_1[b, i] + f_2[b, j]
        logits = f_1 + f_2.transpose(1, 2)             # [batch, nb_nodes, nb_nodes]
        coefs = F.softmax(F.leaky_relu(logits) + bias_mat, dim=-1)

        if self.coef_drop != 0.0 and self.training:
            coefs = F.dropout(coefs, p=self.coef_drop)
        if self.in_drop != 0.0 and self.training:
            seq_fts = F.dropout(seq_fts, p=self.in_drop)

        vals = torch.matmul(coefs, seq_fts)            # [batch, nb_nodes, out_sz]
        ret = vals + self.bias

        if self.residual:
            if self.res_proj is not None:
                ret = ret + self.res_proj(seq)
            else:
                ret = ret + seq

        if self.activation is not None:
            return self.activation(ret)
        return ret


class SpAttentionHead(nn.Module):
    """Sparse graph attention head. Equivalent to sp_attn_head() in the original TF implementation.
    Only supports batch_size=1, matching the original constraint."""

    def __init__(self, in_features, out_sz, in_drop=0.0, coef_drop=0.0, residual=False, activation=F.elu):
        super().__init__()
        self.in_drop = in_drop
        self.coef_drop = coef_drop
        self.residual = residual
        self.activation = activation
        self.out_sz = out_sz

        self.W = nn.Linear(in_features, out_sz, bias=False)
        self.a_1 = nn.Linear(out_sz, 1)
        self.a_2 = nn.Linear(out_sz, 1)
        self.bias = nn.Parameter(torch.zeros(out_sz))

        if residual and in_features != out_sz:
            self.res_proj = nn.Linear(in_features, out_sz, bias=False)
        else:
            self.res_proj = None

    def forward(self, seq, adj_mat, nb_nodes):
        # seq:     [1, nb_nodes, in_features]  (batch_size must be 1)
        # adj_mat: sparse COO tensor [nb_nodes, nb_nodes], values are 1 for edges

        if self.in_drop != 0.0 and self.training:
            seq = F.dropout(seq, p=self.in_drop)

        seq_fts = self.W(seq)                           # [1, nb_nodes, out_sz]

        f_1 = self.a_1(seq_fts).reshape(nb_nodes)      # [nb_nodes]
        f_2 = self.a_2(seq_fts).reshape(nb_nodes)      # [nb_nodes]

        # Get edge (row, col) indices from the sparse adjacency matrix
        indices = adj_mat.coalesce().indices()           # [2, nnz]
        row, col = indices[0], indices[1]               # row=i (source), col=j (neighbor)

        # Attention logit for each edge: e[i,j] = f_1[i] + f_2[j]
        e = f_1[row] + f_2[col]                         # [nnz]
        e = F.leaky_relu(e)

        # Row-wise sparse softmax via torch.sparse.softmax
        sparse_logits = torch.sparse_coo_tensor(
            torch.stack([row, col]), e, (nb_nodes, nb_nodes), device=e.device
        )
        coefs_sparse = torch.sparse.softmax(sparse_logits, dim=1)  # row-wise

        if self.coef_drop != 0.0 and self.training:
            coefs_sparse = torch.sparse_coo_tensor(
                coefs_sparse.coalesce().indices(),
                F.dropout(coefs_sparse.coalesce().values(), p=self.coef_drop),
                coefs_sparse.shape,
                device=e.device,
            )

        if self.in_drop != 0.0 and self.training:
            seq_fts = F.dropout(seq_fts, p=self.in_drop)

        # Sparse matrix multiply: [nb_nodes, nb_nodes] x [nb_nodes, out_sz]
        seq_fts_sq = seq_fts.squeeze(0)                 # [nb_nodes, out_sz]
        vals = torch.sparse.mm(coefs_sparse, seq_fts_sq)  # [nb_nodes, out_sz]
        vals = vals.unsqueeze(0)                        # [1, nb_nodes, out_sz]
        ret = vals + self.bias

        if self.residual:
            if self.res_proj is not None:
                ret = ret + self.res_proj(seq)
            else:
                ret = ret + seq

        if self.activation is not None:
            return self.activation(ret)
        return ret
