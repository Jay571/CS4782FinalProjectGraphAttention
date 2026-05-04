import torch
import torch.nn as nn
import torch.nn.functional as F


class AttentionHead(nn.Module):
    """Dense graph attention head. Equivalent to attn_head() in the original TF implementation.

    Changes from layers.py:
    - LeakyReLU negative_slope set to 0.2 (matches tf.nn.leaky_relu default alpha=0.2)
    - W, a_1, a_2, res_proj initialized with Xavier uniform (matches tf.layers.conv1d default)
    - a_1, a_2 bias initialized to zero (matches TF bias_add default)
    """

    def __init__(self, in_features, out_sz, in_drop=0.0, coef_drop=0.0, residual=False, activation=F.elu):
        super().__init__()
        self.in_drop = in_drop
        self.coef_drop = coef_drop
        self.residual = residual
        self.activation = activation

        self.W = nn.Linear(in_features, out_sz, bias=False)
        nn.init.xavier_uniform_(self.W.weight)

        self.a_1 = nn.Linear(out_sz, 1)
        nn.init.xavier_uniform_(self.a_1.weight)
        nn.init.zeros_(self.a_1.bias)

        self.a_2 = nn.Linear(out_sz, 1)
        nn.init.xavier_uniform_(self.a_2.weight)
        nn.init.zeros_(self.a_2.bias)

        self.bias = nn.Parameter(torch.zeros(out_sz))

        if residual and in_features != out_sz:
            self.res_proj = nn.Linear(in_features, out_sz, bias=True)
            nn.init.xavier_uniform_(self.res_proj.weight)
            nn.init.zeros_(self.res_proj.bias)
        else:
            self.res_proj = None

    def forward(self, seq, bias_mat):
        if self.in_drop != 0.0 and self.training:
            seq = F.dropout(seq, p=self.in_drop)

        seq_fts = self.W(seq)

        f_1 = self.a_1(seq_fts)
        f_2 = self.a_2(seq_fts)
        logits = f_1 + f_2.transpose(1, 2)
        coefs = F.softmax(F.leaky_relu(logits, negative_slope=0.2) + bias_mat, dim=-1)

        if self.coef_drop != 0.0 and self.training:
            coefs = F.dropout(coefs, p=self.coef_drop)
        if self.in_drop != 0.0 and self.training:
            seq_fts = F.dropout(seq_fts, p=self.in_drop)

        vals = torch.matmul(coefs, seq_fts)
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
    Only supports batch_size=1, matching the original constraint.

    Changes from layers.py:
    - LeakyReLU negative_slope set to 0.2 (matches tf.nn.leaky_relu default alpha=0.2)
    - W, a_1, a_2, res_proj initialized with Xavier uniform (matches tf.layers.conv1d default)
    - a_1, a_2 bias initialized to zero (matches TF bias_add default)
    """

    def __init__(self, in_features, out_sz, in_drop=0.0, coef_drop=0.0, residual=False, activation=F.elu):
        super().__init__()
        self.in_drop = in_drop
        self.coef_drop = coef_drop
        self.residual = residual
        self.activation = activation
        self.out_sz = out_sz

        self.W = nn.Linear(in_features, out_sz, bias=False)
        nn.init.xavier_uniform_(self.W.weight)

        self.a_1 = nn.Linear(out_sz, 1)
        nn.init.xavier_uniform_(self.a_1.weight)
        nn.init.zeros_(self.a_1.bias)

        self.a_2 = nn.Linear(out_sz, 1)
        nn.init.xavier_uniform_(self.a_2.weight)
        nn.init.zeros_(self.a_2.bias)

        self.bias = nn.Parameter(torch.zeros(out_sz))

        if residual and in_features != out_sz:
            self.res_proj = nn.Linear(in_features, out_sz, bias=True)
            nn.init.xavier_uniform_(self.res_proj.weight)
            nn.init.zeros_(self.res_proj.bias)
        else:
            self.res_proj = None

    def forward(self, seq, adj_mat, nb_nodes):
        if self.in_drop != 0.0 and self.training:
            seq = F.dropout(seq, p=self.in_drop)

        seq_fts = self.W(seq)

        f_1 = self.a_1(seq_fts).reshape(nb_nodes)
        f_2 = self.a_2(seq_fts).reshape(nb_nodes)

        indices = adj_mat.coalesce().indices()
        row, col = indices[0], indices[1]

        e = f_1[row] + f_2[col]
        e = F.leaky_relu(e, negative_slope=0.2)

        sparse_logits = torch.sparse_coo_tensor(
            torch.stack([row, col]), e, (nb_nodes, nb_nodes), device=e.device
        )
        coefs_sparse = torch.sparse.softmax(sparse_logits, dim=1)

        if self.coef_drop != 0.0 and self.training:
            coefs_sparse = torch.sparse_coo_tensor(
                coefs_sparse.coalesce().indices(),
                F.dropout(coefs_sparse.coalesce().values(), p=self.coef_drop),
                coefs_sparse.shape,
                device=e.device,
            )

        if self.in_drop != 0.0 and self.training:
            seq_fts = F.dropout(seq_fts, p=self.in_drop)

        seq_fts_sq = seq_fts.squeeze(0)
        vals = torch.sparse.mm(coefs_sparse, seq_fts_sq)
        vals = vals.unsqueeze(0)
        ret = vals + self.bias

        if self.residual:
            if self.res_proj is not None:
                ret = ret + self.res_proj(seq)
            else:
                ret = ret + seq

        if self.activation is not None:
            return self.activation(ret)
        return ret
