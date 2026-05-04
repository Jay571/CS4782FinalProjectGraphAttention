import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.layers2 import SpAttentionHead
from models.base_gattn import BaseGAttN


class SpGAT(BaseGAttN):
    """Sparse Graph Attention Network — uses layers2.py (bug-fixed attention heads).
    Only supports batch_size=1.

    Args:
        in_features:  input feature dimension
        nb_classes:   number of output classes
        nb_nodes:     number of graph nodes
        hid_units:    list of hidden units per layer, e.g. [8]
        n_heads:      number of attention heads per layer (len = len(hid_units) + 1)
        attn_drop:    dropout on attention coefficients
        ffd_drop:     dropout on node features
        activation:   activation applied after each hidden layer (default ELU)
        residual:     whether to use residual connections in hidden layers
    """

    def __init__(self, in_features, nb_classes, nb_nodes, hid_units, n_heads,
                 attn_drop=0.0, ffd_drop=0.0, activation=F.elu, residual=False):
        super().__init__()
        self.n_heads = n_heads
        self.hid_units = hid_units
        self.nb_nodes = nb_nodes

        self.hidden_layers = nn.ModuleList()

        layer_0 = nn.ModuleList([
            SpAttentionHead(in_features, hid_units[0],
                            in_drop=ffd_drop, coef_drop=attn_drop,
                            residual=False, activation=activation)
            for _ in range(n_heads[0])
        ])
        self.hidden_layers.append(layer_0)

        for i in range(1, len(hid_units)):
            in_dim = hid_units[i - 1] * n_heads[i - 1]
            layer_i = nn.ModuleList([
                SpAttentionHead(in_dim, hid_units[i],
                                in_drop=ffd_drop, coef_drop=attn_drop,
                                residual=residual, activation=activation)
                for _ in range(n_heads[i])
            ])
            self.hidden_layers.append(layer_i)

        out_in_dim = hid_units[-1] * n_heads[len(hid_units) - 1]
        self.out_layer = nn.ModuleList([
            SpAttentionHead(out_in_dim, nb_classes,
                            in_drop=ffd_drop, coef_drop=attn_drop,
                            residual=False, activation=None)
            for _ in range(n_heads[-1])
        ])

    def forward(self, inputs, adj_mat):
        nb_nodes = self.nb_nodes

        attns = [head(inputs, adj_mat, nb_nodes) for head in self.hidden_layers[0]]
        h = torch.cat(attns, dim=-1)

        for layer in self.hidden_layers[1:]:
            attns = [head(h, adj_mat, nb_nodes) for head in layer]
            h = torch.cat(attns, dim=-1)

        out = [head(h, adj_mat, nb_nodes) for head in self.out_layer]
        logits = torch.stack(out, dim=0).mean(dim=0)

        return logits
