import torch
import torch.nn as nn
import torch.nn.functional as F

# FIRST PASS USING CHATGPT 
# TODO:
    # check __init__ arguments specifically concat, add_self_loops
class GraphAttentionLayer(nn.Module):
    """
    Single-head Graph Attention Layer from the original GAT paper.

    Inputs:
        x   : [N, in_features] node feature matrix
        adj : [N, N] adjacency matrix
              adj[i, j] = 1 if j is a neighbor of i, else 0
              Include self-loops in adj before calling, or set add_self_loops=True.

    Output:
        out : [N, out_features]
    """
    def __init__(
        self,
        in_features: int,
        out_features: int,
        dropout: float = 0.6,
        alpha: float = 0.2,
        concat: bool = True,
        add_self_loops: bool = True,
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.dropout = dropout
        self.concat = concat
        self.add_self_loops = add_self_loops

        # Linear transform W
        self.W = nn.Parameter(torch.empty(in_features, out_features))

        # Attention vector a, applied to concatenated [Wh_i || Wh_j]
        self.a = nn.Parameter(torch.empty(2 * out_features, 1))

        self.leakyrelu = nn.LeakyReLU(alpha)

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.W)
        nn.init.xavier_uniform_(self.a)

    def forward(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        """
        x:   [N, in_features]
        adj: [N, N] (0/1 or bool)
        """
        N = x.size(0)

        if self.add_self_loops:
            adj = adj.clone()
            adj.fill_diagonal_(1)

        # 1) Linear projection: Wh
        Wh = x @ self.W                          # [N, out_features]

        # 2) Compute attention logits e_ij for all pairs
        # Build [Wh_i || Wh_j] for every (i, j)
        Wh_i = Wh.unsqueeze(1).expand(N, N, self.out_features)   # [N, N, F']
        Wh_j = Wh.unsqueeze(0).expand(N, N, self.out_features)   # [N, N, F']
        a_input = torch.cat([Wh_i, Wh_j], dim=-1)                # [N, N, 2F']

        e = self.leakyrelu(torch.matmul(a_input, self.a).squeeze(-1))  # [N, N]

        # 3) Mask out non-neighbors
        # Only attend over j in N_i
        neg_inf = torch.full_like(e, -9e15)
        attention = torch.where(adj > 0, e, neg_inf)

        # 4) Normalize over neighbors with softmax
        attention = F.softmax(attention, dim=1)

        # 5) Dropout on attention coefficients (used in the paper)
        attention = F.dropout(attention, p=self.dropout, training=self.training)

        # 6) Aggregate neighbor features
        h_prime = attention @ Wh   # [N, out_features]

        # Hidden layers in GAT use nonlinearity after aggregation
        if self.concat:
            return F.elu(h_prime)
        else:
            return h_prime