# The results are:
#Cora: 0.8160
#CiteSeer: 0.7020
#PubMed: 0.7910

import torch
import torch.nn as nn
import torch.nn.functional as F

from data_preprocessing import (
    cora_train,
    citeseer_train,
    pubmed_train
)

torch.manual_seed(1)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def normalize_adj(edge_index, num_nodes):
    row, col = edge_index
    values = torch.ones(row.size(0), device=edge_index.device)

    adj = torch.sparse_coo_tensor(
        edge_index,
        values,
        (num_nodes, num_nodes)
    ).coalesce()

    deg = torch.sparse.sum(adj, dim=1).to_dense()
    deg_inv_sqrt = deg.pow(-0.5)
    deg_inv_sqrt[torch.isinf(deg_inv_sqrt)] = 0.0

    row, col = adj.indices()
    norm_values = deg_inv_sqrt[row] * adj.values() * deg_inv_sqrt[col]

    return torch.sparse_coo_tensor(
        adj.indices(),
        norm_values,
        adj.size()
    ).coalesce()


class GraphConvolution(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(in_dim, out_dim))
        nn.init.xavier_uniform_(self.weight)

    def forward(self, x, support):
        x = x @ self.weight
        x = torch.sparse.mm(support, x)
        return x


class GCN(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, dropout=0.5):
        super().__init__()
        self.gc1 = GraphConvolution(in_dim, hidden_dim)
        self.gc2 = GraphConvolution(hidden_dim, out_dim)
        self.dropout = dropout

    def forward(self, x, support):
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = F.relu(self.gc1(x, support))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.gc2(x, support)
        return x


def accuracy(logits, y, mask):
    pred = logits.argmax(dim=1)
    return (pred[mask] == y[mask]).float().mean().item()


def run_gcn(name, data, epochs=200, patience=10):
    data = data.to(device)

    support = normalize_adj(data.edge_index, data.num_nodes).to(device)

    model = GCN(
        in_dim=data.num_node_features,
        hidden_dim=16,
        out_dim=int(data.y.max().item()) + 1,
        dropout=0.5
    ).to(device)

    optimizer = torch.optim.Adam([
        {'params': model.gc1.parameters(), 'weight_decay': 5e-4},
        {'params': model.gc2.parameters(), 'weight_decay': 0.0}
    ], lr=0.01)

    best_val_loss = float("inf")
    best_test_acc = 0.0
    no_improve = 0

    for _ in range(epochs):
        model.train()
        optimizer.zero_grad()

        logits = model(data.x, support)
        loss = F.cross_entropy(logits[data.train_mask], data.y[data.train_mask])

        loss.backward()
        optimizer.step()

        model.eval()
        with torch.no_grad():
            logits = model(data.x, support)
            val_loss = F.cross_entropy(logits[data.val_mask], data.y[data.val_mask])
            test_acc = accuracy(logits, data.y, data.test_mask)

        if val_loss < best_val_loss:
            best_val_loss = val_loss.item()
            best_test_acc = test_acc
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= patience:
                break

    print(f"{name}: {best_test_acc:.4f}")


run_gcn("Cora", cora_train)
run_gcn("CiteSeer", citeseer_train)
run_gcn("PubMed", pubmed_train)