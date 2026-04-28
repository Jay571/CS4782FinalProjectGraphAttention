# The results are: 
#Cora: 0.5610
#CiteSeer: 0.5480
#PubMed: 0.7200
#PPI: 0.4514


import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.loader import DataLoader

from data_preprocessing import (
    cora_train, cora_val, cora_test,
    citeseer_train, citeseer_val, citeseer_test,
    pubmed_train, pubmed_val, pubmed_test,
    ppi_train, ppi_val, ppi_test
)

torch.manual_seed(1)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class MLP(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, dropout=0.6):
        super().__init__()
        self.fc1 = nn.Linear(in_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, out_dim)
        self.dropout = dropout

    def forward(self, x):
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = F.elu(self.fc1(x))
        x = F.dropout(x, p=self.dropout, training=self.training)
        return self.fc2(x)


def accuracy(logits, y, mask):
    pred = logits.argmax(dim=1)
    return (pred[mask] == y[mask]).float().mean().item()


def micro_f1(logits, y):
    pred = (torch.sigmoid(logits) > 0.5).float()
    tp = (pred * y).sum()
    fp = (pred * (1 - y)).sum()
    fn = ((1 - pred) * y).sum()
    return (2 * tp / (2 * tp + fp + fn + 1e-9)).item()


def run_planetoid(name, train_data, val_data, test_data, epochs=1000):
    train_data = train_data.to(device)
    val_data = val_data.to(device)
    test_data = test_data.to(device)

    model = MLP(
        train_data.num_node_features,
        8,
        int(train_data.y.max().item()) + 1
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.005, weight_decay=5e-4)

    best_val = 0
    best_test = 0

    for _ in range(epochs):
        model.train()
        optimizer.zero_grad()
        logits = model(train_data.x)
        loss = F.cross_entropy(logits[train_data.train_mask], train_data.y[train_data.train_mask])
        loss.backward()
        optimizer.step()

        model.eval()
        with torch.no_grad():
            val_acc = accuracy(model(val_data.x), val_data.y, val_data.val_mask)
            test_acc = accuracy(model(test_data.x), test_data.y, test_data.test_mask)

        if val_acc > best_val:
            best_val = val_acc
            best_test = test_acc

    print(f"{name}: {best_test:.4f}")


def run_ppi(epochs=100):
    train_loader = DataLoader(ppi_train, batch_size=2, shuffle=True)
    val_loader = DataLoader(ppi_val, batch_size=2)
    test_loader = DataLoader(ppi_test, batch_size=2)

    model = MLP(ppi_train.num_node_features, 256, ppi_train.num_classes).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.005)

    best_val = 0
    best_test = 0

    for _ in range(epochs):
        model.train()
        for batch in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            loss = F.binary_cross_entropy_with_logits(model(batch.x), batch.y)
            loss.backward()
            optimizer.step()

        model.eval()
        with torch.no_grad():
            val_f1 = eval_ppi(model, val_loader)
            test_f1 = eval_ppi(model, test_loader)

        if val_f1 > best_val:
            best_val = val_f1
            best_test = test_f1

    print(f"PPI: {best_test:.4f}")


@torch.no_grad()
def eval_ppi(model, loader):
    logits_all, y_all = [], []

    for batch in loader:
        batch = batch.to(device)
        logits_all.append(model(batch.x))
        y_all.append(batch.y)

    logits_all = torch.cat(logits_all, dim=0)
    y_all = torch.cat(y_all, dim=0)

    return micro_f1(logits_all, y_all)


run_planetoid("Cora", cora_train, cora_val, cora_test)
run_planetoid("CiteSeer", citeseer_train, citeseer_val, citeseer_test)
run_planetoid("PubMed", pubmed_train, pubmed_val, pubmed_test)
run_ppi()
