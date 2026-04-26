import torch
import torch.nn as nn
import torch.nn.functional as F

from data_preprocessing import cora

torch.manual_seed(1)


class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, dropout=0.6):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        self.dropout = dropout

    def forward(self, x):
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.fc1(x)
        x = F.elu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)


def accuracy(pred, y):
    return (pred == y).sum().item() / y.size(0)


# ✅ 用你 preprocessing 的 data
data = cora.data

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
data = data.to(device)


model = MLP(
    input_dim=data.num_node_features,
    hidden_dim=8,
    output_dim=int(data.y.max().item()) + 1,
    dropout=0.6
).to(device)


optimizer = torch.optim.Adam(
    model.parameters(),
    lr=0.005,
    weight_decay=5e-4
)


for epoch in range(1, 1001):
    model.train()
    optimizer.zero_grad()

    out = model(data.x)
    loss = F.nll_loss(out[data.train_mask], data.y[data.train_mask])

    loss.backward()
    optimizer.step()

    model.eval()
    with torch.no_grad():
        out = model(data.x)
        pred = out.argmax(dim=1)

        train_acc = accuracy(pred[data.train_mask], data.y[data.train_mask])
        val_acc = accuracy(pred[data.val_mask], data.y[data.val_mask])
        test_acc = accuracy(pred[data.test_mask], data.y[data.test_mask])

    if epoch % 50 == 0:
        print(
            f"Epoch {epoch:03d} | "
            f"Loss {loss:.4f} | "
            f"Train {train_acc:.4f} | "
            f"Val {val_acc:.4f} | "
            f"Test {test_acc:.4f}"
        )

print(f"Final Test Accuracy: {test_acc:.4f}")