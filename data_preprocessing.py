import torch
from torch_geometric.datasets import Planetoid, PPI
from torch_geometric.utils import add_self_loops



# To access the splitted datasets:
#  use from data_preprocessing import cora_train









# ========================
# Utils
# ========================

def preprocess_features(features):
    rowsum = features.sum(dim=1, keepdim=True).clamp(min=1e-9)
    return features / rowsum


def add_loops(data):
    data.edge_index, _ = add_self_loops(
        data.edge_index, num_nodes=data.num_nodes
    )
    return data


def split_planetoid(data):
    """Split a single graph into train/val/test node subsets."""
    train_data = data.clone()
    val_data = data.clone()
    test_data = data.clone()

    train_data.train_mask = data.train_mask
    val_data.val_mask = data.val_mask
    test_data.test_mask = data.test_mask

    return train_data, val_data, test_data


# ========================
# Cora
# ========================

_cora = Planetoid(root="./data", name="Cora")[0]
_cora = add_loops(_cora)
_cora.x = preprocess_features(_cora.x)

cora_train, cora_val, cora_test = split_planetoid(_cora)


# ========================
# Citeseer
# ========================

_citeseer = Planetoid(root="./data", name="CiteSeer")[0]
_citeseer = add_loops(_citeseer)
_citeseer.x = preprocess_features(_citeseer.x)

citeseer_train, citeseer_val, citeseer_test = split_planetoid(_citeseer)


# ========================
# Pubmed
# ========================

_pubmed = Planetoid(root="./data", name="PubMed")[0]
_pubmed = add_loops(_pubmed)
_pubmed.x = preprocess_features(_pubmed.x)

pubmed_train, pubmed_val, pubmed_test = split_planetoid(_pubmed)


# ========================
# PPI (inductive)
# ========================

ppi_train = PPI(root="./data/PPI", split="train")
ppi_val = PPI(root="./data/PPI", split="val")
ppi_test = PPI(root="./data/PPI", split="test")