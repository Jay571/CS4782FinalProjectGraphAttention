import torch
from torch_geometric.datasets import Planetoid
from torch_geometric.utils import add_self_loops


def edge_index_to_adj(edge_index, num_nodes):
    """Convert PyG edge_index to a dense binary adjacency matrix [N, N]."""
    adj = torch.zeros(num_nodes, num_nodes)
    adj[edge_index[0], edge_index[1]] = 1.0
    return adj


def adj_to_bias(adj):
    """Convert a binary adjacency matrix to a GAT bias matrix.
    adj: [N, N] or [B, N, N] tensor with self-loops already included.
    Returns same shape: 0.0 for edges, -1e9 for non-edges.
    """
    return -1e9 * (1.0 - adj)


def edge_index_to_sparse_adj(edge_index, num_nodes):
    """Convert PyG edge_index to a sparse COO tensor [N, N] for SpGAT."""
    values = torch.ones(edge_index.size(1))
    return torch.sparse_coo_tensor(edge_index, values, (num_nodes, num_nodes))


def preprocess_features(features):
    """Row-normalize a dense feature matrix [N, F]."""
    rowsum = features.sum(dim=1, keepdim=True).clamp(min=1e-9)
    return features / rowsum


cora = Planetoid(root='./data', name='Cora')
cora.data.edge_index, _ = add_self_loops(cora.data.edge_index, num_nodes=cora.data.num_nodes)

citeseer = Planetoid(root='./data', name='CiteSeer')
citeseer.data.edge_index, _ = add_self_loops(citeseer.data.edge_index, num_nodes=citeseer.data.num_nodes)

pubmed = Planetoid(root='./data', name='PubMed')
pubmed.data.edge_index, _ = add_self_loops(pubmed.data.edge_index, num_nodes=pubmed.data.num_nodes)
