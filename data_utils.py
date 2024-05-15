import torch
from utils import make_edge_attr
from algebra.cliffordalgebra import CliffordAlgebra

def embed_nbody_graphs(batch):
    loc, vel, edge_attr, charges, loc_end, edges = batch

    batch_size, n_nodes, _ = loc.size()
    clifford_algebra = CliffordAlgebra([1, 1, 1])

    loc_mean = loc - loc.mean(dim=1, keepdim=True)
    loc_mean = loc_mean.float().view(-1, *loc_mean.shape[2:])
    loc = loc.float().view(-1, *loc.shape[2:])
    vel = vel.float().view(-1, *vel.shape[2:])
    edge_attr = edge_attr.float().view(-1, *edge_attr.shape[2:])
    charges = charges.float().view(-1, *charges.shape[2:])
    loc_end = loc_end.float().view(-1, *loc_end.shape[2:])

    invariants = charges
    invariants = clifford_algebra.embed(invariants, (0,))

    xv = torch.stack([loc_mean, vel], dim=1)
    covariants = clifford_algebra.embed(xv, (1, 2, 3))

    nodes_in_clifford = torch.cat([invariants[:, None], covariants], dim=1)
    zero_padding_nodes = torch.zeros(5*batch_size, 4, 8)
    full_node_embedding = torch.cat((nodes_in_clifford, zero_padding_nodes), dim=1)

    batch_index = torch.arange(batch_size, device=loc_mean.device)
    edges = edges + n_nodes * batch_index[:, None, None]
    edges = tuple(edges.transpose(0, 1).flatten(1))
    start_nodes, end_nodes = edges

    attention_mask = torch.zeros(25, 25)
    for b in range(batch_size):
        node_start_idx = b * (25)
        edge_start_idx = node_start_idx + 5

        for i in range(20):
            start_node = start_nodes[i].item() + node_start_idx
            end_node = end_nodes[i].item() + node_start_idx
            edge_idx = edge_start_idx + i

            attention_mask[edge_idx, start_node] = 1
            attention_mask[edge_idx, end_node] = 1
            attention_mask[start_node, edge_idx] = 1
            attention_mask[end_node, edge_idx] = 1

    attention_mask = attention_mask.float()
    attention_mask = attention_mask.masked_fill(attention_mask == 0, float('-inf'))
    attention_mask = attention_mask.masked_fill(attention_mask == 1, float(0.0))

    orig_edge_attr_clifford = clifford_algebra.embed(edge_attr[..., None], (0,))
    extra_edge_attr_clifford = make_edge_attr(nodes_in_clifford, edges, batch_size)
    edge_attr_all = torch.cat((orig_edge_attr_clifford, extra_edge_attr_clifford), dim=1)
    zero_padding_edges = torch.zeros(20*batch_size, 3, 8)
    full_edge_embedding = torch.cat((zero_padding_edges, edge_attr_all), dim=1)

    loc_end_clifford = clifford_algebra.embed(loc_end, (1, 2, 3))

    return full_node_embedding, full_edge_embedding, loc_end_clifford, attention_mask

def generate_mock_batch(batch_size):
    n_nodes = 5
    n_features = 3
    n_edges = 20
    n_edge_features = 1

    loc_frame_0 = torch.rand(batch_size, n_nodes, n_features)
    vel_frame_0 = torch.rand(batch_size, n_nodes, n_features)
    edge_attr = torch.rand(batch_size, n_edges, n_edge_features)
    charges = torch.rand(batch_size, n_nodes, 1)
    loc_frame_T = torch.rand(batch_size, n_nodes, n_features)

    rows = torch.randint(0, n_nodes, (n_edges,))
    cols = torch.randint(0, n_nodes, (n_edges,))
    edges = torch.stack([rows, cols], dim=0).repeat(batch_size, 1, 1)

    return loc_frame_0, vel_frame_0, edge_attr, charges, loc_frame_T, edges
