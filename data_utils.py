import torch
from utils import make_edge_attr
from algebra.cliffordalgebra import CliffordAlgebra

clifford_algebra = CliffordAlgebra([1, 1, 1])

def embed_nbody_graphs(batch):
    loc, vel, edge_attr, charges, loc_end, edges = batch

    batch_size, n_nodes, _ = loc.size()
    _, n_edges, _ = edge_attr.size()

    # put the mean of the pointcloud on the origin -> translation invariant
    loc_mean = loc - loc.mean(dim=1, keepdim=True)     # [batch, nodes, dim]

    # All times do [batch, nodes, dim] -> [batch * nodes, dim]
    loc_mean = loc_mean.float().view(-1, *loc_mean.shape[2:])
    loc = loc.float().view(-1, *loc.shape[2:])
    vel = vel.float().view(-1, *vel.shape[2:])
    edge_attr = edge_attr.float().view(-1, *edge_attr.shape[2:])
    charges = charges.float().view(-1, *charges.shape[2:])
    loc_end = loc_end.float().view(-1, *loc_end.shape[2:])

    invariants = charges
    invariants = clifford_algebra.embed(invariants, (0,))

    xv = torch.stack([loc_mean, vel], dim=1) # now of the shape [batch * nodes, 2 (because its loc mean as well as vel), dim]
    covariants = clifford_algebra.embed(xv, (1, 2, 3))

    # Create a vector with [n1, n2, n3, e1, e2, e3, e4]
    nodes_in_clifford = torch.cat([invariants[:, None], covariants], dim=1) # [batch * nodes, #features(charge, loc, vel), dim]
    zero_padding_nodes = torch.zeros(5*batch_size, 4, 8)
    full_node_embedding = torch.cat((nodes_in_clifford, zero_padding_nodes), dim=1)


    # IDK dont really know if this all is needed, just the start and end nodes should be enough but we will see
    batch_index = torch.arange(batch_size, device=loc_mean.device) # torch.arange(batch_size) generates a tensor from 0 to batch_size - 1, creating a sequence that represents each graph in the batch. If batch_size is 3, this tensor will be [0, 1, 2]
    edges = edges + n_nodes * batch_index[:, None, None] # creates separate edge number for every graph. so if edge for graph 1 is between 3 and 4, graph 2 will be between 8 and 9 (if n_nodes = 5)
    edges = tuple(edges.transpose(0, 1).flatten(1)) # where the first element of the tuple contains all start nodes and the second contains all end nodes for edges across the entire batch. ([edges*batch], [edges*batch])
    start_nodes, end_nodes = edges

    # Initialize an attention mask with zeros (disallow all attention initially) NO IDEA IF THIS WORKS FOR MORE IN ONE BATCH
    attention_mask = torch.zeros((n_nodes + n_edges)*batch_size, (n_nodes + n_edges)*batch_size)

    for b in range(batch_size):
        node_start_idx = b * (n_nodes + n_edges)
        edge_start_idx = node_start_idx + 5

        # Nodes can attend to themselves and to all other nodes within the same graph
        for i in range(n_nodes):
            for j in range(n_nodes):
                attention_mask[node_start_idx + i, node_start_idx + j] = 1

        for i in range(n_edges):
            start_node = start_nodes[i].item() + node_start_idx
            end_node = end_nodes[i].item() + node_start_idx
            edge_idx = edge_start_idx + i

            # Edges can attend to their corresponding nodes
            attention_mask[edge_idx, start_node] = 1
            attention_mask[edge_idx, end_node] = 1

            # Nodes can attend to their corresponding edges
            attention_mask[start_node, edge_idx] = 1
            attention_mask[end_node, edge_idx] = 1

    # Convert the mask to float and set masked positions to -inf and allowed positions to 0
    attention_mask = attention_mask.float()
    attention_mask = attention_mask.masked_fill(attention_mask == 0, float('-inf'))
    attention_mask = attention_mask.masked_fill(attention_mask == 1, float(0.0))

    # Create a vector with [n1, n2, n3, e1, e2, e3, e4]
    orig_edge_attr_clifford = clifford_algebra.embed(edge_attr[..., None], (0,)) # now [batch * edges, 1, dim]
    extra_edge_attr_clifford = make_edge_attr(nodes_in_clifford, edges, batch_size) # [batch*edges, #numfeatures, difference + geomprod, dim]
    edge_attr_all = torch.cat((orig_edge_attr_clifford, extra_edge_attr_clifford), dim=1)
    zero_padding_edges = torch.zeros(20*batch_size, 3, 8)
    full_edge_embedding = torch.cat((zero_padding_edges, edge_attr_all), dim=1)

    loc_end_clifford = clifford_algebra.embed(loc_end, (1, 2, 3))

    return full_node_embedding, full_edge_embedding, loc_end_clifford, attention_mask
