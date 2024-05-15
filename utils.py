import torch

def make_edge_attr(node_features, edges, batch_size):
    edge_attributes = []

    total_number_edges = edges[0].shape[0]

    # Loop over all edges
    for i in range(total_number_edges):
        node1 = edges[0][i]
        node2 = edges[1][i]

        # difference between node features
        node_i_features = node_features[node1]  # [#features(charge, loc, vel), dim]
        node_j_features = node_features[node2]  # [#features(charge, loc, vel), dim]
        difference = node_i_features - node_j_features
        edge_representation = difference
        edge_attributes.append(edge_representation)

    edge_attributes = torch.stack(edge_attributes)
    return edge_attributes
