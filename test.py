import torch

def get_attention_mask(batch_size, n_nodes, edges):
    num_edges_per_graph = edges[0].size(0)

    # Total number of elements (nodes + edges)
    total_elements = n_nodes + num_edges_per_graph
    base_attention_mask = torch.zeros(1, total_elements, total_elements, device=edges[0].device)

    # Nodes can send and receive information to/from all nodes (including themselves)
    for i in range(n_nodes):
        for j in range(n_nodes):
            base_attention_mask[0, i, j] = 1

    # Edges can receive information only from adjacent nodes and themselves
    for i in range(num_edges_per_graph):
        start_node = edges[0][i].item()
        end_node = edges[1][i].item()
        edge_idx = n_nodes + i

        # Only adjacent nodes can send information to the edge
        base_attention_mask[0, start_node, edge_idx] = 1
        base_attention_mask[0, end_node, edge_idx] = 1

        # Edges can send information to all nodes
        for node_idx in range(n_nodes):
            base_attention_mask[0, edge_idx, node_idx] = 1

        # Edges can receive information from their corresponding nodes and themselves
        base_attention_mask[0, edge_idx, start_node] = 1
        base_attention_mask[0, edge_idx, end_node] = 1
        base_attention_mask[0, edge_idx, edge_idx] = 1

    # Stack the masks for each batch
    attention_mask = base_attention_mask.repeat(batch_size, 1, 1)

    # Convert the mask to float and set masked positions to -inf and allowed positions to 0
    attention_mask = attention_mask.float()
    attention_mask.masked_fill(attention_mask == 0, float('-inf'))
    attention_mask.masked_fill(attention_mask == 1, float(0.0))

    return attention_mask

# Example with 3 nodes and fully connected graph (3 edges)
batch_size = 2
n_nodes = 5
edges = [
    torch.tensor([0, 0, 0, 0, 1, 1, 1, 2, 2, 3]),
    torch.tensor([1, 2, 3, 4, 0, 2, 4, 0, 4, 0])
]

attention_mask = get_attention_mask(batch_size, n_nodes, edges)
print(attention_mask)
