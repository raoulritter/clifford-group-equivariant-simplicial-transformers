import torch
import numpy as np

# Mock class to hold the function
class MockModel:
    def __init__(self):
        self.triangle_projection = lambda x: x  # Dummy projection
        self.triangle_left = lambda x: x  # Dummy projection
        self.clifford_algebra = self  # For simplicity

    def geometric_product(self, x, y):
        return x * y  # Dummy geometric product

    def get_full_triangle_embedding(self, edge_embeddings, nodes_stack, edges, n_nodes, batch_size):
        # Initialize a list to store the triangle embeddings and a set to store unique triangles
        triangle_embeddings = []

        # Iterate over each batch
        for batch_index in range(batch_size):
            unique_triangles = set()

            # Convert the edges into a dictionary for fast lookup
            edge_dict = {}
            for idx, (u, v) in enumerate(zip(edges[0], edges[1])):
                if u in edge_dict:
                    edge_dict[u].add((v, idx))
                else:
                    edge_dict[u] = {(v, idx)}
                if v in edge_dict:
                    edge_dict[v].add((u, idx))
                else:
                    edge_dict[v] = {(u, idx)}

            # Iterate over each node pair (i, j)
            for i in range(n_nodes):
                for j in range(i + 1, n_nodes):
                    # Find common neighbors of node i and node j
                    if i in edge_dict and j in edge_dict:
                        common_neighbors = {neighbor for neighbor, _ in edge_dict[i]}.intersection(
                            {neighbor for neighbor, _ in edge_dict[j]}
                        )

                        for k in common_neighbors:
                            if k != i and k != j:
                                # Nodes i, j, k form a triangle
                                # Standardize the triangle by sorting the nodes
                                triangle_nodes = tuple(sorted([i, j, k]))

                                # Check if the triangle is already processed
                                if triangle_nodes not in unique_triangles:
                                    unique_triangles.add(triangle_nodes)

                                    # Get the embeddings
                                    node_i_embedding = nodes_stack[batch_index, triangle_nodes[0]]
                                    node_j_embedding = nodes_stack[batch_index, triangle_nodes[1]]
                                    node_k_embedding = nodes_stack[batch_index, triangle_nodes[2]]

                                    edge_ij_idx = next(idx for neighbor, idx in edge_dict[i] if neighbor == j)
                                    edge_jk_idx = next(idx for neighbor, idx in edge_dict[j] if neighbor == k)
                                    edge_ki_idx = next(idx for neighbor, idx in edge_dict[k] if neighbor == i)

                                    edge_ij_embedding = edge_embeddings[batch_index, edge_ij_idx]
                                    edge_jk_embedding = edge_embeddings[batch_index, edge_jk_idx]
                                    edge_ki_embedding = edge_embeddings[batch_index, edge_ki_idx]

                                    # Mean of all features
                                    combined_triangle_embedding = (node_i_embedding + node_j_embedding + node_k_embedding +
                                                                   edge_ij_embedding + edge_jk_embedding + edge_ki_embedding) / 6

                                    triangle_embeddings.append(combined_triangle_embedding)

        full_triangle_embedding = torch.stack(triangle_embeddings)
        triangle_embeddings = self.triangle_projection(full_triangle_embedding)
        triangle_embeddings_left = self.triangle_left(triangle_embeddings)
        full_triangle_embedding = self.clifford_algebra.geometric_product(triangle_embeddings_left, triangle_embeddings)

        full_triangle_embedding = full_triangle_embedding.view(batch_size, -1, 3, 8)

        return full_triangle_embedding, list(unique_triangles)

# Define parameters
n_nodes = 5
batch_size = 10

# Create mock data
edge_embeddings = torch.randn(batch_size, len(range(n_nodes * (n_nodes - 1) // 2)), 3, 8)
nodes_stack = torch.randn(batch_size, n_nodes, 3, 8)

# Create fully connected graph edges
edges = set()
for i in range(n_nodes):
    for j in range(i + 1, n_nodes):
        edges.add((i, j))

# Convert the set to two lists
edge_list_start = [edge[0] for edge in edges]
edge_list_end = [edge[1] for edge in edges]

# Create the final tuple
edges = (edge_list_start, edge_list_end)

# Display the edges tuple
print("Edges tuple:", edges)

# Instantiate the model and run the function
model = MockModel()
full_triangle_embedding, unique_triangles = model.get_full_triangle_embedding(edge_embeddings, nodes_stack, edges, n_nodes, batch_size)

print("Full Triangle Embedding:")
print(full_triangle_embedding.shape)
print("Unique Triangles:")
print(unique_triangles)
