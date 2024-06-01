import torch
from ..original_modules.linear import MVLinear

class NBodyGraphEmbedder:
    def __init__(self, clifford_algebra, in_features, embed_dim, simplex_order, empty_higher_simplices=True):
        self.clifford_algebra = clifford_algebra
        self.node_projection = MVLinear(
            self.clifford_algebra, in_features, embed_dim, subspaces=False
        )
        self.edge_projection = MVLinear(
            self.clifford_algebra, 3, embed_dim, subspaces=False
        )

        self.edge_attr_projection = MVLinear(
            self.clifford_algebra, embed_dim, embed_dim, subspaces=False
        )

        self.edge_attr_left = MVLinear(
            self.clifford_algebra, embed_dim, embed_dim, subspaces=False
        )

        self.triangle_projection = MVLinear(
            self.clifford_algebra, 3, embed_dim, subspaces=False
        )
        self.triangle_left = MVLinear(
            self.clifford_algebra, 3, embed_dim, subspaces=False
        )

        self.embed_dim = embed_dim
        self.empty_simplices = empty_higher_simplices

        #TODO: add this as a parameter
        self.num_nodes = 5

        self.num_edges = self.num_nodes * (self.num_nodes - 1) // 2
        self.simplex_order = simplex_order
        self.num_triangles = self.num_nodes * (self.num_nodes - 1) * (self.num_nodes - 2) // 6

    def embed_nbody_graphs(self, batch):
        loc_mean, vel, edge_attr, charges, edges = self.preprocess(batch)
        # Embed data in Clifford space
        invariants = self.clifford_algebra.embed(charges, (0,))
        xv = torch.stack([loc_mean, vel], dim=1)
        covariants = self.clifford_algebra.embed(xv, (1, 2, 3))
        nodes_stack = torch.cat([invariants[:, None], covariants], dim=1)
        # nodes_stack[:,1,0] += 1
        full_node_embedding = self.node_projection(nodes_stack)
        batch_size, n_nodes, _ = batch[0].size()
        if self.simplex_order == 1:
            full_edge_embedding, edges = self.get_full_edge_embedding(edge_attr, nodes_stack, edges, n_nodes, batch_size)
            attention_mask = self.get_attention_mask(batch_size, n_nodes, edges, None)
            full_embedding = torch.cat((full_node_embedding.reshape(batch_size, n_nodes, self.embed_dim, 8),
                                        full_edge_embedding.reshape(batch_size, self.num_edges, self.embed_dim, 8)),
                                       dim=1)
            return full_embedding, attention_mask
        
        if self.simplex_order == 2:
            full_edge_embedding, edges = self.get_full_edge_embedding(edge_attr, nodes_stack, edges, n_nodes, batch_size)
            full_triangle_embedding, triangles = self.get_full_triangle_embedding(full_edge_embedding, nodes_stack, edges, n_nodes, batch_size)
            attention_mask = self.get_attention_mask(batch_size, n_nodes, edges, triangles)
            full_embedding = torch.cat((full_node_embedding.reshape(batch_size, n_nodes, self.embed_dim, 8),
                                        full_edge_embedding.reshape(batch_size, self.num_edges, self.embed_dim, 8),
                                        full_triangle_embedding.reshape(batch_size, self.num_triangles, self.embed_dim, 8)),
                                       dim=1)
            #TODO: CHECK IF THE TRIANGLE CONCAT MAKES SHAPES FUCK UP
            return full_embedding, attention_mask
        
        return full_node_embedding, None

    #TODO: complete and test this function
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

                                    # node_i_embedding = nodes_stack[batch_index, triangle_nodes[0]]
                                    # node_j_embedding = nodes_stack[batch_index, triangle_nodes[1]]
                                    # node_k_embedding = nodes_stack[batch_index, triangle_nodes[2]]

                                    edge_ij_idx = next(idx for neighbor, idx in edge_dict[i] if neighbor == j)
                                    edge_jk_idx = next(idx for neighbor, idx in edge_dict[j] if neighbor == k)
                                    edge_ki_idx = next(idx for neighbor, idx in edge_dict[k] if neighbor == i)

                                    edge_ij_embedding = edge_embeddings[batch_index, edge_ij_idx]
                                    edge_jk_embedding = edge_embeddings[batch_index, edge_jk_idx]
                                    edge_ki_embedding = edge_embeddings[batch_index, edge_ki_idx]

                                    # Mean of all features
                                    combined_triangle_embedding = (edge_ij_embedding + edge_jk_embedding + edge_ki_embedding) / 3

                                    triangle_embeddings.append(combined_triangle_embedding)

        full_triangle_embedding = torch.stack(triangle_embeddings)
        triangle_embeddings = self.triangle_projection(full_triangle_embedding)
        triangle_embeddings_left = self.triangle_left(triangle_embeddings)
        full_triangle_embedding = self.clifford_algebra.geometric_product(triangle_embeddings_left, triangle_embeddings)

        full_triangle_embedding = full_triangle_embedding.view(batch_size, -1, 3, 8)

        return full_triangle_embedding, list(unique_triangles)

    def get_full_edge_embedding(self,edge_attr, nodes_stack, edges, n_nodes, batch_size):
        edges, indices = self.get_edge_nodes(edges, n_nodes, batch_size)
        start_nodes = edges[0]
        end_nodes = edges[1]
        if self.empty_edges:
            return torch.zeros((batch_size, self.num_edges, self.embed_dim,8)), (start_nodes, end_nodes)
        else:
            if self.undirected_edges:
                edge_attr = edge_attr[:, indices, :]
            full_edge_embedding = self.get_edge_embedding(edge_attr, nodes_stack, (start_nodes, end_nodes))
            return full_edge_embedding, (start_nodes, end_nodes)

    def preprocess(self, batch):
        loc, vel, edge_attr, charges, _, edges = batch
        # print("before",loc.shape, vel.shape, edge_attr.shape, edges.shape, charges.shape)
        loc_mean = self.compute_mean_centered(loc)
        loc_mean, vel, charges = self.flatten_tensors(loc_mean, vel, charges, )
        return loc_mean, vel, edge_attr, charges, edges

    def compute_mean_centered(self, tensor):
        return tensor - tensor.mean(dim=1, keepdim=True)

    def flatten_tensors(self, *tensors):
        return [tensor.float().view(-1, *tensor.shape[2:]) for tensor in tensors]

    def get_edge_nodes(self, edges, n_nodes, batch_size):
        batch_index = torch.arange(batch_size, device=edges.device)
        edges = edges + n_nodes * batch_index[:, None, None]
        if self.undirected_edges:
            edges, indices = self.get_undirected_edges_with_indices(edges[0])
            edges = edges.unsqueeze(0).repeat(batch_size, 1, 1)
            edges = tuple(edges.transpose(0, 1).flatten(1))
            return edges, indices
        else:
            return tuple(edges.transpose(0, 1).flatten(1)), None

    def get_edge_embedding(self, edge_attr, nodes_in_clifford, edges):
        edge_attr = self.make_edge_attr(nodes_in_clifford, edges) 
        projected_edges = self.edge_projection(edge_attr)
        return projected_edges

    def make_edge_attr(self, node_features, edges):
        node1_features = node_features[edges[0]]
        node2_features = node_features[edges[1]]
        edge_attr = (node1_features + node2_features) / 2
        edge_attr = self.edge_attr_projection(edge_attr)
        edge_attr_left = self.edge_attr_left(edge_attr)
        edge_attributes = self.clifford_algebra.geometric_product(edge_attr_left, edge_attr)
        return edge_attributes

    def get_undirected_edges_with_indices(self, tensor):
        edges = set()  # edges before
        undirected_edges = []
        unique_indices = []

        for i, edge in enumerate(tensor.t()):
            node1, node2 = sorted(edge.tolist())
            if (node1, node2) not in edges:
                edges.add((node1, node2))
                undirected_edges.append((node1, node2))
                unique_indices.append(i)

        undirected_edges_tensor = torch.tensor(undirected_edges).t()
        unique_indices_tensor = torch.tensor(unique_indices)

        return undirected_edges_tensor, unique_indices_tensor

    # TODO: test this function
    def get_attention_mask(self, batch_size, n_nodes, edges, triangles):
        num_edges_per_graph = edges[0].size(0) // batch_size

        if triangles is not None:
            number_of_triangles = (n_nodes * (n_nodes - 1) * (n_nodes - 2)) // 6
        else:
            number_of_triangles = 0

        # Initialize an attention mask with zeros for a single batch
        base_attention_mask = torch.zeros(1, n_nodes + num_edges_per_graph, n_nodes + num_edges_per_graph + number_of_triangles,
                                          device=edges[0].device)

        # Nodes can attend to themselves and to all other nodes within the same graph
        for i in range(n_nodes):
            for j in range(n_nodes):
                base_attention_mask[0, i, j] = 1

        for i in range(num_edges_per_graph):
            start_node = edges[0][i].item()
            end_node = edges[1][i].item()
            edge_idx = n_nodes + i

            # Edges can attend to their corresponding nodes
            base_attention_mask[0, edge_idx, start_node] = 1
            base_attention_mask[0, edge_idx, end_node] = 1

            # Nodes can attend to their corresponding edges
            base_attention_mask[0, start_node, edge_idx] = 1
            base_attention_mask[0, end_node, edge_idx] = 1

        # Triangles can attend to all nodes in the graph
        if triangles is not None:
            for triangle_idx, triangle in enumerate(triangles):
                triangle_offset = n_nodes + num_edges_per_graph
                node1, node2, node3 = triangle
                tri_idx = triangle_offset + triangle_idx

                # Triangles can attend to their nodes
                base_attention_mask[0, tri_idx, node1] = 1
                base_attention_mask[0, tri_idx, node2] = 1
                base_attention_mask[0, tri_idx, node3] = 1

                # Nodes can attend to their corresponding triangles
                base_attention_mask[0, node1, tri_idx] = 1
                base_attention_mask[0, node2, tri_idx] = 1
                base_attention_mask[0, node3, tri_idx] = 1

        # Triangles can attend to edges and edges can attend to triangles
        edges_in_triangle = [
            (node1, node2),
            (node2, node3),
            (node3, node1)
        ]
        for edge in edges_in_triangle:
            edge_idx = n_nodes + edges[0].tolist().index(edge[0]) + edges[1].tolist().index(edge[1])
            # Triangle to edge
            base_attention_mask[0, tri_idx, edge_idx] = 1
            # Edge to triangle
            base_attention_mask[0, edge_idx, tri_idx] = 1

        # Convert the mask to float and set masked positions to -inf and allowed positions to 0
        attention_mask = base_attention_mask.float()
        attention_mask[0] = attention_mask[0].masked_fill(attention_mask == 0, float('-inf'))
        attention_mask[0] = attention_mask[0].fill_diagonal_(1)
        attention_mask[0] = attention_mask[0].masked_fill(attention_mask == 1, float(0.0))

        # Stack the masks for each batch
        attention_mask = base_attention_mask.repeat(batch_size, 1, 1)

        return attention_mask