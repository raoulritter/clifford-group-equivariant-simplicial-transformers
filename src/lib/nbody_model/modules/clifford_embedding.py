import torch
from ..original_modules.linear import MVLinear

class NBodyGraphEmbedder:
    def __init__(self, clifford_algebra, in_features, embed_dim, num_edges=10, zero_edges=True):
        self.clifford_algebra = clifford_algebra
        self.node_projection = MVLinear(
            self.clifford_algebra, in_features, embed_dim, subspaces=False
        )
        self.edge_projection = MVLinear(
            self.clifford_algebra, 7, embed_dim, subspaces=False
        )
        self.embed_dim = embed_dim
        self.zero_edges = zero_edges
        self.num_edges = num_edges
        if num_edges == 10:
            self.unique_edges = True
            self.with_edges = True
        elif num_edges == 20:
            self.unique_edges = False
            self.with_edges = True
        else:
            assert num_edges == 0
            self.unique_edges = False
            self.with_edges = False

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
        if self.with_edges:
            full_edge_embedding, edges = self.get_full_edge_embedding(edge_attr, nodes_stack, edges, n_nodes, batch_size)
            attention_mask = self.get_attention_mask(batch_size, n_nodes, edges)
            full_embedding = torch.cat((full_node_embedding.reshape(batch_size, n_nodes, self.embed_dim, 8),
                                        full_edge_embedding.reshape(batch_size, self.num_edges, self.embed_dim, 8)),
                                       dim=1)
            return full_embedding, attention_mask
        else:
            return full_node_embedding, None



    def get_full_edge_embedding(self,edge_attr, nodes_stack, edges, n_nodes, batch_size):
        edges, indices = self.get_edge_nodes(edges, n_nodes, batch_size)
        start_nodes = edges[0]
        end_nodes = edges[1]
        if self.zero_edges:
            return torch.zeros((batch_size, self.num_edges, self.embed_dim,8)), (start_nodes, end_nodes)
        else:
            if self.unique_edges:
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
        if self.unique_edges:
            edges, indices = self.get_unique_edges_with_indices(edges[0])
            edges = edges.unsqueeze(0).repeat(batch_size, 1, 1)
            edges = tuple(edges.transpose(0, 1).flatten(1))
            return edges, indices
        else:
            return tuple(edges.transpose(0, 1).flatten(1)), None

    def get_edge_embedding(self, edge_attr, nodes_in_clifford, edges):
        if self.unique_edges:
            orig_edge_attr_clifford = self.clifford_algebra.embed(edge_attr[..., None], (0,)).view(-1, 1, 8)
        else:
            edge_attr = self.flatten_tensors(edge_attr)[0]  # [batch * edges, dim]
            orig_edge_attr_clifford = self.clifford_algebra.embed(edge_attr[..., None], (0,))

        extra_edge_attr_clifford = self.make_edge_attr(nodes_in_clifford, edges)
        edge_attr_all = torch.cat((orig_edge_attr_clifford, extra_edge_attr_clifford), dim=1)

        projected_edges = self.edge_projection(edge_attr_all)
        return projected_edges

    def make_edge_attr(self, node_features, edges):
        node1_features = node_features[edges[0]]
        node2_features = node_features[edges[1]]
        gp = self.clifford_algebra.geometric_product(node1_features, node2_features)
        gp2 = self.clifford_algebra.geometric_product(node2_features, node1_features)
        gp += gp2
        edge_attributes = torch.cat((node1_features + node2_features,gp), dim=1) # changed
        return edge_attributes

    def get_unique_edges_with_indices(self, tensor):
        edges = set()  # edges before
        unique_edges = []
        unique_indices = []

        for i, edge in enumerate(tensor.t()):
            node1, node2 = sorted(edge.tolist())
            if (node1, node2) not in edges:
                edges.add((node1, node2))
                unique_edges.append((node1, node2))
                unique_indices.append(i)

        unique_edges_tensor = torch.tensor(unique_edges).t()
        unique_indices_tensor = torch.tensor(unique_indices)

        return unique_edges_tensor, unique_indices_tensor

    def get_attention_mask(self, batch_size, n_nodes, edges):
        num_edges_per_graph = edges[0].size(0) // batch_size

        # Initialize an attention mask with zeros for a single batch
        base_attention_mask = torch.zeros(1, n_nodes + num_edges_per_graph, n_nodes + num_edges_per_graph,
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


        # Convert the mask to float and set masked positions to -inf and allowed positions to 0
        attention_mask = base_attention_mask.float()
        attention_mask[0] = attention_mask[0].masked_fill(attention_mask == 0, float('-inf'))
        attention_mask[0] = attention_mask[0].fill_diagonal_(1)
        attention_mask[0] = attention_mask[0].masked_fill(attention_mask == 1, float(0.0))

        # Stack the masks for each batch
        attention_mask = base_attention_mask.repeat(batch_size, 1, 1)

        return attention_mask