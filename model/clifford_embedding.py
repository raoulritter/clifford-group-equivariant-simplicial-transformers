import torch
from models.modules.linear import MVLinear


class NBodyGraphEmbedder:
    def __init__(self, clifford_algebra, in_features, embed_dim, num_edges=10, zero_edges=True):
        self.clifford_algebra = clifford_algebra
        self.node_projection = MVLinear(clifford_algebra, in_features, embed_dim, subspaces=False)
        self.edge_projection = MVLinear(clifford_algebra, 7, embed_dim, subspaces=False)
        self.embed_dim = embed_dim
        self.zero_edges = zero_edges
        self.num_edges = num_edges
        self.with_edges, self.unique_edges = self._determine_edge_settings(num_edges)
        self.batch_size = None

    def _determine_edge_settings(self, num_edges):
        if num_edges == 10:
            return True, True
        elif num_edges == 20:
            return True, False
        else:
            return False, False

    def embed_nbody_graphs(self, batch):
        batch_size, n_nodes, _ = batch[0].size()
        self.batch_size = batch_size
        loc_mean, vel, edge_attr, charges, edges = self.preprocess(batch)

        # Embed into Clifford space
        invariants = self.clifford_algebra.embed(charges, (0,))
        covariants = self.clifford_algebra.embed(torch.stack([loc_mean, vel], dim=1), (1, 2, 3))
        nodes_stack = torch.cat([invariants[:, None], covariants], dim=1)
        nodes_stack[:, 1, 0] += 1
        full_node_embedding = self.node_projection(nodes_stack)

        if self.with_edges:
            start_nodes, end_nodes, indices = self.get_edge_nodes(edges, n_nodes, batch_size)
            if indices is not None:
                edge_attr = edge_attr[:, indices, :]
            full_edge_embedding = self.get_full_edge_embedding(edge_attr, nodes_stack, (start_nodes, end_nodes))
            attention_mask = self.get_attention_mask(batch_size, n_nodes, (start_nodes, end_nodes))
            full_embedding = torch.cat(
                (full_node_embedding.view(batch_size, n_nodes, self.embed_dim, 8),
                 full_edge_embedding.view(batch_size, self.num_edges, self.embed_dim, 8)),
                dim=1)
            return full_embedding, attention_mask
        else:
            return full_node_embedding, None

    def preprocess(self, batch):
        loc, vel, edge_attr, charges, _, edges = batch
        loc_mean = self.compute_mean_centered(loc)
        return (loc_mean.view(-1, *loc_mean.shape[2:]), vel.view(-1, *vel.shape[2:]),
                edge_attr, charges.view(-1, *charges.shape[2:]), edges)

    def compute_mean_centered(self, tensor):
        return tensor - tensor.mean(dim=1, keepdim=True)

    def get_edge_nodes(self, edges, n_nodes, batch_size):
        batch_index = torch.arange(batch_size, device=edges.device)
        edges = edges + n_nodes * batch_index[:, None, None]
        if self.unique_edges:
            edges, indices = self.get_unique_edges_with_indices(edges[0])
            edges = edges.unsqueeze(0).repeat(batch_size, 1, 1)
            edges = tuple(edges.transpose(0, 1).flatten(1))
            return edges[0], edges[1], indices
        else:
            edges = tuple(edges.transpose(0, 1).flatten(1))
            return edges[0], edges[1], None

    def get_full_edge_embedding(self, edge_attr, nodes_stack, edges):
        if self.zero_edges:
            return torch.zeros(self.batch_size, self.num_edges, self.embed_dim, 8)
        edge_attr = edge_attr.view(-1, *edge_attr.shape[2:])  # Flatten the edge attributes
        orig_edge_attr_clifford = self.clifford_algebra.embed(edge_attr[..., None], (0,)).view(-1, 1, 8)
        extra_edge_attr_clifford = self.make_edge_attr(nodes_stack, edges)
        edge_attr_all = torch.cat((orig_edge_attr_clifford, extra_edge_attr_clifford), dim=1)
        return self.edge_projection(edge_attr_all)

    def make_edge_attr(self, node_features, edges):
        node1_features, node2_features = node_features[edges[0]], node_features[edges[1]]
        gp = self.clifford_algebra.geometric_product(node1_features, node2_features)
        gp += self.clifford_algebra.geometric_product(node2_features, node1_features)
        return torch.cat((node1_features + node2_features, gp), dim=1)

    def get_unique_edges_with_indices(self, tensor):
        unique_edges = []
        unique_indices = []
        for i, edge in enumerate(tensor.t()):
            node1, node2 = sorted(edge.tolist())
            if (node1, node2) not in unique_edges:
                unique_edges.append((node1, node2))
                unique_indices.append(i)
        return torch.tensor(unique_edges).t(), torch.tensor(unique_indices)

    def get_attention_mask(self, batch_size, n_nodes, edges):
        num_edges_per_graph = edges[0].size(0) // batch_size
        base_attention_mask = torch.zeros(1, n_nodes + num_edges_per_graph, n_nodes + num_edges_per_graph,
                                          device=edges[0].device)
        base_attention_mask[0, :n_nodes, :n_nodes] = 1
        for i in range(num_edges_per_graph):
            start_node, end_node, edge_idx = edges[0][i].item(), edges[1][i].item(), n_nodes + i
            base_attention_mask[0, edge_idx, [start_node, end_node]] = 1
            base_attention_mask[0, [start_node, end_node], edge_idx] = 1

        attention_mask = base_attention_mask.float()
        attention_mask[0] = attention_mask[0].masked_fill(attention_mask == 0, float('-inf'))
        attention_mask[0] = attention_mask[0].masked_fill(attention_mask == 1, float(0.0))
        attention_mask[0] = attention_mask[0].fill_diagonal_(0)

        return attention_mask.repeat(batch_size, 1, 1)
