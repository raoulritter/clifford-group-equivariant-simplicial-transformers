import torch
import torch.nn as nn
from positional_encoding import PositionalEncoding
from attention import GAST
from models.modules.linear import MVLinear
from algebra.cliffordalgebra import CliffordAlgebra
clifford_algebra = CliffordAlgebra([1, 1, 1])

class NBODY_Transformer(nn.Module):
    def __init__(self, input_dim, d_model, num_heads, num_layers, batch_size, clifford_algebra, channels):
        super(NBODY_Transformer, self).__init__()
        self.positional_encoding = PositionalEncoding(d_model, batch_size)
        self.GAST = GAST(num_layers=num_layers, num_heads=num_heads, channels=channels, num_nodes=5, num_edges=20,
                         clifford_algebra=clifford_algebra)
        self.MVinput = MVLinear(clifford_algebra, input_dim, d_model, subspaces=True)
        self.MVGP = MVLinear(clifford_algebra, d_model * 2, d_model, subspaces=True)
        self.n_nodes = 5
        self.n_edges = 20

    def forward(self, nodes, edges, src_mask, batch_size):
        # POSITIONAL ENCODING left out for now
        # edges_in_clifford = self.positional_encoding(edges_in_clifford)

        # Reshape nodes to [batch_size, n_nodes, n_features, feature_dim]
        nodes = nodes.view(batch_size, self.n_nodes, nodes.size(1), nodes.size(2))

        # Reshape edges to [batch_size, n_edges, n_features, feature_dim]
        edges = edges.view(batch_size, self.n_edges, edges.size(1), edges.size(2))

        combined = torch.cat((nodes, edges), dim=1)  # Should be [batch_size, 25, 7, 8]
        combined = combined.view(batch_size * (self.n_nodes + self.n_edges), combined.size(2),
                                 combined.size(3))  # Should be [batch_size*25, 7, 8]
        src = combined
        # src = torch.cat((nodes, edges), dim=0)

        src_MV = self.MVinput(src)
        src_GP = clifford_algebra.geometric_product(src_MV, src_MV)
        src_cat = torch.cat((src_MV, src_GP), dim=1)
        src = self.MVGP(src_cat)

        enc_output = self.GAST(src, src_mask)
        output = enc_output

        # Reshape the tensor to [batch_size, total_elements, 7, 8]
        reshaped_output = output.view(batch_size, self.n_edges + self.n_nodes, 7, 8)
        nodes = reshaped_output[:, :self.n_nodes, :, :]
        selected_feature = nodes[:, :, 1, :]
        selected_feature = selected_feature.reshape(batch_size * self.n_nodes, 8)

        # return only nodes and only the "pos" feature vector of the nodes
        return selected_feature
