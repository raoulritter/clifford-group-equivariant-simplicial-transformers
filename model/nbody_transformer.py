import torch
import torch.nn as nn
from models.modules.linear import MVLinear
from models.modules.clifford_embedding import NBodyGraphEmbedder
from models.modules.mvsilu import MVSiLU
from model.clifford_embedding import NBodyGraphEmbedder
from model.block import MainBody


class TwoLayerMLP(nn.Module):
    def __init__(self, algebra, in_features_v, hidden_features_v, out_features_v):
        super().__init__()
        self.layer = nn.Sequential(
            MVLinear(algebra, in_features_v, hidden_features_v, subspaces=True, bias=True),
            MVSiLU(algebra, hidden_features_v),
            MVLinear(algebra, hidden_features_v, out_features_v, subspaces=True, bias=True),
        )

    def forward(self, v):
        return self.layer(v)


class NBodyTransformer(nn.Module):
    def __init__(self, input_dim, d_model, num_heads, num_layers,
                 clifford_algebra, num_edges=10, zero_edges=False):
        super(NBodyTransformer, self).__init__()

        # Initialize the transformer with the given parameters
        # and the Clifford algebra
        self.clifford_algebra = clifford_algebra
        self.num_edges = num_edges
        self.d = d_model
        # Initialize the embedding layer
        self.embedding_layer = NBodyGraphEmbedder(self.clifford_algebra, in_features=input_dim,
                                                  embed_dim=d_model, num_edges=num_edges, zero_edges=zero_edges)
        self.GAST = MainBody(num_layers, d_model, num_heads, self.clifford_algebra, num_edges=num_edges)
        self.combined_projection = TwoLayerMLP(self.clifford_algebra, d_model, d_model*4, d_model)
        self.x_left = MVLinear(self.clifford_algebra, d_model, d_model, subspaces=True)

    def forward(self, batch):
        batch_size, n_nodes, _ = batch[0].size()
        loc_end = batch[4]
        loc_start = batch[0]

        # Generate node and edge embeddings along with the attention mask add back attention mask at smoe point please
        full_embeddings, attention_mask = self.embedding_layer.embed_nbody_graphs(
            batch)

        # Apply MVLinear transformation to the combined embeddings
        src = self.combined_projection(full_embeddings.reshape(batch_size * (n_nodes + self.num_edges), self.d, 8))
        # src -> [batch_size * (n_nodes + n_edges), d_model, 8]
        src_left = self.x_left(src)
        #src = self.clifford_algebra.geometric_product(src_left, src)

        # Pass through GAST layers
        output = self.GAST(src, attention_mask)
        output = output.reshape(batch_size, n_nodes + self.num_edges, self.d, 8)

        output_locations = output[:,:n_nodes, 1, 1:4].squeeze(2)
        new_pos = loc_start + output_locations

        return new_pos, loc_end