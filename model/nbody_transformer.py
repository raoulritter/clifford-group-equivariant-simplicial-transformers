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
                 clifford_algebra, unique_edges=False):
        super(NBodyTransformer, self).__init__()
        self.clifford_algebra = clifford_algebra
        self.embedding_layer = NBodyGraphEmbedder(self.clifford_algebra, in_features=input_dim, embed_dim=d_model, unique_edges=unique_edges)
        self.GAST = MainBody(num_layers, d_model, num_heads, self.clifford_algebra, unique_edges=unique_edges)
        self.combined_projection = TwoLayerMLP(self.clifford_algebra, d_model, d_model*4, d_model)
        self.MV_input = MVLinear(self.clifford_algebra, input_dim, d_model, subspaces=True)

    def forward(self, batch):
        batch_size, n_nodes, _ = batch[0].size()
        loc_end = batch[4]
        loc_start = batch[0]

        # Generate node and edge embeddings along with the attention mask add back attention TODO add mask back
        full_embeddings, attention_mask = self.embedding_layer.embed_nbody_graphs(
            batch)

        # Apply MVLinear transformation to the combined embeddings
        src = self.combined_projection(full_embeddings)
        # src -> [batch_size * (n_nodes + n_edges), d_model, 8]

        # Pass through GAST layers
        output = self.GAST(src, attention_mask)

        output_locations = output[:(5 * batch_size), 1, 1:4]
        new_pos = loc_start + output_locations.view(batch_size, 5, 3)

        return new_pos, loc_end