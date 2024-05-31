import torch.nn as nn
from ..original_modules.linear import MVLinear
from ..modules.clifford_embedding import NBodyGraphEmbedder
from ..original_modules.mvsilu import MVSiLU
from ..modules.block import MainBody


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
    def __init__(self, input_dim, d_model, num_heads, num_layers, clifford_algebra, simplex_order=1, empty_higher_simplices=False):
        super().__init__()
        self.clifford_algebra = clifford_algebra
        self.simplex_order = simplex_order
        self.d_model = d_model

        # Initialize embedding and transformer layers
        self.embedding_layer = NBodyGraphEmbedder(clifford_algebra, input_dim, d_model, simplex_order, empty_higher_simplices)
        self.transformer = MainBody(num_layers, d_model, num_heads, clifford_algebra, simplex_order)
        self.combined_projection = TwoLayerMLP(clifford_algebra, d_model, d_model * 4, d_model)
        self.x_left = MVLinear(clifford_algebra, d_model, d_model, subspaces=True)

    def forward(self, batch):
        batch_size, n_nodes, _ = batch[0].size()
        loc_start, loc_end = batch[0], batch[4]

        # Generate embeddings and attention mask
        full_embeddings, attention_mask = self.embedding_layer.embed_nbody_graphs(batch)

        # Apply transformation to embeddings
        src = self.combined_projection(full_embeddings.view(batch_size * (n_nodes + self.simplex_order), self.d_model, 8))
        #src_left = self.x_left(src)
        #src = self.clifford_algebra.geometric_product(src_left, src)

        # Pass through transformer layers
        output = self.transformer(src, attention_mask)
        output = output.view(batch_size, n_nodes + self.simplex_order, self.d_model, 8)

        # Compute new positions
        output_locations = output[:, :n_nodes, 1, 1:4].squeeze(2)
        new_pos = loc_start + output_locations

        return new_pos, loc_end
