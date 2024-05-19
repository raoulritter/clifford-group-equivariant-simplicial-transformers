import torch
import torch.nn as nn
from algebra.cliffordalgebra import CliffordAlgebra
from models.modules.linear import MVLinear
from models.modules.clifford_embedding import NBodyGraphEmbedder
from attention import MainBody


class NBodyTransformer(nn.Module):
    def __init__(self, input_dim, d_model, num_heads, num_layers,
                 clifford_algebra):
        super(NBodyTransformer, self).__init__()
        self.clifford_algebra = clifford_algebra
        self.embedding_layer = NBodyGraphEmbedder(self.clifford_algebra, in_features=input_dim, embed_dim=d_model)
        self.d_model = d_model

        self.GAST = MainBody(num_layers, d_model, num_heads, self.clifford_algebra)
        self.combined_projection = MVLinear(self.clifford_algebra, d_model*2, d_model, subspaces=True)
        self.MV_input = MVLinear(self.clifford_algebra, input_dim, d_model, subspaces=True)
        self.MV_GP = MVLinear(self.clifford_algebra, d_model * 2, d_model, subspaces=True)

    def forward(self, batch):
        batch_size, n_nodes, _ = batch[0].size()

        # Generate node and edge embeddings along with the attention mask add back attention mask at smoe point please
        node_embeddings, edge_embeddings, loc_end_clifford, attention_mask, loc_mean = self.embedding_layer.embed_nbody_graphs(
            batch)

        # nodes -> [batch_size * n_nodes, d_model/2, 8]
        # edges -> [batch_size * n_edges, d_model, 8]
        combined = torch.cat((node_embeddings, edge_embeddings), dim=0)
        combined_gp = self.clifford_algebra.geometric_product(combined, combined)  # do we also for edges?
        #edges = self.clifford_algebra.geometric_product(edge_embeddings, edge_embeddings)

        src_cat = torch.cat((combined, combined_gp), dim=1)
        # Combine nodes and edges after projection

        #nodes = torch.cat((nodes, edge_embeddings), dim=0)
        src = self.combined_projection(src_cat) # check shapes

        # src -> [batch_size * (n_nodes + n_edges), d_model, 8]

        output = self.GAST(src, attention_mask)
        positions = output[:(5 * batch_size), 1, :]
        embedded_loc_mean = self.clifford_algebra.embed(loc_mean, (1, 2, 3))
        pred_end_positions = embedded_loc_mean + positions


        return pred_end_positions, loc_end_clifford
