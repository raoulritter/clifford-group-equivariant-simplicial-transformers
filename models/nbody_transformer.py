import torch
import torch.nn as nn
from positional_encoding import PositionalEncoding
from attention import GAST
from models.modules.linear import MVLinear

class NBODY_Transformer(nn.Module):
    def __init__(self, input_dim, d_model, num_heads, num_layers, batch_size, embed_in_features, embed_out_features, clifford_algebra, channels):
        super(NBODY_Transformer, self).__init__()
        self.positional_encoding = PositionalEncoding(d_model, batch_size)
        self.GAST = GAST(num_layers, d_model, num_heads, clifford_algebra, channels)
        self.embedding = MVLinear(clifford_algebra, embed_in_features, embed_out_features, subspaces=False)
        self.MVinput = MVLinear(clifford_algebra, input_dim, d_model, subspaces=True)
        self.MVGP = MVLinear(clifford_algebra, d_model * 2, d_model, subspaces=True)
        self.clifford_algebra = clifford_algebra

    def forward(self, nodes_in_clifford, edges_in_clifford, src_mask, batch_size):
        src = torch.cat((nodes_in_clifford, edges_in_clifford), dim=0)
        src_MV = self.MVinput(src)
        src_GP = self.clifford_algebra.geometric_product(src_MV, src_MV)

        src_cat = torch.cat((src_MV, src_GP), dim=1)
        src = self.MVGP(src_cat)

        enc_output = self.GAST(src, src_mask)
        output = enc_output

        # return only nodes and only the "pos" feature vector of the nodes
        return output[:(5 * batch_size), 1, :]
