import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from models.modules.linear import MVLinear
from models.modules.mvlayernorm import MVLayerNorm
from models.modules.mvsilu import MVSiLU
from einops import rearrange

class SelfAttentionClifford(nn.Module):
    def __init__(self, num_feat, num_nodes, num_edges, algebra, num_heads=8):
        super(SelfAttentionClifford, self).__init__()
        self.num_feat = num_feat
        self.num_nodes = num_nodes
        self.num_edges = num_edges
        self.algebra = algebra
        self.num_heads = num_heads
        self.q_linear = MVLinear(algebra, num_feat, num_feat * num_heads, subspaces=True)
        self.k_linear = MVLinear(algebra, num_feat, num_feat * num_heads, subspaces=True)
        self.v_linear = MVLinear(algebra, num_feat, num_feat * num_heads, subspaces=True)
        self.output_embedding = MVLinear(algebra, num_feat * num_heads, num_feat, subspaces=True)
        self.cat_output_embedding = MVLinear(algebra, num_feat * 2, num_feat, subspaces=True)
        self.concat_layernorm = MVLayerNorm(algebra, num_feat)

    def forward(self, feature_matrix, attention_mask):
        bs = feature_matrix.size(0) // (self.num_nodes + self.num_edges)

        # Compute query, key, and value matrices
        # feature_matrix -> [batch_size * (n_nodes + n_edges), d_model*2, 8]

        # Compute query, key, and value matrices using einops rearrange
        q = rearrange(self.q_linear(feature_matrix), '(bs n) (h d) c -> bs h n (d c)', bs=bs,
                      n=self.num_nodes + self.num_edges, h=self.num_heads, d=self.num_feat)
        k = rearrange(self.k_linear(feature_matrix), '(bs n) (h d) c -> bs h n (d c)', bs=bs,
                      n=self.num_nodes + self.num_edges, h=self.num_heads, d=self.num_feat)
        v = rearrange(self.v_linear(feature_matrix), '(bs n) (h d) c -> bs h n (d c)', bs=bs,
                      n=self.num_nodes + self.num_edges, h=self.num_heads, d=self.num_feat)
        # q, k, v -> [batch_size, num_heads, n_nodes + n_edges, d_model * 8]

        # Compute dot product for attention
        q = q / math.sqrt(self.num_feat * 8)  # Scale by sqrt(d_k * 8) 8 from CLIFFORD
        attn = torch.matmul(q, k.transpose(-2, -1)) # multiple q and k -> [batch_size, num_heads, n_nodes + n_edges, n_nodes + n_edges]
        #attn = attn + attention_mask.unsqueeze(1).unsqueeze(2)

        # Adjust the attention mask
        attention_mask = attention_mask.unsqueeze(1).repeat(1, self.num_heads, 1,
                                                            1)  # Shape: [batch_size, num_heads, n_nodes + n_edges, n_nodes + n_edges]
        #attn = attn + attention_mask  # Apply the mask
        attn = F.softmax(attn, dim=-1)

        # Apply attention to value
        attention_output = torch.matmul(attn, v) # [batch_size, num_heads, n_nodes + n_edges, d_model, 8]

        attention_output = attention_output.transpose(1, 2).contiguous().view(bs*(self.num_nodes + self.num_edges), self.num_heads * self.num_feat, 8)
        # attention_output -> [batch_size * (n_nodes + n_edges), d_model*num_heads, 8]
        output = self.output_embedding(attention_output)
        # output -> [batch_size * (n_nodes + n_edges), d_model, 8]

        gp_output = self.algebra.geometric_product(output, output)

        output_cat = torch.cat((output, gp_output), dim=1)

        # Output linear transformation
        output = self.cat_output_embedding(output_cat)
        # output -> [batch_size * (n_nodes + n_edges), d_model, 8]

        #TODO: CHECK GEOMETRIC PRODUCT
        # # Apply geometric product to attention output
        # gp_output = self.algebra.geometric_product(attention_output, attention_output)
        # print("gp_output", gp_output.shape)
        # # rearrange output
        # output = output.view(bs * (self.num_nodes + self.num_edges), self.num_feat, 8)
        # output = self.concat_layernorm(output)


        return output


class TransformerBlock(nn.Module):
    def __init__(self, d_model, num_heads, clifford_algebra):
        super(TransformerBlock, self).__init__()
        self.mvlayernorm1 = MVLayerNorm(clifford_algebra, d_model )
        self.self_attn = SelfAttentionClifford(d_model, 5, 20, clifford_algebra, num_heads)
        self.mvlayernorm2 = MVLayerNorm(clifford_algebra, d_model)
        self.mvlayernorm3 = MVLayerNorm(clifford_algebra, d_model)
        self.mlp = nn.Sequential(
            MVLinear(clifford_algebra, d_model, d_model * 4),
            MVSiLU(clifford_algebra, d_model * 4), # replace with geometric product
            MVLinear(clifford_algebra, d_model * 4, d_model) # CE MLP instead of MLP
        )
        # self.dropout = TBD

    def forward(self, src, src_mask=None):
        # src -> [batch_size * (n_nodes + n_edges), d_model*2, 8]
        src_norm1 = self.mvlayernorm1(src)
        # Self-attention
        attended_src = self.self_attn(src_norm1, src_mask)

        # Add and norm
        src = src + attended_src

        src = self.mvlayernorm2(src)

        #TODO: CHECK WITH THE CEMLP INSTEAD??
        ff_src = self.mlp(src)
        # ff_src = self.dropout(ff_src)

        # Add and norm
        src = src + ff_src
        src = self.mvlayernorm3(src)

        return src


class MainBody(nn.Module):
    def __init__(self, num_layers, d_model, num_heads, clifford_algebra):
        super(MainBody, self).__init__()
        self.layers = nn.ModuleList(
            [TransformerBlock(d_model, num_heads, clifford_algebra) for _ in range(num_layers)])

    def forward(self, src, src_mask=None):
        for layer in self.layers:
            src = layer(src, src_mask)
        return src
