import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from original_models.modules.linear import MVLinear
from original_models.modules.mvlayernorm import MVLayerNorm
from original_models.modules.mvsilu import MVSiLU
from einops import rearrange
from transformer.modules.attention import SelfAttentionClifford


class GP_Layer(nn.Module):
        def __init__(self, algebra, in_features_v, hidden_features_v):
            super().__init__()
            self.first_layer = MVLinear(algebra, in_features_v, hidden_features_v, subspaces=True, bias=True)
            self.third_layer = MVLinear(algebra, hidden_features_v, in_features_v, subspaces=True, bias=True)
            self.norm = MVLayerNorm(algebra, in_features_v)
            self.algebra = algebra

        def forward(self, x):
            x = self.first_layer(x)
            x = self.algebra.geometric_product(x, x)
            x = self.third_layer(x)
            x = self.norm(x)

            return x


class TransformerBlock(nn.Module):
    def __init__(self, d_model, num_heads, clifford_algebra, unique_edges=False):
        super(TransformerBlock, self).__init__()
        if unique_edges:
            num_edges = 10
        else:
            num_edges = 20
        self.algebra = clifford_algebra
        self.mvlayernorm1 = MVLayerNorm(clifford_algebra, d_model)
        self.self_attn = SelfAttentionClifford(d_model, 5, num_edges, clifford_algebra, num_heads)
        self.mvlayernorm2 = MVLayerNorm(clifford_algebra, d_model)
        self.mvlayernorm3 = MVLayerNorm(clifford_algebra, d_model)
        self.mlp = nn.Sequential(
            MVLinear(clifford_algebra, d_model, d_model * 2),
            MVSiLU(clifford_algebra, d_model * 2),
            MVLinear(clifford_algebra, d_model * 2, d_model)
        )
        self.gp = GP_Layer(clifford_algebra, d_model, d_model * 2)
        # self.dropout = TBD

    def forward(self, src, src_mask=None):
        # src -> [batch_size * (n_nodes + n_edges), d_model*2, 8]
        # Norm
        src_norm1 = self.mvlayernorm1(src)
        # Self-attention
        attended_src = self.self_attn(src_norm1, src_mask)

        # Add and norm
        src = src + attended_src
        src = self.mvlayernorm2(src)

        # geo prod layer
        gp_src = self.gp(src)
        src = src + gp_src
        src = self.mvlayernorm3(src)

        # MLP
        ff_src = self.mlp(src)

        # Add and norm
        src = src + ff_src

        return src


class MainBody(nn.Module):
    def __init__(self, num_layers, d_model, num_heads, clifford_algebra, unique_edges=False):
        super(MainBody, self).__init__()
        self.layers = nn.ModuleList(
            [TransformerBlock(d_model, num_heads, clifford_algebra, unique_edges=unique_edges) for _ in range(num_layers)])

    def forward(self, src, src_mask=None):
        for layer in self.layers:
            src = layer(src, src_mask)
        return src
