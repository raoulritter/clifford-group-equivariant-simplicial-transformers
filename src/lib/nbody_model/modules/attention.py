import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from ..original_modules.linear import MVLinear
# from ..linear import MVLinear
# from ....models.modules.linear import MVLinear
from einops import rearrange


class SelfAttentionClifford(nn.Module):
    def __init__(self, num_feat, num_nodes, num_edges, algebra, num_heads=8):
        super(SelfAttentionClifford, self).__init__()
        self.num_feat = num_feat
        self.num_nodes = num_nodes
        self.num_edges = num_edges
        self.algebra = algebra
        self.num_heads = num_heads
        self.head_dim = num_feat // num_heads
        self.q_linear = MVLinear(algebra, self.num_feat, self.num_feat, subspaces=True, bias=False)
        self.k_linear = MVLinear(algebra, self.num_feat, self.num_feat, subspaces=True, bias=False)
        self.v_linear = MVLinear(algebra, self.num_feat, self.num_feat, subspaces=True, bias=False)
        self.output_embedding = MVLinear(algebra, self.head_dim * self.num_heads, num_feat, subspaces=True)

    def forward(self, feature_matrix, attention_mask, test=False):
        bs = feature_matrix.size(0) // (self.num_nodes + self.num_edges)
        n = self.num_nodes + self.num_edges

        # Compute query, key, and value using mv linear layers
        q = self.q_linear(feature_matrix)  # q -> [batch_size, n_nodes + n_edges, num_heads, d_model//num_heads 8]
        k = self.k_linear(feature_matrix)  # k -> [batch_size, n_nodes + n_edges, num_heads, d_model//num_heads, 8]
        v = self.v_linear(feature_matrix)

        # rearrange to separate heads and then fold heads into batch dimension
        q = rearrange(q, '(bs n) (h d) c ->  (bs h) n (d c)', bs=bs, n=n, h=self.num_heads, d=self.head_dim)
        k = rearrange(k, '(bs n) (h d) c -> (bs h) n (d c)', bs=bs, n=n, h=self.num_heads, d=self.head_dim)
        v = rearrange(v, '(bs n) (h d) c -> (bs h) n (d c)', bs=bs, n=n, h=self.num_heads, d=self.head_dim)

        # q, k, v -> [batch_size * num_heads, n_nodes + n_edges, head_dim*8]

        # Compute dot product for attention
        q = q / math.sqrt(self.head_dim * 8)  # Scale by sqrt(d_k * 8) 8 from CLIFFORD
        attn = torch.bmm(q, k.transpose(-2, -1)) # multiple q and k -> [batch_size * num_heads, n_nodes + n_edges, n_nodes + n_edges]

        # Adjust the attention mask
        if attention_mask is not None:
            attention_mask = attention_mask.unsqueeze(1).repeat(1, self.num_heads, 1,
                                                            1).view(-1, n, n)  # Shape: [batch_size, num_heads, n_nodes + n_edges, n_nodes + n_edges]
            attn = attn + attention_mask  # Apply the mask

        attn = F.softmax(attn, dim=-1)

        if test:
            return attn
        else:
            # Apply attention to value
            attention_output = torch.bmm(attn, v) # -> [batch_size * num_heads, n_nodes + n_edges, self.head_dim*8] should this be bmm?
            # rearrange to separate heads and then fold heads into feature dimension
            attention_output = rearrange(attention_output, '(bs h) n (d c) -> (bs n) (h d) c', bs=bs, n=n, h=self.num_heads, d=self.head_dim, c=8)

            output = self.output_embedding(attention_output)

            return output

