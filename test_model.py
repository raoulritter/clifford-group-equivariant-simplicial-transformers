import unittest
import torch
import torch.nn.functional as F
from torch import nn
from einops import rearrange
import math
from transformer.modules.attention import SelfAttentionClifford
from algebra.cliffordalgebra import CliffordAlgebra
from transformer.transformer import NBodyTransformer


# Assuming MVLinear and MVLayerNorm are defined elsewhere, import them as well
# from your_module import MVLinear, MVLayerNorm, SelfAttentionClifford
class TestModules(unittest.TestCase):

    def setUp(self):
        self.num_feat = 128
        self.num_nodes = 10
        self.num_edges = 5

        metric = [1, 1, 1]

        self.algebra = CliffordAlgebra(metric)
        self.num_heads = 8
        self.batch_size = 2
        self.model = SelfAttentionClifford(self.num_feat, self.num_nodes, self.num_edges, self.algebra, self.num_heads)

        # Create a random feature matrix and attention mask
        # src -> [batch_size * (n_nodes + n_edges), d_model, 8]
        self.feature_matrix = torch.randn((self.batch_size * (self.num_nodes + self.num_edges), self.num_feat, 8))
        self.attention_mask = torch.zeros(
            (self.batch_size, self.num_nodes + self.num_edges, self.num_nodes + self.num_edges))

    def test_output_shape(self):
        output = self.model(self.feature_matrix, self.attention_mask)
        expected_shape = (self.batch_size * (self.num_nodes + self.num_edges), self.num_feat, 8)
        self.assertEqual(output.shape, expected_shape, "Output shape is incorrect")

    def test_attention_values(self):
        output = self.model(self.feature_matrix, self.attention_mask, test=True)
        # Here we can add checks to validate attention values if needed
        # For example, you could ensure that attention values are non-negative and sum to 1 along the appropriate dimension
        # Ensure attention values are non-negative
        self.assertTrue(torch.all(output >= 0), "Attention values should be non-negative")

        # Ensure attention values sum to 1 along the last dimension
        attn_sum = torch.sum(output, dim=-1)
        self.assertTrue(torch.allclose(attn_sum, torch.ones_like(attn_sum)),
                        "Attention values should sum to 1 along the last dimension")

    def test_model_equivariance(self):
        metric = [1, 1, 1]
        clifford_algebra = CliffordAlgebra(metric)

        # Hyperparameters
        input_dim = 3  # feature_dim
        d_model = 16
        num_heads = 8
        num_layers = 6

        # Create the model
        feature_embedding = NBodyTransformer(input_dim, d_model, num_heads, num_layers, clifford_algebra)

        random_nodes_pos = torch.randn((1, 5, 3))
        random_nodes_pos = random_nodes_pos - random_nodes_pos.mean(dim=1, keepdim=True)
        random_nodes_vel = torch.randn((1, 5, 3))
        random_nodes_vel = random_nodes_vel - random_nodes_vel.mean(dim=1, keepdim=True)
        random_nodes_charges = torch.randn((1, 5, 1))
        algebra = CliffordAlgebra((1, 1, 1))
        edge_attr = torch.randn((1, 20, 1))
        edges = torch.tensor([[[0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3, 4, 4, 4, 4],
                               [1, 2, 3, 4, 0, 2, 3, 4, 0, 1, 3, 4, 0, 1, 2, 4, 0, 1, 2, 3]]])

        batch = [random_nodes_pos, random_nodes_vel, edge_attr, random_nodes_charges, random_nodes_pos, edges]
        # loc, vel, edge_attr, charges, loc_end, edges
        res, _ = feature_embedding.forward(batch)  # [..., 0, 1:4]

        rotor = algebra.versor(order=2)
        random_nodes_pos = random_nodes_pos.reshape(5, 3)
        random_nodes_vel = random_nodes_vel.reshape(5, 3)
        pos = algebra.embed_grade(random_nodes_pos.unsqueeze(1), 1)
        vel = algebra.embed_grade(random_nodes_vel.unsqueeze(1), 1)
        pos = algebra.rho(rotor, pos)[..., 1:4]
        vel = algebra.rho(rotor, vel)[..., 1:4]
        pos = pos.reshape(1, 5, 3)
        vel = vel.reshape(1, 5, 3)
        batch2 = [pos, vel, edge_attr, random_nodes_charges, pos, edges]
        rot_res, _ = feature_embedding.forward(batch2)
        res = algebra.embed_grade(res.unsqueeze(dim=-2), 1)
        res_rot = algebra.rho(rotor, res)[..., 0, 1:4]
        # Compute the sum of the differences
        difference_sum = (res_rot - rot_res).sum()
        print(difference_sum)

        # Check that the sum of the differences is close to 0
        self.assertTrue(abs(difference_sum) < 1e-5,
                        f"Equivariance test failed: difference sum {difference_sum} is not close to 0")


if __name__ == '__main__':
    unittest.main()
