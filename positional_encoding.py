import torch
import torch.nn as nn

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, batch_size):
        super(PositionalEncoding, self).__init__()

        # Adding a binary feature
        # Nodes get a [1, 0] and edges get a [0, 1]
        node_marker = torch.zeros(5*batch_size, 1, 8)

        # Create a tensor of zeros with shape [20*batch_size, 1, 8] with a 1 at the beginning (onehot)
        edge_marker = torch.zeros(20 * batch_size, 1, 8)
        edge_marker[:, :, 0] = 1

        # Concatenate this new feature
        self.pe = torch.cat((node_marker, edge_marker), dim=0)

    def forward(self, x):
        return torch.cat((x, self.pe), dim=1)
