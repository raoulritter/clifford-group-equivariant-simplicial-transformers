import torch
import torch.nn as nn


EPS = 1e-6

class EquivariantDropout(nn.Module):
    """Equivariant dropout for multivectors (and regular dropout for auxiliary scalars).

    Parameters
    ----------
    p : float
        Dropout probability.
    """

    def __init__(self, p: float = 0.0):
        super().__init__()
        self._dropout_prob = p

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """Forward pass. Applies dropout.

        Parameters
        ----------
        input : torch.Tensor with shape (..., hidden_dim, 8)
            Multivector inputs.

        Returns
        -------
        output : torch.Tensor with shape (..., hidden_dim, 8)
            Multivector inputs with dropout applied.
        """
        if not self.training or self._dropout_prob == 0.0:
            return input

        # Generate dropout mask for the multivectors
        mask_shape = input[..., :1].shape
        mask = torch.bernoulli((1 - self._dropout_prob) * torch.ones(mask_shape, device=input.device))
        mask = mask / (1 - self._dropout_prob)  # Scale the mask to keep the expected value the same

        # Apply the mask to the multivector parts
        output = input * mask

        return output