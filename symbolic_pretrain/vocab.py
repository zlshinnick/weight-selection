import torch
import torch.nn as nn


class FrozenTokenEmbedding(nn.Module):
    """A frozen token embedding layer with fixed identity-based weights.

    This embedding layer initializes weights with a scaled identity matrix
    and freezes them, preventing any gradient updates during training.
    """

    def __init__(self, K: int, d: int, scale: float = 0.02):
        """Initialize the frozen token embedding.

        Args:
            K: Number of tokens in the vocabulary.
            d: Dimensionality of the embedding space.
            scale: Scaling factor for the identity matrix initialization. Defaults to 0.02.
        """
        super().__init__()
        self.emb = nn.Embedding(K, d)
        with torch.no_grad():
            W = torch.zeros(K, d)
            W[:K, :K] = torch.eye(K) * scale
            self.emb.weight.copy_(W)
        self.emb.weight.requires_grad = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the embedding layer.

        Args:
            x: Input tensor containing token indices.

        Returns:
            Embedded representation of the input tokens.
        """
        return self.emb(x)
