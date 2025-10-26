import torch
import torch.nn as nn


class FrozenPositionalEmbedding(nn.Module):
    """A frozen positional embedding layer with normalized random vectors.

    This embedding layer initializes position vectors as normalized random
    vectors scaled by a constant factor, then freezes them to prevent
    gradient updates during training.
    """

    def __init__(self, N: int, d: int, scale: float = 0.02):
        """Initialize the frozen positional embedding.

        Args:
            N: Number of positions (sequence length).
            d: Dimensionality of the positional embedding space.
            scale: Scaling factor for the normalized position vectors. Defaults to 0.02.
        """
        super().__init__()
        pos_vecs = torch.randn(N, d)
        pos_vecs = pos_vecs / pos_vecs.norm(dim=1, keepdim=True) * scale
        self.emb = nn.Embedding(N, d)
        with torch.no_grad():
            self.emb.weight.copy_(pos_vecs)
        self.emb.weight.requires_grad_(False)

    def forward(self, idx):
        """Forward pass through the positional embedding layer.

        Args:
            idx: Input tensor containing position indices.

        Returns:
            Positional embeddings corresponding to the input indices.
        """
        return self.emb(idx)
