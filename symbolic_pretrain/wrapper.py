# symbolic_pretrain/wrappers.py
import torch
import torch.nn as nn


class TokenViTFixedPos(nn.Module):
    """Vision Transformer wrapper with token and fixed positional embeddings.

    This module wraps a ViT backbone and applies token embeddings followed by
    fixed positional embeddings to input token IDs, then processes them through
    the ViT architecture.
    """

    def __init__(
        self,
        vit_backbone: nn.Module,
        tok_embed: nn.Module,
        pos_embed: nn.Module,
        H: int,
        W: int,
    ):
        """Initialize the TokenViTFixedPos wrapper.

        Args:
            vit_backbone: The Vision Transformer backbone model.
            tok_embed: Token embedding module. An embedding layer that mapts integer token IDs into vectors of Dimension D.
            pos_embed: Positional embedding module. an embedding layer with fixed (frozen) vectors that represent positions in a grid (H × W).
            H: Height of the token grid.
            W: Width of the token grid.
        """
        super().__init__()
        self.vit = vit_backbone
        self.tok = tok_embed
        self.pos = pos_embed
        self.H, self.W = H, W

    def _forward_core(self, x):
        """Core forward pass through the ViT backbone.

        Prepends class token, applies positional dropout, processes through
        transformer blocks, and applies final normalization.

        Args:
            x: Input tensor of shape (B, N, D) where B is batch size,
               N is sequence length, and D is embedding dimension.

        Returns:
            Processed tensor of shape (B, N+1, D) including class token.
        """
        B, N, _ = x.shape
        cls = self.vit.cls_token.expand(B, -1, -1) # add class token
        x = torch.cat([cls, x], dim=1) # prepend to sequence
        x = self.vit.pos_drop(x) # dropout
        for blk in self.vit.blocks: # pass through transformer blocks
            x = blk(x)
        x = self.vit.norm(x) # final normalization
        return x

    def forward_tokens(self, token_ids):
        """Forward pass returning all token embeddings (excluding class token).

        Args:
            token_ids: Input token IDs of shape (B, N) where B is batch size
                      and N is the number of tokens (must equal H * W).

        Returns:
            Token embeddings of shape (B, N, D) excluding the class token.
        """
        B, N = token_ids.shape
        assert N == self.H * self.W
        x = self.tok(token_ids) # map ids to vector through frozen token embedding layer
        pos_idx = torch.arange(N, device=x.device).unsqueeze(0).expand(B, N) 
        x = x + self.pos(pos_idx) # add fixed positions
        x = self._forward_core(x)
        return x[:, 1:, :]

    def forward_cls(self, token_ids):
        """Forward pass returning only the class token embedding.

        Args:
            token_ids: Input token IDs of shape (B, N) where B is batch size
                      and N is the number of tokens (must equal H * W).

        Returns:
            Class token embeddings of shape (B, D).
        """
        B, N = token_ids.shape
        assert N == self.H * self.W
        x = self.tok(token_ids)
        pos_idx = torch.arange(N, device=x.device).unsqueeze(0).expand(B, N)
        x = x + self.pos(pos_idx)
        x = self._forward_core(x)
        return x[:, 0, :]
