# symbolic_pretrain/wrapper_convit.py
import torch
import torch.nn as nn


class TokenConViTFixedPos(nn.Module):
    """
    Wrapper that feeds symbolic token sequences into a ConViT backbone.

    Flow:
      token_ids (B, H*W) -> frozen tok embedding -> (B, H*W, C)
      + frozen positional embedding -> ConViT blocks (respecting local_up_to_layer)
      -> normalized sequence (B, H*W + 1, C) including the cls token once inserted.
    """

    def __init__(
        self,
        convit_backbone: nn.Module,
        tok_embed: nn.Module,
        pos_embed: nn.Module,
        H: int,
        W: int,
    ):
        super().__init__()
        self.convit = convit_backbone
        self.tok = tok_embed
        self.pos = pos_embed
        self.H, self.W = H, W

        required_attrs = ["cls_token", "pos_drop", "blocks", "norm"]
        missing = [attr for attr in required_attrs if not hasattr(self.convit, attr)]
        if missing:
            raise ValueError(
                f"ConViT backbone missing required attributes: {', '.join(missing)}"
            )

        self._insert_layer = getattr(self.convit, "local_up_to_layer", 0)
        depth = len(self.convit.blocks)
        if self._insert_layer >= depth:
            raise ValueError(
                "ConViT local_up_to_layer must be < number of blocks when using the symbolic wrapper."
            )

        embed_dim = getattr(self.convit, "embed_dim", None)
        if embed_dim is None:
            embed_dim = getattr(self.convit, "num_features", None)
        if embed_dim is None:
            raise ValueError("Unable to infer ConViT embedding dimension.")

        tok_weight = next(self.tok.parameters(), None)
        if tok_weight is None:
            raise ValueError("tok_embed must have parameters to infer embedding dim.")
        tok_dim = tok_weight.shape[-1]
        if tok_dim != embed_dim:
            raise ValueError(
                f"tok_embed dim {tok_dim} != ConViT embed_dim {embed_dim}. "
                "Ensure cfg.model.embed_dim matches the ConViT width."
            )

    def _shape_check(self, token_ids: torch.Tensor):
        B, N = token_ids.shape
        expected = self.H * self.W
        if N != expected:
            raise ValueError(f"Expected sequence length {expected}, got {N}.")
        return B, N

    def _forward_core(self, x: torch.Tensor) -> torch.Tensor:
        """Mirror ConViT forward_features, skipping patch + pos embed stages."""
        x = self.convit.pos_drop(x)
        cls = self.convit.cls_token.expand(x.shape[0], -1, -1)

        for idx, blk in enumerate(self.convit.blocks):
            if idx == self._insert_layer:
                x = torch.cat([cls, x], dim=1)
            x = blk(x)

        x = self.convit.norm(x)
        return x

    def forward_tokens(self, token_ids: torch.Tensor) -> torch.Tensor:
        B, N = self._shape_check(token_ids)
        x = self.tok(token_ids)  # (B, N, C)
        pos_idx = torch.arange(N, device=token_ids.device).unsqueeze(0).expand(B, N)
        x = x + self.pos(pos_idx)  # add frozen spatial bias
        x = self._forward_core(x)
        return x[:, 1:, :]  # drop cls token

    def forward_cls(self, token_ids: torch.Tensor) -> torch.Tensor:
        B, N = self._shape_check(token_ids)
        x = self.tok(token_ids)
        pos_idx = torch.arange(N, device=token_ids.device).unsqueeze(0).expand(B, N)
        x = x + self.pos(pos_idx)
        x = self._forward_core(x)
        return x[:, 0, :]

