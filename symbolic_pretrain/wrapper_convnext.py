# symbolic_pretrain/wrapper_convnext.py
import torch
import torch.nn as nn
from typing import Optional


class TokenConvNeXtFixedPos(nn.Module):
    """
    Wrapper that injects symbolic tokens into a ConvNeXt backbone by replacing image input.

    Flow:
      token_ids (B, H*W) -> tok_embed -> (B, H*W, C_in)
      + optional fixed 2D positional bias -> reshape to (B, C_in, H, W)
      -> run through ConvNeXt stages (no global pool) -> normed feature map (B, C, H', W')
      -> forward_tokens returns (B, H'*W', C)

    Assumptions:
      - `convnext_backbone` is a timm ConvNeXt variant exposing `downsample_layers`, `stages`, and `norm`.
      - The ConvNeXt `in_chans` matches `tok_embed` output dim (C_in).
    """

    def __init__(
        self,
        convnext_backbone: nn.Module,
        tok_embed: nn.Module,
        H: int,
        W: int,
        pos_embed_2d: Optional[nn.Module] = None,
        freeze_tok: bool = True,
        freeze_pos: bool = True,
        return_tokens: bool = True,
    ):
        super().__init__()
        self.conv = convnext_backbone
        self.tok = tok_embed
        self.H, self.W = H, W
        self.return_tokens = return_tokens
        self.pos2d = pos_embed_2d

        if freeze_tok:
            for p in self.tok.parameters():
                p.requires_grad = False
        if self.pos2d is not None and freeze_pos:
            for p in self.pos2d.parameters():
                p.requires_grad = False

        # Infer token embedding dim and check it matches in_chans expected by convnext
        test_weight = next(self.tok.parameters())
        _tok_dim = test_weight.shape[-1]
        if hasattr(self.conv, "stem") and isinstance(self.conv.stem, nn.Module):
            # Cannot easily infer in_chans from stem; assume user passed correct in_chans when building backbone
            pass
        # Some safety: timm convnext exposes feature dims via `num_features`
        if not hasattr(self.conv, "num_features"):
            raise ValueError("Unrecognized ConvNeXt backbone: missing num_features.")

    @torch.no_grad()
    def _shape_check(self, token_ids: torch.Tensor):
        B, N = token_ids.shape
        assert N == self.H * self.W, f"Expected N=H*W={self.H * self.W}, got {N}."

    def _add_pos2d(self, x_flat: torch.Tensor) -> torch.Tensor:
        """
        Add fixed 2D positional embedding to flattened sequence.
        x_flat: (B, N, C)
        """
        if self.pos2d is None:
            return x_flat
        B, N, C = x_flat.shape
        H, W = self.H, self.W
        assert N == H * W
        # If it's an embedding-like module (including our FrozenPositionalEmbedding), call forward with indices
        if isinstance(self.pos2d, nn.Embedding) or callable(
            getattr(self.pos2d, "forward", None)
        ):
            idx = torch.arange(N, device=x_flat.device).unsqueeze(0).expand(B, N)
            pos_vals = self.pos2d(idx)
            return x_flat + pos_vals
        pos = self.pos2d
        if isinstance(pos, nn.Parameter) or isinstance(pos, torch.Tensor):
            if pos.dim() == 4:  # (1, H, W, C)
                pos_hw_c = pos
            elif pos.dim() == 3:  # (H, W, C)
                pos_hw_c = pos.unsqueeze(0)
            else:
                raise ValueError(
                    "pos2d must be (1,H,W,C) or (H,W,C) if tensor/parameter."
                )
            pos_flat = pos_hw_c.view(1, H * W, C).to(x_flat.dtype).to(x_flat.device)
            return x_flat + pos_flat
        raise ValueError("Unsupported pos2d module type.")

    def _forward_convnext_stages(self, x_bchw: torch.Tensor) -> torch.Tensor:
        conv = self.conv
        # Preferred path: models exposing explicit downsample layers and stages
        if hasattr(conv, "downsample_layers") and hasattr(conv, "stages"):
            x = conv.downsample_layers[0](x_bchw)
            for i, stage in enumerate(conv.stages):
                x = stage(x)
                if i + 1 < len(conv.downsample_layers):
                    x = conv.downsample_layers[i + 1](x)
            if hasattr(conv, "norm") and conv.norm is not None:
                x = conv.norm(x)
            return x

        # Fallback path: some timm ConvNeXt variants expose .stem + .stages only
        if hasattr(conv, "stem") and hasattr(conv, "stages"):
            x = conv.stem(x_bchw)
            ds_layers = getattr(conv, "downsample_layers", None)
            for i, stage in enumerate(conv.stages):
                x = stage(x)
                if ds_layers is not None and i + 1 < len(ds_layers):
                    x = ds_layers[i + 1](x)
            if hasattr(conv, "norm") and conv.norm is not None:
                x = conv.norm(x)
            return x

        raise ValueError(
            "ConvNeXt backbone missing expected attributes for stage-wise forward."
        )

    def _forward_core(self, token_ids: torch.Tensor) -> torch.Tensor:
        self._shape_check(token_ids)
        B, N = token_ids.shape
        H, W = self.H, self.W

        # Tokenize and add fixed position
        x = self.tok(token_ids)  # (B, N, C_in)
        x = self._add_pos2d(x)  # (B, N, C_in)
        x = x.view(B, H, W, -1).permute(0, 3, 1, 2).contiguous()  # (B, C_in, H, W)

        # ConvNeXt feature extractor (no global pooling)
        x = self._forward_convnext_stages(x)  # (B, C_feat, H', W')
        return x

    def forward_tokens(self, token_ids: torch.Tensor) -> torch.Tensor:
        x = self._forward_core(token_ids)  # (B, C, H', W')
        B, C, Hp, Wp = x.shape
        x = x.permute(0, 2, 3, 1).reshape(B, Hp * Wp, C)  # (B, H'*W', C)
        return x

    def forward_pooled(self, token_ids: torch.Tensor) -> torch.Tensor:
        x = self._forward_core(token_ids)  # (B, C, H', W')
        x = x.mean(dim=(2, 3))  # (B, C)
        return x

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        x = self._forward_core(token_ids)
        if self.return_tokens:
            B, C, Hp, Wp = x.shape
            return x.permute(0, 2, 3, 1).reshape(B, Hp * Wp, C)
        # pooled → optional head
        x = x.mean(dim=(2, 3))
        if hasattr(self.conv, "head") and self.conv.head is not None:
            x = self.conv.head(x)
        return x
