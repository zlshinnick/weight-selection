# symbolic_pretrain/wrappers_swin.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class TokenSwinFixedPos(nn.Module):
    """
    Wrapper that injects symbolic tokens into a Swin backbone by replacing image patch embedding.

    Assumptions:
      - `swin_backbone` is a timm Swin (SwinTransformer or SwinTransformerV2).
      - `tok_embed` outputs vectors of dim == swin.embed_dim.
      - Optional `pos_embed_2d` provides a fixed (H, W, C) additive bias (kept frozen).
    """

    def __init__(
        self,
        swin_backbone: nn.Module,
        tok_embed: nn.Module,  # e.g., nn.Embedding(K, C)
        H: int,
        W: int,
        pos_embed_2d: Optional[
            nn.Module
        ] = None,  # e.g., nn.Embedding(H*W, C) or nn.Parameter of shape (1, H, W, C)
        freeze_tok: bool = True,
        freeze_pos: bool = True,
        return_tokens: bool = False,  # if True, return (B, H*W, C) after norm; else return pooled/cls-style feature
    ):
        super().__init__()
        self.swin = swin_backbone
        self.tok = tok_embed
        self.H, self.W = H, W
        self.return_tokens = return_tokens

        # Optional fixed positional embedding (2D additive)
        self.pos2d = pos_embed_2d

        # Freeze as in your ViT setup (prevents shortcutting via adaptable embeddings)
        if freeze_tok:
            for p in self.tok.parameters():
                p.requires_grad = False
        if self.pos2d is not None and freeze_pos:
            for p in self.pos2d.parameters():
                p.requires_grad = False

        # Sanity checks
        if hasattr(self.swin, "embed_dim"):
            embed_dim = self.swin.embed_dim
        elif hasattr(self.swin, "num_features"):
            embed_dim = self.swin.num_features
        else:
            raise ValueError("Unrecognized Swin backbone: cannot infer embed_dim.")

        # Check token embedding dimension
        # Try to infer tok embedding dim
        test_weight = next(self.tok.parameters())
        if test_weight.shape[-1] != embed_dim:
            raise ValueError(
                f"tok_embed dim {test_weight.shape[-1]} != swin embed_dim {embed_dim}"
            )

        # Turn off absolute pos/patch embed inside swin (we replace that stage)
        # We will NOT call `self.swin.patch_embed(x_img)` anywhere in this wrapper.
        # Swin uses relative position biases internally; we keep those as-is.

    def _final_hw(self) -> tuple:
        stages = getattr(self.swin, "stages", None) or getattr(
            self.swin, "layers", None
        )
        if stages is None:
            raise ValueError("Swin backbone missing stages/layers.")
        # Swin downsamples between stages: total factor = 2^(num_stages - 1)
        factor = 2 ** (len(stages) - 1)
        return self.H // factor, self.W // factor

    def _add_pos2d(self, x_flat: torch.Tensor) -> torch.Tensor:
        """
        Add fixed 2D positional embedding to flattened sequence.
        x_flat: (B, N, C) where N = H*W
        """
        if self.pos2d is None:
            return x_flat

        B, N, C = x_flat.shape
        H, W = self.H, self.W
        assert N == H * W, "Sequence length must match H*W."

        # Support either embedding over linearized indices OR a direct (1, H, W, C) parameter
        # Support embedding-like modules (including FrozenPositionalEmbedding)
        if isinstance(self.pos2d, nn.Embedding) or callable(
            getattr(self.pos2d, "forward", None)
        ):
            idx = torch.arange(N, device=x_flat.device).unsqueeze(0).expand(B, N)
            x_flat = x_flat + self.pos2d(idx)
        else:
            # Assume a parameter/buffer shaped (1, H, W, C) or (H, W, C)
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
                x_flat = x_flat + pos_flat
            else:
                raise ValueError("Unsupported pos2d module type.")
        return x_flat

    @torch.no_grad()
    def _shape_check(self, token_ids: torch.Tensor):
        B, N = token_ids.shape
        assert N == self.H * self.W, f"Expected N=H*W={self.H * self.W}, got {N}."

    def forward_tokens(self, token_ids: torch.Tensor) -> torch.Tensor:
        """
        Return token features aligned to the input grid (B, H*W, C).
        We upsample the final Swin token map back to (H, W).
        """
        x = self._forward_core(token_ids)  # (B, H'*W', C)
        B, Np, C = x.shape
        Hp, Wp = self._final_hw()
        # reshape → upsample back to (H, W) → flatten
        x_map = x.view(B, Hp, Wp, C).permute(0, 3, 1, 2).contiguous()  # (B, C, Hp, Wp)
        x_up = F.interpolate(
            x_map, size=(self.H, self.W), mode="bilinear", align_corners=False
        )  # (B, C, H, W)
        x_flat = (
            x_up.permute(0, 2, 3, 1).contiguous().view(B, self.H * self.W, C)
        )  # (B, H*W, C)
        return x_flat

    def forward_pooled(self, token_ids: torch.Tensor) -> torch.Tensor:
        """
        Return Swin's pooled feature (global average over tokens after norm),
        matching timm's `forward_features`→`forward_head` convention (pre-head).
        Shape: (B, C)
        """
        x = self.forward_tokens(token_ids)  # (B, H*W, C)
        # Swin typically does: x = x.mean(dim=1) before head
        x = x.mean(dim=1)
        return x

    def _forward_core(self, token_ids: torch.Tensor) -> torch.Tensor:
        """
        Core Swin pass from symbolic tokens. Mimics timm Swin `forward_features`,
        except we skip patch_embed and feed our own (x, H, W).
        Returns normalized token map (B, H*W, C) before the classification head.
        """
        self._shape_check(token_ids)
        B, N = token_ids.shape

        # Tokenize (frozen if configured)
        x = self.tok(token_ids)  # (B, N, C)

        # Optional fixed 2D positional bias (frozen)
        x = self._add_pos2d(x)  # (B, N, C)

        # Dropout like Swin’s pos_drop (it expects B,N,C already)
        if hasattr(self.swin, "pos_drop"):
            x = self.swin.pos_drop(x)

        # Run Swin stages; timm BasicLayer expects/returns x only (it tracks resolution internally)
        stages = getattr(self.swin, "stages", None) or getattr(
            self.swin, "layers", None
        )
        if stages is None:
            raise ValueError("Swin backbone missing stages/layers.")
        for layer in stages:
            x = layer(x)

        # Final norm
        x = self.swin.norm(x)  # (B, H*W', C)
        return x

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        """
        If the underlying Swin has a classification head and you want logits,
        set `return_tokens=False` and call the head here. Otherwise return tokens.
        """
        x = self._forward_core(token_ids)  # (B, H*W, C)

        if self.return_tokens:
            return x  # (B, H*W, C)

        # Default: mirror timm forward() → pooled then head
        x = x.mean(dim=1)  # global average pool (B, C)
        if hasattr(self.swin, "head") and self.swin.head is not None:
            x = self.swin.head(x)  # (B, num_classes) if training a classifier
        return x
