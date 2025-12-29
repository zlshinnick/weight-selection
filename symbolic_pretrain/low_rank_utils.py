"""
Low-rank factorization utilities for transformer models.

This module provides functions to inject low-rank factorized matrices into ViT models.
The LowRankLinear class stores weights as W = A @ B, where A and B are the factorized matrices.

Example usage in cli.py:
    from .low_rank_utils import inject_low_rank_into_vit
    import torch

    # Build model as usual
    model, mlm_head = build_model(cfg, tok, pos)

    # Option 1: Use random initialization with specified ranks (matches ViT init)
    inject_low_rank_into_vit(model, rank_attn=64, rank_mlp=64)

    # Option 2: Use pre-initialized A and B matrices
    # For attention: qkv and proj
    qkv_A = torch.randn(embed_dim * 3, rank)  # [out, rank]
    qkv_B = torch.randn(rank, embed_dim)      # [rank, in]
    proj_A = torch.randn(embed_dim, rank)
    proj_B = torch.randn(rank, embed_dim)

    # For MLP: fc1 and fc2
    fc1_A = torch.randn(hidden_features, rank)
    fc1_B = torch.randn(rank, embed_dim)
    fc2_A = torch.randn(embed_dim, rank)
    fc2_B = torch.randn(rank, hidden_features)

    inject_low_rank_into_vit(
        model,
        attn_A=(qkv_A, proj_A),
        attn_B=(qkv_B, proj_B),
        mlp_A=(fc1_A, fc2_A),
        mlp_B=(fc1_B, fc2_B),
    )
"""

from typing import Optional, Tuple
import torch
import torch.nn as nn

try:
    from timm.models.layers import trunc_normal_
except ImportError:
    # Fallback if timm not available
    def trunc_normal_(tensor, mean=0.0, std=1.0, a=-2.0, b=2.0):
        """Fallback trunc_normal implementation."""
        nn.init.normal_(tensor, mean=mean, std=std)
        with torch.no_grad():
            tensor.clamp_(min=mean + a * std, max=mean + b * std)

try:
    from transformers.pytorch_utils import Conv1D
    from src.pretrain.models.low_rank_linear import LowRankConv1D
except ImportError:
    # If transformers or LowRankConv1D not available, we'll skip GPT-2 functionality
    Conv1D = None
    LowRankConv1D = None


def _make_low_rank_from_conv1d(conv: Conv1D, rank: int) -> "LowRankConv1D":
    """Convert Conv1D to LowRankConv1D (GPT-2 style)."""
    if LowRankConv1D is None:
        raise ImportError(
            "LowRankConv1D not available. Install transformers and src.pretrain.models.low_rank_linear"
        )
    lr = LowRankConv1D(
        out_features=conv.nf,  # Conv1D.nf = out_features
        in_features=conv.nx,  # Conv1D.nx = in_features
        rank=rank,
        bias=(conv.bias is not None),
    )
    return lr


def inject_low_rank_into_gpt2(
    model: nn.Module,
    rank_attn: Optional[int] = None,
    rank_mlp: Optional[int] = None,
) -> None:
    """
    Replace GPT-2 attention + MLP Conv1D layers with LowRankConv1D.
    Assumes HuggingFace-style GPT2 blocks at model.transformer.h.
    """
    if Conv1D is None or LowRankConv1D is None:
        raise ImportError("GPT-2 low rank requires transformers and LowRankConv1D")

    if rank_attn is None and rank_mlp is None:
        return

    transformer = getattr(model, "transformer", model)
    blocks = getattr(transformer, "h", None)
    if blocks is None:
        raise RuntimeError(
            "inject_low_rank_into_gpt2: could not find transformer.h blocks"
        )

    for block in blocks:
        attn = getattr(block, "attn", None)
        mlp = getattr(block, "mlp", None)

        if attn is not None and rank_attn is not None:
            if hasattr(attn, "c_attn") and isinstance(attn.c_attn, Conv1D):
                attn.c_attn = _make_low_rank_from_conv1d(attn.c_attn, rank_attn)
            if hasattr(attn, "c_proj") and isinstance(attn.c_proj, Conv1D):
                attn.c_proj = _make_low_rank_from_conv1d(attn.c_proj, rank_attn)

        if mlp is not None and rank_mlp is not None:
            if hasattr(mlp, "c_fc") and isinstance(mlp.c_fc, Conv1D):
                mlp.c_fc = _make_low_rank_from_conv1d(mlp.c_fc, rank_mlp)
            if hasattr(mlp, "c_proj") and isinstance(mlp.c_proj, Conv1D):
                mlp.c_proj = _make_low_rank_from_conv1d(mlp.c_proj, rank_mlp)


class LowRankLinear(nn.Module):
    """
    Low rank replacement for nn.Linear, parameterizes W as A @ B with rank << min(out, in).

    Forward: x @ B.T @ A.T + bias, which is equivalent to (A @ B) @ x + bias
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        rank: int,
        bias: bool = True,
        A: Optional[torch.Tensor] = None,
        B: Optional[torch.Tensor] = None,
    ):
        """
        Args:
            in_features: Input feature dimension
            out_features: Output feature dimension
            rank: Rank of the low-rank factorization (must be > 0)
            bias: Whether to include a bias term
            A: Optional pre-initialized A matrix [out_features, rank]
            B: Optional pre-initialized B matrix [rank, in_features]
        """
        super().__init__()
        assert rank > 0, "rank must be > 0"
        self.in_features = in_features
        self.out_features = out_features
        self.rank = rank

        # A: (out, rank), B: (rank, in)
        self.A = nn.Parameter(torch.empty(out_features, rank))
        self.B = nn.Parameter(torch.empty(rank, in_features))
        self.bias = nn.Parameter(torch.empty(out_features)) if bias else None

        # Initialize A and B matrices
        if A is not None and B is not None:
            # Use provided matrices
            assert A.shape == (out_features, rank), (
                f"A shape {A.shape} != ({out_features}, {rank})"
            )
            assert B.shape == (rank, in_features), (
                f"B shape {B.shape} != ({rank}, {in_features})"
            )
            self.A.data.copy_(A)
            self.B.data.copy_(B)
        else:
            # Will be initialized by reset_parameters()
            pass

        self._skip_init = A is not None and B is not None
        self.reset_parameters()

    def reset_parameters(self):
        """
        Initialize parameters so that W = A @ B has the same distribution as ViT initialization.
        
        ViT uses trunc_normal_(weight, std=0.02). To achieve this for W = A @ B:
        - If A[i,j] ~ N(0, σ²) and B[j,k] ~ N(0, σ²), then W[i,k] = Σ_j A[i,j] * B[j,k]
        - Var(W[i,k]) = rank * σ⁴
        - To get std(W) = 0.02: σ = (0.02² / rank)^0.25
        """
        if self._skip_init:
            # Skip initialization if A and B were provided
            if self.bias is not None:
                nn.init.zeros_(self.bias)
            return

        target_std_W = 0.02  # Match ViT initialization
        sigma = (target_std_W**2 / self.rank) ** 0.25
        # Use trunc_normal_ to match ViT initialization exactly
        trunc_normal_(self.A, mean=0.0, std=sigma)
        trunc_normal_(self.B, mean=0.0, std=sigma)
        if self.bias is not None:
            nn.init.zeros_(self.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass: x @ B.T @ A.T + bias
        Equivalent to (A @ B) @ x + bias but potentially more memory efficient.
        """
        # x: (..., in_features)
        x_flat = x.view(-1, self.in_features)  # (N, in)
        hidden = x_flat @ self.B.t()  # (N, rank)
        out = hidden @ self.A.t()  # (N, out)
        if self.bias is not None:
            out = out + self.bias
        return out.view(*x.shape[:-1], self.out_features)

    def extra_repr(self) -> str:
        return f"in_features={self.in_features}, out_features={self.out_features}, rank={self.rank}, bias={self.bias is not None}"


def _make_low_rank_from_linear(
    linear: nn.Linear,
    rank: int,
    A: Optional[torch.Tensor] = None,
    B: Optional[torch.Tensor] = None,
    init_from_weights: bool = False,
) -> LowRankLinear:
    """
    Convert a nn.Linear layer to a LowRankLinear layer.

    Args:
        linear: The original nn.Linear layer
        rank: Rank for the low-rank factorization
        A: Optional pre-initialized A matrix [out_features, rank]
        B: Optional pre-initialized B matrix [rank, in_features]
        init_from_weights: If True and A/B are None, initialize A/B using SVD from linear.weight.
                          If False, initialize A/B randomly (matching ViT initialization).

    Returns:
        LowRankLinear layer with the same dimensions
    """
    # If A and B are provided, use them directly
    # Otherwise, optionally initialize from the original weight matrix using SVD
    # (for training from scratch, we use random init instead)
    if A is None or B is None:
        if init_from_weights:
            # Use SVD to initialize from existing weights
            with torch.no_grad():
                weight = linear.weight.data  # [out_features, in_features]
                U, S, Vt = torch.linalg.svd(weight, full_matrices=False)

                # Keep top rank components
                actual_rank = min(rank, U.shape[1])
                U_r = U[:, :actual_rank]  # [out_features, rank]
                S_r = S[:actual_rank]  # [rank]
                Vt_r = Vt[:actual_rank, :]  # [rank, in_features]

                # Factorize: W ≈ U_r @ diag(S_r) @ Vt_r = (U_r @ sqrt(S_r)) @ (sqrt(S_r) @ Vt_r)
                sqrt_S = torch.sqrt(S_r)
                A = U_r @ torch.diag(sqrt_S)  # [out_features, rank]
                B = torch.diag(sqrt_S) @ Vt_r  # [rank, in_features]
        else:
            # Initialize randomly (will use reset_parameters in LowRankLinear)
            A = None
            B = None

    lr = LowRankLinear(
        in_features=linear.in_features,
        out_features=linear.out_features,
        rank=rank,
        bias=(linear.bias is not None),
        A=A,
        B=B,
    )

    # Copy bias if it exists
    if linear.bias is not None and lr.bias is not None:
        lr.bias.data.copy_(linear.bias.data)

    return lr


def inject_low_rank_into_vit(
    model: nn.Module,
    rank_attn: Optional[int] = None,
    rank_mlp: Optional[int] = None,
    attn_A: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    attn_B: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    mlp_A: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    mlp_B: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
) -> None:
    """
    Replace ViT attention + MLP nn.Linear layers with LowRankLinear.

    Args:
        model: VisionTransformer model (or wrapper) with blocks attribute
        rank_attn: Rank for attention layers (qkv, proj). If None, uses A/B shapes.
        rank_mlp: Rank for MLP layers (fc1, fc2). If None, uses A/B shapes.
        attn_A: Optional tuple of (qkv_A, proj_A) matrices for attention
        attn_B: Optional tuple of (qkv_B, proj_B) matrices for attention
        mlp_A: Optional tuple of (fc1_A, fc2_A) matrices for MLP
        mlp_B: Optional tuple of (fc1_B, fc2_B) matrices for MLP

    Notes:
        - If A and B are provided, rank_attn/rank_mlp are inferred from their shapes
        - If A and B are None, uses random initialization matching ViT (trunc_normal, std=0.02)
        - Attention matrices: qkv [embed_dim*3, embed_dim], proj [embed_dim, embed_dim]
        - MLP matrices: fc1 [hidden_features, embed_dim], fc2 [embed_dim, hidden_features]
        - A and B are initialized so that W = A @ B has the same distribution as standard ViT weights
    """
    if rank_attn is None and rank_mlp is None and attn_A is None and mlp_A is None:
        return

    # Get blocks from model (could be wrapped)
    blocks = None
    if hasattr(model, "blocks"):
        blocks = model.blocks
    elif hasattr(model, "vit") and hasattr(model.vit, "blocks"):
        blocks = model.vit.blocks  # TokenViTFixedPos wrapper
    elif hasattr(model, "vit_backbone") and hasattr(model.vit_backbone, "blocks"):
        blocks = model.vit_backbone.blocks
    elif hasattr(model, "transformer") and hasattr(model.transformer, "blocks"):
        blocks = model.transformer.blocks

    if blocks is None:
        raise RuntimeError(
            "inject_low_rank_into_vit: could not find blocks attribute. "
            "Expected model.blocks, model.vit.blocks, model.vit_backbone.blocks, or model.transformer.blocks"
        )

    for block_idx, block in enumerate(blocks):
        attn = getattr(block, "attn", None)
        mlp = getattr(block, "mlp", None)

        # Process attention layers (qkv and proj)
        if attn is not None and (rank_attn is not None or attn_A is not None):
            # Handle qkv
            if hasattr(attn, "qkv") and isinstance(attn.qkv, nn.Linear):
                qkv_A = attn_A[0] if attn_A is not None and len(attn_A) > 0 else None
                qkv_B = attn_B[0] if attn_B is not None and len(attn_B) > 0 else None

                if qkv_A is not None:
                    rank_qkv = qkv_A.shape[1]
                else:
                    rank_qkv = rank_attn

                attn.qkv = _make_low_rank_from_linear(
                    attn.qkv, rank=rank_qkv, A=qkv_A, B=qkv_B
                )

            # Handle proj
            if hasattr(attn, "proj") and isinstance(attn.proj, nn.Linear):
                proj_A = attn_A[1] if attn_A is not None and len(attn_A) > 1 else None
                proj_B = attn_B[1] if attn_B is not None and len(attn_B) > 1 else None

                if proj_A is not None:
                    rank_proj = proj_A.shape[1]
                else:
                    rank_proj = rank_attn

                attn.proj = _make_low_rank_from_linear(
                    attn.proj, rank=rank_proj, A=proj_A, B=proj_B
                )

        # Process MLP layers (fc1 and fc2)
        if mlp is not None and (rank_mlp is not None or mlp_A is not None):
            # Handle fc1
            if hasattr(mlp, "fc1") and isinstance(mlp.fc1, nn.Linear):
                fc1_A = mlp_A[0] if mlp_A is not None and len(mlp_A) > 0 else None
                fc1_B = mlp_B[0] if mlp_B is not None and len(mlp_B) > 0 else None

                if fc1_A is not None:
                    rank_fc1 = fc1_A.shape[1]
                else:
                    rank_fc1 = rank_mlp

                mlp.fc1 = _make_low_rank_from_linear(
                    mlp.fc1, rank=rank_fc1, A=fc1_A, B=fc1_B
                )

            # Handle fc2
            if hasattr(mlp, "fc2") and isinstance(mlp.fc2, nn.Linear):
                fc2_A = mlp_A[1] if mlp_A is not None and len(mlp_A) > 1 else None
                fc2_B = mlp_B[1] if mlp_B is not None and len(mlp_B) > 1 else None

                if fc2_A is not None:
                    rank_fc2 = fc2_A.shape[1]
                else:
                    rank_fc2 = rank_mlp

                mlp.fc2 = _make_low_rank_from_linear(
                    mlp.fc2, rank=rank_fc2, A=fc2_A, B=fc2_B
                )
