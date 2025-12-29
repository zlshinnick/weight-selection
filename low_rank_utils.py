"""
Utility functions for applying SVD-based low-rank approximation (weight truncation) to models.
Used in main.py
"""

import torch


def apply_svd_low_rank(weight: torch.Tensor, rank: int = 64) -> torch.Tensor:
    """
    Apply SVD to a weight matrix and return a rank-r approximation.

    Args:
        weight (torch.Tensor): Weight matrix of shape [m, n]
        rank (int): Target rank

    Returns:
        torch.Tensor: Approximated weight matrix of shape [m, n] with rank <= rank
    """
    m, n = weight.shape
    actual_rank = min(rank, m, n)

    # Perform SVD: W = U @ S @ V^T
    U, S, Vt = torch.linalg.svd(weight, full_matrices=False)

    # Keep only top 'actual_rank' singular values and vectors
    U_r = U[:, :actual_rank]  # [m, r]
    S_r = S[:actual_rank]  # [r]
    Vt_r = Vt[:actual_rank, :]  # [r, n]

    # Reconstruct: W_approx = U_r @ diag(S_r) @ Vt_r
    weight_approx = U_r @ torch.diag(S_r) @ Vt_r

    return weight_approx


def apply_weight_truncation_to_model(
    model,
    rank: int = 64,
    attention_only: bool = False,
    mlp_only: bool = False,
) -> None:
    """
    Apply SVD-based low-rank approximation (weight truncation) to attention and/or MLP matrices.

    Note: Biases are preserved as-is (not decomposed) since they are vectors
    and are added after matrix multiplication: output = (A @ B @ input) + bias.

    Args:
        model: VisionTransformer model
        rank (int): Target rank for low-rank approximation
        attention_only (bool): If True, only apply to attention matrices (qkv, proj)
        mlp_only (bool): If True, only apply to MLP matrices (fc1, fc2)
    """
    if attention_only and mlp_only:
        raise ValueError("Cannot set both attention_only and mlp_only to True")

    # Default: apply to both if neither flag is set
    apply_attention = not mlp_only
    apply_mlp = not attention_only

    model.eval()

    with torch.no_grad():
        for block_idx, block in enumerate(model.blocks):
            if apply_attention:
                # Process Attention matrices
                # 1. qkv: [embed_dim * 3, embed_dim] = [576, 192] for vit_tiny
                if hasattr(block.attn, "qkv"):
                    qkv_weight = block.attn.qkv.weight.data.clone()
                    qkv_weight_lr = apply_svd_low_rank(qkv_weight, rank=rank)
                    block.attn.qkv.weight.data.copy_(qkv_weight_lr)
                    print(
                        f"Block {block_idx}: qkv weight shape {qkv_weight.shape} -> rank {rank}"
                    )

                # 2. proj: [embed_dim, embed_dim] = [192, 192] for vit_tiny
                if hasattr(block.attn, "proj"):
                    proj_weight = block.attn.proj.weight.data.clone()
                    proj_weight_lr = apply_svd_low_rank(proj_weight, rank=rank)
                    block.attn.proj.weight.data.copy_(proj_weight_lr)
                    print(
                        f"Block {block_idx}: proj weight shape {proj_weight.shape} -> rank {rank}"
                    )

            if apply_mlp:
                # Process MLP matrices
                # 3. fc1: [hidden_features, embed_dim] = [768, 192] for vit_tiny (mlp_ratio=4)
                if hasattr(block.mlp, "fc1"):
                    fc1_weight = block.mlp.fc1.weight.data.clone()
                    fc1_weight_lr = apply_svd_low_rank(fc1_weight, rank=rank)
                    block.mlp.fc1.weight.data.copy_(fc1_weight_lr)
                    print(
                        f"Block {block_idx}: fc1 weight shape {fc1_weight.shape} -> rank {rank}"
                    )

                # 4. fc2: [embed_dim, hidden_features] = [192, 768] for vit_tiny
                if hasattr(block.mlp, "fc2"):
                    fc2_weight = block.mlp.fc2.weight.data.clone()
                    fc2_weight_lr = apply_svd_low_rank(fc2_weight, rank=rank)
                    block.mlp.fc2.weight.data.copy_(fc2_weight_lr)
                    print(
                        f"Block {block_idx}: fc2 weight shape {fc2_weight.shape} -> rank {rank}"
                    )
