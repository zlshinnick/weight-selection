"""Width expansion via SVD + row-wise tiling.

Expands 2D weight matrices to larger dimensions using SVD factorization
and row-wise tiling of factor matrices, while preserving function on
the original subspace at initialization.

The approach:
1. Compute SVD: W = U @ diag(S) @ Vt
2. Form factors: A = U * sqrt(S), B = V * sqrt(S)
3. Tile rows of A if expanding output dimension
4. Tile rows of B if expanding input dimension
5. Reconstruct: W_expanded = A_tiled @ B_tiled.T
6. Optionally rescale to match original Frobenius norm

This reuses learned singular directions and provides structured
initialization for new dimensions.
"""

import torch


def _tile_rows(
    matrix: torch.Tensor, tgt_rows: int, mode: str = "cyclic"
) -> torch.Tensor:
    """Tile rows of a matrix to reach target number of rows.

    Args:
        matrix: Input matrix of shape (src_rows, cols).
        tgt_rows: Target number of rows (must be >= src_rows).
        mode: Tiling mode - "cyclic" or "repeat".
            - "cyclic": rows are repeated cyclically (0, 1, 2, ..., 0, 1, 2, ...)
            - "repeat": each row is repeated before moving to next

    Returns:
        Matrix of shape (tgt_rows, cols).
    """
    src_rows = matrix.shape[0]

    if tgt_rows < src_rows:
        raise ValueError(f"Cannot tile from {src_rows} to {tgt_rows} rows")

    if tgt_rows == src_rows:
        return matrix.clone()

    if mode == "cyclic":
        # Cyclic indexing: 0, 1, 2, ..., n-1, 0, 1, 2, ...
        indices = torch.arange(tgt_rows, device=matrix.device) % src_rows
    elif mode == "repeat":
        # Repeat indexing: 0, 0, ..., 1, 1, ..., 2, 2, ...
        repeats_per_row = tgt_rows // src_rows
        remainder = tgt_rows % src_rows
        # Each row gets at least repeats_per_row copies
        # First 'remainder' rows get one extra copy
        counts = torch.full((src_rows,), repeats_per_row, dtype=torch.long)
        counts[:remainder] += 1
        indices = torch.repeat_interleave(
            torch.arange(src_rows, device=matrix.device), counts.to(matrix.device)
        )
    else:
        raise ValueError(f"Unknown tile mode: {mode}. Use 'cyclic' or 'repeat'.")

    return matrix[indices]


def expand_weight_svd_tile(
    W: torch.Tensor,
    tgt_out_dim: int,
    tgt_in_dim: int,
    tile_mode: str = "cyclic",
    scale_mode: str = "fro",
    pad_mode: str = "tile",
) -> torch.Tensor:
    """Expand a 2D weight matrix via SVD factorization + row-wise tiling or padding.

    This is a function-preserving expansion: for inputs restricted to the
    original input dimensions (zero-padded), the expanded weight reproduces
    the original computation on those output dimensions.

    Args:
        W: Original weight matrix of shape (out_dim, in_dim).
        tgt_out_dim: Target output dimension (must be >= out_dim).
        tgt_in_dim: Target input dimension (must be >= in_dim).
        tile_mode: "cyclic" or "repeat" for row tiling (only used when pad_mode="tile").
        scale_mode: "fro" to rescale to match Frobenius norm, "none" for no scaling.
        pad_mode: Expansion strategy for A and B factor matrices:
            - "tile": Row-wise tiling of factors (default).
            - "zeros": Pad factors with zeros.
            - "random": Pad factors with scaled random values (GPT-2 init after A @ B.T).

    Returns:
        Expanded weight matrix of shape (tgt_out_dim, tgt_in_dim).

    Notes:
        - Does NOT use low-rank truncation (all singular values are kept).
        - All modes preserve the original subspace via SVD factorization.
        - For pad_mode="tile": W_expanded[:out_dim, :in_dim] ≈ W (approximately).
        - For pad_mode="zeros": W_expanded[:out_dim, :in_dim] = W (exactly).
        - For pad_mode="random": Padded regions have N(0, 0.02) distribution after A @ B.T.
    """
    out_dim, in_dim = W.shape

    if tgt_out_dim < out_dim:
        raise ValueError(f"Cannot shrink output dim from {out_dim} to {tgt_out_dim}")
    if tgt_in_dim < in_dim:
        raise ValueError(f"Cannot shrink input dim from {in_dim} to {tgt_in_dim}")

    # Handle case where no expansion needed
    if tgt_out_dim == out_dim and tgt_in_dim == in_dim:
        return W.clone()

    # Compute full SVD (no truncation) for all modes
    # W = U @ diag(S) @ Vt
    # U: (out_dim, k), S: (k,), Vt: (k, in_dim) where k = min(out_dim, in_dim)
    U, S, Vt = torch.linalg.svd(W, full_matrices=False)
    k = S.shape[0]

    # Form factors with sqrt(S) distributed to both
    # A = U * sqrt(S): (out_dim, k)
    # B = V * sqrt(S) = Vt.T * sqrt(S): (in_dim, k)
    sqrt_S = torch.sqrt(S)
    A = U * sqrt_S.unsqueeze(0)  # (out_dim, k)
    B = Vt.T * sqrt_S.unsqueeze(0)  # (in_dim, k)

    if pad_mode == "tile":
        # Tile rows of A if expanding output dimension
        A_expanded = _tile_rows(A, tgt_out_dim, mode=tile_mode)  # (tgt_out_dim, k)

        # Tile rows of B if expanding input dimension
        B_expanded = _tile_rows(B, tgt_in_dim, mode=tile_mode)  # (tgt_in_dim, k)

    elif pad_mode == "zeros":
        # Pad A and B with zeros to preserve subspace
        A_expanded = torch.zeros(tgt_out_dim, k, dtype=W.dtype, device=W.device)
        A_expanded[:out_dim, :] = A

        B_expanded = torch.zeros(tgt_in_dim, k, dtype=W.dtype, device=W.device)
        B_expanded[:in_dim, :] = B

    elif pad_mode == "random":
        # Pad A and B with random values scaled for GPT-2 init after reconstruction
        # Since W = A @ B.T, each element is sum of k products.
        # To get N(0, 0.02) in the final matrix, we scale by 0.02 / sqrt(k)
        pad_std = 0.02 / (k**0.5)

        A_expanded = (
            torch.randn(tgt_out_dim, k, dtype=W.dtype, device=W.device) * pad_std
        )
        A_expanded[:out_dim, :] = A

        B_expanded = (
            torch.randn(tgt_in_dim, k, dtype=W.dtype, device=W.device) * pad_std
        )
        B_expanded[:in_dim, :] = B

    else:
        raise ValueError(
            f"Unknown pad_mode: {pad_mode}. Use 'tile', 'zeros', or 'random'."
        )

    # Reconstruct expanded weight
    # W_expanded = A_expanded @ B_expanded.T: (tgt_out_dim, tgt_in_dim)
    W_expanded = A_expanded @ B_expanded.T

    # Optionally rescale to match original Frobenius norm
    if scale_mode == "fro":
        orig_norm = torch.linalg.norm(W, ord="fro")
        expanded_norm = torch.linalg.norm(W_expanded, ord="fro")
        if expanded_norm > 0:
            W_expanded = W_expanded * (orig_norm / expanded_norm)
    elif scale_mode != "none":
        raise ValueError(f"Unknown scale_mode: {scale_mode}. Use 'fro' or 'none'.")

    return W_expanded


def expand_qkv_weight(
    qkv_weight: torch.Tensor,
    src_embed_dim: int,
    tgt_embed_dim: int,
    tile_mode: str = "cyclic",
    scale_mode: str = "fro",
    pad_mode: str = "tile",
) -> torch.Tensor:
    """Expand QKV weight matrix which has shape (3 * embed_dim, embed_dim).

    The QKV weight is structured as stacked [Q, K, V] projections.
    We expand each section independently and concatenate to preserve
    the structure expected by multi-head attention.

    Args:
        qkv_weight: Original QKV weight of shape (3 * src_embed_dim, src_embed_dim).
        src_embed_dim: Source embedding dimension.
        tgt_embed_dim: Target embedding dimension.
        tile_mode: Tiling mode for SVD expansion (only used when pad_mode="tile").
        scale_mode: Scaling mode for SVD expansion.
        pad_mode: Expansion strategy: "tile", "zeros", or "random".

    Returns:
        Expanded QKV weight of shape (3 * tgt_embed_dim, tgt_embed_dim).
    """
    # Split into Q, K, V sections (each is embed_dim x embed_dim)
    q_weight = qkv_weight[:src_embed_dim]
    k_weight = qkv_weight[src_embed_dim : 2 * src_embed_dim]
    v_weight = qkv_weight[2 * src_embed_dim :]

    # Expand each section independently
    q_expanded = expand_weight_svd_tile(
        q_weight, tgt_embed_dim, tgt_embed_dim, tile_mode, scale_mode, pad_mode
    )
    k_expanded = expand_weight_svd_tile(
        k_weight, tgt_embed_dim, tgt_embed_dim, tile_mode, scale_mode, pad_mode
    )
    v_expanded = expand_weight_svd_tile(
        v_weight, tgt_embed_dim, tgt_embed_dim, tile_mode, scale_mode, pad_mode
    )

    # Concatenate back into QKV structure
    return torch.cat([q_expanded, k_expanded, v_expanded], dim=0)


def expand_fc1_weight(
    fc1_weight: torch.Tensor,
    src_embed_dim: int,
    tgt_embed_dim: int,
    src_mlp_dim: int,
    tgt_mlp_dim: int,
    tile_mode: str = "cyclic",
    scale_mode: str = "fro",
    pad_mode: str = "tile",
) -> torch.Tensor:
    """Expand MLP fc1 weight matrix.

    fc1 projects from embed_dim to mlp_dim (typically 4 * embed_dim).
    Shape: (mlp_dim, embed_dim) -> (tgt_mlp_dim, tgt_embed_dim)

    Args:
        fc1_weight: Original fc1 weight of shape (src_mlp_dim, src_embed_dim).
        src_embed_dim: Source embedding dimension.
        tgt_embed_dim: Target embedding dimension.
        src_mlp_dim: Source MLP hidden dimension.
        tgt_mlp_dim: Target MLP hidden dimension.
        tile_mode: Tiling mode for SVD expansion (only used when pad_mode="tile").
        scale_mode: Scaling mode for SVD expansion.
        pad_mode: Expansion strategy: "tile", "zeros", or "random".

    Returns:
        Expanded fc1 weight of shape (tgt_mlp_dim, tgt_embed_dim).
    """
    return expand_weight_svd_tile(
        fc1_weight, tgt_mlp_dim, tgt_embed_dim, tile_mode, scale_mode, pad_mode
    )


def expand_fc2_weight(
    fc2_weight: torch.Tensor,
    src_embed_dim: int,
    tgt_embed_dim: int,
    src_mlp_dim: int,
    tgt_mlp_dim: int,
    tile_mode: str = "cyclic",
    scale_mode: str = "fro",
    pad_mode: str = "tile",
) -> torch.Tensor:
    """Expand MLP fc2 weight matrix.

    fc2 projects from mlp_dim back to embed_dim.
    Shape: (embed_dim, mlp_dim) -> (tgt_embed_dim, tgt_mlp_dim)

    Args:
        fc2_weight: Original fc2 weight of shape (src_embed_dim, src_mlp_dim).
        src_embed_dim: Source embedding dimension.
        tgt_embed_dim: Target embedding dimension.
        src_mlp_dim: Source MLP hidden dimension.
        tgt_mlp_dim: Target MLP hidden dimension.
        tile_mode: Tiling mode for SVD expansion (only used when pad_mode="tile").
        scale_mode: Scaling mode for SVD expansion.
        pad_mode: Expansion strategy: "tile", "zeros", or "random".

    Returns:
        Expanded fc2 weight of shape (tgt_embed_dim, tgt_mlp_dim).
    """
    return expand_weight_svd_tile(
        fc2_weight, tgt_embed_dim, tgt_mlp_dim, tile_mode, scale_mode, pad_mode
    )


def expand_proj_weight(
    proj_weight: torch.Tensor,
    src_embed_dim: int,
    tgt_embed_dim: int,
    tile_mode: str = "cyclic",
    scale_mode: str = "fro",
    pad_mode: str = "tile",
) -> torch.Tensor:
    """Expand attention projection weight matrix.

    proj projects from embed_dim to embed_dim.
    Shape: (embed_dim, embed_dim) -> (tgt_embed_dim, tgt_embed_dim)

    Args:
        proj_weight: Original proj weight of shape (src_embed_dim, src_embed_dim).
        src_embed_dim: Source embedding dimension.
        tgt_embed_dim: Target embedding dimension.
        tile_mode: Tiling mode for SVD expansion (only used when pad_mode="tile").
        scale_mode: Scaling mode for SVD expansion.
        pad_mode: Expansion strategy: "tile", "zeros", or "random".

    Returns:
        Expanded proj weight of shape (tgt_embed_dim, tgt_embed_dim).
    """
    return expand_weight_svd_tile(
        proj_weight, tgt_embed_dim, tgt_embed_dim, tile_mode, scale_mode, pad_mode
    )
