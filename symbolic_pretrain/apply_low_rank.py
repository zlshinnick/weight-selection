"""Apply low-rank factorization to models and handle weight tying re-application."""

from typing import Union, List

from .low_rank_utils import inject_low_rank_into_vit
from .model_factory import tie_vit_layer_weights, tie_weights_across_models


def apply_low_rank_factorization(
    model: Union[object, List[object]],
    cfg: object,
    total_params: int,
) -> None:
    """
    Apply low-rank factorization to model(s) if requested in config.

    This function:
    1. Applies low-rank factorization to attention and/or MLP matrices
    2. Re-applies weight tying if it was enabled (low-rank injection creates new modules)
    3. Recalculates and prints parameter counts after low-rank injection

    Args:
        model: Single model or list of parallel models
        cfg: Configuration object with model settings
        total_params (int): Original total parameter count before low-rank
    """
    rank_attn = getattr(cfg.model, "rank_attn", None)
    rank_mlp = getattr(cfg.model, "rank_mlp", None)
    if rank_attn is None and rank_mlp is None:
        return

    print("[model] Applying low-rank factorization...")
    if isinstance(model, list):
        # Parallel models
        for m in model:
            inject_low_rank_into_vit(m, rank_attn=rank_attn, rank_mlp=rank_mlp)
        print(
            f"[model] ✓ Applied low-rank: rank_attn={rank_attn}, rank_mlp={rank_mlp} "
            f"(to {len(model)} models)"
        )
    else:
        # Single model
        inject_low_rank_into_vit(model, rank_attn=rank_attn, rank_mlp=rank_mlp)
        print(
            f"[model] ✓ Applied low-rank: rank_attn={rank_attn}, rank_mlp={rank_mlp}"
        )

    # Re-apply weight tying if it was enabled (low-rank injection creates new modules)
    if getattr(cfg.model, "tie_weights", False):
        tie_attention_mlp_only = getattr(cfg.model, "tie_attention_mlp_only", False)
        tie_weight_groups = getattr(cfg.model, "tie_weight_groups", None)
        if isinstance(model, list):
            # Re-apply within-model weight tying
            for m in model:
                tie_vit_layer_weights(
                    m.vit,
                    tie_attention_mlp_only=tie_attention_mlp_only,
                    tie_weight_groups=tie_weight_groups,
                )
            # Re-apply across-model weight tying
            tie_weights_across_models(model)
            print(
                "[model] ✓ Re-applied weight tying (within and across models) after low-rank injection"
            )
        else:
            tie_vit_layer_weights(
                model.vit,
                tie_attention_mlp_only=tie_attention_mlp_only,
                tie_weight_groups=tie_weight_groups,
            )
            print("[model] ✓ Re-applied weight tying after low-rank injection")

    # Recalculate parameter counts after low-rank injection
    if isinstance(model, list):
        total_params_lr = sum(p.numel() for m in model for p in m.parameters())
        trainable_params_lr = sum(
            p.numel() for m in model for p in m.parameters() if p.requires_grad
        )
        print(
            f"[model] After low-rank - Total params: {total_params_lr:,}, "
            f"Trainable: {trainable_params_lr:,} (reduced by "
            f"{100 * (1 - total_params_lr / total_params):.1f}%)\n"
        )
    else:
        total_params_lr = sum(p.numel() for p in model.parameters())
        trainable_params_lr = sum(
            p.numel() for p in model.parameters() if p.requires_grad
        )
        print(
            f"[model] After low-rank - Total params: {total_params_lr:,}, "
            f"Trainable: {trainable_params_lr:,} (reduced by "
            f"{100 * (1 - total_params_lr / total_params):.1f}%)\n"
        )


