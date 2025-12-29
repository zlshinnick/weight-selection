"""Factory functions for building model components."""

import sys
from pathlib import Path
import torch.nn as nn

# Add parent directory to path to import utils
sys.path.insert(0, str(Path(__file__).parent.parent))
import utils

from .wrapper import TokenViTFixedPos
from .wrapper_swin import TokenSwinFixedPos
from .wrapper_convnext import TokenConvNeXtFixedPos
from .wrapper_convit import TokenConViTFixedPos
from .utils import set_all_seeds


def _tie_block_to_reference(
    target_block, reference_block, tie_attention_mlp_only=False
):
    """Helper function to tie a target block's weights to a reference block.

    Args:
        target_block: The block whose weights will be tied (tied to reference)
        reference_block: The reference block (source of weights)
        tie_attention_mlp_only: If True, only tie attention and MLP weights
    """
    if tie_attention_mlp_only:
        # Only tie attention and MLP weights, not norms or other parameters
        # Get reference attention and MLP parameters
        ref_attn_params = dict(reference_block.attn.named_parameters())
        ref_attn_buffers = dict(reference_block.attn.named_buffers())
        ref_mlp_params = dict(reference_block.mlp.named_parameters())
        ref_mlp_buffers = dict(reference_block.mlp.named_buffers())

        # Tie attention parameters
        for name, param in list(target_block.attn.named_parameters()):
            if name in ref_attn_params:
                parts = name.split(".")
                module = target_block.attn
                for part in parts[:-1]:
                    module = getattr(module, part)
                param_name = parts[-1]
                ref_param = ref_attn_params[name]
                if param_name in module._parameters:
                    del module._parameters[param_name]
                module._parameters[param_name] = ref_param
                setattr(module, param_name, ref_param)


        # Tie MLP parameters
        for name, param in list(target_block.mlp.named_parameters()):
            if name in ref_mlp_params:
                parts = name.split(".")
                module = target_block.mlp
                for part in parts[:-1]:
                    module = getattr(module, part)
                param_name = parts[-1]
                ref_param = ref_mlp_params[name]
                if param_name in module._parameters:
                    del module._parameters[param_name]
                module._parameters[param_name] = ref_param
                setattr(module, param_name, ref_param)


    else:
        # Tie all parameters (original behavior)
        ref_params = dict(reference_block.named_parameters())
        
        # Tie all parameters by replacing them with references
        for name, param in list(target_block.named_parameters()):
            if name in ref_params:
                # Get the module path and parameter name
                parts = name.split(".")
                module = target_block
                for part in parts[:-1]:
                    module = getattr(module, part)
                param_name = parts[-1]

                # Replace the parameter with a reference to the reference block's parameter
                ref_param = ref_params[name]
                # Remove old parameter and register the shared one
                if param_name in module._parameters:
                    del module._parameters[param_name]
                module._parameters[param_name] = ref_param
                # Also update the attribute
                setattr(module, param_name, ref_param)



def _verify_weight_tying(model, tie_weight_groups, tie_attention_mlp_only=False):
    """Verify that weights are actually tied by checking object identity.

    Args:
        model: A VisionTransformer model with a 'blocks' attribute.
        tie_weight_groups: List of lists specifying layer groups.
        tie_attention_mlp_only: Whether only attention/MLP weights were tied.

    Raises:
        AssertionError: If weights are not properly tied.
    """
    blocks = model.blocks

    for group in tie_weight_groups:
        if len(group) == 1:
            continue  # Single layer group - nothing to verify

        ref_idx = group[0]
        reference_block = blocks[ref_idx]

        # Verify all other layers in the group are tied to the reference
        for target_idx in group[1:]:
            target_block = blocks[target_idx]

            if tie_attention_mlp_only:
                # Only check attention and MLP parameters
                ref_attn_params = dict(reference_block.attn.named_parameters())
                target_attn_params = dict(target_block.attn.named_parameters())
                ref_mlp_params = dict(reference_block.mlp.named_parameters())
                target_mlp_params = dict(target_block.mlp.named_parameters())

                # Verify attention parameters are tied
                for name, ref_param in ref_attn_params.items():
                    if name in target_attn_params:
                        target_param = target_attn_params[name]
                        assert ref_param is target_param, (
                            f"Layer {target_idx} attention.{name} is not tied to layer {ref_idx}. "
                            f"Expected same object (id: {id(ref_param)}), got id: {id(target_param)}"
                        )

                # Verify MLP parameters are tied
                for name, ref_param in ref_mlp_params.items():
                    if name in target_mlp_params:
                        target_param = target_mlp_params[name]
                        assert ref_param is target_param, (
                            f"Layer {target_idx} mlp.{name} is not tied to layer {ref_idx}. "
                            f"Expected same object (id: {id(ref_param)}), got id: {id(target_param)}"
                        )
            else:
                # Check all parameters
                ref_params = dict(reference_block.named_parameters())
                target_params = dict(target_block.named_parameters())

                for name, ref_param in ref_params.items():
                    if name in target_params:
                        target_param = target_params[name]
                        assert ref_param is target_param, (
                            f"Layer {target_idx}.{name} is not tied to layer {ref_idx}. "
                            f"Expected same object (id: {id(ref_param)}), got id: {id(target_param)}"
                        )


def tie_vit_layer_weights(
    model, tie_attention_mlp_only=False, tie_weight_groups=None, verify=True
):
    """Tie weights of layers in a ViT model according to specified groups.

    Args:
        model: A VisionTransformer model with a 'blocks' attribute.
        tie_attention_mlp_only: If True, only tie attention and MLP weights,
            leaving norms and other parameters independent. If False, tie all parameters.
        tie_weight_groups: Optional list of lists specifying layer groups to tie.
            Each inner list represents a group of layer indices that will be tied together,
            where the first layer in each group is used as the reference.
            Example: [[0], [1, 2, 3, 4, 5, 6, 7, 8, 9, 10], [11]] means:
                - Layer 0 is independent
                - Layers 1-10 are tied (layer 1 is reference)
                - Layer 11 is independent
            If None, uses default behavior: layer 0 untied, layer 1 reference, layers 2+ tied to layer 1.
        verify: If True, verify that weights are actually tied using assertions. Default: True.
    """
    if not hasattr(model, "blocks"):
        raise ValueError(
            "Model does not have a 'blocks' attribute. Expected ViT model."
        )

    blocks = model.blocks
    num_layers = len(blocks)

    # Determine weight tying groups
    if tie_weight_groups is None:
        # Default behavior: layer 0 untied, layer 1 reference, layers 2+ tied to layer 1
        if num_layers < 3:
            raise ValueError(
                "Model must have at least 3 layers to tie weights (layer 0 untied, layer 1 reference, layers 2+ tied)."
            )
        tie_weight_groups = [
            [0],
            list(range(1, num_layers)),
        ]  # [0] independent, [1, 2, ..., n-1] tied
    else:
        # Validate custom groups
        all_indices = []
        for group in tie_weight_groups:
            if not isinstance(group, list) or len(group) == 0:
                raise ValueError(f"Each group must be a non-empty list, got: {group}")
            all_indices.extend(group)

        # Check for duplicates
        if len(all_indices) != len(set(all_indices)):
            raise ValueError("Duplicate layer indices found in tie_weight_groups")

        # Check all indices are valid
        max_idx = max(all_indices) if all_indices else -1
        if max_idx >= num_layers:
            raise ValueError(
                f"Layer index {max_idx} exceeds model depth ({num_layers})"
            )
        if min(all_indices) < 0:
            raise ValueError("Layer indices must be non-negative")

    # Apply weight tying for each group
    tied_groups = []
    independent_layers = []

    for group in tie_weight_groups:
        if len(group) == 1:
            # Single layer group - no tying needed (layer is independent)
            independent_layers.append(group[0])
            continue

        # First layer in group is the reference
        ref_idx = group[0]
        reference_block = blocks[ref_idx]

        tied_groups.append((ref_idx, group[1:]))

        # Tie all other layers in the group to the reference
        for target_idx in group[1:]:
            target_block = blocks[target_idx]
            _tie_block_to_reference(
                target_block,
                reference_block,
                tie_attention_mlp_only=tie_attention_mlp_only,
            )

    # Log the weight tying configuration
    print("[model] Weight tying groups applied:")
    if independent_layers:
        print(f"  - Independent layers: {independent_layers}")
    for ref_idx, tied_indices in tied_groups:
        tie_type = (
            "attention & MLP only" if tie_attention_mlp_only else "all parameters"
        )
        print(f"  - Layers {tied_indices} tied to layer {ref_idx} ({tie_type})")

    # Verify weight tying if requested
    if verify:
        _verify_weight_tying(model, tie_weight_groups, tie_attention_mlp_only)
        print("[model] ✓ Weight tying verification passed (all weights properly tied)")


def tie_weights_across_models(models):
    """Tie attention and MLP weight matrices across n models.

    For each layer, ties only:
    - Attention: qkv.weight (or qkv.A, qkv.B for LowRankLinear), qkv.bias (if exists),
                 proj.weight (or proj.A, proj.B for LowRankLinear), proj.bias (if exists)
    - MLP: fc1.weight (or fc1.A, fc1.B for LowRankLinear), fc1.bias (if exists),
           fc2.weight (or fc2.A, fc2.B for LowRankLinear), fc2.bias (if exists)
    """
    if len(models) < 2:
        raise ValueError("At least 2 models are required to tie weights")

    # Get the first model's structure as reference
    ref_model = models[0]
    ref_blocks = ref_model.vit.blocks
    num_layers = len(ref_blocks)

    # For each layer, tie only attention and MLP weight matrices across all models
    for layer_idx in range(num_layers):
        ref_block = ref_blocks[layer_idx]

        # Get reference attention weight matrices: qkv and proj
        # Handle both nn.Linear (has .weight) and LowRankLinear (has .A and .B)
        ref_qkv = ref_block.attn.qkv
        ref_qkv_weight = getattr(ref_qkv, "weight", None)
        ref_qkv_A = getattr(ref_qkv, "A", None)
        ref_qkv_B = getattr(ref_qkv, "B", None)
        ref_qkv_bias = (
            ref_block.attn.qkv.bias if ref_block.attn.qkv.bias is not None else None
        )
        ref_proj = ref_block.attn.proj
        ref_proj_weight = getattr(ref_proj, "weight", None)
        ref_proj_A = getattr(ref_proj, "A", None)
        ref_proj_B = getattr(ref_proj, "B", None)
        ref_proj_bias = (
            ref_block.attn.proj.bias if ref_block.attn.proj.bias is not None else None
        )

        # Get reference MLP weight matrices: fc1 and fc2
        ref_fc1 = ref_block.mlp.fc1
        ref_fc1_weight = getattr(ref_fc1, "weight", None)
        ref_fc1_A = getattr(ref_fc1, "A", None)
        ref_fc1_B = getattr(ref_fc1, "B", None)
        ref_fc1_bias = (
            ref_block.mlp.fc1.bias if ref_block.mlp.fc1.bias is not None else None
        )
        ref_fc2 = ref_block.mlp.fc2
        ref_fc2_weight = getattr(ref_fc2, "weight", None)
        ref_fc2_A = getattr(ref_fc2, "A", None)
        ref_fc2_B = getattr(ref_fc2, "B", None)
        ref_fc2_bias = (
            ref_block.mlp.fc2.bias if ref_block.mlp.fc2.bias is not None else None
        )

        # Tie weights in all other models to the reference model
        for model_idx in range(1, len(models)):
            target_model = models[model_idx]
            target_block = target_model.vit.blocks[layer_idx]

            # Tie attention qkv weight and bias
            target_qkv = target_block.attn.qkv
            if ref_qkv_weight is not None:
                # Regular Linear layer
                if "weight" in target_qkv._parameters:
                    del target_qkv._parameters["weight"]
                target_qkv._parameters["weight"] = ref_qkv_weight
                target_qkv.weight = ref_qkv_weight
            elif ref_qkv_A is not None and ref_qkv_B is not None:
                # LowRankLinear layer - tie A and B
                if "A" in target_qkv._parameters:
                    del target_qkv._parameters["A"]
                target_qkv._parameters["A"] = ref_qkv_A
                target_qkv.A = ref_qkv_A
                if "B" in target_qkv._parameters:
                    del target_qkv._parameters["B"]
                target_qkv._parameters["B"] = ref_qkv_B
                target_qkv.B = ref_qkv_B

            if ref_qkv_bias is not None:
                if "bias" in target_qkv._parameters:
                    del target_qkv._parameters["bias"]
                target_qkv._parameters["bias"] = ref_qkv_bias
                target_qkv.bias = ref_qkv_bias

            # Tie attention proj weight and bias
            target_proj = target_block.attn.proj
            if ref_proj_weight is not None:
                # Regular Linear layer
                if "weight" in target_proj._parameters:
                    del target_proj._parameters["weight"]
                target_proj._parameters["weight"] = ref_proj_weight
                target_proj.weight = ref_proj_weight
            elif ref_proj_A is not None and ref_proj_B is not None:
                # LowRankLinear layer - tie A and B
                if "A" in target_proj._parameters:
                    del target_proj._parameters["A"]
                target_proj._parameters["A"] = ref_proj_A
                target_proj.A = ref_proj_A
                if "B" in target_proj._parameters:
                    del target_proj._parameters["B"]
                target_proj._parameters["B"] = ref_proj_B
                target_proj.B = ref_proj_B

            if ref_proj_bias is not None:
                if "bias" in target_proj._parameters:
                    del target_proj._parameters["bias"]
                target_proj._parameters["bias"] = ref_proj_bias
                target_proj.bias = ref_proj_bias

            # Tie MLP fc1 weight and bias
            target_fc1 = target_block.mlp.fc1
            if ref_fc1_weight is not None:
                # Regular Linear layer
                if "weight" in target_fc1._parameters:
                    del target_fc1._parameters["weight"]
                target_fc1._parameters["weight"] = ref_fc1_weight
                target_fc1.weight = ref_fc1_weight
            elif ref_fc1_A is not None and ref_fc1_B is not None:
                # LowRankLinear layer - tie A and B
                if "A" in target_fc1._parameters:
                    del target_fc1._parameters["A"]
                target_fc1._parameters["A"] = ref_fc1_A
                target_fc1.A = ref_fc1_A
                if "B" in target_fc1._parameters:
                    del target_fc1._parameters["B"]
                target_fc1._parameters["B"] = ref_fc1_B
                target_fc1.B = ref_fc1_B

            if ref_fc1_bias is not None:
                if "bias" in target_fc1._parameters:
                    del target_fc1._parameters["bias"]
                target_fc1._parameters["bias"] = ref_fc1_bias
                target_fc1.bias = ref_fc1_bias

            # Tie MLP fc2 weight and bias
            target_fc2 = target_block.mlp.fc2
            if ref_fc2_weight is not None:
                # Regular Linear layer
                if "weight" in target_fc2._parameters:
                    del target_fc2._parameters["weight"]
                target_fc2._parameters["weight"] = ref_fc2_weight
                target_fc2.weight = ref_fc2_weight
            elif ref_fc2_A is not None and ref_fc2_B is not None:
                # LowRankLinear layer - tie A and B
                if "A" in target_fc2._parameters:
                    del target_fc2._parameters["A"]
                target_fc2._parameters["A"] = ref_fc2_A
                target_fc2.A = ref_fc2_A
                if "B" in target_fc2._parameters:
                    del target_fc2._parameters["B"]
                target_fc2._parameters["B"] = ref_fc2_B
                target_fc2.B = ref_fc2_B

            if ref_fc2_bias is not None:
                if "bias" in target_fc2._parameters:
                    del target_fc2._parameters["bias"]
                target_fc2._parameters["bias"] = ref_fc2_bias
                target_fc2.bias = ref_fc2_bias

    # Verify that weights are correctly tied across all models
    ref_blocks = ref_model.vit.blocks
    for layer_idx in range(num_layers):
        ref_block = ref_blocks[layer_idx]
        ref_qkv = ref_block.attn.qkv
        ref_proj = ref_block.attn.proj
        ref_fc1 = ref_block.mlp.fc1
        ref_fc2 = ref_block.mlp.fc2

        # Verify all models share the same weight tensors (by reference)
        for model_idx in range(1, len(models)):
            target_block = models[model_idx].vit.blocks[layer_idx]
            target_qkv = target_block.attn.qkv
            target_proj = target_block.attn.proj
            target_fc1 = target_block.mlp.fc1
            target_fc2 = target_block.mlp.fc2

            # Check qkv (handle both Linear and LowRankLinear)
            if hasattr(ref_qkv, "weight"):
                assert id(target_qkv.weight) == id(ref_qkv.weight), (
                    f"Layer {layer_idx}: Model {model_idx} qkv.weight not tied to reference model"
                )
            elif hasattr(ref_qkv, "A"):
                assert id(target_qkv.A) == id(ref_qkv.A), (
                    f"Layer {layer_idx}: Model {model_idx} qkv.A not tied to reference model"
                )
                assert id(target_qkv.B) == id(ref_qkv.B), (
                    f"Layer {layer_idx}: Model {model_idx} qkv.B not tied to reference model"
                )

            # Check proj
            if hasattr(ref_proj, "weight"):
                assert id(target_proj.weight) == id(ref_proj.weight), (
                    f"Layer {layer_idx}: Model {model_idx} proj.weight not tied to reference model"
                )
            elif hasattr(ref_proj, "A"):
                assert id(target_proj.A) == id(ref_proj.A), (
                    f"Layer {layer_idx}: Model {model_idx} proj.A not tied to reference model"
                )
                assert id(target_proj.B) == id(ref_proj.B), (
                    f"Layer {layer_idx}: Model {model_idx} proj.B not tied to reference model"
                )

            # Check fc1
            if hasattr(ref_fc1, "weight"):
                assert id(target_fc1.weight) == id(ref_fc1.weight), (
                    f"Layer {layer_idx}: Model {model_idx} fc1.weight not tied to reference model"
                )
            elif hasattr(ref_fc1, "A"):
                assert id(target_fc1.A) == id(ref_fc1.A), (
                    f"Layer {layer_idx}: Model {model_idx} fc1.A not tied to reference model"
                )
                assert id(target_fc1.B) == id(ref_fc1.B), (
                    f"Layer {layer_idx}: Model {model_idx} fc1.B not tied to reference model"
                )

            # Check fc2
            if hasattr(ref_fc2, "weight"):
                assert id(target_fc2.weight) == id(ref_fc2.weight), (
                    f"Layer {layer_idx}: Model {model_idx} fc2.weight not tied to reference model"
                )
            elif hasattr(ref_fc2, "A"):
                assert id(target_fc2.A) == id(ref_fc2.A), (
                    f"Layer {layer_idx}: Model {model_idx} fc2.A not tied to reference model"
                )
                assert id(target_fc2.B) == id(ref_fc2.B), (
                    f"Layer {layer_idx}: Model {model_idx} fc2.B not tied to reference model"
                )

            # Check biases if they exist
            if ref_block.attn.qkv.bias is not None:
                assert id(target_block.attn.qkv.bias) == id(ref_block.attn.qkv.bias), (
                    f"Layer {layer_idx}: Model {model_idx} qkv.bias not tied to reference model"
                )
            if ref_block.attn.proj.bias is not None:
                assert id(target_block.attn.proj.bias) == id(
                    ref_block.attn.proj.bias
                ), (
                    f"Layer {layer_idx}: Model {model_idx} proj.bias not tied to reference model"
                )
            if ref_block.mlp.fc1.bias is not None:
                assert id(target_block.mlp.fc1.bias) == id(ref_block.mlp.fc1.bias), (
                    f"Layer {layer_idx}: Model {model_idx} fc1.bias not tied to reference model"
                )
            if ref_block.mlp.fc2.bias is not None:
                assert id(target_block.mlp.fc2.bias) == id(ref_block.mlp.fc2.bias), (
                    f"Layer {layer_idx}: Model {model_idx} fc2.bias not tied to reference model"
                )


class _ArgsAdapter:
    """Adapter to convert cfg structure to args structure for utils.build_model."""

    def __init__(self, cfg):
        self.model = cfg.model.name
        self.nb_classes = cfg.model.num_classes
        self.drop_path = cfg.model.drop_path
        self.layer_scale_init_value = cfg.model.layer_scale_init_value
        self.head_init_scale = cfg.model.head_init_scale
        # Override model depth if specified
        self.depth = getattr(cfg.model, "depth", None)
        # For ConvNeXt, set in_chans to token embed dim so wrapper can feed BCHW
        if str(cfg.model.name).startswith("convnext"):
            self.in_chans = cfg.model.embed_dim
        # Pass mimetic init flags through to utils.build_model
        self.mimetic_init = getattr(cfg.model, "mimetic_init", False)
        self.mimetic_alpha = getattr(cfg.model, "mimetic_alpha", 0.4)
        self.mimetic_beta = getattr(cfg.model, "mimetic_beta", 0.4)
        self.mimetic_dist = getattr(cfg.model, "mimetic_dist", "uniform")


def make_backbone(cfg):
    """Create a Vision Transformer backbone using utils.build_model.

    Args:
        cfg: Configuration object containing model settings.

    Returns:
        A model instance created via utils.build_model.
        Model is always initialized from scratch (pretrained=False) to match main.py behavior.
    """
    args_adapter = _ArgsAdapter(cfg)
    return utils.build_model(args_adapter)


def make_mlm_head(embed_dim: int, K: int):
    """Create a masked language modeling head.

    Args:
        embed_dim: Dimension of the input embeddings.
        K: Vocabulary size (number of output classes).

    Returns:
        A linear layer mapping from embed_dim to K classes.
    """
    return nn.Linear(embed_dim, K)


def build_model(cfg, tok_embed: nn.Module, pos_embed: nn.Module):
    """Build the complete model with backbone, embeddings, and MLM head.

    Args:
        cfg: Configuration object containing model, grid, and vocab settings.
        tok_embed: Token embedding module.
        pos_embed: Positional embedding module.

    Returns:
        A tuple of (wrapped_model, mlm_head) where:
            - wrapped_model: TokenViTFixedPos wrapper around the ViT backbone.
            - mlm_head: Linear layer for masked language modeling predictions.
    """
    backbone = make_backbone(cfg)

    model_name = str(cfg.model.name).lower()
    use_swin = bool(getattr(cfg.model, "use_swin", False))
    if model_name.startswith("convnext"):
        # Expect ConvNeXt-like structure
        if not (hasattr(backbone, "num_features")):
            raise ValueError(
                "cfg.model.name startswith 'convnext' but backbone missing num_features."
            )
        wrap = TokenConvNeXtFixedPos(
            convnext_backbone=backbone,
            tok_embed=tok_embed,
            H=cfg.grid.H,
            W=cfg.grid.W,
            pos_embed_2d=pos_embed,
            return_tokens=True,
        )
        feat_dim = getattr(backbone, "num_features")
        mlm = make_mlm_head(feat_dim, cfg.vocab.K)
        return wrap, mlm
    elif model_name.startswith("convit"):
        if not hasattr(backbone, "blocks"):
            raise ValueError(
                "cfg.model.name startswith 'convit' but backbone missing transformer blocks."
            )
        wrap = TokenConViTFixedPos(
            convit_backbone=backbone,
            tok_embed=tok_embed,
            pos_embed=pos_embed,
            H=cfg.grid.H,
            W=cfg.grid.W,
        )
        feat_dim = getattr(
            backbone, "embed_dim", getattr(backbone, "num_features", None)
        )
        if feat_dim is None:
            raise ValueError("Unable to infer ConViT feature dimension.")
        mlm = make_mlm_head(feat_dim, cfg.vocab.K)
        return wrap, mlm
    elif use_swin:
        # Basic sanity: expect Swin-like structure
        if not hasattr(backbone, "layers") or not hasattr(backbone, "norm"):
            raise ValueError(
                "cfg.model.use_swin=True but cfg.model.name is not a Swin model. "
                "Set cfg.model.name to a timm Swin variant (e.g., 'swin_tiny_patch4_window7_224')."
            )
        wrap = TokenSwinFixedPos(
            swin_backbone=backbone,
            tok_embed=tok_embed,
            H=cfg.grid.H,
            W=cfg.grid.W,
            pos_embed_2d=pos_embed,
        )
        feat_dim = getattr(backbone, "num_features", getattr(backbone, "embed_dim"))
        mlm = make_mlm_head(feat_dim, cfg.vocab.K)
        return wrap, mlm
    else:
        vit = backbone
        # Apply weight tying if requested (only for ViT models)
        if getattr(cfg.model, "tie_weights", False):
            tie_attention_mlp_only = getattr(cfg.model, "tie_attention_mlp_only", False)
            tie_weight_groups = getattr(cfg.model, "tie_weight_groups", None)
            tie_vit_layer_weights(
                vit,
                tie_attention_mlp_only=tie_attention_mlp_only,
                tie_weight_groups=tie_weight_groups,
            )
        wrap = TokenViTFixedPos(
            vit_backbone=vit,
            tok_embed=tok_embed,
            pos_embed=pos_embed,
            H=cfg.grid.H,
            W=cfg.grid.W,
        )
        mlm = make_mlm_head(cfg.model.embed_dim, cfg.vocab.K)
        return wrap, mlm


def build_parallel_models(
    cfg, tok_embed: nn.Module, pos_embed: nn.Module, n_models: int
):
    """Build n parallel models with different seeds and tie weights across them.

    Args:
        cfg: Configuration object containing model, grid, and vocab settings.
        tok_embed: Token embedding module (shared across models).
        pos_embed: Positional embedding module (shared across models).
        n_models: Number of parallel models to create.

    Returns:
        A tuple of (models, mlm_heads) where:
            - models: List of n wrapped models
            - mlm_heads: List of n MLM heads
    """
    if n_models < 2:
        raise ValueError("n_models must be at least 2 for parallel training")

    base_seed = cfg.seed
    models = []
    mlm_heads = []

    # Build n models with different seeds
    for i in range(n_models):
        # Set seed for this model (base_seed + i)
        set_all_seeds(base_seed + i)

        # Build model and head
        model, mlm_head = build_model(cfg, tok_embed, pos_embed)
        models.append(model)
        mlm_heads.append(mlm_head)

    # Tie weights across all models (attention, MLP, and first layer)
    tie_weights_across_models(models)

    return models, mlm_heads
