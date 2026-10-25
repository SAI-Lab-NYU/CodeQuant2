import gc
from typing import Dict

import torch
import torch.nn as nn
from tqdm import tqdm


def _rotate_input(layer: nn.Module, r1_rotation_dict_cache: Dict[str, torch.Tensor]) -> None:
    """Replace the weight W with W @ R1 so the layer consumes rotated inputs x @ R1."""
    dtype = layer.weight.data.dtype
    device = layer.weight.data.device

    R1 = r1_rotation_dict_cache[str(device)]

    W_ = layer.weight.data.to(device=device, dtype=torch.float64)
    R1 = R1.to(device=device, dtype=torch.float64)
    layer.weight.data = torch.matmul(W_, R1).to(device=device, dtype=dtype)
    del W_


def _rotate_output(layer: nn.Linear, r1_rotation_dict_cache: Dict[str, torch.Tensor]) -> None:
    """Replace the weight W with R1^T @ W, and the bias b with R1^T b, so the layer emits rotated outputs."""
    dtype = layer.weight.data.dtype
    device = layer.weight.data.device

    R1 = r1_rotation_dict_cache[str(device)]

    W_ = layer.weight.data.to(device=device, dtype=torch.float64)
    R1 = R1.to(device=device, dtype=torch.float64)
    layer.weight.data = torch.matmul(R1.T, W_).to(device=device, dtype=dtype)
    if layer.bias is not None:
        b = layer.bias.data.to(device=device, dtype=torch.float64)
        layer.bias.data = torch.matmul(R1.T, b).to(device=device, dtype=dtype)
        del b
    del W_


def fuse_rotation(model: nn.Module,
                  r1_rotation_dict_cache: Dict[str, torch.Tensor],
                  model_name) -> None:
    """Fold the shared rotation R1 into the embeddings, the head and every decoder layer in place.

    Args:
        model: Causal LM whose decoder layers expose ``self_attn`` and ``mlp``.
        r1_rotation_dict_cache: The rotation matrix keyed by ``str(device)`` for
            every device that holds model weights.
        model_name: Model name. Phi models use fused ``qkv_proj`` and
            ``gate_up_proj`` projections; every other model uses separate
            ``q/k/v_proj`` and ``gate/up_proj`` projections.
    """
    _rotate_input(model.model.embed_tokens, r1_rotation_dict_cache)
    _rotate_input(model.lm_head, r1_rotation_dict_cache)

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    layers = model.model.layers
    is_phi = 'Phi' in model_name

    for idx in tqdm(range(len(layers)), unit="layer", desc="Rotating"):
        layer = layers[idx]

        if is_phi:
            _rotate_input(layer.self_attn.qkv_proj, r1_rotation_dict_cache)
        else:
            for W in [layer.self_attn.q_proj, layer.self_attn.k_proj, layer.self_attn.v_proj]:
                _rotate_input(W, r1_rotation_dict_cache)
        _rotate_output(layer.self_attn.o_proj, r1_rotation_dict_cache)

        if is_phi:
            _rotate_input(layer.mlp.gate_up_proj, r1_rotation_dict_cache)
        else:
            _rotate_input(layer.mlp.up_proj, r1_rotation_dict_cache)
            _rotate_input(layer.mlp.gate_proj, r1_rotation_dict_cache)
        _rotate_output(layer.mlp.down_proj, r1_rotation_dict_cache)
        del layer

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
