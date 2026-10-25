import os

import torch

from rotation.fuse_rotation import fuse_rotation
from rotation.rotation_matrix import load_or_create_R1
from utils.rotation_utils import fuse_weight


def apply_finetuned_rotation(model, model_name: str, model_prefix: str, hidden_size: int, rotation_dir: str, device: str) -> str:
    """Fold the RMSNorm weights, the saved R1 rotation and the saved scaling vectors into ``model`` in place.

    Args:
        model: Causal LM to transform.
        model_name: Hugging Face model name; selects the Phi or Qwen layer layout.
        model_prefix: Model name without the organization, used in the checkpoint file name.
        hidden_size: Hidden size of ``model``.
        rotation_dir: Directory holding ``<model_prefix>_r1_finetune.pt``.
        device: Device the rotation matrix is loaded on.

    Returns:
        Path of the rotation checkpoint that was applied.
    """
    fuse_weight(model, model_name)
    checkpoint_path = os.path.join(rotation_dir, f"{model_prefix}_r1_finetune.pt")
    rotation_matrix = load_or_create_R1(
        mode="offline",
        dim=hidden_size,
        device=device,
        checkpoint_path=checkpoint_path,
    ).weight.detach()
    rotation_by_device = {
        f"cuda:{device_idx}": rotation_matrix.to(f"cuda:{device_idx}")
        for device_idx in range(torch.cuda.device_count())
    }
    fuse_rotation(model, rotation_by_device, model_name)

    checkpoint = torch.load(checkpoint_path, weights_only=False, map_location="cpu")
    scaling_vectors = checkpoint.get("scaling_vectors")
    if scaling_vectors:
        apply_scaling_vectors(model, model_name, scaling_vectors)
    return checkpoint_path


def apply_scaling_vectors(model, model_name: str, scaling_vectors: dict) -> None:
    """Divide each RMSNorm by its scaling vector and multiply the linear layers it feeds by the same vector.

    Args:
        model: Causal LM to transform in place.
        model_name: Hugging Face model name; selects the Phi or Qwen layer layout.
        scaling_vectors: Mapping from RMSNorm module name to its scaling vector.
    """
    is_phi = "Phi" in model_name

    with torch.no_grad():
        for name, module in model.named_modules():
            attention_scale = scaling_vectors.get(name + ".input_layernorm")
            if attention_scale is not None:
                _apply_attention_scaling(module, attention_scale, is_phi)

            ffn_scale = scaling_vectors.get(name + ".post_attention_layernorm")
            if ffn_scale is not None:
                _apply_ffn_scaling(module, ffn_scale, is_phi)


def _apply_attention_scaling(module, scale: torch.Tensor, is_phi: bool) -> None:
    ln = module.input_layernorm
    ln.weight.data.div_(scale.to(ln.weight.device))
    if hasattr(ln, "bias") and ln.bias is not None:
        ln.bias.data.div_(scale.to(ln.bias.device))

    if is_phi:
        fc = module.self_attn.qkv_proj
        fc.weight.data.mul_(scale.view(1, -1).to(dtype=fc.weight.dtype, device=fc.weight.device))
        return

    for fc in [module.self_attn.q_proj, module.self_attn.k_proj, module.self_attn.v_proj]:
        fc.weight.data.mul_(scale.view(1, -1).to(dtype=fc.weight.dtype, device=fc.weight.device))


def _apply_ffn_scaling(module, scale: torch.Tensor, is_phi: bool) -> None:
    ln = module.post_attention_layernorm
    ln.weight.data.div_(scale.to(ln.weight.device))
    if hasattr(ln, "bias") and ln.bias is not None:
        ln.bias.data.div_(scale.to(ln.bias.device))

    if is_phi:
        fc = module.mlp.gate_up_proj
        fc.weight.data.mul_(scale.view(1, -1).to(dtype=fc.weight.dtype, device=fc.weight.device))
        return

    for fc in [module.mlp.gate_proj, module.mlp.up_proj]:
        fc.weight.data.mul_(scale.view(1, -1).to(dtype=fc.weight.dtype, device=fc.weight.device))
