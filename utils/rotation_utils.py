import torch
import torch.nn as nn


def _fuse_scale_into_linear(linear: nn.Linear, scale: torch.Tensor, tag=None):
    W = linear.weight.data
    if W.shape[1] != scale.numel():
        print(f"[WARNING] {tag} W.in={W.shape[1]} scale={scale.numel()} W.shape={tuple(W.shape)} device={W.device} dtype={W.dtype}")
        return
    W *= scale.unsqueeze(0).to(W.device, dtype=W.dtype)


def _set_rms_weight_ones(rms: nn.Module):
    w = getattr(rms, "weight", None)
    if isinstance(w, torch.Tensor):
        w.data.fill_(1.0)


def fuse_weight(model: nn.Module, model_prefix):
    """Fold every RMSNorm weight into the linear layers it feeds and reset the norm weight to ones.

    Args:
        model: Causal LM whose decoder layers expose ``input_layernorm``,
            ``post_attention_layernorm``, ``self_attn`` and ``mlp``.
        model_prefix: Model name. Phi models use fused ``qkv_proj`` and
            ``gate_up_proj`` projections; every other model uses separate
            ``q/k/v_proj`` and ``gate/up_proj`` projections.
    """
    layers = model.model.layers

    for i, layer in enumerate(layers):
        if 'Phi' in model_prefix:
            in_norm = getattr(layer, "input_layernorm")

            # self-attention
            attn = getattr(layer, "self_attn")
            scale = in_norm.weight.detach()

            prof_lists = ["qkv_proj"]
            for proj_name in prof_lists:
                proj = getattr(attn, proj_name)
                _fuse_scale_into_linear(proj, scale, f"layer_{i}_{proj_name}")
            _set_rms_weight_ones(in_norm)

            # mlp
            post_norm = getattr(layer, "post_attention_layernorm")
            scale = post_norm.weight.detach()
            mlp = getattr(layer, "mlp")
            up = getattr(mlp, "gate_up_proj")
            _fuse_scale_into_linear(up, scale, f"layer_{i}_up_proj")
            _set_rms_weight_ones(post_norm)
        else:
            in_norm = getattr(layer, "input_layernorm")

            # self-attention
            attn = getattr(layer, "self_attn")
            scale = in_norm.weight.detach()

            prof_lists = ["q_proj", "k_proj", "v_proj"]
            for proj_name in prof_lists:
                proj = getattr(attn, proj_name)
                _fuse_scale_into_linear(proj, scale, f"layer_{i}_{proj_name}")
            _set_rms_weight_ones(in_norm)

            # mlp
            post_norm = getattr(layer, "post_attention_layernorm")
            scale = post_norm.weight.detach()

            mlp = getattr(layer, "mlp")
            up = getattr(mlp, "up_proj")
            gate = getattr(mlp, "gate_proj")
            if isinstance(up, nn.Linear):
                _fuse_scale_into_linear(up, scale, f"layer_{i}_up_proj")
            if isinstance(gate, nn.Linear):
                _fuse_scale_into_linear(gate, scale, f"layer_{i}_gate_proj")
            _set_rms_weight_ones(post_norm)

    lang_model = model.model
    norm = getattr(lang_model, "norm")
    scale = norm.weight.detach()
    proj = getattr(model, "lm_head")
    _fuse_scale_into_linear(proj, scale, f"lm_head")
    _set_rms_weight_ones(norm)

    torch.cuda.empty_cache()
