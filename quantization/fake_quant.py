import torch


def int_quantizer(tensor: torch.Tensor,
                  quantization_bit: int,
                  dim: int = -1,
                  ste: bool = True) -> torch.Tensor:
    """Fake-quantize ``tensor`` with asymmetric min/max integer levels along ``dim``.

    Args:
        tensor: Values to quantize.
        quantization_bit: Number of bits per value.
        dim: Dimension over which each min/max range is taken.
        ste: If True, round with a straight-through estimator so the gradient
            with respect to ``tensor`` is the identity. If False, use plain
            rounding so the gradient flows only through the min/max range.

    Returns:
        Dequantized values with the same shape and dtype as ``tensor``.
    """
    quant_max = 2 ** quantization_bit - 1
    if ste:
        with torch.no_grad():
            min_val, _ = tensor.min(dim=dim, keepdim=True)
            max_val, _ = tensor.max(dim=dim, keepdim=True)
            scale = (max_val - min_val) / quant_max
            scale.clamp_(min=1e-5)
        scaled_tensor = (tensor - min_val) / scale
        quantized_scaled = (
            torch.round(scaled_tensor).clamp(0, quant_max) - scaled_tensor
        ).detach() + scaled_tensor
    else:
        min_val, _ = tensor.min(dim=dim, keepdim=True)
        max_val, _ = tensor.max(dim=dim, keepdim=True)
        scale = (max_val - min_val) / quant_max
        scale.clamp_(min=1e-5)
        quantized_scaled = torch.round((tensor - min_val) / scale).clamp(0, quant_max)
    return quantized_scaled * scale + min_val


def activation_quantizer(inputs: torch.Tensor,
                         quantization_bit: int = 4,
                         ste: bool = True) -> torch.Tensor:
    """Fake-quantize activations per token, i.e. one min/max range per last-dim slice.

    Args:
        inputs: Activations of any shape; the last dimension is the channel axis.
        quantization_bit: Number of bits per value.
        ste: Passed through to :func:`int_quantizer`.

    Returns:
        Dequantized activations with the same shape and dtype as ``inputs``.
    """
    original_shape = inputs.shape
    flattened_inputs = inputs.reshape(-1, original_shape[-1])
    quantized_inputs = int_quantizer(flattened_inputs, quantization_bit, dim=-1, ste=ste)
    return quantized_inputs.reshape(original_shape)
