from typing import Tuple

import torch

from utils.tensor_utils import group, degroup


def int_quantizer(tensor: torch.Tensor,
                  quantization_bit: int,
                  dim: int,
                  min_val: float,
                  max_val: float,
                  mask: torch.Tensor,
                  method: str) -> torch.Tensor:
    if method == "minmax":
        if min_val is None and max_val is None:
            masked_tensor = tensor.clone().masked_fill(~mask, float("inf"))
            min_val, _ = masked_tensor.min(dim=dim, keepdim=True)
            masked_tensor = masked_tensor.masked_fill(~mask, float("-inf"))
            max_val, _ = masked_tensor.max(dim=dim, keepdim=True)

        quant_max = 2**quantization_bit - 1
        scale = (max_val - min_val) / quant_max
        scale.clamp_(min=1e-5)

        quantized_scaled = torch.round((tensor - min_val) / scale).clamp(0, quant_max)
        quantized_tensor = quantized_scaled * scale + min_val
    elif method == "absmax":
        scale = tensor.clone().masked_fill(~mask, float("-inf")).abs().max(dim=dim, keepdim=True)[0]
        quant_max = 2**(quantization_bit - 1) - 1
        scale.clamp_(min=1e-5).div_(quant_max)
        quantized_tensor = tensor.div(scale).round_().mul_(scale)
    else:
        raise ValueError(f"Unsupported quantization method: {method}")

    return quantized_tensor


def activation_quantizer(input_args: Tuple[torch.Tensor, ...],
                         group_size: int = 1024,
                         quantization_bit: int = 4) -> torch.Tensor:
    original_shape = input_args[0].shape

    flattened_inputs = input_args[0].reshape(-1, original_shape[-1])
    grouped_inputs, mask = group(flattened_inputs, group_size=group_size)
    quantized_grouped_inputs = int_quantizer(grouped_inputs, quantization_bit, -1, None, None, mask, method="minmax")
    quantized_inputs = degroup(quantized_grouped_inputs, group_size, flattened_inputs.shape)
    return quantized_inputs.reshape(original_shape)
