"""
vLLM quantization plugin that fake-quantizes input activations to INT levels.

Inherits from UnquantizedLinearMethod so weight creation and loading are
handled by the parent class. Only apply() is overridden to quantize the input
activation before the matmul.

Quantization is applied only to layers whose names match _TARGET_PATTERN
(q/k/v/gate/up projections and their fused qkv/gate_up variants). Weights stay
in their original dtype.

Usage
-----
Set ACT_QUANT_BITS *before* importing this module (and before vllm):

    import os
    os.environ["ACT_QUANT_BITS"] = "4"

    import act_quant_plugin          # registers "act_quant" with vLLM
    from vllm import LLM

    llm = LLM(model="/path/to/model", quantization="act_quant")
"""

import os
import re
import sys
from typing import Optional

import torch
from torch import Tensor

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from quantization.fake_quant import int_quantizer

from vllm.model_executor.layers.linear import LinearBase, UnquantizedLinearMethod
from vllm.model_executor.layers.quantization import register_quantization_config
from vllm.model_executor.layers.quantization.base_config import (
    QuantizationConfig,
    QuantizeMethodBase,
)

_TARGET_PATTERN = re.compile(
    r".*(q_proj|k_proj|v_proj|qkv_proj|gate_proj|up_proj|gate_up_proj).*"
)


def _default_bits() -> int:
    return int(os.environ.get("ACT_QUANT_BITS", "8"))


class ActQuantLinearMethod(UnquantizedLinearMethod):
    """
    Inherits weight creation/loading from UnquantizedLinearMethod.
    Only apply() is overridden to quantize input activations before
    delegating to the parent's linear operation.
    """

    def __init__(self, bits: int):
        self.bits = bits

    def apply(
        self,
        layer: torch.nn.Module,
        x: Tensor,
        bias: Optional[Tensor] = None,
    ) -> Tensor:
        x = self._quantize_activation(x)
        return super().apply(layer, x, bias)

    def _quantize_activation(self, x: Tensor) -> Tensor:
        original_shape = x.shape
        flattened_x = x.reshape(-1, original_shape[-1])
        quantized_x = int_quantizer(flattened_x, quantization_bit=self.bits, dim=-1)
        return quantized_x.reshape(original_shape)


@register_quantization_config("act_quant")
class ActQuantConfig(QuantizationConfig):
    """
    Activation-only INT quantization.
    Weights stay in original dtype; only activations of target layers are quantized.
    """

    def __init__(self, bits: int = None):
        self.bits = bits if bits is not None else _default_bits()

    def get_name(self) -> str:
        return "act_quant"

    def get_supported_act_dtypes(self) -> list:
        return [torch.bfloat16, torch.float16]

    @classmethod
    def get_min_capability(cls) -> int:
        return 60  # Volta and above

    @staticmethod
    def get_config_filenames() -> list:
        return []  # no quantization_config.json needed

    @classmethod
    def from_config(cls, config: dict) -> "ActQuantConfig":
        return cls(bits=config.get("bits", _default_bits()))

    def get_quant_method(
        self, layer: torch.nn.Module, prefix: str
    ) -> Optional[QuantizeMethodBase]:
        if isinstance(layer, LinearBase):
            if _TARGET_PATTERN.search(prefix):
                return ActQuantLinearMethod(self.bits)
            return UnquantizedLinearMethod()
        return None

    def get_scaled_act_names(self) -> list:
        return []
