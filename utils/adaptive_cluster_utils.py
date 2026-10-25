from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from cluster.clustering import kmeans
from quantization.fake_quant import activation_quantizer


@torch.no_grad()
def reconstruct_from_centroids(weight: torch.Tensor, k: int, iters: int = 20) -> torch.Tensor:
    """Reconstruct a weight matrix via k-means centroid lookup."""
    centroids, assignments = kmeans(weight, k, iters)
    out_features, in_features = weight.size(0), weight.size(1)
    centroid_ids = assignments.to(dtype=torch.long).clamp(0, k - 1)
    row_ids = torch.arange(out_features).unsqueeze(1).expand(out_features, in_features)
    return centroids[row_ids, centroid_ids]


def get_layer_idx(name: str) -> int:
    parts = name.split(".")
    for i, part in enumerate(parts):
        if part == "layers" and i + 1 < len(parts):
            try:
                return int(parts[i + 1])
            except ValueError:
                pass
    return -1


def get_layer_latency(
    name: str,
    k: int,
    self_attn_latency: Dict[str, float],
    mlp_latency: Dict[str, float],
) -> float:
    lat_table = mlp_latency if "mlp" in name else self_attn_latency
    return float(lat_table[str(k)])


def tokenize_batch(texts: List[str], tokenizer, max_length: int, device: str) -> Dict:
    """Wrap each calibration text in the chat template and tokenize the batch."""
    processed = [
        tokenizer.apply_chat_template(
            [{"role": "user", "content": text}],
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        for text in texts
    ]
    return tokenizer(
        processed,
        return_tensors="pt",
        padding="max_length",
        truncation=True,
        max_length=max_length,
    ).to(device)


def reconstruction_loss(
    label: torch.Tensor,
    output: torch.Tensor,
    reduction: str = "mean",
) -> torch.Tensor:
    loss = (label - output).pow(2)
    if reduction == "mean":
        return loss.mean()
    if reduction == "sum":
        return loss.sum()
    if reduction == "none":
        return loss
    raise ValueError("reduction must be mean | sum | none")


class PreLinear(nn.Module):
    """Wraps a linear layer with k-means-reconstructed weights for each candidate cluster count.

    The forward pass uses the weight reconstructed with ``pinned_k`` centroids
    (or the original weight when ``pinned_k == -1``) on activation-quantized
    inputs, and records the input and the output gradient so that
    ``accumulate_importance`` can score a change of cluster count.
    """

    def __init__(
        self,
        linear: nn.Linear,
        name: str,
        activation_bit: int = 4,
        k_list: Tuple[int, ...] = (8, 16, 32),
        kmeans_iters: int = 100,
    ):
        super().__init__()
        self.linear = linear
        self.name = name
        self.k_list = tuple(sorted(set(map(int, k_list))))
        self.small, self.base, self.big = self.k_list[0], self.k_list[1], self.k_list[-1]
        self.activation_bit = activation_bit
        self.kmeans_iters = kmeans_iters
        self.pinned_k = self.base
        self._Wk: Dict[int, torch.Tensor] = {}
        self._last_x: Optional[torch.Tensor] = None
        self._last_gy: Optional[torch.Tensor] = None
        self.reset_scores()

    def reset_scores(self):
        self._sum_sq_ip_16_8 = 0.0
        self._sum_sq_ip_32_16 = 0.0
        self._count_16 = 0
        self._count_32 = 0

    @torch.no_grad()
    def build_Wk_cache(self):
        weight = self.linear.weight.detach()
        self._Wk = {
            k: reconstruct_from_centroids(weight, k=k, iters=self.kmeans_iters)
            for k in self.k_list
        }

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.pinned_k == -1:
            return F.linear(x, self.linear.weight, self.linear.bias)

        if self.pinned_k not in self._Wk:
            raise RuntimeError(f"[{self.name}] W{self.pinned_k} missing. Call build_Wk_cache() before forward.")

        xq = activation_quantizer(inputs=x, quantization_bit=self.activation_bit)
        self._last_x = xq.detach()

        weight = self._Wk[self.pinned_k].to(device=xq.device, dtype=xq.dtype)
        bias = self.linear.bias
        if bias is not None:
            bias = bias.to(device=xq.device, dtype=xq.dtype)

        output = F.linear(xq, weight, bias)
        if output.requires_grad:
            output.register_hook(self._save_grad)
        return output

    def _save_grad(self, grad: torch.Tensor) -> torch.Tensor:
        self._last_gy = grad.detach()
        return grad

    @torch.no_grad()
    def _accumulate_pair(self, weight_ref: torch.Tensor, weight_target: torch.Tensor) -> float:
        if self._last_x is None or self._last_gy is None:
            return 0.0
        x = self._last_x.reshape(-1, self._last_x.shape[-1])
        gy = self._last_gy.reshape(-1, self._last_gy.shape[-1])
        delta_weight = (weight_ref - weight_target).to(device=x.device, dtype=x.dtype)
        inner_product = (F.linear(x, delta_weight) * gy).sum(dim=-1)
        return float(inner_product.square().sum().item())

    @torch.no_grad()
    def accumulate_importance(self):
        if self._last_x is None or self._last_gy is None:
            return
        if self.pinned_k not in self._Wk:
            raise RuntimeError(f"[{self.name}] Wk cache missing. Call build_Wk_cache() first.")

        pinned_weight = self._Wk[self.pinned_k]
        n = self._last_x.reshape(-1, self._last_x.shape[-1]).shape[0]

        self._count_16 += n
        self._sum_sq_ip_16_8 += self._accumulate_pair(pinned_weight, self._Wk[self.small])
        self._count_32 += n
        self._sum_sq_ip_32_16 += self._accumulate_pair(pinned_weight, self._Wk[self.big])

        self._last_x = None
        self._last_gy = None

    def get_scores(self) -> Dict[str, float]:
        return {
            "I16_to_8": self._sum_sq_ip_16_8 / max(1, self._count_16),
            "I32_to_16": -self._sum_sq_ip_32_16 / max(1, self._count_32),
            "dim": float(self.linear.weight.numel()),
            "count_16": float(self._count_16),
            "count_32": float(self._count_32),
        }


def replace_linear_with_prelinear(
    model: nn.Module,
    k_list: Tuple[int, ...] = (8, 16, 32),
    kmeans_iters: int = 300,
    activation_bit: int = 4,
) -> Dict[str, PreLinear]:
    """Wrap every ``nn.Linear`` except ``lm_head``, ``down_proj`` and ``o_proj`` in a ``PreLinear``."""
    replaced = {}
    for name, module in list(model.named_modules()):
        if "lm_head" in name:
            continue
        if "down_proj" in name or "o_proj" in name:
            continue
        if not isinstance(module, nn.Linear):
            continue

        parts = name.split(".")
        parent = model
        for attr in parts[:-1]:
            parent = getattr(parent, attr)

        wrapper = PreLinear(
            module, name=name, activation_bit=activation_bit,
            k_list=k_list, kmeans_iters=kmeans_iters,
        )
        setattr(parent, parts[-1], wrapper)
        replaced[name] = wrapper
    return replaced


def allocate_cluster_latency(
    layer_scores: Dict[str, Dict[str, float]],
    self_attn_latency: Dict[str, float],
    mlp_latency: Dict[str, float],
    choices: Tuple[int, ...] = (12, 16, 20),
    latency_budget: float = 0.0,
    change_ratio: float = 0.15,
) -> Tuple[Dict[str, int], Dict[str, Any]]:
    """Pair one upgrade to the largest cluster count with one downgrade to the smallest.

    Matrices are visited in order of their importance scores; a pair is
    accepted only if it lowers the summed score, and at most ``change_ratio``
    of the matrices change in total.

    Returns:
        Tuple of the per-matrix cluster count and a summary dict with the
        latency and bit-width bookkeeping of the allocation.
    """
    small_k, base_k, big_k = choices[0], choices[1], choices[-1]
    name_small = f"I{base_k}_to_{small_k}"
    name_big = f"I{big_k}_to_{base_k}"

    result: Dict[str, int] = {name: base_k for name in layer_scores}
    scores_upgrade = {name: float(item.get(name_big, 0.0)) for name, item in layer_scores.items()}
    scores_downgrade = {name: float(item.get(name_small, 0.0)) for name, item in layer_scores.items()}
    sorted_upgrade = sorted(layer_scores.keys(), key=lambda n: scores_upgrade[n])
    sorted_downgrade = sorted(layer_scores.keys(), key=lambda n: scores_downgrade[n])

    max_changes = int(len(layer_scores) * change_ratio)
    changed = set()
    available = set(layer_scores.keys())
    latency_cost = 0.0
    upgrade_idx = 0
    downgrade_idx = 0

    while len(changed) < max_changes and upgrade_idx < len(sorted_upgrade) and downgrade_idx < len(sorted_downgrade):
        upgrade_name = None
        while upgrade_idx < len(sorted_upgrade):
            candidate = sorted_upgrade[upgrade_idx]
            upgrade_idx += 1
            if candidate in available:
                upgrade_name = candidate
                break
        if upgrade_name is None:
            break

        downgrade_name = None
        while downgrade_idx < len(sorted_downgrade):
            candidate = sorted_downgrade[downgrade_idx]
            downgrade_idx += 1
            if candidate in available and candidate != upgrade_name:
                downgrade_name = candidate
                break
        if downgrade_name is None:
            break

        net_delta = scores_upgrade[upgrade_name] + scores_downgrade[downgrade_name]
        if net_delta >= 0:
            continue

        available.remove(upgrade_name)
        result[upgrade_name] = big_k
        changed.add(upgrade_name)
        latency_cost += get_layer_latency(upgrade_name, big_k, self_attn_latency, mlp_latency) \
            - get_layer_latency(upgrade_name, base_k, self_attn_latency, mlp_latency)

        available.remove(downgrade_name)
        result[downgrade_name] = small_k
        changed.add(downgrade_name)
        latency_cost += get_layer_latency(downgrade_name, small_k, self_attn_latency, mlp_latency) \
            - get_layer_latency(downgrade_name, base_k, self_attn_latency, mlp_latency)

    base_latency = sum(get_layer_latency(n, base_k, self_attn_latency, mlp_latency) for n in layer_scores)
    total_latency = base_latency + latency_cost
    small_layers = [n for n, k in result.items() if k == small_k]
    big_layers = [n for n, k in result.items() if k == big_k]
    total_dim = sum(float(layer_scores[n].get("dim", 0.0)) for n in layer_scores)
    total_bits = sum(float(layer_scores[n].get("dim", 0.0)) * k for n, k in result.items())

    info = {
        "latency_budget": latency_budget,
        "change_ratio": change_ratio,
        "latency_cost": latency_cost,
        "base_latency": base_latency,
        "total_latency": total_latency,
        "small": len(small_layers),
        "large": len(big_layers),
        "changed": len(changed),
        "total_weights": len(layer_scores),
        "small_layers": small_layers,
        "big_layers": big_layers,
        "speedup": base_latency / total_latency if total_latency > 0 else 1.0,
        "average": total_bits / total_dim if total_dim > 0 else float(base_k),
    }
    return result, info
