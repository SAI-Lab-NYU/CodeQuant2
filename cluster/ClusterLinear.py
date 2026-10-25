import gc
from typing import Any, Dict, List

import torch
import torch.nn as nn
import torch.nn.functional as F

from cluster.clustering import kmeans
from quantization.fake_quant import activation_quantizer


class ClusterLinear(nn.Linear):
    """Linear layer whose weight is a per-row codebook of ``cluster_num`` centroids.

    Centroids and hard assignments are initialized with k-means. Every forward
    call runs one SGD step on the centroids using the local reconstruction loss
    between the clustered layer and the original layer, then returns the
    original layer's output. When ``use_assignment_lora`` is set, the
    assignments are relaxed into a softmax over the k-means one-hot plus a
    LoRA-parameterized update; those LoRA parameters receive gradients from the
    same loss and are stepped by the caller's optimizer.
    """

    def __init__(self, weight: torch.Tensor,
                 bias: torch.Tensor,
                 cluster_num: int,
                 layer_name: str,
                 lr: float = 0.0001,
                 use_assignment_lora: bool = False,
                 a_lora_r: int = 64,
                 activation_bit: int = 4,
                 start_temp: float = 2.0,
                 end_temp: float = 2.0,
                 hard_ratio: float = 1.0,
                 kmeans_parameterize_steps: int = 300,
                 soft: bool = False,
                 loss1_weight: float = 0.5,
                 ):
        super().__init__(in_features=weight.size(1),
                         out_features=weight.size(0),
                         bias=(bias is not None))
        with torch.no_grad():
            self.weight.data = self.weight.data.to(device=weight.device, dtype=weight.dtype)
            self.weight.copy_(weight)
            self.weight.requires_grad_(False)
            if bias is not None:
                self.bias.copy_(bias)
                self.bias.requires_grad_(False)
            self.ori_weight = self.weight.data
        self.device = weight.device
        self.to(weight.device)

        del weight, bias

        self.layer_name = layer_name
        self.cluster_num = cluster_num
        self.lr = lr

        self.activation_bit = activation_bit
        self.temperature = float(start_temp)
        self.start_temperature = float(start_temp)
        self.end_temperature = float(end_temp)
        self.hard_ratio = float(hard_ratio)

        self.kmeans_parameterize_steps = int(kmeans_parameterize_steps)
        self.use_assignment_lora = bool(use_assignment_lora)
        self.a_lora_r = int(a_lora_r)
        self.soft = bool(soft)
        self.loss1_weight = float(loss1_weight)

        self.centroids = None
        self.centroids_optimizer = None

        self.A_assignment = None
        self.B_assignment = None

        self.register_buffer("assignment_index", None, persistent=True)

        self._cluster()

        self.cached_input = None

    @property
    def cluster_weight(self) -> torch.Tensor:
        return self.get_cluster_weight()

    def _cluster(self):
        centroids, assignments = kmeans(
            self.ori_weight, self.cluster_num, self.kmeans_parameterize_steps,
        )
        centroids = centroids.to(self.device, dtype=self.weight.dtype)
        self.centroids = nn.Parameter(centroids.bfloat16(), requires_grad=True)

        assignments = assignments.to(self.device, dtype=torch.uint8)
        self.assignment_index = assignments
        self.out_features, self.in_features = self.assignment_index.shape

        if self.use_assignment_lora:
            self.A_assignment = nn.Parameter(
                torch.zeros(
                    self.out_features * self.cluster_num, self.a_lora_r,
                    device=self.device, dtype=self.weight.dtype
                ),
                requires_grad=True,
            )
            self.B_assignment = nn.Parameter(
                torch.zeros(
                    self.a_lora_r, self.in_features,
                    device=self.device, dtype=self.weight.dtype
                ),
                requires_grad=True,
            )
            nn.init.kaiming_uniform_(self.A_assignment, a=5 ** 0.5)
            nn.init.zeros_(self.B_assignment)

        del centroids, assignments
        torch.cuda.empty_cache()

        self.centroids_optimizer = torch.optim.SGD([self.centroids], lr=self.lr, momentum=0.9, weight_decay=0)

    def set_temperature(self, ratio: float) -> None:
        """Anneal the assignment softmax temperature linearly with training progress.

        Args:
            ratio: Training progress in [0, 1]. The temperature moves from
                ``start_temp`` to ``end_temp`` as ``ratio`` goes from 0 to
                ``hard_ratio``; from ``hard_ratio`` on the layer uses hard
                assignments.
        """
        ratio = max(0.0, min(1.0, float(ratio)))
        self.temperature = self.start_temperature + (ratio / self.hard_ratio) * (
            self.end_temperature - self.start_temperature
        )
        self.soft = ratio < self.hard_ratio

    def get_cluster_weight(self, eval_mode: bool = False) -> torch.Tensor:
        """Reconstruct the full weight matrix from centroids and assignments.

        Args:
            eval_mode: If True, use hard argmax assignments even when the
                layer trains soft assignments.

        Returns:
            Weight tensor of shape (out_features, in_features).
        """
        centroids = self.centroids.to(dtype=self.weight.dtype, device=self.device)

        out_features, in_features = self.weight.shape
        cluster_count = self.cluster_num

        if not self.use_assignment_lora:
            centroid_ids = self.assignment_index.to(device=self.device, dtype=torch.long)
            centroid_ids = centroid_ids.clamp_min(0).clamp_max(cluster_count - 1)
            row_ids = torch.arange(out_features, device=self.device).unsqueeze(1).expand(out_features, in_features)
            return centroids[row_ids, centroid_ids]

        assignment_delta = self.A_assignment @ self.B_assignment
        assignment_logits = assignment_delta.view(
            out_features, cluster_count, in_features
        ).transpose(1, 2).contiguous()

        centroid_ids = self.assignment_index.to(device=self.device, dtype=torch.long)
        base_assignment = torch.zeros(
            (out_features, in_features, cluster_count),
            device=assignment_logits.device,
            dtype=assignment_logits.dtype,
        )
        base_assignment.scatter_(-1, centroid_ids.unsqueeze(-1), 1.0)
        logits = assignment_logits + base_assignment
        logits = logits / max(float(self.temperature), 1e-8)

        if self.soft and (not eval_mode):
            assignment_weights = F.softmax(logits, dim=-1)
        else:
            soft_assignment = F.softmax(logits, dim=-1)
            with torch.no_grad():
                hard_assignment = torch.zeros_like(soft_assignment)
                hard_assignment.scatter_(-1, soft_assignment.argmax(dim=-1, keepdim=True), 1.0)
            assignment_weights = hard_assignment.detach()

        return torch.einsum("nmk,nk->nm", assignment_weights, centroids)

    def forward(self, inputs):
        self.cached_input = inputs.detach().cpu()
        self._loss(self.cached_input)
        return F.linear(inputs, self.ori_weight.to(inputs.dtype), self.bias).detach()

    def _mse(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        element_mse = ((pred - target) ** 2).mean(-1)  # (batch, seq_len)
        return element_mse.mean()

    def _loss(self, cached_inputs):
        cached_inputs = cached_inputs.to(self.cluster_weight.device).detach()
        quantized_inputs = activation_quantizer(
            inputs=cached_inputs,
            quantization_bit=self.activation_bit
        )
        quantized_output = F.linear(quantized_inputs, self.cluster_weight, self.bias)
        clean_output = F.linear(cached_inputs, self.cluster_weight, self.bias)
        target_output = F.linear(cached_inputs, self.ori_weight.to(quantized_inputs.dtype), self.bias)
        quantized_loss = self._mse(quantized_output, target_output).to(cached_inputs.dtype)
        clean_loss = self._mse(clean_output, target_output).to(cached_inputs.dtype)
        loss = self.loss1_weight * quantized_loss + (1.0 - self.loss1_weight) * clean_loss

        self._step_loss = loss.item()
        self._step_nmse = loss.item() / target_output.pow(2).mean().clamp(min=1e-12).item()
        self.centroids_optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer_params = [p for group in self.centroids_optimizer.param_groups for p in group["params"]]
        torch.nn.utils.clip_grad_norm_(optimizer_params, max_norm=1.0)
        self.centroids_optimizer.step()

        del loss, quantized_output, cached_inputs
        torch.cuda.empty_cache()


def replace_linear(
        model: nn.Module,
        target_layers: List[str],
        args: Dict[str, Any],
        dtype,
):
    """Replace every ``target_layers`` attribute that is an ``nn.Linear`` with a ``ClusterLinear``.

    Args:
        model: Model to modify in place.
        target_layers: Attribute names such as ``q_proj`` to replace.
        args: Keys ``cluster_assignment`` (per-layer cluster counts or None),
            ``cluster_num``, ``centroids_lr``, ``activation_quantization_bit``,
            ``start_temperature``, ``end_temperature``, ``hard_ratio``,
            ``use_assignment_lora``, ``a_lora_r``, ``soft`` and ``loss1_weight``.
        dtype: Dtype the new layers are cast to.
    """
    adaptive_assignment = args["cluster_assignment"] is not None
    replaced_layers = {}

    for name, module in model.named_modules():
        for attr in target_layers:
            if not hasattr(module, attr):
                continue
            linear_layer = getattr(module, attr)
            if not isinstance(linear_layer, nn.Linear):
                continue

            module_name = f"{name}.{attr}" if name else attr

            with torch.no_grad():
                weight = linear_layer.weight.detach().contiguous()
                bias = linear_layer.bias.detach().contiguous() if linear_layer.bias is not None else None

            setattr(module, attr, None)
            del linear_layer
            torch.cuda.empty_cache()

            if adaptive_assignment and module_name in args["cluster_assignment"].keys():
                cluster_number = args["cluster_assignment"][module_name]
            else:
                cluster_number = args["cluster_num"]

            new_linear = ClusterLinear(
                weight=weight,
                bias=bias,
                cluster_num=cluster_number,
                lr=args["centroids_lr"],
                layer_name=module_name,
                activation_bit=args["activation_quantization_bit"],
                start_temp=args["start_temperature"],
                end_temp=args["end_temperature"],
                hard_ratio=args["hard_ratio"],
                use_assignment_lora=args.get("use_assignment_lora", False),
                a_lora_r=args.get("a_lora_r", 64),
                soft=args["soft"],
                loss1_weight=args.get("loss1_weight", 0.5),
            ).to(weight.device).to(dtype)
            setattr(module, attr, new_linear)
            replaced_layers[module_name] = new_linear

    print(f"[INFO] Replaced {len(replaced_layers)} Linear layer(s) with ClusterLinear.")

    gc.collect()
    torch.cuda.empty_cache()


def cluster_linear_post_process(model: nn.Module, target_layers: List[str]):
    """Write each ``ClusterLinear``'s hard-assigned clustered weight back into its dense ``weight``."""
    fused_count = 0
    for name, module in model.named_modules():
        for attr in target_layers:
            if not hasattr(module, attr):
                continue
            linear_layer = getattr(module, attr)
            if not isinstance(linear_layer, ClusterLinear):
                continue
            with torch.no_grad():
                linear_layer.weight.data.copy_(linear_layer.get_cluster_weight(eval_mode=True).detach())
            fused_count += 1
    print(f"[INFO] Fused {fused_count} ClusterLinear layer(s) into dense weights.")
