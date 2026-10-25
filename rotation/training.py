from __future__ import annotations

import re
from typing import Callable, Optional, Tuple, Union

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from quantization.fake_quant import activation_quantizer
from rotation.rotation_matrix import create_optimizer, load_or_create_R1
from utils.reasoning_data_utils import (
    GenAnswerDataset,
    MixedDataset,
    WikiTextDataset,
    format_reasoning_batch,
)


def _mse(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    element_mse = ((pred - target) ** 2).mean(-1)
    return element_mse.mean()


def quantization_loss_hook(
    module_name: str,
    rotation: Union[nn.Module, torch.Tensor],
    scaling_vector: Optional[nn.Parameter],
    activation_quantization_bit: int,
    loss_dict: dict,
):
    """Build a forward hook that records the activation quantization error of a RMSNorm output.

    The hooked output is detached, rotated with ``rotation`` (an orthogonal
    ``nn.Module`` whose ``weight`` receives gradients, or a frozen matrix),
    divided by ``scaling_vector`` when one is given, and compared against its
    fake-quantized version. The MSE is written to ``loss_dict[module_name]``.

    Args:
        module_name: Key under which the loss is stored.
        rotation: Rotation module or matrix applied to the hooked output.
        scaling_vector: Optional per-channel divisor applied after the rotation.
        activation_quantization_bit: Bit-width of the fake quantization.
        loss_dict: Dictionary that receives the loss of every hooked module.

    Returns:
        A forward hook accepting ``(module, input_args, output)``.
    """

    def hook(
        module: nn.Module,
        input_args: Tuple[torch.Tensor, ...],
        output_tensor_or_tuple: Union[torch.Tensor, Tuple],
    ) -> None:
        if isinstance(output_tensor_or_tuple, torch.Tensor):
            rmsnorm_output = output_tensor_or_tuple.detach()
        elif isinstance(output_tensor_or_tuple, tuple):
            if len(output_tensor_or_tuple) > 0 and isinstance(output_tensor_or_tuple[0], torch.Tensor):
                rmsnorm_output = output_tensor_or_tuple[0].detach()

        rmsnorm_output.requires_grad_(True)
        rotation_matrix = rotation.weight if isinstance(rotation, nn.Module) else rotation
        rot_output = rmsnorm_output.float() @ rotation_matrix.to(rmsnorm_output.device)
        if scaling_vector is not None:
            rot_output = rot_output / scaling_vector.to(rot_output.device)
        quant_rot_output = activation_quantizer(rot_output, activation_quantization_bit, ste=False)
        loss_dict[module_name] = _mse(rot_output, quant_rot_output)

    return hook


def _build_calibration_loader(
    gen_data_path: str,
    calibration_samples: int,
    wiki_dataset_name: str | None,
    wiki_ratio: float,
    batch_size: int,
) -> DataLoader:
    generated_answer_dataset = GenAnswerDataset(
        gen_data_path,
        limit=calibration_samples,
        skip_empty_prediction=True,
    )
    if wiki_dataset_name:
        wiki_dataset = WikiTextDataset(wiki_dataset_name, limit=calibration_samples)
        dataset = MixedDataset(
            generated_answer_dataset,
            wiki_dataset,
            total=calibration_samples,
            wiki_ratio=wiki_ratio,
            primary_label="gsm",
        )
        wiki_count = sum(1 for r in dataset.records if r["source"] == "wiki2")
        generated_count = len(dataset) - wiki_count
        print(
            f"[INFO] Mixed dataset: {len(dataset)} samples "
            f"({generated_count} generated + {wiki_count} WikiText-2, wiki_ratio={wiki_ratio})"
        )
    else:
        dataset = generated_answer_dataset

    generator = torch.Generator()
    generator.manual_seed(42)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        generator=generator,
    )


def _tokenize_rotation_batch(batch: dict, tokenizer, max_length: int, device: str) -> dict:
    processed = format_reasoning_batch(batch, tokenizer)
    return tokenizer(
        processed,
        return_tensors="pt",
        padding="max_length",
        truncation=True,
        max_length=max_length,
    ).to(device)


def rotation_fine_tune(
    model: nn.Module,
    tokenizer: nn.Module,
    activation_quantization_bit: int,
    gen_data_path: str,
    calibration_samples: int,
    batch_size: int,
    max_length: int,
    epochs: int,
    lr: float,
    device: str,
    wiki_dataset_name: str = None,
    wiki_ratio: float = 0.5,
    pretrained_rotation_path: str = None,
    log_metrics: Optional[Callable] = None,
):
    """Fit the shared rotation R1, then one scaling vector per RMSNorm, on calibration activations.

    Both stages minimize the fake-quantization error of the rotated (and
    scaled) RMSNorm outputs over the calibration set.

    Args:
        model: Causal LM with RMSNorm weights already folded into the linear layers.
        tokenizer: Tokenizer matching ``model``.
        activation_quantization_bit: Bit-width of the activation fake quantization.
        gen_data_path: JSONL file of reasoning traces from ``script/gen_vllm.py``.
        calibration_samples: Number of calibration sequences.
        batch_size: Calibration batch size.
        max_length: Token truncation length.
        epochs: Number of passes over the calibration set for each stage.
        lr: Adam learning rate for both stages.
        device: Device the rotation and scaling parameters live on.
        wiki_dataset_name: Optional WikiText dataset mixed into the calibration set.
        wiki_ratio: Fraction of WikiText samples when ``wiki_dataset_name`` is set.
        pretrained_rotation_path: Optional checkpoint whose ``r1`` is loaded
            instead of training the rotation.
        log_metrics: Optional callable, such as ``wandb.log``, receiving a
            metrics dict per step.

    Returns:
        Tuple of the rotation module and a mapping from RMSNorm name to its
        scaling vector.
    """
    rotation_module = load_or_create_R1(mode="online", device=device, dim=model.config.hidden_size)
    if pretrained_rotation_path is not None:
        checkpoint = torch.load(pretrained_rotation_path, map_location=device)
        rotation_module.load_state_dict(checkpoint["r1"])
        rotation_module = rotation_module.to(device)
        print(f"[INFO] Loaded pretrained rotation matrix from {pretrained_rotation_path}, skipping rotation training.")
    else:
        rotation_optimizer = create_optimizer(rotation_module, lr=lr)

    hidden_size = model.config.hidden_size
    scaling_vectors: dict[str, nn.Parameter] = {}
    output_regex = re.compile(r".*(input_layernorm|post_attention_layernorm).*")
    dataloader = _build_calibration_loader(
        gen_data_path=gen_data_path,
        calibration_samples=calibration_samples,
        wiki_dataset_name=wiki_dataset_name,
        wiki_ratio=wiki_ratio,
        batch_size=batch_size,
    )

    model.eval()

    if pretrained_rotation_path is None:
        rotation_losses = {}
        handles = {}
        for name, module in model.named_modules():
            if output_regex.search(name):
                hook = quantization_loss_hook(
                    name,
                    rotation_module,
                    None,
                    activation_quantization_bit,
                    rotation_losses,
                )
                handles[name] = module.register_forward_hook(hook)
        print(f"[INFO] Registered {len(handles)} rotation hook(s).")

        step = 0
        print("[INFO] Starting rotation optimization.")
        for epoch_idx in tqdm(range(epochs), leave=False):
            for step_in_epoch, batch in enumerate(tqdm(dataloader, leave=False)):
                rotation_losses.clear()
                inputs = _tokenize_rotation_batch(batch, tokenizer, max_length, device)

                model(**inputs)
                rotation_optimizer.zero_grad()

                gpu_count = torch.cuda.device_count()
                rotation_loss = torch.stack([
                    loss.to(f"cuda:{gpu_count - 1}")
                    for loss in rotation_losses.values()
                    if loss is not None
                ]).sum()
                rotation_loss.backward()

                rotation_grad_norm = torch.norm(rotation_module.parametrizations.weight.original.grad)
                log_dict = {
                    "rotation/epoch": epoch_idx,
                    "rotation/step": step,
                    "rotation/loss": rotation_loss.item(),
                    "rotation/R1_grad_norm": rotation_grad_norm.item(),
                }
                for module_name, loss in rotation_losses.items():
                    if loss is not None:
                        log_dict[f"rotation/loss_{module_name}"] = loss.item()
                if log_metrics is not None:
                    log_metrics(log_dict, step=step)
                step += 1

                rotation_optimizer.step()

        for handle in handles.values():
            handle.remove()
        print("[INFO] Rotation optimization finished.")

    scaling_loss_dict = {}
    handles = {}
    for name, module in model.named_modules():
        if output_regex.search(name):
            init_sv = torch.ones(hidden_size, dtype=torch.float32, device=device)
            init_sv = init_sv + 0.01 * torch.randn_like(init_sv)
            scaling_vectors[name] = nn.Parameter(init_sv)
            frozen_rotation = rotation_module.weight.detach()
            hook = quantization_loss_hook(
                name,
                frozen_rotation,
                scaling_vectors[name],
                activation_quantization_bit,
                scaling_loss_dict,
            )
            handles[name] = module.register_forward_hook(hook)
    print(f"[INFO] Registered {len(handles)} scaling hook(s).")

    scaling_optimizers = {
        name: torch.optim.Adam([scaling_vector], lr=lr)
        for name, scaling_vector in scaling_vectors.items()
    }

    scaling_step = 0
    print("[INFO] Starting scaling-vector optimization.")
    for epoch_idx in tqdm(range(epochs), leave=False):
        for step_in_epoch, batch in enumerate(tqdm(dataloader, leave=False)):
            scaling_loss_dict.clear()
            inputs = _tokenize_rotation_batch(batch, tokenizer, max_length, device)

            model(**inputs)

            for name, loss in scaling_loss_dict.items():
                scaling_optimizers[name].zero_grad()
                loss.backward()
                scaling_optimizers[name].step()

            total_loss = sum(loss.item() for loss in scaling_loss_dict.values())
            log_dict = {
                "scaling/epoch": epoch_idx,
                "scaling/step": scaling_step,
                "scaling/total_loss": total_loss,
            }
            for module_name, loss in scaling_loss_dict.items():
                if loss is not None:
                    log_dict[f"scaling/loss_{module_name}"] = loss.item()
            if log_metrics is not None:
                log_metrics(log_dict)
            scaling_step += 1

    print("[INFO] Scaling-vector optimization finished.")
    return rotation_module, scaling_vectors
