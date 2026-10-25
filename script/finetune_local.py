import argparse
import gc
import os
import sys
from pathlib import Path

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from rotation.pipeline import apply_finetuned_rotation
from utils.entrypoint_utils import load_yaml_config
from utils.model_loading_utils import load_causal_lm
from utils.reasoning_data_utils import (
    GenAnswerDataset,
    MixedDataset,
    WikiTextDataset,
    format_reasoning_batch,
)


def parse_args():
    parser = argparse.ArgumentParser(description="fine-tune clustered weights with local reconstruction losses")
    parser.add_argument("--config", type=str, required=True, help="config file name or path")
    parser.add_argument("--wandb", action="store_true", help="log training curves to Weights & Biases")
    return parser.parse_args()


def set_seed(seed: int = 42) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)


def build_dataloader(experiment_config: dict, model_prefix: str, size: int, batch_size: int) -> DataLoader:
    gen_data_path = os.path.join(experiment_config["path"]["rotation"], f"{model_prefix}_reasoning_gen.jsonl")
    print(f"[INFO] Loading generated answers from {gen_data_path}")
    gen_dataset = GenAnswerDataset(gen_data_path, limit=size)

    training_config = experiment_config["training_para"]
    wiki_dataset_name = training_config.get("wiki_dataset", None)
    wiki_ratio = training_config.get("wiki_ratio", 0.0)
    if wiki_dataset_name and wiki_ratio > 0.0:
        wiki_dataset = WikiTextDataset(wiki_dataset_name, limit=size)
        train_dataset = MixedDataset(gen_dataset, wiki_dataset, total=size, wiki_ratio=wiki_ratio)
        n_wiki = sum(1 for r in train_dataset.records if r["source"] == "wiki2")
        n_gen = len(train_dataset) - n_wiki
        print(
            f"[INFO] Mixed dataset: {len(train_dataset)} samples "
            f"({n_gen} gen + {n_wiki} wiki, wiki_ratio={wiki_ratio})"
        )
    else:
        train_dataset = gen_dataset

    return DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=2,
        pin_memory=True,
        drop_last=False,
    )


def load_cluster_assignment(experiment_config: dict, model_prefix: str):
    if not experiment_config["bits"]["assignment_cluster_number"]:
        return None

    assignment_path = os.path.join(
        experiment_config["path"]["clustering"],
        f"{model_prefix}_cluster_assignment.yaml",
    )
    with open(assignment_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def replace_model_linears(model, experiment_config: dict, cluster_assignment, dtype):
    from cluster.ClusterLinear import ClusterLinear, cluster_linear_post_process, replace_linear

    training_config = experiment_config["training_para"]
    cluster_linear_args = {
        "activation_quantization_bit": experiment_config["bits"]["activation"],
        "cluster_num": experiment_config["bits"]["cluster_num"],
        "cluster_assignment": cluster_assignment,
        "start_temperature": training_config["start_temp"],
        "end_temperature": training_config["end_temp"],
        "hard_ratio": training_config["hard_ratio"],
        "use_assignment_lora": training_config["use_assignment_lora"],
        "a_lora_r": training_config["a_lora_r"],
        "soft": training_config["soft"],
        "loss1_weight": training_config.get("loss1_weight", 0.5),
        "centroids_lr": training_config["lr_c"],
    }
    replace_linear(
        model=model,
        target_layers=training_config["target_layers"],
        args=cluster_linear_args,
        dtype=dtype,
    )
    return ClusterLinear, cluster_linear_post_process


def collect_assignment_params(model) -> list:
    return [
        param
        for name, param in model.named_parameters()
        if param.requires_grad and "assignment" in name
    ]


def log_param_counts(model, assignment_params: list) -> None:
    total_params = sum(p.numel() for p in model.parameters())
    trainable_all = sum(p.numel() for p in model.parameters() if p.requires_grad)
    trainable_asgn = sum(p.numel() for p in assignment_params)
    trainable_cent = sum(
        p.numel() for name, p in model.named_parameters() if p.requires_grad and "centroids" in name
    )
    print(f"\n[PARAMS] Total model params      : {total_params:>15,}")
    print(f"[PARAMS] Trainable (all)          : {trainable_all:>15,}  ({100 * trainable_all / total_params:.3f}%)")
    print(f"[PARAMS]   assignment params      : {trainable_asgn:>15,}  ({100 * trainable_asgn / total_params:.3f}%)")
    print(f"[PARAMS]   centroids params       : {trainable_cent:>15,}  ({100 * trainable_cent / total_params:.3f}%)\n")


def set_all_temperature(model, ratio: float) -> float:
    """Apply the temperature schedule at training progress ``ratio`` to every ClusterLinear layer."""
    temperature = float("nan")
    for _, module in model.named_modules():
        if hasattr(module, "set_temperature"):
            module.set_temperature(ratio)
            temperature = module.temperature
    return temperature


def collect_cluster_losses(model, cluster_linear_cls, log_dict: dict):
    total_linear_loss = 0.0
    for name, module in model.named_modules():
        if isinstance(module, cluster_linear_cls) and hasattr(module, "_step_loss"):
            total_linear_loss += module._step_loss
            log_dict[f"nmse/linear/{name}"] = module._step_nmse
            module._step_loss = 0.0
            module._step_nmse = 0.0
    return total_linear_loss


def train_assignment(
    model,
    tokenizer,
    dataloader: DataLoader,
    experiment_config: dict,
    cluster_linear_cls,
    assignment_params: list,
    optimizer,
    epochs: int,
    max_length: int,
    device: str,
    log_metrics,
) -> None:
    gc.collect()
    torch.cuda.empty_cache()
    model.eval()

    loss_log_path = os.path.join(experiment_config["path"]["clustering"], "step_loss.txt")
    total_steps = max(epochs * len(dataloader), 1)
    step = 0
    print("[INFO] Starting local ClusterLinear fine-tuning.")
    with open(loss_log_path, "w") as loss_log:
        loss_log.write("step\ttotal_loss\tlinear_loss\tassignment_grad_norm\ttemperature\n")
        for epoch_idx in tqdm(range(epochs), leave=False):
            for _, batch in enumerate(tqdm(dataloader, leave=False)):
                processed = format_reasoning_batch(batch, tokenizer)
                inputs = tokenizer(
                    processed,
                    return_tensors="pt",
                    padding="max_length",
                    truncation=True,
                    max_length=max_length,
                ).to(device)

                optimizer.zero_grad(set_to_none=True)
                temperature = set_all_temperature(model, step / total_steps)
                model(**inputs)

                assignment_grad_norm = torch.nn.utils.clip_grad_norm_(assignment_params, max_norm=1.0)
                optimizer.step()

                log_dict = {"train/epoch": epoch_idx, "train/step": step, "train/temperature": temperature}
                linear_loss = collect_cluster_losses(model, cluster_linear_cls, log_dict)
                log_dict["train/linear_loss"] = linear_loss
                log_dict["train/total_loss"] = linear_loss
                log_dict["train/assignment_grad_norm"] = assignment_grad_norm.item()

                loss_log.write(
                    f"{step}\t{linear_loss:.6f}\t{linear_loss:.6f}\t{assignment_grad_norm.item():.4f}"
                    f"\t{temperature:.6f}\n"
                )
                loss_log.flush()
                if log_metrics is not None:
                    log_metrics(log_dict, step=step)
                step += 1
                del inputs
    print(f"[INFO] Local fine-tuning finished. Step losses saved to {loss_log_path}")


def main(experiment_config: dict, use_wandb: bool) -> None:
    set_seed(42)

    device = "cuda"
    model_name = experiment_config["model_name"]
    model_prefix = model_name.split("/")[1]
    tie = experiment_config["tie"]

    training_config = experiment_config["training_para"]
    batch_size = int(training_config.get("batch_size", 16))
    epochs = int(training_config.get("epochs", 4))
    size = int(training_config.get("size", 512))
    max_length = int(training_config.get("max_length", 512))

    print("[INFO] Loading model for finetune phase...")
    config_ft, model, tokenizer = load_causal_lm(
        model_name,
        tie=tie,
        dtype="auto",
        device_map="auto",
        attn_implementation="eager",
    )
    dtype = model.dtype

    apply_finetuned_rotation(
        model=model,
        model_name=model_name,
        model_prefix=model_prefix,
        hidden_size=config_ft.hidden_size,
        rotation_dir=experiment_config["path"]["rotation"],
        device=device,
    )

    gc.collect()
    torch.cuda.empty_cache()

    for _, param in model.named_parameters():
        param.requires_grad_(False)

    dataloader = build_dataloader(experiment_config, model_prefix, size, batch_size)
    cluster_assignment = load_cluster_assignment(experiment_config, model_prefix)
    cluster_linear_cls, cluster_linear_post_process = replace_model_linears(
        model,
        experiment_config,
        cluster_assignment,
        dtype,
    )

    assignment_params = collect_assignment_params(model)
    optimizer = torch.optim.SGD(
        [{"params": assignment_params, "lr": 1e-5, "weight_decay": 0.0}],
        momentum=0.9,
    )
    log_param_counts(model, assignment_params)

    log_metrics = None
    if use_wandb:
        import wandb

        wandb.init(
            project="clusquant",
            name="finetune",
            config={**experiment_config, "stage": "finetune"},
        )
        log_metrics = wandb.log

    train_assignment(
        model=model,
        tokenizer=tokenizer,
        dataloader=dataloader,
        experiment_config=experiment_config,
        cluster_linear_cls=cluster_linear_cls,
        assignment_params=assignment_params,
        optimizer=optimizer,
        epochs=epochs,
        max_length=max_length,
        device=device,
        log_metrics=log_metrics,
    )

    cluster_linear_post_process(
        model=model,
        target_layers=experiment_config["training_para"]["target_layers"],
    )

    save_dir = os.path.join(experiment_config["path"]["clustering"], "final_model")
    model.save_pretrained(save_dir)
    tokenizer.save_pretrained(save_dir)


if __name__ == "__main__":
    args = parse_args()
    main(load_yaml_config(args.config), args.wandb)
