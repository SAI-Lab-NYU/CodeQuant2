import argparse
import os
import random
import sys
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from rotation.training import rotation_fine_tune
from utils.entrypoint_utils import load_yaml_config
from utils.model_loading_utils import load_causal_lm
from utils.rotation_utils import fuse_weight


def parse_args():
    parser = argparse.ArgumentParser(description="fine-tune the rotation matrix and scaling vectors")
    parser.add_argument("--config", type=str, required=True, help="config file name or path")
    parser.add_argument("--wandb", action="store_true", help="log training curves to Weights & Biases")
    return parser.parse_args()


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def main(experiment_config: dict, use_wandb: bool) -> None:
    set_seed(42)

    model_name = experiment_config["model_name"]
    device = experiment_config["device"]
    tie = experiment_config["tie"]
    model_type = model_name.split("/")[1]

    print("[INFO] Loading model for finetune phase...")
    _, model, tokenizer = load_causal_lm(
        model_name,
        tie=tie,
        dtype="auto",
        device_map="auto",
        attn_implementation="eager",
    )

    fuse_weight(model, model_name)
    print(f"[INFO] model {model_name} fused.")

    rotation_config = experiment_config["rotation"]
    save_path = experiment_config["path"]["rotation"]
    os.makedirs(save_path, exist_ok=True)
    gen_data_path = os.path.join(save_path, f"{model_type}_reasoning_gen.jsonl")
    print(f"[INFO] Loading generated answers from {gen_data_path}")

    log_metrics = None
    if use_wandb:
        import wandb

        wandb.init(
            project="codequant2",
            name="rotation_fine_tune",
            config={**experiment_config, "stage": "rotation_fine_tune"},
        )
        log_metrics = wandb.log

    pretrained_rotation_path = rotation_config.get("pretrained_rotation_path", None)
    if pretrained_rotation_path:
        print(f"[INFO] pretrained_rotation_path set: {pretrained_rotation_path}, skipping rotation training.")

    rotation_matrix, scaling_vectors = rotation_fine_tune(
        model=model,
        tokenizer=tokenizer,
        activation_quantization_bit=experiment_config["bits"]["activation"],
        gen_data_path=gen_data_path,
        calibration_samples=rotation_config["max_sample"],
        batch_size=rotation_config["batch_size"],
        max_length=rotation_config["max_length"],
        epochs=rotation_config["epochs"],
        lr=rotation_config["fine_tune_lr"],
        device=device,
        wiki_dataset_name=rotation_config.get("wiki_dataset", None),
        wiki_ratio=rotation_config.get("wiki_ratio", 0.5),
        pretrained_rotation_path=pretrained_rotation_path,
        log_metrics=log_metrics,
    )

    rotation_checkpoint = {
        "r1": rotation_matrix.cpu().state_dict(),
        "dim": model.config.hidden_size,
        "scaling_vectors": {name: sv.detach().cpu() for name, sv in scaling_vectors.items()},
    }
    torch.save(rotation_checkpoint, os.path.join(save_path, f"{model_type}_r1_finetune.pt"))


if __name__ == "__main__":
    args = parse_args()
    main(load_yaml_config(args.config), args.wandb)
