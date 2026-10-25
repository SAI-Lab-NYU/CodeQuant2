import argparse
import os
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
import yaml
from tqdm import tqdm

from torch.utils.data import DataLoader

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from rotation.pipeline import apply_finetuned_rotation
from utils.adaptive_cluster_utils import (
    allocate_cluster_latency,
    get_layer_idx,
    reconstruction_loss,
    replace_linear_with_prelinear,
    tokenize_batch,
)
from utils.dataset_utils import load_calibration_dataset
from utils.entrypoint_utils import load_yaml_config
from utils.model_loading_utils import load_causal_lm


CLUSTER_CHOICES = (8, 16, 32)
# Profiled decode-time LUT kernel latency of one target matrix for each
# cluster count; the allocation charges every change against these numbers.
SELF_ATTN_LATENCY = {"8": 0.0386, "16": 0.0391, "32": 0.0429}
MLP_LATENCY = {"8": 0.0387, "16": 0.0418, "32": 0.0661}
CALIBRATION_DATASET = "Self-GRIT/wikitext-2-raw-v1-preprocessed"
CALIBRATION_SIZE = 1024
CALIBRATION_EPOCHS = 1
LAYERS_PER_GROUP = 10


def main(
    experiment_config,
    kmeans_iters: int = 300,
    seed: int = 42,
) -> Dict[str, Dict[str, float]]:
    """Score each target linear layer and save its adaptive cluster assignment."""

    default_cluster_count = CLUSTER_CHOICES[1]

    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

    device = "cuda"
    model_name = experiment_config["model_name"]
    model_prefix = model_name.split("/")[1]
    tie = experiment_config["tie"]

    model_config, model, tokenizer = load_causal_lm(
        model_name,
        tie=tie,
        dtype="auto",
        device_map="auto",
        attn_implementation="eager",
    )

    adaptive_config = experiment_config.get("adap_clus_para", {})
    batch_size = int(adaptive_config.get("batch_size", 16))
    max_length = int(adaptive_config.get("max_length", 256))

    os.makedirs(experiment_config["path"]["clustering"], exist_ok=True)

    apply_finetuned_rotation(
        model=model,
        model_name=model_name,
        model_prefix=model_prefix,
        hidden_size=model_config.hidden_size,
        rotation_dir=experiment_config["path"]["rotation"],
        device=device,
    )

    train_dataset = load_calibration_dataset(CALIBRATION_DATASET, size=CALIBRATION_SIZE)
    dataloader_seed = torch.Generator()
    dataloader_seed.manual_seed(42)
    dataloader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=2,
        generator=dataloader_seed,
    )

    for parameter in model.parameters():
        parameter.requires_grad_(True)

    cluster_linears = replace_linear_with_prelinear(
        model,
        k_list=CLUSTER_CHOICES,
        kmeans_iters=kmeans_iters,
        activation_bit=int(experiment_config["bits"]["activation"]),
    )
    for cluster_linear in tqdm(cluster_linears.values(), desc="Building Wk cache"):
        cluster_linear.build_Wk_cache()

    # Score ten transformer layers at a time to keep calibration memory bounded.
    layer_groups: Dict[int, List[str]] = defaultdict(list)
    for name in cluster_linears:
        group_id = max(0, get_layer_idx(name)) // LAYERS_PER_GROUP
        layer_groups[group_id].append(name)

    sorted_group_ids = sorted(layer_groups.keys())
    print(f"[INFO] Total groups: {len(sorted_group_ids)}, ~{LAYERS_PER_GROUP} transformer layers each")

    final_assignment: Dict[str, int] = {name: default_cluster_count for name in cluster_linears}
    layer_scores: Dict[str, Dict[str, float]] = {}

    for group_id in sorted_group_ids:
        group_names = set(layer_groups[group_id])
        print(
            f"\n[Group {group_id}] Scoring layers "
            f"{group_id * LAYERS_PER_GROUP}-{group_id * LAYERS_PER_GROUP + LAYERS_PER_GROUP - 1} "
            f"({len(group_names)} weights)"
        )

        for name in group_names:
            cluster_linears[name].reset_scores()

        for _ in range(CALIBRATION_EPOCHS):
            for batch in tqdm(dataloader, desc=f"Group {group_id} calibration"):
                inputs = tokenize_batch(batch["prompt"], tokenizer, max_length, device)

                with torch.no_grad():
                    for cluster_linear in cluster_linears.values():
                        cluster_linear.pinned_k = -1
                    fp_logits = model(**inputs).logits.detach()

                for name, cluster_linear in cluster_linears.items():
                    cluster_linear.pinned_k = (
                        default_cluster_count if name in group_names else final_assignment[name]
                    )

                model.zero_grad(set_to_none=True)
                quantized_output = model(**inputs)
                reconstruction_loss(label=fp_logits, output=quantized_output.logits).backward()
                del fp_logits, quantized_output

                for name in group_names:
                    cluster_linears[name].accumulate_importance()

                for name, cluster_linear in cluster_linears.items():
                    if name not in group_names:
                        cluster_linear._last_x = None
                        cluster_linear._last_gy = None

                del inputs

        group_scores = {name: cluster_linears[name].get_scores() for name in group_names}
        layer_scores.update(group_scores)

        group_assignment, info = allocate_cluster_latency(
            group_scores, SELF_ATTN_LATENCY, MLP_LATENCY, choices=CLUSTER_CHOICES
        )
        print(
            f"[Group {group_id}] Allocation finished: changed={info['changed']}, "
            f"small={info['small']}, large={info['large']}, "
            f"avg_bits={info['average']:.2f}, speedup={info['speedup']:.4f}"
        )

        for name, k in group_assignment.items():
            final_assignment[name] = k
            cluster_linears[name].pinned_k = k

    save_path = os.path.join(
        experiment_config["path"]["clustering"],
        f"{model_prefix}_cluster_assignment.yaml",
    )
    with open(save_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(final_assignment, f, sort_keys=True)
    print(f"[INFO] Saved cluster assignment to: {save_path}")

    return layer_scores


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="allocate cluster counts per target matrix")
    parser.add_argument("--config", type=str, required=True, help="config file name or path")
    args = parser.parse_args()

    main(load_yaml_config(args.config))
