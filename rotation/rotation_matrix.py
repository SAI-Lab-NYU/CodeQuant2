import torch
import torch.nn as nn


def load_or_create_R1(mode: str,
                      device: str,
                      checkpoint_path: str = None,
                      dim: int = None):
    if mode == "offline":
        checkpoint = torch.load(checkpoint_path, weights_only=False, map_location="cpu")
        hidden_size = checkpoint["dim"]
        rotation_module = nn.utils.parametrizations.orthogonal(
            nn.Linear(hidden_size, hidden_size, bias=False, dtype=torch.float32),
            orthogonal_map="cayley",
        )
        rotation_module.load_state_dict(checkpoint["r1"])

        return rotation_module.to(device)
    elif mode == "online":
        assert dim is not None, "dim must be specified"

        rotation_module = nn.utils.parametrizations.orthogonal(
            nn.Linear(dim, dim, bias=False, dtype=torch.float32),
            orthogonal_map="cayley",
        )

        return rotation_module.to(device)
    else:
        raise ValueError("Unknown mode")


def create_optimizer(matrix: nn.Module,
                     lr: float):
    return torch.optim.Adam([matrix.parametrizations.weight.original], lr=lr)
