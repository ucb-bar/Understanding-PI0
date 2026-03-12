from __future__ import annotations

import platform
import random
from dataclasses import dataclass

import torch


@dataclass
class RuntimeInfo:
    python: str
    torch_version: str
    cuda_available: bool
    cuda_device: str | None
    cuda_capability: tuple[int, int] | None


def seed_all(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_runtime_info(device: str = "cpu") -> RuntimeInfo:
    cuda_available = torch.cuda.is_available()
    cuda_device = None
    cuda_capability = None

    if cuda_available and str(device).startswith("cuda"):
        idx = torch.device(device).index
        if idx is None:
            idx = torch.cuda.current_device()
        cuda_device = torch.cuda.get_device_name(idx)
        cuda_capability = torch.cuda.get_device_capability(idx)

    return RuntimeInfo(
        python=platform.python_version(),
        torch_version=torch.__version__,
        cuda_available=cuda_available,
        cuda_device=cuda_device,
        cuda_capability=cuda_capability,
    )


def mx_execution_supported(device: str = "cpu") -> bool:
    if not torch.cuda.is_available():
        return False
    if not str(device).startswith("cuda"):
        return False
    idx = torch.device(device).index
    if idx is None:
        idx = torch.cuda.current_device()
    major, _minor = torch.cuda.get_device_capability(idx)
    return major >= 10  # SM100+


def warn_if_mx_execution_unavailable(device: str = "cpu") -> None:
    if not mx_execution_supported(device):
        print(
            "[warning] TorchAO MX execution is documented for NVIDIA SM100+ / Blackwell. "
            "Structural quantization/export may still work, but eager MX execution can fail here."
        )


def print_runtime_info(device: str = "cpu") -> None:
    info = get_runtime_info(device)
    print("[runtime]")
    print(f"  python          : {info.python}")
    print(f"  torch           : {info.torch_version}")
    print(f"  cuda_available  : {info.cuda_available}")
    print(f"  cuda_device     : {info.cuda_device}")
    print(f"  cuda_capability : {info.cuda_capability}")