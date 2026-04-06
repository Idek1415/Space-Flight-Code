"""Shared PyTorch device selection (NVIDIA CUDA when available)."""

import torch


def get_torch_device_str() -> str:
    """Return ``cuda`` when an NVIDIA GPU is usable, else ``cpu``."""
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def get_torch_device() -> torch.device:
    return torch.device(get_torch_device_str())
