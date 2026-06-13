from __future__ import annotations

import sys
from functools import lru_cache

import torch


def _get_gpu_arg() -> str:
    argv = sys.argv
    for index, arg in enumerate(argv[1:], start=1):
        if arg == "--gpu" and index + 1 < len(argv):
            return str(argv[index + 1]).strip()
        if str(arg).startswith("--gpu="):
            return str(arg).split("=", 1)[1].strip()
    return ""


def _device_type_from_string(device: str) -> str:
    device = str(device or "").strip().lower()
    if not device:
        return ""
    if device.isdigit():
        return "cuda"
    return device.split(":", 1)[0]


def _device_index(device) -> int | None:
    if isinstance(device, torch.device):
        return device.index
    value = str(device or "").strip().lower()
    if value.isdigit():
        return int(value)
    if ":" in value:
        index = value.split(":", 1)[1]
        if index.isdigit():
            return int(index)
    return None


def is_mps_available() -> bool:
    return bool(
        sys.platform == "darwin"
        and hasattr(torch.backends, "mps")
        and torch.backends.mps.is_available()
    )


def is_xpu_available() -> bool:
    return bool(hasattr(torch, "xpu") and torch.xpu.is_available())


def get_preferred_device(device=None) -> str:
    if isinstance(device, torch.device):
        return str(device)
    value = str(device or "").strip()
    if not value:
        value = _get_gpu_arg()
    if value:
        return f"cuda:{value}" if value.isdigit() else value
    if torch.cuda.is_available():
        return "cuda"
    if is_xpu_available():
        return "xpu"
    if is_mps_available():
        return "mps"
    return "cuda"


def get_accelerator_type(device=None) -> str:
    if isinstance(device, torch.device):
        return device.type
    return _device_type_from_string(get_preferred_device(device))


def get_device_capability(device=None) -> tuple[int, int]:
    if get_accelerator_type(device) != "cuda" or not torch.cuda.is_available():
        return (0, 0)
    try:
        preferred_device = get_preferred_device(device)
        index = _device_index(preferred_device)
        return torch.cuda.get_device_capability(index)
    except Exception:
        return (0, 0)


@lru_cache(maxsize=8)
def _xpu_bfloat16_supported(device: str) -> bool:
    if not is_xpu_available():
        return False
    try:
        with torch.no_grad():
            sample = torch.ones((1, 1), device=device, dtype=torch.bfloat16)
            result = sample @ sample
            synchronize(device)
        return result.dtype == torch.bfloat16
    except Exception:
        return False


def is_bfloat16_supported(device=None) -> bool:
    accelerator = get_accelerator_type(device)
    if accelerator == "cuda":
        major, _ = get_device_capability(device)
        return major >= 8
    if accelerator == "xpu":
        return _xpu_bfloat16_supported(get_preferred_device(device))
    return False


def set_device(device=None) -> None:
    accelerator = get_accelerator_type(device)
    preferred_device = get_preferred_device(device)
    try:
        if accelerator == "cuda" and torch.cuda.is_available():
            index = _device_index(preferred_device)
            if index is not None:
                torch.cuda.set_device(index)
        elif accelerator == "xpu" and is_xpu_available() and hasattr(torch.xpu, "set_device"):
            torch.xpu.set_device(preferred_device)
    except Exception:
        pass


def empty_cache(device=None) -> None:
    accelerator = get_accelerator_type(device)
    try:
        if accelerator == "cuda" and torch.cuda.is_available():
            torch.cuda.empty_cache()
        elif accelerator == "xpu" and is_xpu_available() and hasattr(torch.xpu, "empty_cache"):
            torch.xpu.empty_cache()
        elif accelerator == "mps" and is_mps_available() and hasattr(torch.mps, "empty_cache"):
            torch.mps.empty_cache()
    except Exception:
        pass


def synchronize(device=None) -> None:
    accelerator = get_accelerator_type(device)
    try:
        if accelerator == "cuda" and torch.cuda.is_available():
            torch.cuda.synchronize()
        elif accelerator == "xpu" and is_xpu_available() and hasattr(torch.xpu, "synchronize"):
            torch.xpu.synchronize()
        elif accelerator == "mps" and is_mps_available() and hasattr(torch.mps, "synchronize"):
            torch.mps.synchronize()
    except Exception:
        pass


def manual_seed_all(seed: int) -> None:
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if is_xpu_available() and hasattr(torch.xpu, "manual_seed_all"):
        torch.xpu.manual_seed_all(seed)
    if is_mps_available() and hasattr(torch.mps, "manual_seed"):
        torch.mps.manual_seed(seed)


def _device_properties(device=None):
    accelerator = get_accelerator_type(device)
    index = _device_index(get_preferred_device(device))
    try:
        if accelerator == "cuda" and torch.cuda.is_available():
            return torch.cuda.get_device_properties(index)
        if accelerator == "xpu" and is_xpu_available() and hasattr(torch.xpu, "get_device_properties"):
            return torch.xpu.get_device_properties(0 if index is None else index)
    except Exception:
        return None
    return None


def get_device_total_memory_mb(device=None) -> float:
    properties = _device_properties(device)
    total_memory = getattr(properties, "total_memory", 0) if properties is not None else 0
    return total_memory / 1048576 if total_memory else 0


def get_device_memory_allocated(device=None) -> int:
    accelerator = get_accelerator_type(device)
    index = _device_index(get_preferred_device(device))
    try:
        if accelerator == "cuda" and torch.cuda.is_available():
            return int(torch.cuda.memory_allocated(index))
        if accelerator == "xpu" and is_xpu_available() and hasattr(torch.xpu, "memory_allocated"):
            return int(torch.xpu.memory_allocated(0 if index is None else index))
        if accelerator == "mps" and is_mps_available() and hasattr(torch.mps, "current_allocated_memory"):
            return int(torch.mps.current_allocated_memory())
    except Exception:
        return 0
    return 0


def get_device_memory_reserved(device=None) -> int:
    accelerator = get_accelerator_type(device)
    index = _device_index(get_preferred_device(device))
    try:
        if accelerator == "cuda" and torch.cuda.is_available():
            return int(torch.cuda.memory_reserved(index))
        if accelerator == "xpu" and is_xpu_available() and hasattr(torch.xpu, "memory_reserved"):
            return int(torch.xpu.memory_reserved(0 if index is None else index))
        if accelerator == "mps" and is_mps_available() and hasattr(torch.mps, "driver_allocated_memory"):
            return int(torch.mps.driver_allocated_memory())
    except Exception:
        return 0
    return 0
