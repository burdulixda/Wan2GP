"""Intel XPU compatibility patch for CUDA-oriented offload code.

Import before MMGP so its CUDA-shaped offload hooks can target native
``torch.xpu`` without enabling CUDA-only optional kernels.
"""
from __future__ import annotations

import sys

_PATCHED = False


def _arg_requests_xpu(argv: list[str] | None = None) -> bool:
    argv = sys.argv if argv is None else argv
    for index, arg in enumerate(argv[1:], start=1):
        if arg == "--gpu" and index + 1 < len(argv):
            return str(argv[index + 1]).strip().lower().split(":", 1)[0] == "xpu"
        if str(arg).startswith("--gpu="):
            return str(arg).split("=", 1)[1].strip().lower().split(":", 1)[0] == "xpu"
    return False


def should_apply_xpu_patch() -> bool:
    return _arg_requests_xpu()


def apply_xpu_patch() -> bool:
    global _PATCHED

    import torch as _torch

    if _PATCHED:
        return True
    if not hasattr(_torch, "xpu") or not _torch.xpu.is_available():
        return False

    preferred_device = _preferred_xpu_device(_torch)
    preferred_index = 0 if preferred_device.index is None else preferred_device.index
    try:
        _torch.xpu.set_device(preferred_device)
    except Exception:
        pass

    device_name = _safe_call(lambda: _torch.xpu.get_device_name(preferred_index), "Intel XPU")
    properties = _safe_call(lambda: _torch.xpu.get_device_properties(preferred_index), None)
    total_memory = int(getattr(properties, "total_memory", 0) or 0)

    print(f"[XPU Patch] Detected: {device_name}")
    print(f"[XPU Patch] Redirecting CUDA compatibility calls to {preferred_device}")

    _cuda = _torch.cuda

    class _DummyGraph:
        replay = lambda self, *a, **kw: None
        capture_begin = lambda self, *a, **kw: None
        capture_end = lambda self, *a, **kw: None

    def _device_index(device=None) -> int:
        device = _replace_cuda_device(_torch, device, preferred_device)
        if isinstance(device, _torch.device) and device.index is not None:
            return int(device.index)
        value = str(device or "").strip().lower()
        if ":" in value:
            value = value.split(":", 1)[1]
        return int(value) if value.isdigit() else preferred_index

    def _xpu_device(device=None):
        return _torch.device("xpu", _device_index(device))

    def _set_xpu_device(device=None):
        try:
            return _torch.xpu.set_device(_xpu_device(device))
        except Exception:
            return None

    # Keep CUDA feature gates disabled while making explicit torch.cuda calls
    # from MMGP-style offload code land on native XPU APIs.
    _cuda.is_available = lambda: False
    _cuda._is_compiled = lambda: False
    _cuda.empty_cache = _torch.xpu.empty_cache
    _cuda.synchronize = lambda device=None: _torch.xpu.synchronize(_xpu_device(device))
    _cuda.get_device_capability = lambda device=None: (0, 0)
    _cuda.manual_seed_all = _torch.xpu.manual_seed_all
    _cuda.manual_seed = lambda seed: _torch.xpu.manual_seed(int(seed))
    _cuda.current_stream = lambda device=None: _torch.xpu.current_stream(_xpu_device(device))
    _cuda.default_stream = lambda device=None: _torch.xpu.current_stream(_xpu_device(device))
    _cuda.get_device_properties = lambda device=None: _torch.xpu.get_device_properties(_device_index(device))
    _cuda.set_device = _set_xpu_device
    _cuda.current_device = _torch.xpu.current_device
    _cuda.device_count = _torch.xpu.device_count
    _cuda.device = _torch.xpu.device
    _cuda.Stream = _torch.xpu.Stream
    _cuda.stream = _torch.xpu.stream
    _cuda.Event = _torch.xpu.Event
    _cuda.is_bf16_supported = lambda device=None: True
    _cuda.bfloat16_supported = lambda device=None: True
    _cuda.is_current_stream_capturing = getattr(_torch.xpu, "is_current_stream_capturing", lambda: False)
    _cuda.graph = lambda *a, **kw: _DummyGraph()
    _cuda.CUDAGraph = _DummyGraph
    _cuda.graph_pool_handle = lambda: None
    _cuda.mem_get_info = lambda device=None: _torch.xpu.mem_get_info(_device_index(device))
    _cuda.memory_allocated = lambda device=None: _torch.xpu.memory_allocated(_device_index(device))
    _cuda.memory_reserved = lambda device=None: _torch.xpu.memory_reserved(_device_index(device))
    _cuda.max_memory_allocated = lambda device=None: _torch.xpu.max_memory_allocated(_device_index(device))
    _cuda.max_memory_reserved = lambda device=None: _torch.xpu.max_memory_reserved(_device_index(device))
    _cuda.reset_peak_memory_stats = lambda device=None: _torch.xpu.reset_peak_memory_stats(_device_index(device))
    _cuda.memory_stats = lambda device=None: _torch.xpu.memory_stats(_device_index(device))
    _cuda.ipc_collect = lambda: None
    _cuda.is_initialized = _torch.xpu.is_initialized
    _cuda._lazy_init = lambda: None
    if not hasattr(_cuda, "CudaError"):
        _cuda.CudaError = RuntimeError

    _patch_autocast(_torch)
    default_device_holder = _patch_device_moves(_torch, preferred_device)
    _patch_tensor_creation(_torch, preferred_device, default_device_holder)

    print("[XPU Patch] Applied successfully")
    if total_memory:
        print(f"[XPU Patch] Device memory: {total_memory / 1024 ** 3:.1f}GB")
    _PATCHED = True
    return True


def _preferred_xpu_device(torch_module):
    for index, arg in enumerate(sys.argv[1:], start=1):
        value = None
        if arg == "--gpu" and index + 1 < len(sys.argv):
            value = sys.argv[index + 1]
        elif str(arg).startswith("--gpu="):
            value = str(arg).split("=", 1)[1]
        if value is None:
            continue
        value = str(value).strip().lower()
        if value == "xpu":
            return torch_module.device("xpu", 0)
        if value.startswith("xpu:"):
            suffix = value.split(":", 1)[1]
            if suffix.isdigit():
                return torch_module.device("xpu", int(suffix))
    return torch_module.device("xpu", 0)


def _replace_cuda_device(torch_module, value, preferred_device):
    if isinstance(value, str):
        lower_value = value.strip().lower()
        if lower_value == "cuda":
            return preferred_device
        if lower_value.startswith("cuda:"):
            suffix = lower_value.split(":", 1)[1]
            if suffix.isdigit():
                return torch_module.device("xpu", int(suffix))
            return preferred_device
    if isinstance(value, torch_module.device) and value.type == "cuda":
        return torch_module.device("xpu", 0 if value.index is None else value.index)
    return value


def _replace_map_location(torch_module, map_location, preferred_device):
    if isinstance(map_location, dict):
        return {
            key: _replace_cuda_device(torch_module, value, preferred_device)
            for key, value in map_location.items()
        }
    return _replace_cuda_device(torch_module, map_location, preferred_device)


def _patch_autocast(torch_module) -> None:
    orig_autocast = torch_module.autocast

    class _XpuAutocast:
        def __init__(self, enabled=True, dtype=None, device_type="xpu", cache_enabled=None):
            self._autocast = orig_autocast("xpu", enabled=enabled, dtype=dtype, cache_enabled=cache_enabled)

        def __enter__(self):
            return self._autocast.__enter__()

        def __exit__(self, *args):
            return self._autocast.__exit__(*args)

    class _autocast_mode_mod:
        autocast = _XpuAutocast

    class _amp_common:
        @staticmethod
        def amp_definitely_not_available():
            return False

    class _PatchedAMP:
        autocast = _XpuAutocast
        autocast_mode = _autocast_mode_mod
        common = _amp_common

        class GradScaler:
            def __init__(self, *args, **kwargs):
                pass

            def step(self, *args, **kwargs):
                return args[0] if args else None

            def update(self, *args, **kwargs):
                pass

            def unscale_(self, *args, **kwargs):
                pass

            def get_scale(self):
                return 1.0

            def state_dict(self):
                return {}

            def load_state_dict(self, *args):
                pass

    def patched_autocast(device_type=None, *args, **kwargs):
        if device_type == "cuda":
            device_type = "xpu"
        if kwargs.get("device_type") == "cuda":
            kwargs["device_type"] = "xpu"
        return orig_autocast(device_type, *args, **kwargs)

    torch_module.autocast = patched_autocast
    torch_module.amp.autocast = patched_autocast
    try:
        import torch.cuda.amp as cuda_amp

        cuda_amp.autocast = _XpuAutocast
        cuda_amp.GradScaler = _PatchedAMP.GradScaler
        torch_module.cuda.amp = cuda_amp
    except Exception:
        torch_module.cuda.amp = _PatchedAMP


def _patch_device_moves(torch_module, preferred_device) -> dict:
    orig_tensor_to = torch_module.Tensor.to
    orig_module_to = torch_module.nn.Module.to
    orig_load = torch_module.load
    orig_set_default_device = torch_module.set_default_device
    orig_generator = torch_module.Generator
    default_device_holder = {"device": torch_module.device("cpu")}

    def patched_tensor_cuda(self, device=None, *args, **kwargs):
        return orig_tensor_to(self, _replace_cuda_device(torch_module, device or "cuda", preferred_device), *args, **kwargs)

    def patched_module_cuda(self, device=None):
        return orig_module_to(self, _replace_cuda_device(torch_module, device or "cuda", preferred_device))

    def patched_tensor_to(self, *args, **kwargs):
        args = tuple(_replace_cuda_device(torch_module, arg, preferred_device) for arg in args)
        if "device" in kwargs:
            kwargs["device"] = _replace_cuda_device(torch_module, kwargs["device"], preferred_device)
        return orig_tensor_to(self, *args, **kwargs)

    def patched_module_to(self, *args, **kwargs):
        args = tuple(_replace_cuda_device(torch_module, arg, preferred_device) for arg in args)
        if "device" in kwargs:
            kwargs["device"] = _replace_cuda_device(torch_module, kwargs["device"], preferred_device)
        return orig_module_to(self, *args, **kwargs)

    def patched_load(*args, **kwargs):
        if "map_location" in kwargs:
            kwargs["map_location"] = _replace_map_location(torch_module, kwargs["map_location"], preferred_device)
        elif len(args) >= 2:
            args = (args[0], _replace_map_location(torch_module, args[1], preferred_device), *args[2:])
        return orig_load(*args, **kwargs)

    def patched_set_default_device(device):
        device = _replace_cuda_device(torch_module, device, preferred_device)
        default_device_holder["device"] = device
        return orig_set_default_device(device)

    class _PatchedGenerator(orig_generator):
        def __new__(cls, device=None):
            device = _replace_cuda_device(torch_module, device, preferred_device)
            if device is None:
                return super().__new__(cls)
            return super().__new__(cls, device=device)

    torch_module.Tensor.cuda = patched_tensor_cuda
    torch_module.nn.Module.cuda = patched_module_cuda
    torch_module.Tensor.to = patched_tensor_to
    torch_module.nn.Module.to = patched_module_to
    torch_module.load = patched_load
    torch_module.set_default_device = patched_set_default_device
    torch_module.Generator = _PatchedGenerator
    return default_device_holder


def _patch_tensor_creation(torch_module, preferred_device, default_device_holder) -> None:
    for fn_name in (
        "zeros",
        "ones",
        "randn",
        "rand",
        "tensor",
        "arange",
        "linspace",
        "empty",
        "full",
        "eye",
        "zeros_like",
        "ones_like",
        "randn_like",
        "rand_like",
        "empty_like",
        "full_like",
        "as_tensor",
        "from_numpy",
    ):
        if not hasattr(torch_module, fn_name):
            continue
        original = getattr(torch_module, fn_name)

        def make_patcher(orig, name):
            def patched(*args, **kwargs):
                if "device" in kwargs:
                    kwargs["device"] = _replace_cuda_device(torch_module, kwargs["device"], preferred_device)
                elif name != "from_numpy":
                    default_device = default_device_holder["device"]
                    if isinstance(default_device, torch_module.device) and default_device.type == "xpu":
                        kwargs["device"] = default_device
                return orig(*args, **kwargs)

            return patched

        setattr(torch_module, fn_name, make_patcher(original, fn_name))


def _safe_call(func, default):
    try:
        return func()
    except Exception:
        return default
