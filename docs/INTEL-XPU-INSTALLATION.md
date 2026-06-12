# Intel XPU Installation Guide

This guide covers the initial Intel GPU path for Wan2GP using native PyTorch XPU
wheels. Start with the conservative settings below; optimized CUDA-only kernels
are not part of the initial Intel baseline.

## Supported Baseline

Use this path for Intel Arc, Arc Pro, and other Intel GPUs supported by current
PyTorch XPU wheels.

The initial target is:

- PyTorch XPU, not Intel Extension for PyTorch.
- SDPA attention.
- Memory profile 5 for first smoke tests.
- FP16 fallback if BF16 support or kernels are unstable on a specific system.

CUDA-only acceleration packages such as Sage Attention, Sparge Attention, Flash
Attention, Nunchaku, Lightx2v, GGUF CUDA kernels, and CUDA graph paths should be
treated as unsupported on Intel XPU until tested individually.

## Install

The installer includes an `INTEL_XPU` hardware profile. On Intel GPU systems,
run the normal installer and select the detected Intel XPU profile when prompted:

```bash
python setup.py install
```

For a manual PyTorch install inside an already active environment:

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/xpu
pip install -r requirements.txt
```

## Verify PyTorch XPU

Run:

```bash
python -c "import torch; print(torch.__version__); print(torch.xpu.is_available()); print(torch.xpu.get_device_name(0) if torch.xpu.is_available() else 'No XPU')"
```

`torch.xpu.is_available()` must print `True`. If it prints `False`, check Intel
GPU driver installation before debugging Wan2GP.

## First Wan2GP Smoke Test

Start with SDPA and the lowest-memory profile:

```bash
python wgp.py --gpu xpu --attention sdpa --profile 5 --fp16
```

If the app starts and a small generation succeeds, try removing `--fp16` to use
the default dtype policy.

## Notes

PyTorch documents Intel GPU support through the `torch.xpu` API, including
`torch.xpu.is_available()`, `.to("xpu")`, `torch.autocast(device_type="xpu")`,
and `torch.xpu.synchronize()`. Wan2GP support should follow that native PyTorch
path first.

References:

- PyTorch Intel GPU getting started: https://docs.pytorch.org/docs/stable/notes/get_start_xpu.html
- PyTorch XPU API: https://docs.pytorch.org/docs/stable/xpu.html
