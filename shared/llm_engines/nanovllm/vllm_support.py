import copy
import os
from typing import Any, Callable, Optional

_PROBE_CACHE = None
_WARNED_REQUESTED_VLLM_NOT_SUPPORTED = False
_TRITON_SMOKE_CACHE = {}
_XPU_TRITON_FALLBACK_CACHE = None
XPU_TRITON_FALLBACK_ENV = "WGP_XPU_TRITON_FALLBACKS"
_WINDOWS_DLL_DIRECTORY_HANDLES = []
_XPU_TRITON_ENV_CACHE = None


def _env_enabled(name, default=True):
    raw = str(os.environ.get(name, "1" if default else "0")).strip().lower()
    return raw in ("1", "true", "yes", "y", "on")


def _is_mps_available():
    try:
        import torch
        return hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
    except Exception:
        return False


def _short_error_message(exc):
    msg = str(exc).replace("\n", " ").strip()
    if len(msg) > 260:
        msg = msg[:260] + "..."
    return msg


def _ensure_windows_intel_runtime_dll_dirs() -> None:
    if os.name != "nt" or _WINDOWS_DLL_DIRECTORY_HANDLES:
        return
    import sys

    candidates = [
        os.path.join(sys.prefix, "Library", "bin"),
        os.path.join(sys.prefix, "Library", "lib"),
    ]
    for candidate in candidates:
        if not os.path.isdir(candidate):
            continue
        try:
            _WINDOWS_DLL_DIRECTORY_HANDLES.append(os.add_dll_directory(candidate))
        except Exception:
            pass


def _clean_tool_path(raw_path):
    if not raw_path:
        return None
    return str(raw_path).strip().strip('"')


def _normalize_windows_cl_path(path):
    path = _clean_tool_path(path)
    if not path:
        return None
    if os.name == "nt" and os.path.basename(path).lower() in ("cl", "cl.exe"):
        return os.path.join(os.path.dirname(path), "cl.EXE")
    return path


def _which_tool(*names):
    import shutil

    for name in names:
        candidate = _clean_tool_path(name)
        if not candidate:
            continue
        if os.path.isfile(candidate):
            return _normalize_windows_cl_path(candidate)
        found = shutil.which(candidate)
        if found:
            return _normalize_windows_cl_path(found)
    return None


def _prepend_windows_path(entry):
    entry = _clean_tool_path(entry)
    if not entry or not os.path.isdir(entry):
        return
    current = os.environ.get("PATH") or os.environ.get("Path", "")
    entry_norm = os.path.normcase(os.path.abspath(entry))
    parts = [part for part in current.split(os.pathsep) if part]
    for part in parts:
        try:
            if os.path.normcase(os.path.abspath(part)) == entry_norm:
                return
        except Exception:
            continue
    updated = entry + (os.pathsep + current if current else "")
    os.environ["PATH"] = updated
    os.environ["Path"] = updated


def _is_windows_cl(path):
    path = _clean_tool_path(path)
    return os.name == "nt" and bool(path) and os.path.basename(path).lower() in ("cl", "cl.exe")


def _setdefault_directory_env(name, path):
    if os.environ.get(name):
        return
    try:
        path = os.path.abspath(path)
        os.makedirs(path, exist_ok=True)
    except Exception:
        return
    os.environ[name] = path


def _configure_xpu_triton_compiler_env():
    cc = _which_tool(os.environ.get("CC"))
    cxx = _which_tool(os.environ.get("CXX"))
    if cc:
        os.environ["CC"] = cc
    if cxx:
        os.environ["CXX"] = cxx

    if not cc:
        cc = _which_tool("icx", "icpx", "clang", "gcc", "cl")
        if cc:
            os.environ["CC"] = cc
    if not cxx:
        cxx = _which_tool("icpx", "clang++", "g++", "cl")
        if cxx:
            os.environ["CXX"] = cxx

    if os.name == "nt" and _is_windows_cl(cxx or cc):
        cl_path = _normalize_windows_cl_path(cxx or cc)
        _prepend_windows_path(os.path.dirname(cl_path))
        os.environ["CC"] = cl_path
        os.environ["CXX"] = cl_path
        cc = cxx = cl_path

    return cc, cxx


def _ensure_xpu_triton_runtime_env():
    global _XPU_TRITON_ENV_CACHE
    if _XPU_TRITON_ENV_CACHE is not None:
        return _XPU_TRITON_ENV_CACHE

    _ensure_windows_intel_runtime_dll_dirs()
    os.environ.setdefault("ONEAPI_DEVICE_SELECTOR", "level_zero:0")
    os.environ.setdefault("UR_L0_USE_RELAXED_ALLOCATION_LIMITS", "1")
    os.environ.setdefault("SYCL_CACHE_PERSISTENT", "1")
    _setdefault_directory_env("SYCL_CACHE_DIR", os.path.join("cache", "sycl-xpu"))
    _setdefault_directory_env("TRITON_CACHE_DIR", os.path.join("cache", "triton-xpu"))

    cc, cxx = _configure_xpu_triton_compiler_env()
    if not cc or not cxx:
        _XPU_TRITON_ENV_CACHE = (
            False,
            "No C/C++ compiler is visible through CC, CXX, or PATH; prepare the compiler environment before launching Wan2GP",
        )
        return _XPU_TRITON_ENV_CACHE

    if _is_windows_cl(cxx) and (not os.environ.get("INCLUDE") or not os.environ.get("LIB")):
        _XPU_TRITON_ENV_CACHE = (
            False,
            "MSVC cl.exe is visible, but INCLUDE/LIB are not; launch from a Visual Studio Developer Command Prompt or run VsDevCmd before Wan2GP",
        )
        return _XPU_TRITON_ENV_CACHE

    _XPU_TRITON_ENV_CACHE = (True, "ok")
    return _XPU_TRITON_ENV_CACHE


def _runtime_device_for_smoke(torch, device_type: str):
    device_type = str(device_type or "").strip().lower()
    if device_type == "cuda":
        if not torch.cuda.is_available():
            return None, "CUDA is not available"
        return torch.device("cuda", torch.cuda.current_device()), "ok"
    if device_type == "xpu":
        if not hasattr(torch, "xpu") or not torch.xpu.is_available():
            return None, "XPU is not available"
        index = torch.xpu.current_device() if hasattr(torch.xpu, "current_device") else 0
        return torch.device("xpu", index), "ok"
    return None, f"Unsupported Triton smoke device: {device_type}"


def _synchronize_runtime_device(torch, device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device=device)
    elif device.type == "xpu" and hasattr(torch, "xpu"):
        torch.xpu.synchronize(device)


def _check_triton_runtime_smoke(device_type="cuda"):
    global _TRITON_SMOKE_CACHE
    device_type = str(device_type or "cuda").strip().lower()
    if device_type in _TRITON_SMOKE_CACHE:
        return _TRITON_SMOKE_CACHE[device_type]
    try:
        import torch
        import triton
        import triton.language as tl

        _ensure_windows_intel_runtime_dll_dirs()

        device, device_msg = _runtime_device_for_smoke(torch, device_type)
        if device is None:
            _TRITON_SMOKE_CACHE[device_type] = (False, device_msg)
            return _TRITON_SMOKE_CACHE[device_type]

        @triton.jit
        def _smoke_add_one_kernel(x_ptr, y_ptr, n_elements, BLOCK: tl.constexpr):
            pid = tl.program_id(0)
            offs = pid * BLOCK + tl.arange(0, BLOCK)
            mask = offs < n_elements
            x = tl.load(x_ptr + offs, mask=mask, other=0.0)
            tl.store(y_ptr + offs, x + 1.0, mask=mask)

        n_elements = 128
        block_size = 128
        x = torch.arange(n_elements, dtype=torch.float32, device=device)
        y = torch.empty_like(x)
        grid = (triton.cdiv(n_elements, block_size),)
        _smoke_add_one_kernel[grid](x, y, n_elements, BLOCK=block_size)
        _synchronize_runtime_device(torch, device)
        if not torch.allclose(y, x + 1.0, atol=1e-5, rtol=1e-5):
            _TRITON_SMOKE_CACHE[device_type] = (False, f"Triton {device_type} runtime smoke test failed: incorrect output from smoke kernel")
            return _TRITON_SMOKE_CACHE[device_type]
    except Exception as exc:
        _TRITON_SMOKE_CACHE[device_type] = (False, f"Triton {device_type} runtime smoke test failed: {_short_error_message(exc)}")
        return _TRITON_SMOKE_CACHE[device_type]
    _TRITON_SMOKE_CACHE[device_type] = (True, "ok")
    return _TRITON_SMOKE_CACHE[device_type]


def _check_triton():
    try:
        import triton  # noqa: F401
        import triton.language as tl  # noqa: F401
    except Exception as exc:
        return False, f"Triton import failed: {exc}"
    if _env_enabled("WGP_VLLM_TRITON_SMOKE", default=True):
        smoke_ok, smoke_msg = _check_triton_runtime_smoke("cuda")
        if not smoke_ok:
            return False, smoke_msg
    return True, "ok"


def probe_xpu_triton_fallbacks(force=False):
    global _XPU_TRITON_FALLBACK_CACHE
    if _XPU_TRITON_FALLBACK_CACHE is not None and not force:
        return copy.deepcopy(_XPU_TRITON_FALLBACK_CACHE)

    checks = {}
    kernels = {
        "rmsnorm": False,
        "kv_cache": False,
    }

    if not _env_enabled(XPU_TRITON_FALLBACK_ENV, default=True):
        result = {
            "supported": False,
            "kernels": kernels,
            "checks": {"disabled": {"ok": False, "message": f"disabled by {XPU_TRITON_FALLBACK_ENV}"}},
        }
        _XPU_TRITON_FALLBACK_CACHE = result
        return copy.deepcopy(result)

    try:
        import torch
    except Exception as exc:
        result = {
            "supported": False,
            "kernels": kernels,
            "checks": {"torch": {"ok": False, "message": f"Torch import failed: {exc}"}},
        }
        _XPU_TRITON_FALLBACK_CACHE = result
        return copy.deepcopy(result)

    device, device_msg = _runtime_device_for_smoke(torch, "xpu")
    if device is None:
        result = {
            "supported": False,
            "kernels": kernels,
            "checks": {"xpu": {"ok": False, "message": device_msg}},
        }
        _XPU_TRITON_FALLBACK_CACHE = result
        return copy.deepcopy(result)

    compiler_ok, compiler_msg = _ensure_xpu_triton_runtime_env()
    checks["xpu_triton_env"] = {"ok": compiler_ok, "message": compiler_msg}
    if not compiler_ok:
        result = {
            "supported": False,
            "kernels": kernels,
            "checks": checks,
        }
        _XPU_TRITON_FALLBACK_CACHE = result
        return copy.deepcopy(result)

    triton_ok, triton_msg = _check_triton_runtime_smoke("xpu")
    checks["triton_xpu"] = {"ok": triton_ok, "message": triton_msg}

    if triton_ok:
        try:
            from shared.llm_engines.nanovllm.layers.layernorm import smoke_triton_rmsnorm
            from shared.llm_engines.nanovllm.layers.attention import smoke_triton_kv_cache
        except Exception as exc:
            checks["kernel_imports"] = {"ok": False, "message": f"Kernel smoke imports failed: {_short_error_message(exc)}"}
        else:
            rmsnorm_ok, rmsnorm_msg = smoke_triton_rmsnorm(device)
            checks["rmsnorm"] = {"ok": rmsnorm_ok, "message": rmsnorm_msg}
            kernels["rmsnorm"] = bool(rmsnorm_ok)

            kv_cache_ok, kv_cache_msg = smoke_triton_kv_cache(device)
            checks["kv_cache"] = {"ok": kv_cache_ok, "message": kv_cache_msg}
            kernels["kv_cache"] = bool(kv_cache_ok)

    result = {
        "supported": any(kernels.values()),
        "kernels": kernels,
        "checks": checks,
    }
    _XPU_TRITON_FALLBACK_CACHE = result
    return copy.deepcopy(result)


def _check_flash_attention_2():
    try:
        import flash_attn
        from flash_attn import flash_attn_varlen_func  # noqa: F401
        from flash_attn import flash_attn_with_kvcache  # noqa: F401
        version = str(getattr(flash_attn, "__version__", ""))
    except ModuleNotFoundError:
        return False, "non installed"
    except Exception as exc:
        if "no module named 'flash_attn'" in str(exc).strip().lower():
            return False, "non installed"
        return False, f"FlashAttention import failed: {exc}"

    major = None
    if len(version) > 0:
        try:
            major = int(version.split(".", 1)[0])
        except Exception:
            major = None
    if major is not None and major < 2:
        return False, f"FlashAttention major version is {major}, expected >= 2"
    return True, "ok"


def probe_vllm_runtime(force=False):
    global _PROBE_CACHE
    if _PROBE_CACHE is not None and not force:
        return _PROBE_CACHE.copy()

    checks = {}

    triton_ok, triton_msg = _check_triton()
    checks["triton"] = {"ok": triton_ok, "message": triton_msg}

    flash_ok, flash_msg = _check_flash_attention_2()
    checks["flash_attention_2"] = {"ok": flash_ok, "message": flash_msg}

    supported = triton_ok and flash_ok
    result = {
        "supported": supported,
        "preferred_engine": "vllm" if supported else "legacy",
        "checks": checks,
    }

    _PROBE_CACHE = result.copy()
    return result


def resolve_lm_decoder_engine(requested_engine, engines_available = []):
    requested_engine = str(requested_engine or "").strip().lower()
    if _is_mps_available():
        return "legacy"
    probe_result = probe_vllm_runtime()
    supported = bool(probe_result.get("supported", False))
    cg_available = "cg" in engines_available
    vllm_available= "vllm" in engines_available
    default_engine = "cg" if cg_available else "legacy"
    if requested_engine == "vllm":
        if supported:
            if vllm_available: return "vllm"
            requested_engine = default_engine
        elif not vllm_available:
            requested_engine = default_engine
        else:
            global _WARNED_REQUESTED_VLLM_NOT_SUPPORTED
            if not _WARNED_REQUESTED_VLLM_NOT_SUPPORTED:
                checks = probe_result.get("checks", {})
                reasons = []
                if isinstance(checks, dict):
                    for check_name, check_data in checks.items():
                        if isinstance(check_data, dict) and not check_data.get("ok", False):
                            msg = str(check_data.get("message", "failed")).replace("\n", " ").strip()
                            if len(msg) > 220:
                                msg = msg[:220] + "..."
                            reasons.append(f"{check_name}={msg}")
                reason_text = "; ".join(reasons) if len(reasons) > 0 else "unknown reason"
                # print(f"[LM] Requested decoder engine 'vllm' is not supported at startup ({reason_text}).")
                print(f"[LM] Requested decoder engine 'vllm' is not supported (triton & flash attention 2 are needed).")
                _WARNED_REQUESTED_VLLM_NOT_SUPPORTED = True
            return default_engine
    if requested_engine == "":
        return "vllm" if supported and vllm_available else default_engine
    if requested_engine in ("legacy", "cg"):
        if not cg_available:
            return "legacy"
        return requested_engine
    print(f"[LM] Unknown decoder engine '{requested_engine}', falling back to 'legacy'.")
    return "legacy"


def _clear_inductor_cuda_pools():
    try:
        from torch._inductor import cudagraph_trees as cgt
    except Exception:
        return

    clear_cublass_cache = getattr(cgt, "clear_cublass_cache", None)
    if callable(clear_cublass_cache):
        try:
            clear_cublass_cache()
        except Exception:
            pass


class NanoVllmTextEngine:
    keep_loaded_for_phase2 = True

    def __init__(self, model, model_path: str, tokenizer, enforce_eager: bool = False, graph_pool_handle=None):
        self.model = model
        self.model_path = model_path
        self.tokenizer = tokenizer
        self.enforce_eager = bool(enforce_eager)
        self.graph_pool_handle = graph_pool_handle
        self.hf_config = getattr(model, "config", None)
        self._llm = None
        self._sampling_params_cls = None
        self._max_model_len_hint = None
        self._max_num_seqs_hint = None
        self._max_num_batched_tokens_hint = None
        self._last_failure_reason = ""

    @staticmethod
    def _compute_runtime_hints(prompt_len: int, max_tokens: int, cfg_scale: float):
        max_model_len = max(8, int(prompt_len) + int(max_tokens))
        max_num_seqs = 2 if cfg_scale and cfg_scale > 1.0 else 1
        max_num_batched_tokens = max_model_len * max_num_seqs
        return max_model_len, max_num_seqs, max_num_batched_tokens

    def _get_min_model_len_hint(self):
        min_model_len = getattr(self.model, "_prompt_enhancer_min_model_len_hint", None)
        if min_model_len is None:
            return 8
        try:
            return max(8, int(min_model_len))
        except Exception:
            return 8

    def _ensure_runtime_capacity(self, max_model_len: int, max_num_seqs: int, max_num_batched_tokens: int):
        if self._max_model_len_hint is None:
            self._max_model_len_hint = max_model_len
            self._max_num_seqs_hint = max_num_seqs
            self._max_num_batched_tokens_hint = max_num_batched_tokens
            return

        need_grow = (
            max_model_len > int(self._max_model_len_hint)
            or max_num_seqs > int(self._max_num_seqs_hint)
            or max_num_batched_tokens > int(self._max_num_batched_tokens_hint)
        )
        if not need_grow:
            return

        self._max_model_len_hint = max_model_len
        self._max_num_seqs_hint = max_num_seqs
        self._max_num_batched_tokens_hint = max_num_batched_tokens
        self.close()

    def reserve_runtime(self, prompt_len: int, max_tokens: int, cfg_scale: float):
        req_model_len, req_num_seqs, req_num_batched = self._compute_runtime_hints(
            prompt_len=prompt_len,
            max_tokens=max_tokens,
            cfg_scale=cfg_scale,
        )
        req_model_len = max(req_model_len, self._get_min_model_len_hint())
        req_num_batched = max(req_num_batched, req_model_len * req_num_seqs)
        self._ensure_runtime_capacity(req_model_len, req_num_seqs, req_num_batched)

    def _ensure_llm(self):
        if self._llm is not None:
            return
        try:
            import torch._inductor.config as inductor_config

            if bool(getattr(inductor_config, "split_reductions", False)):
                inductor_config.split_reductions = False
        except Exception:
            pass
        try:
            from shared.llm_engines.nanovllm import LLM, SamplingParams
        except Exception as exc:  # pragma: no cover
            raise RuntimeError(f"nano-vllm is not available for vllm engine: {exc}") from exc
        if not self.model_path:
            raise RuntimeError("vllm engine requires a model_path")
        max_model_len = self._max_model_len_hint or 4096
        max_num_seqs = self._max_num_seqs_hint or 1
        max_num_batched_tokens = self._max_num_batched_tokens_hint or (max_model_len * max_num_seqs)
        hf_config = self.hf_config
        if hf_config is not None and bool(getattr(self.model, "_prompt_enhancer_allow_extended_context", False)):
            requested_max_model_len = int(self._max_model_len_hint or 0)
            configured_max_position_embeddings = int(getattr(hf_config, "max_position_embeddings", 0) or 0)
            if requested_max_model_len > configured_max_position_embeddings:
                hf_config = copy.deepcopy(hf_config)
                hf_config.max_position_embeddings = requested_max_model_len
        self._llm = LLM(
            model=self.model_path,
            enforce_eager=self.enforce_eager,
            tensor_parallel_size=1,
            max_model_len=max_model_len,
            max_num_seqs=max_num_seqs,
            max_num_batched_tokens=max_num_batched_tokens,
            hf_config=hf_config,
            tokenizer=self.tokenizer,
            model_object=self.model,
            graph_pool_handle=self.graph_pool_handle,
        )
        self._sampling_params_cls = SamplingParams

    def release_runtime_allocations(self):
        if self._llm is None:
            return
        try:
            self._llm.reset_runtime_state()
        except Exception:
            pass

    def close(self):
        llm = getattr(self, "_llm", None)
        self._llm = None
        if llm is not None:
            try:
                close_fn = getattr(llm, "close", None)
                if callable(close_fn):
                    close_fn()
                else:
                    try:
                        llm.reset_runtime_state()
                    except Exception:
                        pass
                    try:
                        llm.clear_graph_cache()
                    except Exception:
                        pass
                    exit_fn = getattr(llm, "exit", None)
                    if callable(exit_fn):
                        exit_fn()
            except Exception:
                pass
            try:
                del llm
            except Exception:
                pass
        self._sampling_params_cls = None
        try:
            _clear_inductor_cuda_pools()
        except Exception:
            pass

    def __del__(self):
        self.close()

    @staticmethod
    def _extract_text_and_tokens(output_obj) -> tuple[str, list[int]]:
        if output_obj is None:
            return "", []
        if isinstance(output_obj, dict):
            text = str(output_obj.get("text", "") or "")
            token_ids = output_obj.get("token_ids", []) or []
            return text, [int(x) for x in token_ids]
        if hasattr(output_obj, "outputs"):
            outputs = getattr(output_obj, "outputs", None)
            if outputs and len(outputs) > 0:
                text = str(getattr(outputs[0], "text", "") or "")
                token_ids = getattr(outputs[0], "token_ids", None)
                if token_ids is None:
                    token_ids = getattr(outputs[0], "token_ids_list", []) or []
                return text, [int(x) for x in token_ids] if token_ids else []
        text = str(getattr(output_obj, "text", "") or "")
        token_ids = getattr(output_obj, "token_ids", None)
        if token_ids is None:
            token_ids = []
        return text, [int(x) for x in token_ids]

    def get_last_failure_reason(self) -> str:
        return self._last_failure_reason

    def generate_text(
        self,
        prompt: str,
        prompt_negative: str,
        max_tokens: int,
        temperature: Optional[float],
        top_p: Optional[float],
        top_k: Optional[int],
        cfg_scale: float,
        seed: Optional[int],
        callback=None,
        abort_fn: Optional[Callable[[], bool]] = None,
        logits_processor: Optional[Any] = None,
        logits_processor_update_state: Optional[Callable[[int], None]] = None,
        stop_checker: Optional[Callable[[list[int], int], bool]] = None,
        progress_label: str = "LM text",
        release_vram_after: bool = True,
        ignore_eos: bool = False,
    ):
        del stop_checker
        if abort_fn is not None and abort_fn():
            if release_vram_after:
                self.release_runtime_allocations()
            return None
        try:
            prompt_len = len(self.tokenizer.encode(prompt))
        except Exception:
            prompt_len = 0
        if cfg_scale > 1.0 and prompt_negative:
            try:
                prompt_len = max(prompt_len, len(self.tokenizer.encode(prompt_negative)))
            except Exception:
                pass

        req_model_len, req_num_seqs, req_num_batched = self._compute_runtime_hints(
            prompt_len=prompt_len,
            max_tokens=max_tokens,
            cfg_scale=cfg_scale,
        )
        self._ensure_runtime_capacity(req_model_len, req_num_seqs, req_num_batched)
        self._ensure_llm()
        if self._llm is None:
            return None
        try:
            self._llm.reset()
        except Exception:
            pass

        if callback is not None:
            callback(
                step_idx=-1,
                override_num_inference_steps=max_tokens,
                denoising_extra=f"{progress_label} 0/{max_tokens}",
                progress_unit="tokens",
            )

        seed_value = None
        if seed is not None:
            try:
                seed_value = int(seed)
            except Exception:
                seed_value = None
            if seed_value is not None and seed_value < 0:
                seed_value = None

        temp = temperature if temperature is not None and temperature > 0 else 1e-5
        sampling_params = self._sampling_params_cls(
            temperature=temp,
            max_tokens=max_tokens,
            cfg_scale=max(cfg_scale, 1.0),
            top_k=top_k if top_k is not None and top_k > 0 else None,
            top_p=top_p if top_p is not None and 0.0 < top_p < 1.0 else None,
            ignore_eos=bool(ignore_eos),
            logits_processor=logits_processor,
            logits_processor_update_state=logits_processor_update_state,
            seed=seed_value,
        )

        text = ""
        token_ids: list[int] = []
        try:
            outputs = self._llm.generate(
                prompts=[prompt],
                sampling_params=sampling_params,
                use_tqdm=True,
                unconditional_prompts=[prompt_negative] if cfg_scale > 1.0 else None,
            )
            if outputs:
                text, token_ids = self._extract_text_and_tokens(outputs[0])
            if (not text) and token_ids:
                try:
                    text = self.tokenizer.decode(token_ids, skip_special_tokens=False)
                except Exception:
                    text = ""
            self._last_failure_reason = ""
        except Exception as exc:
            self._last_failure_reason = str(exc)
            raise
        finally:
            if release_vram_after:
                self.release_runtime_allocations()

        if callback is not None:
            callback(
                step_idx=max(0, max_tokens - 1),
                override_num_inference_steps=max_tokens,
                denoising_extra=f"{progress_label} {max_tokens}/{max_tokens}",
                progress_unit="tokens",
            )

        return {"token_ids": token_ids, "text": text}

    def generate_embedded(
        self,
        prompt_token_ids: list[int],
        prompt_embeds,
        prompt_position_ids,
        max_tokens: int,
        temperature: Optional[float],
        top_p: Optional[float],
        top_k: Optional[int],
        cfg_scale: float,
        seed: Optional[int],
        use_tqdm: bool = True,
        release_vram_after: bool = True,
        ignore_eos: bool = False,
        position_offset: int = 0,
    ):
        req_model_len, req_num_seqs, req_num_batched = self._compute_runtime_hints(
            prompt_len=len(prompt_token_ids),
            max_tokens=max_tokens,
            cfg_scale=cfg_scale,
        )
        self._ensure_runtime_capacity(req_model_len, req_num_seqs, req_num_batched)
        self._ensure_llm()
        if self._llm is None:
            return None
        try:
            self._llm.reset()
        except Exception:
            pass

        seed_value = None
        if seed is not None:
            try:
                seed_value = int(seed)
            except Exception:
                seed_value = None
            if seed_value is not None and seed_value < 0:
                seed_value = None

        temp = temperature if temperature is not None and temperature > 0 else 1e-5
        sampling_params = self._sampling_params_cls(
            temperature=temp,
            max_tokens=max_tokens,
            cfg_scale=max(cfg_scale, 1.0),
            top_k=top_k if top_k is not None and top_k > 0 else None,
            top_p=top_p if top_p is not None and 0.0 < top_p < 1.0 else None,
            ignore_eos=bool(ignore_eos),
            seed=seed_value,
        )

        text = ""
        token_ids: list[int] = []
        try:
            outputs = self._llm.generate_embedded(
                prompts=[prompt_token_ids],
                prompt_embeds=[prompt_embeds],
                prompt_position_ids=[prompt_position_ids],
                position_offsets=[int(position_offset)],
                sampling_params=sampling_params,
                use_tqdm=use_tqdm,
            )
            if outputs:
                text, token_ids = self._extract_text_and_tokens(outputs[0])
            if (not text) and token_ids:
                try:
                    text = self.tokenizer.decode(token_ids, skip_special_tokens=False)
                except Exception:
                    text = ""
            self._last_failure_reason = ""
        except Exception as exc:
            self._last_failure_reason = str(exc)
            raise
        finally:
            if release_vram_after:
                self.release_runtime_allocations()

        return {"token_ids": token_ids, "text": text}
