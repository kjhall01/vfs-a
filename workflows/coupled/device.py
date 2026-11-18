"""Explicit device and floating-point precision for the coupled workflow.

Mesh construction and file I/O stay on the host. Prognostic tensors and the
registered model geometry live on one selected device. This is not a global
default-device switch: every initializer still declares its tensor ownership.
"""
from contextlib import contextmanager
import os
import torch


def device_spec(value):
    """Validate a configuration without requiring that its hardware is present."""
    if not isinstance(value, (str, torch.device)):
        raise ValueError("device must be cpu, cuda, cuda:N, or mps")
    try:
        device = torch.device(value)
    except (TypeError, RuntimeError) as error:
        raise ValueError("device must be cpu, cuda, cuda:N, or mps") from error
    if (device.type not in ("cpu", "cuda", "mps")
            or (device.type == "cpu" and device.index is not None)
            or (device.type == "mps" and device.index not in (None, 0))):
        raise ValueError("coupled execution supports cpu, cuda, cuda:N, or mps")
    return device


def dtype_spec(value):
    """Only FP32/FP64 are supported; half precision is not a solver policy."""
    for name, dtype in (("float32", torch.float32), ("float64", torch.float64)):
        if value == name or value == dtype:
            return dtype
    raise ValueError("dtype must be float32 or float64")


def validate_precision(device, dtype):
    """Check the pair even on a host without the requested accelerator."""
    device, dtype = device_spec(device), dtype_spec(dtype)
    if device.type == "mps" and dtype != torch.float32:
        raise ValueError("MPS does not support float64; explicitly select --dtype float32")
    return dtype


def mixing_budget_rtol(dtype):
    """Relative roundoff allowance for the coupled lagged TKE ledger.

    Its original 1e-9 threshold is below one FP32 ulp. Allow 32 eps for the
    tridiagonal solve and local products/differences; retain the FP64 default.
    This changes acceptance precision, never the fluxes or stored state.
    """
    return max(1e-9, 32 * torch.finfo(dtype_spec(dtype)).eps)


@contextmanager
def initialization_dtype(dtype):
    """Scope constructors that use PyTorch's default; never change the caller."""
    previous = torch.get_default_dtype()
    try:
        torch.set_default_dtype(dtype)
        yield
    finally:
        torch.set_default_dtype(previous)


def resolve_device(value):
    """Resolve available hardware; never silently fall back to CPU."""
    device = device_spec(value)
    if device.type == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but unavailable; use a GPU allocation and a CUDA-enabled PyTorch installation")
        index = torch.cuda.current_device() if device.index is None else device.index
        if index >= torch.cuda.device_count():
            raise ValueError(f"CUDA device {index} is outside the {torch.cuda.device_count()} visible GPUs")
        device = torch.device("cuda", index)
    elif device.type == "mps":
        if not torch.backends.mps.is_available():
            raise RuntimeError("MPS requested but unavailable; use an MPS-enabled PyTorch installation on supported Apple hardware")
        device = torch.device("mps", 0)
    return device


@contextmanager
def execution_context(device):
    """Require deterministic GPU algorithms for the existing exact replay check.

Unsupported operations fail visibly. Keep CPU defaults and restore the caller's
algorithm policy on exit. cuBLAS workspace is configured before the first model
operation; an incompatible user setting is reported rather than overwritten.
"""
    device = resolve_device(device)
    if device.type == "cpu":
        yield
        return
    if device.type == "mps" and os.environ.get("PYTORCH_ENABLE_MPS_FALLBACK") == "1":
        raise ValueError("unset PYTORCH_ENABLE_MPS_FALLBACK before starting Python; coupled MPS execution must expose unsupported operators")
    workspace = os.environ.get("CUBLAS_WORKSPACE_CONFIG")
    if device.type == "cuda" and workspace not in (None, ":4096:8", ":16:8"):
        raise ValueError("CUDA replay requires CUBLAS_WORKSPACE_CONFIG=:4096:8 (or :16:8)")
    if device.type == "cuda" and workspace is None:
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    deterministic = torch.are_deterministic_algorithms_enabled()
    warn_only = torch.is_deterministic_algorithms_warn_only_enabled()
    try:
        torch.use_deterministic_algorithms(True)
        if device.type == "cuda":
            with torch.cuda.device(device):
                yield
        else:
            yield
    finally:
        torch.use_deterministic_algorithms(deterministic, warn_only=warn_only)
        if device.type == "cuda" and workspace is None:
            os.environ.pop("CUBLAS_WORKSPACE_CONFIG", None)


def runtime_manifest(device, dtype=torch.float64):
    device = resolve_device(device)
    dtype = validate_precision(device, dtype)
    result = dict(device=str(device), dtype=str(dtype).removeprefix("torch."),
                  deterministic_algorithms=torch.are_deterministic_algorithms_enabled(),
                  mixing_budget_rtol=mixing_budget_rtol(dtype))
    if device.type == "cuda":
        props = torch.cuda.get_device_properties(device)
        result.update(name=props.name, capability=list(torch.cuda.get_device_capability(device)),
                      total_memory_bytes=props.total_memory, cuda=torch.version.cuda,
                      cublas_workspace=os.environ.get("CUBLAS_WORKSPACE_CONFIG"))
    elif device.type == "mps":
        import platform
        result.update(macos=platform.mac_ver()[0], machine=platform.machine(),
                      mps_fallback=os.environ.get("PYTORCH_ENABLE_MPS_FALLBACK", "0"))
    return result


def synchronize(device):
    """Time completed GPU work, including kernels queued by compiled regions."""
    if torch.device(device).type == "cuda":
        torch.cuda.synchronize(device)
    elif torch.device(device).type == "mps":
        torch.mps.synchronize()
