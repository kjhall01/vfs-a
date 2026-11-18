"""Explicit CPU/CUDA execution for the float64 coupled workflow.

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
        raise ValueError("device must be cpu, cuda, or cuda:N")
    try:
        device = torch.device(value)
    except (TypeError, RuntimeError) as error:
        raise ValueError("device must be cpu, cuda, or cuda:N") from error
    if device.type not in ("cpu", "cuda") or (device.type == "cpu" and device.index is not None):
        raise ValueError("coupled float64 execution supports cpu, cuda, or cuda:N")
    return device


def resolve_device(value):
    """Resolve CUDA against visible GPUs; never silently fall back to CPU."""
    device = device_spec(value)
    if device.type == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but unavailable; use a GPU allocation and a CUDA-enabled PyTorch installation")
        index = torch.cuda.current_device() if device.index is None else device.index
        if index >= torch.cuda.device_count():
            raise ValueError(f"CUDA device {index} is outside the {torch.cuda.device_count()} visible GPUs")
        device = torch.device("cuda", index)
    return device


@contextmanager
def execution_context(device):
    """Require deterministic CUDA algorithms for the existing exact replay check.

Unsupported operations fail visibly. Keep CPU defaults and restore the caller's
algorithm policy on exit. cuBLAS workspace is configured before the first model
operation; an incompatible user setting is reported rather than overwritten.
"""
    device = resolve_device(device)
    if device.type == "cpu":
        yield
        return
    workspace = os.environ.get("CUBLAS_WORKSPACE_CONFIG")
    if workspace not in (None, ":4096:8", ":16:8"):
        raise ValueError("CUDA replay requires CUBLAS_WORKSPACE_CONFIG=:4096:8 (or :16:8)")
    if workspace is None:
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    deterministic = torch.are_deterministic_algorithms_enabled()
    warn_only = torch.is_deterministic_algorithms_warn_only_enabled()
    try:
        torch.use_deterministic_algorithms(True)
        with torch.cuda.device(device):
            yield
    finally:
        torch.use_deterministic_algorithms(deterministic, warn_only=warn_only)
        if workspace is None:
            os.environ.pop("CUBLAS_WORKSPACE_CONFIG", None)


def runtime_manifest(device):
    device = resolve_device(device)
    result = dict(device=str(device), dtype="float64",
                  deterministic_algorithms=torch.are_deterministic_algorithms_enabled())
    if device.type == "cuda":
        props = torch.cuda.get_device_properties(device)
        result.update(name=props.name, capability=list(torch.cuda.get_device_capability(device)),
                      total_memory_bytes=props.total_memory, cuda=torch.version.cuda,
                      cublas_workspace=os.environ.get("CUBLAS_WORKSPACE_CONFIG"))
    return result


def synchronize(device):
    """Time completed GPU work, including kernels queued by compiled regions."""
    if torch.device(device).type == "cuda":
        torch.cuda.synchronize(device)
