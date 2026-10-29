"""Synthetic passive-tracer fields on a spherical Voronoi mesh."""

from __future__ import annotations

import math
from typing import Any

import torch


def _mesh_xyz(mesh, *, dtype: torch.dtype | None = None, device: torch.device | str | None = None) -> torch.Tensor:
    xyz = mesh.cell_xyz
    if dtype is not None or device is not None:
        xyz = xyz.to(dtype=dtype if dtype is not None else xyz.dtype, device=device if device is not None else xyz.device)
    return xyz


def _lat_lon_from_xyz(xyz: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    lat = torch.asin(xyz[..., 2].clamp(-1.0, 1.0))
    lon = torch.atan2(xyz[..., 1], xyz[..., 0])
    return lat, lon


def _normalize(v: torch.Tensor) -> torch.Tensor:
    return v / v.norm(dim=-1, keepdim=True).clamp_min(torch.finfo(v.dtype).tiny)


def _random_unit_vectors(
    count: int,
    *,
    seed: int,
    dtype: torch.dtype,
    device: torch.device | str,
) -> torch.Tensor:
    gen = torch.Generator(device="cpu")
    gen.manual_seed(int(seed))
    v = torch.randn(int(count), 3, generator=gen, dtype=torch.float64)
    return _normalize(v).to(dtype=dtype, device=device)


def _rand(
    shape: tuple[int, ...],
    *,
    seed: int,
    dtype: torch.dtype,
    device: torch.device | str,
) -> torch.Tensor:
    gen = torch.Generator(device="cpu")
    gen.manual_seed(int(seed))
    return torch.rand(shape, generator=gen, dtype=torch.float64).to(dtype=dtype, device=device)


def _angular_distance(xyz: torch.Tensor, center: torch.Tensor) -> torch.Tensor:
    return torch.acos((xyz @ center).clamp(-1.0, 1.0))


def _rescale(
    q: torch.Tensor,
    *,
    min_value: float = 0.0,
    max_value: float = 1.0,
) -> torch.Tensor:
    q_min = q.amin()
    q_max = q.amax()
    span = (q_max - q_min).clamp_min(torch.finfo(q.dtype).tiny)
    out = (q - q_min) / span
    return float(min_value) + (float(max_value) - float(min_value)) * out


def _expand_levels_tracers(base: torch.Tensor, n_levels: int, n_tracers: int) -> torch.Tensor:
    q = base[:, None, None].expand(base.shape[0], int(n_levels), int(n_tracers)).clone()
    if int(n_levels) > 1:
        z = torch.linspace(0.0, 1.0, int(n_levels), dtype=base.dtype, device=base.device)
        modulation = 0.9 + 0.1 * torch.cos(math.pi * z).square()
        q = q * modulation[None, :, None]
    if int(n_tracers) > 1:
        factors = torch.linspace(1.0, 0.75, int(n_tracers), dtype=base.dtype, device=base.device)
        q = q * factors[None, None, :]
    return q


def gaussian_blob(
    mesh,
    *,
    center: torch.Tensor | None = None,
    width_rad: float = 0.35,
    baseline: float = 0.02,
    amplitude: float = 1.0,
    n_levels: int = 1,
    n_tracers: int = 1,
    dtype: torch.dtype | None = None,
    device: torch.device | str | None = None,
) -> torch.Tensor:
    """Return a positive Gaussian/cosine-bell-like blob concentration."""

    xyz = _mesh_xyz(mesh, dtype=dtype, device=device)
    if center is None:
        center = torch.tensor([1.0, 0.0, 0.0], dtype=xyz.dtype, device=xyz.device)
    else:
        center = _normalize(center.to(dtype=xyz.dtype, device=xyz.device).reshape(1, 3))[0]
    dist = _angular_distance(xyz, center)
    q = float(baseline) + float(amplitude) * torch.exp(-0.5 * (dist / float(width_rad)).square())
    return _expand_levels_tracers(q, n_levels, n_tracers)


def cosine_bell(
    mesh,
    *,
    center: torch.Tensor | None = None,
    radius_rad: float = 0.55,
    baseline: float = 0.0,
    amplitude: float = 1.0,
    n_levels: int = 1,
    n_tracers: int = 1,
    dtype: torch.dtype | None = None,
    device: torch.device | str | None = None,
) -> torch.Tensor:
    """Return a compact cosine bell on the sphere."""

    xyz = _mesh_xyz(mesh, dtype=dtype, device=device)
    if center is None:
        center = torch.tensor([1.0, 0.0, 0.0], dtype=xyz.dtype, device=xyz.device)
    else:
        center = _normalize(center.to(dtype=xyz.dtype, device=xyz.device).reshape(1, 3))[0]
    dist = _angular_distance(xyz, center)
    inside = dist <= float(radius_rad)
    bell = 0.5 * (1.0 + torch.cos(math.pi * dist / float(radius_rad)))
    q = float(baseline) + float(amplitude) * torch.where(inside, bell, torch.zeros_like(bell))
    return _expand_levels_tracers(q, n_levels, n_tracers)


def multi_blob(
    mesh,
    *,
    seed: int = 0,
    n_blobs: int = 5,
    width_rad: float = 0.25,
    n_levels: int = 1,
    n_tracers: int = 1,
    dtype: torch.dtype | None = None,
    device: torch.device | str | None = None,
) -> torch.Tensor:
    """Return a reproducible mixture of random Gaussian blobs."""

    xyz = _mesh_xyz(mesh, dtype=dtype, device=device)
    centers = _random_unit_vectors(n_blobs, seed=seed, dtype=xyz.dtype, device=xyz.device)
    amps = 0.4 + _rand((int(n_blobs),), seed=seed + 101, dtype=xyz.dtype, device=xyz.device)
    widths = float(width_rad) * (0.65 + 0.7 * _rand((int(n_blobs),), seed=seed + 202, dtype=xyz.dtype, device=xyz.device))
    q = torch.zeros(xyz.shape[0], dtype=xyz.dtype, device=xyz.device)
    for center, amp, width in zip(centers, amps, widths):
        dist = _angular_distance(xyz, center)
        q = q + amp * torch.exp(-0.5 * (dist / width).square())
    q = _rescale(q, min_value=0.02, max_value=1.0)
    return _expand_levels_tracers(q, n_levels, n_tracers)


def slotted_cylinder(
    mesh,
    *,
    center: torch.Tensor | None = None,
    radius_rad: float = 0.42,
    slot_width_rad: float = 0.12,
    slot_depth_rad: float = 0.34,
    n_levels: int = 1,
    n_tracers: int = 1,
    dtype: torch.dtype | None = None,
    device: torch.device | str | None = None,
) -> torch.Tensor:
    """Return a spherical slotted-cylinder-like sharp blob."""

    xyz = _mesh_xyz(mesh, dtype=dtype, device=device)
    if center is None:
        center = _normalize(torch.tensor([[0.75, 0.45, 0.48]], dtype=xyz.dtype, device=xyz.device))[0]
    else:
        center = _normalize(center.to(dtype=xyz.dtype, device=xyz.device).reshape(1, 3))[0]
    lat0, lon0 = _lat_lon_from_xyz(center)
    east = torch.stack([-torch.sin(lon0), torch.cos(lon0), torch.zeros_like(lon0)])
    north = torch.stack(
        [
            -torch.sin(lat0) * torch.cos(lon0),
            -torch.sin(lat0) * torch.sin(lon0),
            torch.cos(lat0),
        ]
    )
    tangent = xyz - (xyz @ center)[:, None] * center[None, :]
    xloc = tangent @ east
    yloc = tangent @ north
    dist = _angular_distance(xyz, center)
    cap = dist <= float(radius_rad)
    slot = (xloc.abs() < float(slot_width_rad)) & (yloc > -0.1 * float(radius_rad)) & (yloc < float(slot_depth_rad))
    q = torch.where(cap & ~slot, torch.ones_like(dist), torch.zeros_like(dist))
    return _expand_levels_tracers(q, n_levels, n_tracers)


def front_field(
    mesh,
    *,
    seed: int = 0,
    width_rad: float = 0.08,
    sharpness: float = 1.0,
    n_levels: int = 1,
    n_tracers: int = 1,
    dtype: torch.dtype | None = None,
    device: torch.device | str | None = None,
) -> torch.Tensor:
    """Return a random great-circle front with optional waviness."""

    xyz = _mesh_xyz(mesh, dtype=dtype, device=device)
    normal = _random_unit_vectors(1, seed=seed, dtype=xyz.dtype, device=xyz.device)[0]
    wave_dir = _random_unit_vectors(1, seed=seed + 33, dtype=xyz.dtype, device=xyz.device)[0]
    signed = xyz @ normal
    waviness = 0.2 * torch.sin(5.0 * (xyz @ wave_dir))
    q = torch.sigmoid((signed + waviness) / max(float(width_rad), 1.0e-6))
    if float(sharpness) != 1.0:
        q = torch.sigmoid(float(sharpness) * (q - 0.5))
        q = _rescale(q, min_value=0.0, max_value=1.0)
    q = 0.01 + 0.99 * q
    return _expand_levels_tracers(q, n_levels, n_tracers)


def filament_field(
    mesh,
    *,
    seed: int = 0,
    width_rad: float = 0.09,
    n_levels: int = 1,
    n_tracers: int = 1,
    dtype: torch.dtype | None = None,
    device: torch.device | str | None = None,
) -> torch.Tensor:
    """Return a narrow ring/filament-like positive field."""

    xyz = _mesh_xyz(mesh, dtype=dtype, device=device)
    normal = _random_unit_vectors(1, seed=seed, dtype=xyz.dtype, device=xyz.device)[0]
    along = _random_unit_vectors(1, seed=seed + 44, dtype=xyz.dtype, device=xyz.device)[0]
    distance_to_great_circle = torch.asin((xyz @ normal).clamp(-1.0, 1.0)).abs()
    modulation = 0.55 + 0.45 * torch.cos(8.0 * (xyz @ along)).square()
    q = torch.exp(-0.5 * (distance_to_great_circle / float(width_rad)).square()) * modulation
    q = _rescale(q, min_value=0.01, max_value=1.0)
    return _expand_levels_tracers(q, n_levels, n_tracers)


def red_noise_field(
    mesh,
    *,
    seed: int = 0,
    n_modes: int = 32,
    spectral_slope: float = 2.0,
    smoothness_scale: float = 0.25,
    intermittency: float = 1.0,
    min_value: float = 0.0,
    max_value: float = 1.0,
    n_levels: int = 1,
    n_tracers: int = 1,
    dtype: torch.dtype | None = None,
    device: torch.device | str | None = None,
) -> torch.Tensor:
    """Return a reproducible red-noise field from random spherical modes."""

    xyz = _mesh_xyz(mesh, dtype=dtype, device=device)
    n_modes = int(n_modes)
    dirs = _random_unit_vectors(n_modes, seed=seed, dtype=xyz.dtype, device=xyz.device)
    phases = 2.0 * math.pi * _rand((n_modes,), seed=seed + 77, dtype=xyz.dtype, device=xyz.device)
    raw_k = 1.0 + torch.floor(12.0 * _rand((n_modes,), seed=seed + 88, dtype=xyz.dtype, device=xyz.device))
    amplitude = raw_k.pow(-float(spectral_slope))
    if float(smoothness_scale) > 0.0:
        amplitude = amplitude * torch.exp(-0.5 * (raw_k * float(smoothness_scale) / math.pi).square())
    projection = xyz @ dirs.T
    angular = torch.acos(projection.clamp(-1.0, 1.0))
    field = (amplitude[None, :] * torch.cos(raw_k[None, :] * angular + phases[None, :])).sum(dim=-1)
    field = (field - field.mean()) / field.std().clamp_min(torch.finfo(field.dtype).tiny)
    if float(intermittency) != 1.0:
        field = torch.tanh(float(intermittency) * field)
    q = _rescale(field, min_value=min_value, max_value=max_value)
    return _expand_levels_tracers(q, n_levels, n_tracers)


def make_tracer_field(
    mesh,
    kind: str,
    *,
    seed: int = 0,
    n_levels: int = 1,
    n_tracers: int = 1,
    dtype: torch.dtype | None = None,
    device: torch.device | str | None = None,
    **kwargs: Any,
) -> torch.Tensor:
    """Dispatch a named synthetic tracer generator."""

    name = str(kind).lower()
    common = {
        "n_levels": int(n_levels),
        "n_tracers": int(n_tracers),
        "dtype": dtype,
        "device": device,
    }
    if name in {"gaussian", "gaussian_blob", "blob", "cosine_bell"}:
        if name == "cosine_bell":
            return cosine_bell(mesh, **common, **kwargs)
        return gaussian_blob(mesh, **common, **kwargs)
    if name in {"multi_blob", "multi_blobs"}:
        return multi_blob(mesh, seed=seed, **common, **kwargs)
    if name in {"slotted", "slotted_cylinder", "cylinder"}:
        return slotted_cylinder(mesh, **common, **kwargs)
    if name in {"front", "sharp_front"}:
        return front_field(mesh, seed=seed, **common, **kwargs)
    if name in {"filament", "ring"}:
        return filament_field(mesh, seed=seed, **common, **kwargs)
    if name in {"red_noise", "rednoise", "climate_like"}:
        return red_noise_field(mesh, seed=seed, **common, **kwargs)
    raise ValueError(
        "tracer kind must be one of gaussian, multi_blob, slotted_cylinder, "
        f"front, filament, red_noise; got {kind!r}"
    )


__all__ = [
    "cosine_bell",
    "filament_field",
    "front_field",
    "gaussian_blob",
    "make_tracer_field",
    "multi_blob",
    "red_noise_field",
    "slotted_cylinder",
]
