"""YAML/mapping construction helpers for the surface system."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

import torch

from .fractions import SurfaceFractions
from .glacier_land_ice import GlacierLandIce
from .icebergs import IcebergTracer
from .runoff_router import RunoffRouter
from .sea_ice import SeaIceThermo, SeaIceTransport
from .state import SurfaceState
from .system import SurfaceSystem


def _load_mapping(path_or_mapping: str | Path | Mapping[str, Any]) -> dict[str, Any]:
    if isinstance(path_or_mapping, Mapping):
        return dict(path_or_mapping)
    import yaml

    with Path(path_or_mapping).open() as f:
        loaded = yaml.safe_load(f) or {}
    if not isinstance(loaded, Mapping):
        raise ValueError("surface system config must be a mapping")
    return dict(loaded)


def _component_enabled(spec: Any, default: bool = False) -> bool:
    if isinstance(spec, bool):
        return spec
    if spec is None:
        return default
    if isinstance(spec, Mapping):
        return bool(spec.get("enabled", default))
    return bool(spec)


def _component_kwargs(spec: Any) -> dict[str, Any]:
    if isinstance(spec, Mapping):
        kwargs = dict(spec.get("kwargs", {}))
        for key, value in spec.items():
            if key not in {"enabled", "initial_state", "kwargs", "type"}:
                kwargs.setdefault(str(key), value)
        return kwargs
    return {}


def _component_initial_state(spec: Any) -> dict[str, Any]:
    if isinstance(spec, Mapping):
        initial = spec.get("initial_state", {})
        if isinstance(initial, Mapping):
            return dict(initial)
    return {}


def _infer_n_cells(config: Mapping[str, Any], *, mesh: Any | None, n_cells: int | None) -> int:
    if n_cells is not None:
        return int(n_cells)
    if mesh is not None:
        return int(mesh.nCells)
    if "n_cells" in config:
        return int(config["n_cells"])
    fractions = config.get("fractions", {})
    if isinstance(fractions, Mapping):
        for key in ("land_fraction", "ocean_fraction"):
            value = fractions.get(key)
            if value is not None and not isinstance(value, (float, int)):
                return int(torch.as_tensor(value).numel())
    raise ValueError("n_cells or mesh is required for surface system config")


def surface_system_from_config(
    path_or_mapping: str | Path | Mapping[str, Any],
    *,
    mesh: Any | None = None,
    topography: torch.Tensor | None = None,
    ocean_cell: torch.Tensor | None = None,
    n_cells: int | None = None,
    dtype: torch.dtype = torch.float64,
    device: torch.device | str | None = None,
) -> tuple[SurfaceSystem, SurfaceState]:
    """Build ``SurfaceSystem`` and its initial standalone state from YAML/mapping."""

    # Keep this import local: ``vfs.land`` depends on the lightweight
    # ``vfs.surface.packets`` module, and surface configuration is not needed
    # while land is being initialized.
    from ..land import LandSurface

    config = _load_mapping(path_or_mapping)
    n = _infer_n_cells(config, mesh=mesh, n_cells=n_cells)
    components = dict(config.get("components", {}))
    frac_spec = dict(config.get("fractions", {}))
    if ocean_cell is not None and "land_fraction" not in frac_spec:
        ocean_tensor = torch.as_tensor(ocean_cell, dtype=dtype, device=device).reshape(-1)
        frac_spec["ocean_fraction"] = ocean_tensor
        frac_spec["land_fraction"] = 1.0 - ocean_tensor
    land_fraction = frac_spec.get("land_fraction", 0.0)
    ocean_fraction = frac_spec.get("ocean_fraction")
    fractions = SurfaceFractions.from_land_ocean(
        land_fraction,
        ocean_fraction,
        sea_ice_fraction=frac_spec.get("sea_ice_fraction", 0.0),
        glacier_fraction=frac_spec.get("glacier_fraction", 0.0),
        snow_fraction=frac_spec.get("snow_fraction", 0.0),
        n_cells=n,
        dtype=dtype,
        device=device,
    )

    land = LandSurface(**_component_kwargs(components.get("land"))) if _component_enabled(components.get("land")) else None
    sea_ice = (
        SeaIceThermo(**_component_kwargs(components.get("sea_ice")))
        if _component_enabled(components.get("sea_ice"))
        else None
    )
    sea_ice_transport = (
        SeaIceTransport(**_component_kwargs(components.get("sea_ice_transport")))
        if _component_enabled(components.get("sea_ice_transport"))
        else None
    )
    glacier = (
        GlacierLandIce(**_component_kwargs(components.get("glacier")))
        if _component_enabled(components.get("glacier"))
        else None
    )
    icebergs = (
        IcebergTracer(**_component_kwargs(components.get("icebergs")))
        if _component_enabled(components.get("icebergs"))
        else None
    )
    router = None
    router_spec = components.get("runoff_router")
    if _component_enabled(router_spec):
        kwargs = _component_kwargs(router_spec)
        topo = kwargs.pop("topography", topography)
        ocean = kwargs.pop("ocean_cell", ocean_cell)
        if topo is None or ocean is None or mesh is None:
            raise ValueError("runoff_router config requires mesh, topography, and ocean_cell")
        router = RunoffRouter.from_mesh_topography(mesh, topo, ocean, **kwargs)

    system_kwargs = dict(config.get("system", {}))
    system = SurfaceSystem(
        land=land,
        sea_ice=sea_ice,
        sea_ice_transport=sea_ice_transport,
        glacier=glacier,
        icebergs=icebergs,
        runoff_router=router,
        **system_kwargs,
    )
    state = system.initial_state(n, fractions=fractions, dtype=dtype, device=device)
    if land is not None:
        land_initial = _component_initial_state(components.get("land"))
        if land_initial:
            state = state.with_components(land=land.initial_state(n, dtype=dtype, device=device, **land_initial))
    state.validate()
    return system, state


load_surface_system_config = surface_system_from_config


__all__ = ["load_surface_system_config", "surface_system_from_config"]
