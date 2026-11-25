"""Explicit example configuration; no data download or package-default changes."""

from pathlib import Path
import numpy as np
import torch
from pyam21.constants import GRAV
from pyam21.namelist import read_namelist
from pyam21.sat_vapor_pres import get_sat_vapor_pres
from vfs import (
    MoistAtmosphere,
    VoronoiMesh,
    CharneyPhillipsColumn,
    make_hybrid_coefficients,
)
from .adapter import AM2Adapter
from workflows.coupled.device import initialization_dtype, resolve_device, validate_precision


_SEA_LEVEL_PRESSURE = 100000.0  # Pa


def _moist_profile(pressure, sea_level_temperature):
    temperature = 205 + (sea_level_temperature - 205) * (pressure / _SEA_LEVEL_PRESSURE) ** 0.7
    vapor = 0.7 * get_sat_vapor_pres().compute_qs(temperature, pressure)
    return temperature, vapor


def _terrain_surface_pressure(phi_s, sea_level_temperature, thermo, top_pressure_pa):
    """Invert moist hydrostatic balance from sea level to the terrain in each column."""
    nodes, weights = np.polynomial.legendre.leggauss(12)
    nodes = phi_s.new_tensor((nodes + 1) / 2)
    weights = phi_s.new_tensor(weights / 2)
    phi_s = phi_s.reshape(-1, 1)

    def gas_constant_times_temperature(log_pressure_ratio):
        pressure = _SEA_LEVEL_PRESSURE * log_pressure_ratio.exp()
        temperature, vapor = _moist_profile(pressure, sea_level_temperature)
        return (thermo.R_d + (thermo.R_v - thermo.R_d) * vapor) * temperature

    # With x = log(p_s / p0), the terrain geopotential is -integral_0^x R_mix T dx.
    # Its endpoint derivative supplies the Newton step. Sea level stays at x = 0.
    def residual(x):
        return phi_s + x * (weights * gas_constant_times_temperature(x * nodes)).sum(-1, keepdim=True)

    lid = phi_s.new_tensor(np.log(top_pressure_pa / _SEA_LEVEL_PRESSURE))
    if (residual(lid) >= 0).any():
        raise ValueError("terrain reaches the atmospheric pressure lid")
    x = -phi_s / (thermo.R_d * sea_level_temperature)
    for _ in range(5):
        x = x - residual(x) / gas_constant_times_temperature(x)
    ps = _SEA_LEVEL_PRESSURE * x.exp()
    if not torch.isfinite(ps).all() or (ps <= top_pressure_pa).any():
        raise ValueError("invalid hydrostatic surface pressure above terrain")
    return ps


@initialization_dtype(torch.float64)
def example(
    source,
    *,
    cells=32,
    levels=24,
    top_pressure_pa=1000.0,
    aquaplanet=False,
    energy_mode="conservative",
    radiation_data=None,
    radiation_step=3600,
    climatology_width_degrees=1.0,
    start=(2000, 1, 1, 0, 0, 0),
    mesh_options=None,
    surface_height_m=None,
    namelist_overrides=None,
    device="cpu",
    dtype=torch.float64,
):
    """Analytic moist profile with BOMEX physics options, not the BOMEX case.

    There is no SCM large-scale forcing. Radiation is enabled only with an
    explicit data directory. Surface fluxes, RAS convection, stratiform clouds
    and mixing are calculated by AM2.
    Positive upper-air vapor and a positive pressure lid are explicit initial
    conditions required by VFS, not hidden modifications to a reference case.
    """
    dtype = validate_precision(device, dtype)
    device = resolve_device(device)
    torch.set_num_threads(1)
    nml = read_namelist(str(Path(source) / "reference/input/gcss_bomex/input.nml"))
    if radiation_data is not None:
        # BOMEX disables aerosols, leaves the dataset blank and omits optical
        # model names. Take the complete optical configuration from upstream's
        # radiating ASTEX case; retain BOMEX's composition/climatology and all
        # other physics choices. Explicit caller overrides still apply below.
        aerosol = read_namelist(
            str(Path(source) / "reference/input/gcss_astex/input.nml")
        )["aerosolrad_package_nml"]
        nml.setdefault("aerosolrad_package_nml", {}).update(
            {
                key: aerosol[key]
                for key in (
                    "do_lwaerosol",
                    "do_swaerosol",
                    "aerosol_data_set",
                    "optical_filename",
                    "aerosol_optical_names",
                )
            }
        )
    for group, values in (namelist_overrides or {}).items():
        nml.setdefault(group, {}).update(values)
    mesh = VoronoiMesh(
        n_cells=cells, **({"lloyd_iters": 0, "verbose": False} | (mesh_options or {}))
    )
    if not 0.0 < top_pressure_pa < _SEA_LEVEL_PRESSURE:
        raise ValueError("top_pressure_pa must lie between 0 and reference surface pressure")
    A, B = make_hybrid_coefficients(levels, p_top=top_pressure_pa)
    if callable(surface_height_m):
        surface_height_m = surface_height_m(mesh)
    phi_s = 0.0 if surface_height_m is None else GRAV * torch.as_tensor(surface_height_m, dtype=torch.float64)
    column = CharneyPhillipsColumn(levels, A, B, phi_s=phi_s, phi_full_mode="direct_u_gl2_adjoint")
    model = MoistAtmosphere(mesh, column).to(dtype=torch.float64)
    # Explicit workflow configuration BEFORE any prognostic state exists. The
    # native dynamics and every mass/energy diagnostic use this same parameter.
    model.dynamics.kernel.gravity = GRAV
    rad = None
    if radiation_data is not None:
        from .radiation import build_radiation

        if (
            isinstance(radiation_step, bool)
            or not isinstance(radiation_step, int)
            or radiation_step <= 0
        ):
            raise ValueError("radiation_step must be positive integer seconds")
        nml.setdefault("physics_driver_nml", {})["do_radiation"] = True
        nml.setdefault("radiation_driver_nml", {}).update(
            rad_time_step=radiation_step,
            lw_rad_time_step=radiation_step,
            sw_rad_time_step=radiation_step,
            renormalize_sw_fluxes=True,
        )
        rad = build_radiation(
            nml, model, radiation_data, width_degrees=climatology_width_degrees
        )
    # Construct geometry and input-data sampling on CPU, then move registered
    # geometry BEFORE building maps, physics latitude/longitude or state.
    model.to(device=device, dtype=dtype)
    adapter = AM2Adapter(
        model,
        nml,
        energy_mode=energy_mode,
        start=start,
        radiation=None if rad is None else rad.driver,
        radiation_id=None if rad is None else rad.identity,
    )
    adapter.radiation_manifest = None if rad is None else rad.manifest
    sine_lat = mesh.cell_xyz[:, 2]
    surface = 300.0 - (25 * sine_lat.square() if aquaplanet else 0 * sine_lat)
    ps = _SEA_LEVEL_PRESSURE
    if surface_height_m is not None:
        ps = _terrain_surface_pressure(
            model.column.phi_s, surface[:, None], model.dynamics.thermodynamics, top_pressure_pa
        )
    p = model.interface_pressure(ps)
    T, vapor = _moist_profile(p, surface[:, None])
    atmosphere = model.state_from_fields(
        temperature=T, surface_pressure=ps, normal_wind=0.0, vapor=vapor
    )
    if surface_height_m is not None:
        surface = T[:, -1]
    state = adapter.initialize(atmosphere, sst=surface + 1.0)
    return adapter, state
