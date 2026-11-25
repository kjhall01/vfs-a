"""365-day coupled aquaplanet campaign with explicit device and precision."""

import argparse
import csv
from concurrent.futures import ThreadPoolExecutor
from contextlib import ExitStack
from dataclasses import asdict, dataclass, replace
import hashlib
import io
import json
import logging
import math
from pathlib import Path
import sys
import time
import numpy as np
import torch
import yaml

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))
from workflows.pyam21.setup import example
from workflows.pyam21.dynamics import advance_dynamics, pressure_velocity
from workflows.coupled.driver import exchange_step
from workflows.coupled.schedule import ocean_events, event_due
from workflows.coupled.ice_temperature import (
    ThermalIceReceiver,
    ThermalSnowOceanIce,
    ThermalIceState,
)
from workflows.pyam21.state import copy_tree
from workflows.coupled.compilation import configure_compilation
from workflows.coupled.energy_trace import EnergyTrace
from workflows.coupled.restart_compat import (
    verify_compatible_source_delta, canonical_manifest_json,
)
from workflows.coupled.device import (validate_precision, resolve_device, execution_context,
    runtime_manifest, synchronize, initialization_dtype, mixing_budget_rtol)
from ocean_ice import OceanIce
from ridging import CategoryIce, RidgingState
from snow import SnowState
from roquet import RoquetOcean, RoquetMixing
from eddy import GMRedi
from eddy_energy import EddyMixingLength, EddyDissipation
from backscatter import EddyBackscatter
from vfs.land import LandState
from vfs.ocean.domain import OceanDomain
from vfs.ocean.stratified.remap import ThermalProjectionInfeasible
from workflows.coupled.coast import CoastalReceiver
from vfs.ocean import OceanColumn

DAY = 86400
VAPOR_SUMS = ("vapor_correction_events", "vapor_correction_kg_m2")
VAPOR_PEAKS = ("vapor_correction_proxy_residual_J_m2",)
OCEAN_SUMS = (
    "gmredi_steps",
    "gm_limiting_passes",
    "gm_limited_interfaces",
    "redi_limited_fluxes",
    "eddy_decay_steps",
    "ale_steps",
    "ale_changed_columns",
    "ale_deferred_columns",
    "gm_exchange_J_m2",
    "redi_exchange_J_m2",
    "eddy_dissipation_J_m2",
    "ale_energy_change_J_m2",
)
OCEAN_PEAKS = (
    "gm_outflow_fraction",
    "redi_step_number",
    "gmredi_local_residual_J_m2",
    "eddy_decay_local_residual_J_m2",
    "ale_column_residual_J_m2",
)


@dataclass(frozen=True)
class CampaignConfig:
    days: int = 365
    start_year: int = 2001  # non-leap year on upstream Julian calendar
    cells: int = 256
    atmosphere_levels: int = 24
    atmosphere_top_pa: float = 1000.0
    ocean_levels: int = 12
    ocean_depth_m: float = 1000.0
    top_layer_m: float = 20.0
    lloyd_iterations: int = 10
    physics_dt: int = 1200
    atmosphere_dt: int = 240
    ocean_dt: int = 1200
    ocean_dynamics_integrator: str = "ssprk3"
    mixing_dt: int | None = None  # None = every ocean dynamics step
    mesoscale_dt: int | None = None  # None = every ocean dynamics step
    ale_dt: int | None = None  # None = every physics exchange
    radiation_dt: int = 3600
    solar_constant: float = 1361.0
    obliquity_degrees: float = 23.44
    eccentricity: float = 0.01671
    ice_temperature_K: float = 265.0
    mixing_length_m: float = 2.0
    mixing_substeps: int = 10
    gm_redi: bool = True
    gm_energy_scheme: str = "midpoint"
    gm_length_m: float = 10000.0
    gm_efficiency: float = 0.1
    gm_kinetic_fraction: float = 0.5
    redi_diffusivity_m2_s: float = 100.0
    shelf_redi: bool = False
    partial_cell_layers: bool = False
    redi_positive_transport: bool = True
    eddy_initial_energy_J_kg: float = 0.1
    eddy_substeps: int = 10
    eddy_dissipation: bool = True
    eddy_decay_days: float = 30.0
    backscatter: bool = True
    backscatter_viscosity_m2_s: float = 10000.0
    backscatter_days: float = 1.0
    backscatter_fraction: float = 0.25
    biharmonic_viscosity_m4_s: float = 1e13
    ale_remap: bool = True
    threads: int = 1
    compile: bool = False
    device: str = "cpu"
    dtype: str = "float64"
    diagnostic_hours: int = 6
    restart_days: int = 5
    overlap_dynamics: bool = False

    def validate(self):
        validate_precision(self.device, self.dtype)
        for key in (
            "days",
            "start_year",
            "cells",
            "atmosphere_levels",
            "ocean_levels",
            "physics_dt",
            "atmosphere_dt",
            "ocean_dt",
            "radiation_dt",
            "mixing_substeps",
            "eddy_substeps",
            "threads",
            "diagnostic_hours",
            "restart_days",
        ):
            if type(getattr(self, key)) is not int or getattr(self, key) <= 0:
                raise ValueError(f"{key} must be a positive integer")
        for key in (
            "gm_redi",
            "redi_positive_transport",
            "shelf_redi",
            "partial_cell_layers",
            "eddy_dissipation",
            "backscatter",
            "ale_remap",
            "compile",
            "overlap_dynamics",
        ):
            if type(getattr(self, key)) is not bool:
                raise ValueError(f"{key} must be a YAML boolean (true/false)")
        if self.overlap_dynamics and torch.device(self.device).type != "cpu":
            raise ValueError("overlap_dynamics requires CPU components")
        if self.start_year % 4 == 0 or self.days > 365:
            raise ValueError(
                "campaign requires a non-leap start year and at most 365 days"
            )
        if self.cells < 16 or self.atmosphere_levels < 4 or self.ocean_levels < 2:
            raise ValueError("insufficient cells or levels")
        if type(self.lloyd_iterations) is not int or self.lloyd_iterations < 0:
            raise ValueError("lloyd_iterations must be a nonnegative integer")
        for key in (
            "atmosphere_top_pa",
            "ocean_depth_m",
            "top_layer_m",
            "solar_constant",
            "mixing_length_m",
            "ice_temperature_K",
            "gm_length_m",
            "gm_efficiency",
            "gm_kinetic_fraction",
            "eddy_decay_days",
            "backscatter_days",
        ):
            value = getattr(self, key)
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"invalid {key}")
        for key in (
            "redi_diffusivity_m2_s",
            "eddy_initial_energy_J_kg",
            "backscatter_viscosity_m2_s",
            "backscatter_fraction",
            "biharmonic_viscosity_m4_s",
        ):
            value = getattr(self, key)
            if not math.isfinite(value) or value < 0:
                raise ValueError(f"{key} must be finite and nonnegative")
        if self.gm_energy_scheme not in GMRedi.energy_schemes:
            raise ValueError("unknown gm_energy_scheme")
        if self.ocean_dynamics_integrator not in ("ssprk3", "ssprk43"):
            raise ValueError("ocean_dynamics_integrator must be ssprk3 or ssprk43")
        if self.gm_kinetic_fraction > 1 or self.backscatter_fraction > 1:
            raise ValueError("eddy kinetic and backscatter fractions must be <= 1")
        if (
            self.backscatter
            and self.ocean_dt * self.backscatter_fraction > self.backscatter_days * DAY
        ):
            raise ValueError("backscatter requires ocean_dt*fraction <= timescale")
        if (
            not 0.0 < self.atmosphere_top_pa < 100000.0
            or self.ocean_depth_m <= self.top_layer_m
            or not 0 < self.ice_temperature_K <= 273.15
        ):
            raise ValueError("invalid depth or ice temperature")
        if not 0 <= self.obliquity_degrees < 90 or not 0 <= self.eccentricity < 1:
            raise ValueError("invalid orbital parameters")
        if any(
            interval % self.physics_dt
            for interval in (
                6 * 3600,
                DAY,
                self.radiation_dt,
                self.diagnostic_hours * 3600,
                self.restart_days * DAY,
            )
        ):
            raise ValueError("physics_dt must divide radiation and output cadences")
        if self.physics_dt % self.atmosphere_dt or self.physics_dt % self.ocean_dt:
            raise ValueError("component timesteps must divide physics_dt")
        mixing_dt = self.ocean_dt if self.mixing_dt is None else self.mixing_dt
        mesoscale_dt = self.ocean_dt if self.mesoscale_dt is None else self.mesoscale_dt
        # Fail closed for unsupported, non-commensurate closure schedules.
        # Every closure interval completes before the next coupled checkpoint.
        if self.physics_dt % mixing_dt or self.physics_dt % mesoscale_dt:
            raise ValueError("mixing_dt and mesoscale_dt must divide physics_dt")
        tuple(ocean_events(self.physics_dt, self.ocean_dt, mixing_dt, mesoscale_dt))
        ale_dt = self.physics_dt if self.ale_dt is None else self.ale_dt
        if (type(ale_dt) is not int or ale_dt < self.physics_dt
                or ale_dt % self.physics_dt):
            raise ValueError("ale_dt must be a positive multiple of physics_dt")
        return self


def snapshot_times(days=365):
    """0,6h,...,day14,day15,...,day365. Includes day14 exactly once."""
    end = days * DAY
    early = list(range(0, min(14 * DAY, end) + 1, 6 * 3600))
    return early + list(range(15 * DAY, end + 1, DAY))


def rest_thickness(config):
    """Shared initial/ALE layer distribution, with a nominal fixed top depth."""
    tail = torch.logspace(0, 1, config.ocean_levels - 1, dtype=torch.float64)
    return torch.cat(
        (
            torch.tensor([config.top_layer_m], dtype=torch.float64),
            tail / tail.sum() * (config.ocean_depth_m - config.top_layer_m),
        )
    )


def gm_redi_operator(config):
    factory = GMRedi
    options = {}
    if config.shelf_redi:
        from workflows.coupled.shelf_redi import ShelfGMRedi
        factory = ShelfGMRedi
        options["nominal"] = rest_thickness(config)
    return factory(
        **options,
        kappa_redi=config.redi_diffusivity_m2_s,
        positive_redi=config.redi_positive_transport,
        gm_energy_scheme=config.gm_energy_scheme,
        energy_closure=EddyMixingLength(
            config.gm_length_m, config.gm_efficiency, config.gm_kinetic_fraction
        ),
    )


def mesoscale_step(receiver, sea, dt, config, *, on_event=None):
    """Explicit GM/Redi then eddy decay; both exchange energy internally.

    Fixed-h tracer changes do not alter ice loading/KE or its cold content.
    Operators own parameters only: all prognostic energy is in the restart state.
    """
    model = receiver.H.model
    area = sea.coupled.ocean.ocean.mesh.cell_area
    report = dict.fromkeys(OCEAN_SUMS + OCEAN_PEAKS, 0.0)
    ocean = sea.coupled.ocean
    if config.gm_redi:
        eddy = gm_redi_operator(config)
        for substep in range(config.eddy_substeps):
            ocean, r = eddy.apply(model, ocean, dt / config.eddy_substeps)
            if on_event is not None:
                on_event("gmredi", ocean, {"substep": substep + 1, "substeps": config.eddy_substeps})
            report["gmredi_steps"] += 1
            report["gm_limiting_passes"] += r.gm_limiting_passes
            report["gm_limited_interfaces"] += r.gm_limited_interfaces
            report["redi_limited_fluxes"] += r.redi_limited_fluxes
            report["gm_exchange_J_m2"] += float(r.gm_reservoir_exchange / area.sum())
            report["redi_exchange_J_m2"] += float(
                r.redi_reservoir_exchange / area.sum()
            )
            report["gm_outflow_fraction"] = max(
                report["gm_outflow_fraction"], float(r.gm_outflow_fraction.max())
            )
            report["redi_step_number"] = max(
                report["redi_step_number"], float(r.redi_step_number.max())
            )
            for residual in (r.gm_local_energy_residual, r.redi_local_energy_residual):
                report["gmredi_local_residual_J_m2"] = max(
                    report["gmredi_local_residual_J_m2"],
                    float((residual / area[:, None]).abs().max()),
                )
    if config.eddy_dissipation:
        ocean, r = EddyDissipation(config.eddy_decay_days * DAY).apply(model, ocean, dt)
        if on_event is not None:
            on_event("eddy_dissipation", ocean, {})
        report["eddy_decay_steps"] = 1
        report["eddy_dissipation_J_m2"] = float(r.dissipated_energy / area.sum())
        report["eddy_decay_local_residual_J_m2"] = float(
            (r.local_energy_residual / area[:, None]).abs().max()
        )
    return replace(sea, coupled=replace(sea.coupled, ocean=ocean)), report


def ale_step(receiver, sea, config):
    """Post-exchange adaptive thermal ALE, retaining the current water depth.

    Returning to the initial stretched fractions does not restore water mass.
    Ice energy depends on total liquid depth, not its layer partition: the
    existing ocean column-energy remapper therefore also closes combined H.
    """
    report = dict.fromkeys(OCEAN_SUMS + OCEAN_PEAKS, 0.0)
    if not config.ale_remap:
        return sea, report
    model = receiver.H.model
    initial = sea.coupled.ocean
    s = initial.ocean
    fractions = rest_thickness(config).to(s.h)
    fractions = fractions / fractions.sum()
    if s.column.layer_reference is not None:
        reference = s.column.layer_reference
        fractions = reference / reference.sum(-1, keepdim=True)
    target = s.h.sum(-1, keepdim=True) * fractions
    if s.column.active_layers is None:
        ocean, r = model.remap_with_report(initial, target)
    else:
        domain = model.ocean.domain
        left, right = s.mesh.cellsOnEdge[domain.active_edge].T
        # ALE is optional motion of the computational interfaces. A nearly
        # homogeneous shelf can lack two independent thermal directions for
        # simultaneously conserving H and maintaining its entropy. Keep that
        # column and the other ends of its velocity faces at their current
        # geometry, then retry the remaining target from the original state.
        requested = domain.wet_cell & (target != s.h).any(-1)
        deferred = torch.zeros_like(requested)
        while True:
            try:
                ocean, r = model.remap_with_report(initial, torch.where(deferred[:, None], s.h, target))
                break
            except ThermalProjectionInfeasible as exc:
                touching = exc.columns[left] | exc.columns[right]
                hold = exc.columns.clone()
                hold[left[touching]] = True
                hold[right[touching]] = True
                additional = requested & hold & ~deferred
                if not additional.any():
                    raise
                deferred |= additional
        report["ale_deferred_columns"] = int(deferred.sum())
    report["ale_steps"] = 1
    report["ale_changed_columns"] = r.changed_columns
    # Keep remap error visible in the numerical ledger, never external forcing.
    report["ale_energy_change_J_m2"] = float(
        (model.total_energy(ocean) - model.total_energy(initial))
        / s.mesh.cell_area.sum()
    )
    report["ale_column_residual_J_m2"] = float(
        (r.column_energy_residual / s.mesh.cell_area).abs().max()
    )
    return replace(sea, coupled=replace(sea.coupled, ocean=ocean)), report


def accumulate_ocean(report, addition):
    for key in OCEAN_SUMS:
        report[key] += addition[key]
    for key in OCEAN_PEAKS:
        report[key] = max(report[key], addition[key])
    for key in ("runoff_kg_m2", "runoff_gravity_heat_J_m2"):
        if key in addition:
            report[key] = report.get(key, 0.) + addition[key]


def load_mesh(config, path):
    """Load saved unit-sphere generators, without additional relaxation.

    Read/hash the same bytes. File location is provenance only; moving an
    unchanged mesh file must not invalidate a restart.
    """
    if path is None:
        return config, None, dict(kind="generated", cells=config.cells)
    raw = Path(path).read_bytes()
    xyz = np.load(io.BytesIO(raw), allow_pickle=False)
    if (
        not isinstance(xyz, np.ndarray)
        or xyz.ndim != 2
        or xyz.shape[1] != 3
        or len(xyz) < 16
        or xyz.dtype.kind != "f"
        or not np.isfinite(xyz).all()
        or not np.allclose(np.linalg.norm(xyz, axis=1), 1.0, rtol=0, atol=1e-6)
        or len(np.unique(xyz, axis=0)) != len(xyz)
    ):
        raise ValueError(
            "--mesh requires a .npy array of N>=16 distinct finite floating-point unit 3-vectors"
        )
    effective = replace(config, cells=len(xyz), lloyd_iterations=0).validate()
    metadata = dict(
        kind="saved_generators",
        cells=len(xyz),
        path=str(Path(path).resolve()),
        sha256=hashlib.sha256(raw).hexdigest(),
        bytes=len(raw),
        additional_lloyd_iterations=0,
    )
    return effective, xyz, metadata


def island_height(mesh):
    radius = ((mesh.cell_xyz[:, 0] - 0.65) / 0.35).clamp_min(0.)
    return 1000. * radius.square()


def load_geography(path, *, mesh_sha256=None):
    if path is None:
        return None
    with np.load(path, allow_pickle=False) as archive:
        if "metadata" in archive:
            expected = json.loads(str(archive["metadata"])).get("mesh_sha256")
            if expected is not None and expected != mesh_sha256:
                raise ValueError("geography mesh_sha256 differs from the supplied mesh")
        return {k: archive[k] for k in ("land", "height_m", "depth_m")}


def build(
    config, source, radiation_data, *, smoke_no_radiation=False, generator_xyz=None, geography=None, island=False
):
    config.validate()
    device = resolve_device(config.device)
    dtype = validate_precision(device, config.dtype)
    if radiation_data is None and not smoke_no_radiation:
        raise ValueError(
            "--radiation-data /path/to/sea_esf is required for the seasonal campaign"
        )
    nml = dict(
        astronomy_nml=dict(
            obliq=config.obliquity_degrees,
            ecc=config.eccentricity,
            period=365 * DAY,
            year_ae=config.start_year,
            month_ae=9,
            day_ae=23,
            hour_ae=0,
            minute_ae=0,
            second_ae=0,
        ),
        radiation_driver_nml=dict(
            solar_constant=config.solar_constant,
            solar_dataset_entry=[1, 1, 1, 0, 0, 0],
            time_varying_solar_constant=False,
            zenith_spec="diurnally_varying",
        ),
    )
    if config.partial_cell_layers and not config.shelf_redi:
        raise ValueError("partial_cell_layers requires shelf_redi")
    if config.shelf_redi and (not config.gm_redi or (geography is None and not island)):
        raise ValueError('shelf_redi requires GM/Redi and coastal geography or an island')
    if island and geography is not None:
        raise ValueError("choose either island or supplied geography")
    height = None
    if island:
        # Generate terrain after any mesh relaxation.
        height = island_height
    elif geography is not None:
        mask, height, depth = (np.asarray(geography[k]) for k in ("land", "height_m", "depth_m"))
        if (mask.dtype != np.bool_ or any(v.shape != (config.cells,) for v in (mask, height, depth))
                or not mask.any() or mask.all() or not np.isfinite(height).all() or not np.isfinite(depth).all()
                or (height[~mask] != 0).any() or (height[mask] < 0).any() or (depth[~mask] <= 0).any()):
            raise ValueError("invalid coastal geometry: use mesh-ordered land mask, nonnegative terrain and positive wet depths")
    adapter, air = example(
        source,
        cells=config.cells,
        levels=config.atmosphere_levels,
        top_pressure_pa=config.atmosphere_top_pa,
        aquaplanet=True,
        radiation_data=radiation_data,
        radiation_step=config.radiation_dt,
        start=(config.start_year, 1, 1, 0, 0, 0),
        namelist_overrides=nml,
        surface_height_m=height,
        device=device,
        dtype=dtype,
        mesh_options=dict(
            lloyd_iters=config.lloyd_iterations,
            generator_xyz=generator_xyz,
            rotation_scheme="compatible",
            verbose=False,
        ),
    )
    torch.set_num_threads(config.threads)
    mesh = adapter.model.mesh
    domain = None
    column_options = dict(total_depth=config.ocean_depth_m)
    if height is not None:
        land = adapter.model.column.phi_s > 0 if island else torch.as_tensor(mask, device=device)
        depths = torch.full_like(mesh.cell_area, config.ocean_depth_m)
        if not island:
            depths[~land] = torch.as_tensor(depth[~mask], dtype=dtype, device=device)
        domain = OceanDomain(mesh, ~land, depth=depths).to(device=device, dtype=dtype)
        column_options = dict(total_depth=depths, rest_thickness=rest_thickness(config),
                              partial_cells=config.partial_cell_layers)
    with initialization_dtype(torch.float64):
        column = OceanColumn(
            config.ocean_levels, gravity=9.8, **column_options
        ).to(device=device, dtype=dtype)
    model = RoquetOcean(
        mesh,
        column,
        domain=domain,
        closure=RoquetMixing(config.mixing_length_m, substeps=config.mixing_substeps,
                            budget_rtol=mixing_budget_rtol(dtype)),
        backscatter=(
            EddyBackscatter(
                config.backscatter_viscosity_m2_s,
                config.backscatter_days * DAY,
                config.backscatter_fraction,
                config.biharmonic_viscosity_m4_s,
            )
            if config.backscatter
            else None
        ),
    )
    if config.shelf_redi:
        from workflows.coupled.shelf_transport import ShelfTemperatureOcean
        model.dynamics = ShelfTemperatureOcean(model.ocean)
    dz = rest_thickness(config).to(mesh.cell_area)
    h = dz[None].expand(mesh.nCells, -1).clone()
    if config.partial_cell_layers:
        h = column.layer_reference.clone()
    elif domain is not None:
        h = (h / dz.sum()) * depths[:, None]
    z = h.cumsum(-1) - h / 2
    sinlat = mesh.cell_xyz[:, 2]
    sst = (28 - 37 * sinlat.square()).clamp_min(-0.054 * 34.0)
    deep = torch.minimum(sst, torch.full_like(sst, 2.0))
    ct = deep[:, None] + (sst - deep)[:, None] * torch.exp(-(z - z[:, :1]) / 250.0)
    ocean = model.state(
        u_e=h.new_zeros(mesh.nEdges, config.ocean_levels),
        h=h,
        conservative_temperature=ct,
        absolute_salinity=torch.full_like(h, 34.0),
        tracer=torch.ones_like(h)[..., None],
        tke=1e-4 if domain is None else 1e-4 * domain.wet_cell[:, None],
        eddy_energy=config.eddy_initial_energy_J_kg,
    )
    # Smooth polar ice cap, with no prescribed velocity or ongoing restoring.
    cover = ((sinlat.abs() - 0.90) / 0.08).clamp(0, 0.7)
    if domain is not None:
        cover = domain.apply_cell(cover)
    area = cover[:, None] * cover.new_tensor([0.8, 0.2])
    ice = CategoryIce(
        area, area * area.new_tensor([0.3, 1.0]), area.new_zeros(mesh.nEdges, 1)
    )
    owner = ThermalSnowOceanIce(OceanIce(model))
    bare = SnowState(RidgingState(ocean, ice), torch.zeros_like(area))
    sea = owner.initialize(bare, config.ice_temperature_K)
    receiver = ThermalIceReceiver(owner)
    if domain is not None:
        receiver = CoastalReceiver(owner, adapter, land)
        air = replace(air, surface={**air.surface, "land": land})
        soil = receiver.land_surface.column.initialize(air.surface["t_surf"][land])
        sea = replace(sea, land=soil)
        digest = hashlib.sha256()
        for field in (land, depths):
            digest.update(field.detach().cpu().numpy().tobytes())
        adapter.geography_hash = digest.hexdigest()
    # Equilibrate the liquidus and latent/sensible storage without external heat.
    sea, _, _ = owner.phase_step(sea, 1, torch.zeros_like(area[:, :1]))
    configure_compilation(adapter, receiver, enabled=config.compile)
    return adapter, air, receiver, sea


def _advance_air(adapter, air, receiver, sea, dt, config, trace):
    evolved = air
    for elapsed in range(0, dt, config.atmosphere_dt):
        evolved = advance_dynamics(
            adapter, evolved, min(config.atmosphere_dt, dt - elapsed)
        )
        if trace is not None:
            trace.observe("atmosphere_dynamics", adapter, evolved, receiver, sea,
                          details={"substep_seconds": min(config.atmosphere_dt, dt - elapsed)})
    return evolved


def _advance_water(adapter, evolved, receiver, sea, dt, config, trace):
    water = sea
    external = 0.0
    ocean_report = dict.fromkeys(OCEAN_SUMS + OCEAN_PEAKS, 0.0)
    A = adapter.model.mesh.cell_area
    mix_dt = config.ocean_dt if config.mixing_dt is None else config.mixing_dt
    meso_dt = config.ocean_dt if config.mesoscale_dt is None else config.mesoscale_dt
    for event in ocean_events(dt, config.ocean_dt, mix_dt, meso_dt):
        step_dt = event.dynamics_dt
        ocean_callback = None
        if trace is not None and trace.ocean_stages:
            def ocean_callback(stage, intermediate, external_energy):
                trace.observe(stage, adapter, evolved, receiver, intermediate,
                              external=float(external_energy / A.sum()),
                              details={"substep_seconds": step_dt})
        # Preserve the historical within-step mixing placement when the
        # mixing cadence equals ocean_dt. A slower mixing cadence deliberately
        # uses operator splitting after the completed ocean/ice update.
        inline_mixing = mix_dt == config.ocean_dt
        water, r = receiver.snow.step(
            water, step_dt, stage_callback=ocean_callback,
            dynamics_integrator=config.ocean_dynamics_integrator,
            apply_mixing=inline_mixing,
        )
        snow_external = float(r.external_energy / A.sum())
        external += snow_external
        if event.mixing_dt is not None and not inline_mixing:
            water = receiver.snow.mixing_step(
                water, event.mixing_dt, stage_callback=ocean_callback
            )
        if trace is not None and not trace.ocean_stages:
            trace.observe("ocean_ice_snow_dynamics", adapter, evolved, receiver, water,
                          external=snow_external,
                          details={"substep_seconds": step_dt,
                                   "mixing_elapsed_seconds": event.mixing_dt})
        if event.mesoscale_dt is not None:
            callback = None
            if trace is not None and trace.substeps:
                def callback(stage, ocean, detail):
                    intermediate = replace(water, coupled=replace(water.coupled, ocean=ocean))
                    trace.observe(stage, adapter, evolved, receiver, intermediate,
                                  details=detail)
            water, r = mesoscale_step(
                receiver, water, event.mesoscale_dt, config, on_event=callback
            )
            if trace is not None and not trace.substeps:
                trace.observe("gmredi_and_eddy_dissipation", adapter, evolved, receiver,
                              water, details={"substep_seconds": event.mesoscale_dt})
            accumulate_ocean(ocean_report, r)
    return water, external, ocean_report


def advance(adapter, air, receiver, sea, dt, config, *, trace=None):
    """Join the independent dynamics advances before the atomic surface exchange."""
    if trace is not None:
        trace.begin(adapter, air, receiver, sea)
    # Energy traces observe a serial sequence of whole coupled states. Keep
    # autograd and autocast calls on their calling thread as well.
    overlap = (config.overlap_dynamics and trace is None and not torch.is_grad_enabled()
               and not torch.is_autocast_enabled("cpu"))
    if overlap:
        inference = torch.is_inference_mode_enabled()

        def advance_air_worker():
            with torch.inference_mode(inference), torch.no_grad():
                return _advance_air(adapter, air, receiver, sea, dt, config, None)

        with ThreadPoolExecutor(max_workers=1, thread_name_prefix="vfs-air") as executor:
            future = executor.submit(advance_air_worker)
            water, external, ocean_report = _advance_water(adapter, air, receiver, sea, dt, config, None)
            evolved = future.result()
    else:
        evolved = _advance_air(adapter, air, receiver, sea, dt, config, trace)
        water, external, ocean_report = _advance_water(adapter, evolved, receiver, sea, dt, config, trace)
    A = adapter.model.mesh.cell_area
    evolved, water, x, ar, sr = exchange_step(
        adapter, evolved, receiver, water, dt, omega=pressure_velocity(adapter, evolved)
    )
    exchange_external = float((A * x.toa).sum() / A.sum())
    exchange_upstream = float((A * ar.energy_changes["am2_energy_residual"]).sum() / A.sum())
    vapor_report = dict.fromkeys(VAPOR_SUMS + VAPOR_PEAKS, 0.0)
    if ar.vapor_correction is not None:
        correction = ar.vapor_correction
        cell = correction["indices"][:, 0]
        vapor_report.update(
            vapor_correction_events=cell.numel(),
            vapor_correction_kg_m2=float(
                (A[cell] * correction["vapor_added_kg_m2"]).sum() / A.sum()
            ),
            vapor_correction_proxy_residual_J_m2=float(
                correction["am2_energy_residual_J_m2"].abs().max()
            ),
        )
    if trace is not None:
        trace.observe("atmosphere_surface_exchange", adapter, evolved, receiver,
                      water, external=exchange_external, upstream=exchange_upstream,
                      details={**vapor_report,
                               "surface_residual_max_J": float(sr.energy_residual.abs().max()),
                               "air_residual_max_J": float(ar.energy_transfer["layer_residual"].abs().max()),
                               "radiation_residual_max_W": float(
                                   ar.energy_changes["radiation_flux_residual"].abs().max() / dt)})
    # Immediately consume latent excess at the melting constraint, rather than
    # leaving an artificially warm ice-covered top ocean layer until next step.
    water, _, _ = receiver.snow.phase_step(
        water, dt, torch.zeros_like(water.snow_mass[:, :1])
    )
    if trace is not None:
        trace.observe("post_exchange_ice_phase", adapter, evolved, receiver, water)
    ale_dt = config.physics_dt if config.ale_dt is None else config.ale_dt
    # Preserve historical behavior on a truncated final interval when ALE
    # remains at its default every-exchange cadence.
    if (event_due(evolved.time_seconds, ale_dt)
            or (config.ale_dt is None and dt != config.physics_dt)):
        water, r = ale_step(receiver, water, config)
        if trace is not None:
            trace.observe("ale_remap", adapter, evolved, receiver, water,
                          details={"ale_energy_change_J_m2": r["ale_energy_change_J_m2"],
                                   "ale_changed_columns": r["ale_changed_columns"],
                                   "ale_deferred_columns": r["ale_deferred_columns"]})
        accumulate_ocean(ocean_report, r)
    if sr.runoff_received is not None:
        ocean_report.update(runoff_kg_m2=float((sr.runoff_received * A).sum() / A.sum()),
                            runoff_gravity_heat_J_m2=float((sr.runoff_gravity_heat * A).sum() / A.sum()))
    return (
        evolved,
        water,
        dict(
            **ocean_report,
            **vapor_report,
            external=external + exchange_external,
            upstream=exchange_upstream,
            surface_residual=float(sr.energy_residual.abs().max()),
            air_residual=float(ar.energy_transfer["layer_residual"].abs().max()),
            radiation_residual=float(
                ar.energy_changes["radiation_flux_residual"].abs().max() / dt
            ),
            radiation_energy_residual=float(
                (A * ar.energy_changes["radiation_flux_residual"]).sum() / A.sum()
            ),
        ),
    )


def inventories(adapter, air, receiver, sea):
    A = adapter.model.mesh.cell_area
    s = sea.coupled.ocean.ocean
    air_diag = adapter.model.diagnostics(air.atmosphere)
    wet_area = A
    land_energy = land_water = 0.
    if sea.land is not None:
        l = receiver.land_surface
        wet_area = A * l.wet
        land_energy = (A[l.land] * l.column.energy(sea.land)).sum()
        land_water = (A[l.land] * sea.land.water()).sum()
    return dict(
        energy=float((land_energy + air_diag["energy"] + receiver.snow.total_energy(sea)) / A.sum()),
        water=float(
            (
                land_water + air_diag["species"].sum()
                + (
                    wet_area
                    * (s.column.rho0 * s.h.sum(-1) + receiver.H.mass(sea.coupled)[:, 0])
                ).sum()
            )
            / A.sum()
        ),
        salt=float((wet_area[:, None] * s.salt_h).sum() / A.sum()),
    )


def diagnostics(adapter, air, receiver, sea, ledger):
    current = inventories(adapter, air, receiver, sea)
    s = sea.coupled.ocean.ocean
    ice = sea.coupled.ice
    T = receiver.snow.temperature(sea)
    present = ice.area > 0
    A = adapter.model.mesh.cell_area
    wet = torch.ones_like(A, dtype=torch.bool) if sea.land is None else receiver.land_surface.wet
    land_diag = {}
    if sea.land is not None:
        l = receiver.land_surface
        land_diag = dict(land_water_kg_m2_global=float((A[l.land] * sea.land.water()).sum() / A.sum()),
                         runoff_kg_m2=ledger.get("runoff_kg_m2", 0.),
                         runoff_gravity_heat_J_m2=ledger.get("runoff_gravity_heat_J_m2", 0.))
    wet_layers = wet[:, None] & (s.h > 0)
    return dict(
        **land_diag,
        seconds=air.time_seconds,
        day=air.time_seconds / DAY,
        energy_minus_toa_J_m2=current["energy"]
        - ledger["initial"]["energy"]
        - ledger["external"],
        upstream_residual_J_m2=ledger["upstream"],
        radiation_energy_residual_J_m2=ledger["radiation_energy_residual"],
        upstream_minus_radiation_residual_J_m2=ledger["upstream"]
        - ledger["radiation_energy_residual"],
        numerical_energy_residual_J_m2=current["energy"]
        - ledger["initial"]["energy"]
        - ledger["external"]
        - ledger["upstream"],
        water_residual_kg_m2=current["water"] - ledger["initial"]["water"],
        salt_relative=(current["salt"] - ledger["initial"]["salt"])
        / ledger["initial"]["salt"],
        sst_min_K=float(s.temperature()[wet, 0].min() + 273.15),
        sst_max_K=float(s.temperature()[wet, 0].max() + 273.15),
        ice_min_K=float(T[present].min()) if present.any() else None,
        ice_max_K=float(T[present].max()) if present.any() else None,
        ice_mass_kg=float((A[:, None] * receiver.H.mass(sea.coupled)).sum()),
        snow_mass_kg=float((A[:, None] * sea.snow_mass).sum()),
        ice_sensible_J_m2=float((A[:, None] * sea.cold_content).sum() / A.sum()),
        max_air_speed_m_s=float(air.atmosphere.u_e.abs().max()),
        max_ocean_speed_m_s=float(s.u_e.abs().max()),
        max_ice_speed_m_s=float(ice.velocity.abs().max()),
        min_ocean_thickness_m=float(s.h[wet_layers].min()),
        min_eddy_energy_J_kg=float(sea.coupled.ocean.specific_eddy_energy()[wet_layers].min()),
        max_eddy_energy_J_kg=float(sea.coupled.ocean.specific_eddy_energy()[wet_layers].max()),
        **{k: ledger[k] for k in OCEAN_SUMS + OCEAN_PEAKS},
        **{k: ledger.get(k, 0.0) for k in VAPOR_SUMS + VAPOR_PEAKS},
        **{
            k: ledger[k]
            for k in ("surface_residual", "air_residual", "radiation_residual")
        },
    )


def energy_trace_measure(adapter, air, receiver, sea):
    """Diagnostic-only decomposition of the campaign's energy inventory."""
    inv = inventories(adapter, air, receiver, sea)
    A = adapter.model.mesh.cell_area.sum()
    atm = float(adapter.model.diagnostics(air.atmosphere)["energy"] / A)
    ocean = float(receiver.H.model.total_energy(sea.coupled.ocean) / A)
    sea_total = float(receiver.snow.total_energy(sea) / A)
    if sea.land is not None:
        land = receiver.land_surface
        inv["land_energy"] = float((land.mesh.cell_area[land.land] * land.column.energy(sea.land)).sum() / A)
    return dict(inv, atmosphere_energy=atm, ocean_energy=ocean,
                ice_snow_energy=sea_total - ocean)


def energy_trace_extrema(adapter, air, receiver, sea):
    """Global velocity extrema, indices and state-admissibility indicators."""
    ocean = sea.coupled.ocean.ocean
    def peak(x):
        f = x.abs().reshape(-1)
        k = int(f.argmax().item())
        return float(f[k]), k
    va, ia = peak(air.atmosphere.u_e)
    vo, io = peak(ocean.u_e)
    vi, ii = peak(sea.coupled.ice.velocity)
    wet = torch.ones_like(ocean.h[:, 0], dtype=torch.bool) if sea.land is None else receiver.land_surface.wet
    wet_layers = wet[:, None] & (ocean.h > 0)
    e = sea.coupled.ocean.specific_eddy_energy()[wet_layers]
    return dict(max_air_speed_m_s=va, max_air_edge=ia // air.atmosphere.u_e.shape[-1],
                max_air_level=ia % air.atmosphere.u_e.shape[-1],
                max_ocean_speed_m_s=vo, max_ocean_edge=io // ocean.u_e.shape[-1],
                max_ocean_level=io % ocean.u_e.shape[-1],
                max_ice_speed_m_s=vi, max_ice_flat_index=ii,
                min_ocean_thickness_m=float(ocean.h[wet_layers].min()),
                max_eddy_energy_J_kg=float(e.max()), min_eddy_energy_J_kg=float(e.min()),
                sst_min_K=float(ocean.temperature()[wet, 0].min() + 273.15),
                sst_max_K=float(ocean.temperature()[wet, 0].max() + 273.15))


def source_hashes():
    import vfs

    result = {}
    roots = {
        "workflows": ROOT / "workflows",
        "validation": ROOT / "validation",
        "vfs": Path(vfs.__file__).parent,
    }
    for label, root in roots.items():
        for p in sorted(root.rglob("*.py")):
            result[f"{label}/{p.relative_to(root)}"] = hashlib.sha256(
                p.read_bytes()
            ).hexdigest()
    return result


def fingerprint(config, adapter, hashes, smoke_no_radiation, mesh_sha256=None):
    payload = dict(
        schema=1,
        config=asdict(config),
        atmosphere=adapter.fingerprint,
        source=hashes,
        torch=str(torch.__version__),
        numpy=np.__version__,
        smoke_no_radiation=smoke_no_radiation,
        mesh_sha256=mesh_sha256,
        compilation=adapter.compilation_manifest,
        execution=runtime_manifest(adapter.model.mesh.cell_area.device, adapter.model.mesh.cell_area.dtype),
    )
    if hasattr(adapter, "geography_hash"):
        payload["geography_hash"] = adapter.geography_hash
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()


def pack(adapter, air, receiver, sea, identity, ledger):
    receiver.snow.check(sea)
    s = sea.coupled.ocean.ocean
    return dict(
        **({"land": {k: v.clone() for k, v in vars(sea.land).items()}} if sea.land is not None else {}),
        schema=1,
        fingerprint=identity,
        air=adapter.pack(air),
        ledger=copy_tree(ledger),
        ocean={
            k: getattr(s, k).clone()
            for k in ("u_e", "h", "temp_h", "salt_h", "tracer_h")
        },
        tke_h=sea.coupled.ocean.tke_h.clone(),
        eddy_h=sea.coupled.ocean.eddy_h.clone(),
        ice={
            k: getattr(sea.coupled.ice, k).clone()
            for k in ("area", "volume", "velocity")
        },
        snow_mass=sea.snow_mass.clone(),
        cold_content=sea.cold_content.clone(),
    )


def unpack(payload, adapter, receiver, template, identity):
    if payload.get("schema") != 1 or payload.get("fingerprint") != identity:
        raise ValueError("campaign restart configuration/source/runtime does not match")
    p = copy_tree(payload, device=adapter.model.mesh.cell_area.device)
    air = adapter.unpack(p["air"])
    physical = template.coupled.ocean.ocean.with_tensors(**p["ocean"])
    ocean = replace(
        template.coupled.ocean, ocean=physical, tke_h=p["tke_h"], eddy_h=p["eddy_h"]
    )
    sea = ThermalIceState(
        RidgingState(ocean, CategoryIce(**p["ice"])), p["snow_mass"], p["cold_content"]
    )
    if template.land is not None:
        land = LandState(**p["land"])
        receiver.land_surface.column.check(land)
        if land.skin.shape != template.land.skin.shape:
            raise ValueError("restart land geometry mismatch")
        sea = replace(sea, land=land)
    receiver.snow.check(sea)
    return air, sea, p["ledger"]


def atomic_json(path, value):
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temp.replace(path)


def checkpoint(path, payload):
    temp = path.with_suffix(path.suffix + ".tmp")
    # Portable host storage; unpack explicitly restores the selected device.
    torch.save(copy_tree(payload, device="cpu"), temp)
    temp.replace(path)


def snapshot(path, adapter, air, receiver, sea):
    """Native mesh NetCDF; instantaneous snapshots, never time averages."""
    import xarray as xr

    mesh = adapter.model.mesh
    f = adapter.model.physical_fields(air.atmosphere)
    s = sea.coupled.ocean.ocean

    def array(x):
        return x.detach().cpu().numpy()

    data = {
        "cell_area": (("cell",), array(mesh.cell_area), {"units": "m2"}),
        "latitude": (
            ("cell",),
            array(torch.asin(mesh.cell_xyz[:, 2]) * 180 / math.pi),
            {"units": "degrees_north"},
        ),
        "longitude": (
            ("cell",),
            array(
                torch.atan2(mesh.cell_xyz[:, 1], mesh.cell_xyz[:, 0]) * 180 / math.pi
            ),
            {"units": "degrees_east"},
        ),
        "cells_on_edge": (
            ("edge", "pair"),
            array(mesh.cellsOnEdge),
            {"start_index": 0},
        ),
        "cell_xyz": (("cell", "xyz"), array(mesh.cell_xyz)),
        "edge_xyz": (("edge", "xyz"), array(mesh.edge_xyz)),
        "air_temperature": (
            ("cell", "air_interface"),
            array(f["temperature"]),
            {"units": "K"},
        ),
        "air_pressure": (
            ("cell", "air_interface"),
            array(adapter.model.column.interface_pressure(air.atmosphere.ps)),
            {"units": "Pa"},
        ),
        "surface_pressure": (
            ("cell",),
            array(air.atmosphere.ps.reshape(-1)),
            {"units": "Pa"},
        ),
        "air_normal_wind": (
            ("edge", "air_level"),
            array(air.atmosphere.u_e),
            {"units": "m s-1"},
        ),
        "cloud_fraction": (("cell", "air_interface"), array(air.cloud_fraction)),
        "ocean_temperature": (
            ("cell", "ocean_level"),
            array(s.temperature()),
            {"units": "degC", "long_name": "Conservative Temperature"},
        ),
        "ocean_salinity": (
            ("cell", "ocean_level"),
            array(s.salinity()),
            {"units": "g kg-1"},
        ),
        "ocean_thickness": (("cell", "ocean_level"), array(s.h), {"units": "m"}),
        "ocean_normal_wind": (
            ("edge", "ocean_level"),
            array(s.u_e),
            {"units": "m s-1"},
        ),
        "ocean_tke": (
            ("cell", "ocean_level"),
            array(sea.coupled.ocean.specific_tke()),
            {"units": "m2 s-2"},
        ),
        "ocean_eddy_energy": (
            ("cell", "ocean_level"),
            array(sea.coupled.ocean.specific_eddy_energy()),
            {"units": "J kg-1", "long_name": "mesoscale eddy energy reservoir"},
        ),
        "ice_area": (("cell", "category"), array(sea.coupled.ice.area)),
        "ice_volume": (
            ("cell", "category"),
            array(sea.coupled.ice.volume),
            {"units": "m"},
        ),
        "ice_temperature": (
            ("cell", "category"),
            array(receiver.snow.temperature(sea)),
            {
                "units": "K",
                "comment": "273.15 on absent categories; mask using ice_area",
            },
        ),
        "ice_cold_content": (
            ("cell", "category"),
            array(sea.cold_content),
            {"units": "J m-2"},
        ),
        "ice_normal_wind": (
            ("edge", "ice_level"),
            array(sea.coupled.ice.velocity),
            {"units": "m s-1"},
        ),
        "snow_mass": (("cell", "category"), array(sea.snow_mass), {"units": "kg m-2"}),
    }
    if sea.land is not None:
        l = receiver.land_surface
        if s.column.layer_reference is not None:
            active = s.column.active_layers & l.wet[:, None]
            data["ocean_active_layer_count"] = (("cell",), array(active.sum(-1)).astype("int32"))
        data["land_mask"] = (("cell",), array(l.land).astype("int8"))
        data["surface_height_m"] = (("cell",), array(adapter.model.column.phi_s / s.column.gravity))
        data["ocean_depth_m"] = (("cell",), array(receiver.H.model.ocean.domain.depth[:, 0] * l.wet))
        data["land_cell_index"] = (("land_cell",), array(l.land.nonzero()[:, 0]))
        for name, field in vars(sea.land).items():
            dims = ("land_cell", "soil_layer") if field.ndim == 2 else ("land_cell",)
            data["land_" + name] = (dims, array(field))
    for k in ("vapor", "liquid", "ice"):
        data["air_" + k] = (
            ("cell", "air_interface"),
            array(f[k]),
            {"units": "kg kg-1"},
        )
    start = adapter.configuration["start"]
    ds = xr.Dataset(
        data,
        attrs=dict(
            description="VFS coupled instantaneous state", calendar="julian"
        ),
    )
    ds = ds.expand_dims(time=[air.time_seconds])
    ds.time.attrs.update(
        units=f"seconds since {start[0]:04d}-01-01 00:00:00", calendar="julian"
    )
    temp = path.with_suffix(".nc.tmp")
    ds.to_netcdf(
        temp,
        engine="netcdf4",
        encoding={k: {"zlib": True, "complevel": 1} for k in ds.data_vars},
    )
    temp.replace(path)


def run_logger(output, verbose):
    """One run owns its handlers; no global logging configuration is changed."""
    logger = logging.Logger("vfs.coupled.campaign", level=logging.INFO)
    logger.propagate = False
    formatter = logging.Formatter(
        "%(asctime)s %(levelname)s %(message)s", "%Y-%m-%dT%H:%M:%SZ"
    )
    formatter.converter = time.gmtime
    file_handler = logging.FileHandler(output / "run.log", encoding="utf-8")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    if verbose:
        console = logging.StreamHandler(sys.stdout)
        console.setFormatter(logging.Formatter("%(message)s"))
        logger.addHandler(console)
    return logger


@torch.no_grad()
def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path(__file__).with_name("coupled_aquaplanet_1yr.yaml"),
    )
    parser.add_argument(
        "--source",
        type=Path,
        required=True,
        help="pinned PyAM21 checkout containing the physics namelist",
    )
    coast = parser.add_mutually_exclusive_group()
    coast.add_argument("--island", action="store_true", help="idealized island and flat-bottom active ocean")
    coast.add_argument("--geography", type=Path, help="mesh-ordered .npz arrays land, height_m and depth_m")
    parser.add_argument("--radiation-data", type=Path)
    parser.add_argument(
        "--mesh",
        type=Path,
        help="Saved (N,3) .npy unit-sphere generators; infers cells and disables extra Lloyd relaxation",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--resume", type=Path)
    parser.add_argument(
        "--resume-compatible-manifest", type=Path,
        help="Opt-in audited source-compatible restart bridge; requires prior run manifest.json and --resume",
    )
    parser.add_argument("--device", help="cpu, cuda, cuda:N, or mps (overrides YAML)")
    parser.add_argument("--dtype", choices=("float32", "float64"), help="Floating-point precision; MPS requires float32 (overrides YAML)")
    parser.add_argument("--overlap-dynamics", action=argparse.BooleanOptionalAction, default=None,
                        help="Run CPU atmosphere and ocean dynamics concurrently before surface exchange")
    parser.add_argument("--threads", type=int, help="CPU thread count, including host work for GPU (overrides YAML)")
    parser.add_argument(
        "--compile", action=argparse.BooleanOptionalAction, default=None,
        help="Compile shared mesh and atmosphere/ocean thermal kernels with Inductor; first calls include compilation (overrides YAML)",
    )
    parser.add_argument(
        "--preflight",
        action="store_true",
        help="build and execute one coupled interval, including restart replay",
    )
    parser.add_argument(
        "--stop-after-days",
        type=float,
        help="stop at this elapsed model day; retain full campaign restart identity",
    )
    parser.add_argument(
        "--smoke-no-radiation",
        action="store_true",
        help="explicit radiation-OFF plumbing test, limited to one day",
    )
    parser.add_argument("--energy-trace", action="store_true",
                        help="Opt-in per-operator energy ledger to energy_stages.jsonl (computationally expensive)")
    parser.add_argument("--energy-trace-substeps", action="store_true",
                        help="Trace each GM–Redi substep; requires --energy-trace")
    parser.add_argument("--energy-trace-ocean-stages", action="store_true",
                        help="Trace snow/phase/drag/SSPRK3/TKE mixing separately; requires --energy-trace")
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Also print initialization/progress to the terminal; run.log is always written",
    )
    args = parser.parse_args(argv)
    if args.energy_trace_substeps and not args.energy_trace:
        parser.error("--energy-trace-substeps requires --energy-trace")
    if args.energy_trace_ocean_stages and not args.energy_trace:
        parser.error("--energy-trace-ocean-stages requires --energy-trace")
    if args.resume_compatible_manifest and not args.resume:
        parser.error("--resume-compatible-manifest requires --resume")
    if args.preflight and args.energy_trace:
        parser.error("--energy-trace does not support --preflight")
    overrides = {name: getattr(args, name) for name in ("device", "dtype", "threads", "compile", "overlap_dynamics")
                 if getattr(args, name) is not None}
    config = CampaignConfig(**(yaml.safe_load(args.config.read_text()) | overrides)).validate()
    end = (
        config.days * DAY
        if args.stop_after_days is None
        else args.stop_after_days * DAY
    )
    if (
        not math.isfinite(end)
        or not 0 < end <= config.days * DAY
        or not float(end).is_integer()
    ):
        parser.error("stop time must be positive integer seconds within the campaign")
    end = int(end)
    if args.smoke_no_radiation and (
        args.radiation_data is not None or (not args.preflight and end > DAY)
    ):
        parser.error(
            "radiation-OFF smoke requires --stop-after-days <=1 and no --radiation-data"
        )
    output = args.output
    if output.exists() and any(output.iterdir()):
        parser.error(
            "output directory must be new/empty; resume into a new segment directory"
        )
    output.mkdir(parents=True, exist_ok=True)
    atomic_json(output / "configuration.json", asdict(config))
    air = sea = adapter = receiver = identity = ledger = None
    energy_trace = None
    start = time.perf_counter()
    logger = run_logger(output, args.verbose)
    execution = ExitStack()
    try:
        device = resolve_device(config.device)
        config = replace(config, device=str(device))
        execution.enter_context(execution_context(device))
        runtime = runtime_manifest(device, config.dtype)
        logger.info("Execution: %s", json.dumps(runtime))
        logger.info(
            "Starting %s; target %.6f model days; radiation %s",
            "preflight" if args.preflight else "campaign",
            end / DAY,
            "OFF (explicit smoke test)" if args.smoke_no_radiation else "SeaEsfRad",
        )
        config, generators, mesh_info = load_mesh(config, args.mesh)
        atomic_json(output / "configuration.json", asdict(config))
        logger.info(
            "Initializing %d cells; mesh=%s; additional Lloyd iterations=%d",
            config.cells,
            mesh_info["kind"],
            config.lloyd_iterations,
        )
        if args.mesh:
            logger.info("Mesh %s; SHA256 %s", args.mesh, mesh_info["sha256"])
        if config.compile:
            logger.info("Enabling regional torch.compile with Inductor; first kernel calls can take several minutes")
        geography = load_geography(args.geography, mesh_sha256=mesh_info.get("sha256"))
        adapter, air, receiver, sea = build(
            config,
            args.source,
            args.radiation_data,
            smoke_no_radiation=args.smoke_no_radiation,
            generator_xyz=generators, island=args.island, geography=geography,
        )
        if config.shelf_redi:
            if config.partial_cell_layers:
                ocean_state = sea.coupled.ocean.ocean
                counts = ocean_state.column.active_layers.sum(-1)[receiver.land_surface.wet]
                logger.info("Prognostic wet ocean layers per column: %d to %d", counts.min(), counts.max())
            logger.info("Resolved and GM transport use physical wet openings; Redi uses partial-cell geometry")
        hashes = source_hashes()
        identity = fingerprint(
            config, adapter, hashes, args.smoke_no_radiation, mesh_info.get("sha256")
        )
        logger.info("Initialization complete; restart fingerprint %s", identity)
        logger.info("Compilation: %s", json.dumps(adapter.compilation_manifest))
        if config.compile:
            logger.info("Initial diagnostics and first intervals include compilation; later new tensor shapes may compile once too")
        logger.info(
            "Ocean processes: GM-Redi=%s (Redi %.8g m2/s; %d substeps); eddy decay=%s (%g days); backscatter=%s (nu2=%g m2/s, nu4=%g m4/s); ALE=%s (every %d s)",
            config.gm_redi,
            config.redi_diffusivity_m2_s,
            config.eddy_substeps,
            config.eddy_dissipation,
            config.eddy_decay_days,
            config.backscatter,
            config.backscatter_viscosity_m2_s,
            config.biharmonic_viscosity_m4_s,
            config.ale_remap,
            config.ale_dt or config.physics_dt,
        )
        logger.info(
            "Ocean–ice explicit dynamics integrator=%s",
            config.ocean_dynamics_integrator,
        )
        logger.info(
            "GM energy scheme=%s; Redi conservative positive-inventory transport=%s; initial eddy energy=%g J/kg",
            config.gm_energy_scheme,
            config.redi_positive_transport,
            config.eddy_initial_energy_J_kg,
        )
        atomic_json(
            output / "manifest.json",
            dict(
                fingerprint=identity,
                source_sha256=hashes,
                radiation=adapter.radiation_manifest,
                mesh=mesh_info,
                geography_sha256=getattr(adapter, "geography_hash", None),
                configuration=asdict(config),
                radiation_enabled=not args.smoke_no_radiation,
                compilation=adapter.compilation_manifest,
                execution=runtime,
                torch=str(torch.__version__),
            ),
        )
        ledger = dict(
            **dict.fromkeys(OCEAN_SUMS + OCEAN_PEAKS, 0.0),
            **dict.fromkeys(VAPOR_SUMS + VAPOR_PEAKS, 0.0),
            initial=inventories(adapter, air, receiver, sea),
            external=0.0,
            upstream=0.0,
            radiation_energy_residual=0.0,
            surface_residual=0.0,
            air_residual=0.0,
            radiation_residual=0.0,
            steps=0,
        )
        if args.resume:
            payload = torch.load(args.resume, map_location="cpu", weights_only=True)
            expected_restart_identity = identity
            if args.resume_compatible_manifest:
                prior = json.loads(args.resume_compatible_manifest.read_text())
                if prior.get("configuration") != asdict(config):
                    raise ValueError("compatible restart: prior campaign configuration differs")
                if prior.get("mesh", {}).get("sha256") != mesh_info.get("sha256"):
                    raise ValueError("compatible restart: mesh SHA256 differs")
                if canonical_manifest_json(prior.get("radiation")) != canonical_manifest_json(adapter.radiation_manifest):
                    raise ValueError("compatible restart: radiation tables/metadata differ")
                if prior.get("compilation") != adapter.compilation_manifest:
                    raise ValueError("compatible restart: compilation policy differs")
                if prior.get("execution") != runtime:
                    raise ValueError("compatible restart: execution runtime differs")
                if prior.get("radiation_enabled") != (not args.smoke_no_radiation):
                    raise ValueError("compatible restart: radiation mode differs")
                source_delta = verify_compatible_source_delta(
                    prior.get("source_sha256"), hashes
                )
                expected_restart_identity = fingerprint(
                    config, adapter, prior["source_sha256"],
                    args.smoke_no_radiation, mesh_info.get("sha256")
                )
                if (prior.get("fingerprint") != expected_restart_identity
                        or payload.get("fingerprint") != expected_restart_identity):
                    raise ValueError(
                        "compatible restart: checkpoint and original manifest identity "
                        "do not match the reconstructed fingerprint"
                    )
                audit = dict(
                    prior_manifest=str(args.resume_compatible_manifest),
                    prior_checkpoint=str(args.resume),
                    prior_fingerprint=expected_restart_identity,
                    new_fingerprint=identity,
                    **source_delta,
                )
                atomic_json(output / "compatible_resume_audit.json", audit)
                if "modified_instrumentation_files" in source_delta:
                    atomic_json(output / "instrumentation_resume_audit.json", audit)
                logger.info(
                    "Audited compatible restart: prior %s -> current %s; allowed changes %s",
                    expected_restart_identity, identity,
                    json.dumps(source_delta),
                )
            air, sea, ledger = unpack(
                payload, adapter, receiver, sea, expected_restart_identity
            )
            for key in VAPOR_SUMS + VAPOR_PEAKS + ("ale_deferred_columns",):
                ledger.setdefault(key, 0.0)
            logger.info("Resumed %s at day %.6f", args.resume, air.time_seconds / DAY)
        if air.time_seconds >= end:
            raise ValueError("restart is already at or beyond requested end")
        if args.energy_trace:
            energy_trace = EnergyTrace(output / "energy_stages.jsonl",
                                       energy_trace_measure, energy_trace_extrema,
                                       substeps=args.energy_trace_substeps,
                                       ocean_stages=args.energy_trace_ocean_stages)
            logger.info("Opt-in energy trace enabled, extra diagnostics may be slow")
        if args.preflight:
            logger.info(
                "Advancing one complete coupled interval and replaying its restart"
            )
            payload = pack(adapter, air, receiver, sea, identity, ledger)
            checkpoint(output / "preflight_restart.pt", payload)
            with torch.no_grad():
                a, s, r = advance(
                    adapter, air, receiver, sea, config.physics_dt, config
                )
                ra, rs, _ = unpack(
                    torch.load(output / "preflight_restart.pt", map_location="cpu", weights_only=True),
                    adapter, receiver, sea, identity,
                )
                b, t, u = advance(adapter, ra, receiver, rs, config.physics_dt, config)
                synchronize(device)
            left = pack(adapter, a, receiver, s, identity, ledger)
            right = pack(adapter, b, receiver, t, identity, ledger)

            def equal(x, y):
                if isinstance(x, torch.Tensor):
                    return torch.equal(x, y)
                if isinstance(x, dict):
                    return x.keys() == y.keys() and all(equal(x[k], y[k]) for k in x)
                if isinstance(x, (tuple, list)):
                    return len(x) == len(y) and all(equal(i, j) for i, j in zip(x, y))
                return x == y

            if not equal(left, right) or r != u:
                raise AssertionError("restart replay differs")
            atomic_json(
                output / "status.json",
                dict(
                    status="preflight_passed",
                    radiation_enabled=not args.smoke_no_radiation,
                    restart_replay="bitwise",
                    wall_seconds=time.perf_counter() - start,
                    interval=r,
                ),
            )
            logger.info("Preflight passed; restart replay bitwise; %s", json.dumps(r))
            print((output / "status.json").read_text(), flush=True)
            return 0
        scheduled = set(snapshot_times(config.days))
        snapshot_dir = output / "snapshots"
        snapshot_dir.mkdir()
        restarts = output / "restarts"
        restarts.mkdir()
        if air.time_seconds in scheduled:
            snapshot(
                snapshot_dir / f"state_{air.time_seconds:08d}.nc",
                adapter,
                air,
                receiver,
                sea,
            )
            logger.info("Wrote initial snapshot at day %.6f", air.time_seconds / DAY)
        checkpoint(
            restarts / f"restart_{air.time_seconds:08d}.pt",
            pack(adapter, air, receiver, sea, identity, ledger),
        )
        logger.info("Wrote initial restart at day %.6f", air.time_seconds / DAY)
        last_log = time.perf_counter()
        with (output / "diagnostics.csv").open("w") as f, torch.no_grad():
            row = diagnostics(adapter, air, receiver, sea, ledger)
            writer = csv.DictWriter(f, fieldnames=list(row), lineterminator="\n")
            writer.writeheader()
            writer.writerow(row)
            f.flush()
            while air.time_seconds < end:
                dt = min(config.physics_dt, end - air.time_seconds)
                candidate, water, report = advance(
                    adapter, air, receiver, sea, dt, config, trace=energy_trace
                )
                # An asynchronous CUDA failure must be observed before commit.
                synchronize(device)
                air, sea = candidate, water
                ledger["steps"] += 1
                ledger["external"] += report["external"]
                ledger["upstream"] += report["upstream"]
                ledger["radiation_energy_residual"] += report["radiation_energy_residual"]
                accumulate_ocean(ledger, report)
                for k in VAPOR_SUMS:
                    ledger[k] += report[k]
                for k in VAPOR_PEAKS:
                    ledger[k] = max(ledger[k], report[k])
                for k in ("surface_residual", "air_residual", "radiation_residual"):
                    ledger[k] = max(ledger[k], report[k])
                seconds = air.time_seconds
                if energy_trace is not None:
                    energy_trace.commit(ledger, seconds=seconds, report=report)
                if seconds in scheduled:
                    snapshot(
                        snapshot_dir / f"state_{seconds:08d}.nc",
                        adapter,
                        air,
                        receiver,
                        sea,
                    )
                    logger.info("Wrote snapshot at day %.6f", seconds / DAY)
                if seconds % (config.diagnostic_hours * 3600) == 0 or seconds == end:
                    row = diagnostics(adapter, air, receiver, sea, ledger)
                    writer.writerow(row)
                    f.flush()
                    atomic_json(
                        output / "status.json",
                        dict(
                            status="running",
                            wall_seconds=time.perf_counter() - start,
                            latest=row,
                        ),
                    )
                    logger.info(
                        "Diagnostics at day %.6f: numerical energy %.8g J/m2; water %.8g kg/m2; salt relative %.8g",
                        seconds / DAY,
                        row["numerical_energy_residual_J_m2"],
                        row["water_residual_kg_m2"],
                        row["salt_relative"],
                    )
                if seconds % (config.restart_days * DAY) == 0 or seconds == end:
                    checkpoint(
                        restarts / f"restart_{seconds:08d}.pt",
                        pack(adapter, air, receiver, sea, identity, ledger),
                    )
                    logger.info("Wrote restart at day %.6f", seconds / DAY)
                if time.perf_counter() - last_log >= 30 or seconds == end:
                    logger.info(
                        "day %.6f/%g; %d total steps; %.1f wall seconds",
                        seconds / DAY,
                        end / DAY,
                        ledger["steps"],
                        time.perf_counter() - start,
                    )
                    last_log = time.perf_counter()
        atomic_json(
            output / "status.json",
            dict(
                status="completed" if end == config.days * DAY else "segment_completed",
                wall_seconds=time.perf_counter() - start,
                latest=diagnostics(adapter, air, receiver, sea, ledger),
            ),
        )
        logger.info(
            "Run finished at day %.6f; %.1f wall seconds",
            air.time_seconds / DAY,
            time.perf_counter() - start,
        )
        print((output / "status.json").read_text(), flush=True)
        return 0
    except (Exception, KeyboardInterrupt) as exc:
        status = dict(
            status=(
                "failed" if not isinstance(exc, KeyboardInterrupt) else "interrupted"
            ),
            error=f"{type(exc).__name__}: {exc}",
            last_committed_seconds=None if air is None else air.time_seconds,
            wall_seconds=time.perf_counter() - start,
        )
        logger.exception(
            "Run %s at last committed second %s",
            status["status"],
            status["last_committed_seconds"],
        )
        if identity is not None and ledger is not None:
            try:
                checkpoint(
                    output / "last_good_restart.pt",
                    pack(adapter, air, receiver, sea, identity, ledger),
                )
                logger.info("Saved last_good_restart.pt at second %d", air.time_seconds)
            except Exception as checkpoint_error:
                status["checkpoint_error"] = str(checkpoint_error)
                logger.exception("Could not save last-good restart")
        atomic_json(output / "status.json", status)
        print(json.dumps(status, indent=2), file=sys.stderr, flush=True)
        return 1
    finally:
        if energy_trace is not None:
            energy_trace.close()
        execution.close()
        for handler in logger.handlers[:]:
            handler.close()
            logger.removeHandler(handler)


if __name__ == "__main__":
    raise SystemExit(main())
