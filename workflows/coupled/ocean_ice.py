"""Reciprocal receiver for the research SnowOceanIce Hamiltonian.

Same mesh, all wet. No independent evaporation, wind drag, or radiation law.
Water/momentum are applied first; one local first-law thermal solve realizes
exactly the shared surface energy. Upstream atmospheric imbalance is untouched.
"""

from dataclasses import dataclass, replace
from pathlib import Path
import sys
import torch

# Research dependencies are explicitly loaded, never installed into vfs.
ROOT = Path(__file__).resolve().parents[2]
for directory in ("sea_ice", "temperature_coordinate", "ocean_column"):
    path = str(ROOT / "validation" / directory)
    if path not in sys.path:
        sys.path.insert(0, path)
from roquet import RoquetWorkspace
from ocean_ice import safe
from vfs.ocean.energy import ConstantDensityEnergy
from workflows.exchange import SurfaceTiles
from workflows.momentum import KineticWindMap


@dataclass(frozen=True)
class SurfaceReport:
    energy_residual: torch.Tensor  # J/m2 by cell
    water_residual: torch.Tensor  # kg/m2 by cell
    momentum_residual: torch.Tensor  # native edge generalized momentum
    thermal_increment: torch.Tensor  # J/m2 first-law thermal solve
    ocean_impulse: torch.Tensor
    ice_impulse: torch.Tensor


class OceanIceReceiver:
    def __init__(self, snow):
        self.snow = snow
        self.H = snow.hamiltonian
        self.wind = KineticWindMap(self.H.model.ocean.mesh)

    def check_geometry(self, adapter, atmosphere, state):
        if atmosphere.surface["land"].any():
            raise ValueError("shared ocean/ice exchange supports all-wet surfaces only")
        ocean = state.coupled.ocean.ocean
        if ocean.mesh is not adapter.model.mesh:
            raise ValueError("shared exchange currently requires the identical mesh object")
        if ocean.column.gravity != adapter.model.dynamics.kernel.gravity:
            raise ValueError("component gravity must match")

    def tiles(self, state, ice_temperature):
        """Explicit CT+273.15 SST approximation; caller supplies ice skin K.

        Porous snow controls albedo. Ice skin temperature/conduction must be
        provided by a surface model; this receiver does not invent one.
        """
        self.snow.check(state)
        s, ice = state.coupled.ocean.ocean, state.coupled.ice
        if self.H.model.ocean.domain is not None:
            raise ValueError("first shared exchange requires an all-wet mesh")
        if ice_temperature.shape != ice.area.shape:
            raise ValueError("ice skin temperature must match categories")
        hi, hs = self.snow.depths(state)
        p = self.snow.parameters
        alpha = p.ice_albedo + (p.albedo - p.ice_albedo) * hs / (hs + p.patch_depth)
        opened = 1 - ice.area.sum(-1, keepdim=True)
        temp = torch.cat((s.temperature()[:, :1] + 273.15, ice_temperature), -1)
        vwater = self.wind.vector(s.u_e)[:, 0]
        vice = self.wind.vector(ice.velocity)[:, 0]
        tiles = SurfaceTiles(
            torch.cat((opened, ice.area), -1),
            temp,
            torch.stack((vwater, vice, vice), 1),
            torch.cat((torch.full_like(opened, p.water_albedo), alpha), -1),
        )
        tiles.validate(s.mesh)
        return tiles

    def prepare(self, initial, exchange, tiles):
        """Mass and reciprocal impulses; finalize common finite stress work.

        No state is mutated. Rain passes through ice; snow is intercepted by
        old cover. Deposition on ice joins snow; sublimation consumes snow first.
        Frozen mass is part of category volume, not an additional mass store.
        """
        self.snow.check(initial)
        H = self.H
        old = initial.coupled
        s, ice = old.ocean.ocean, old.ice
        mesh = s.mesh
        tiles.validate(mesh)
        exchange.validate(mesh)
        if (
            (tiles.fraction == 0)
            & (
                (exchange.vapor != 0)
                | (exchange.sensible != 0)
                | (exchange.stress != 0).any(-1)
            )
        ).any():
            raise ValueError("exchange acts on an absent tile")
        if not torch.equal(tiles.fraction[:, 1:], ice.area):
            raise ValueError("tile fractions must match recipient at interval start")
        if self.H.model.ocean.domain is not None:
            raise ValueError("first shared exchange requires all-wet mesh")
        A = mesh.cell_area[:, None]
        rho = H.slab.parameters.density
        snowfall = exchange.snow[:, None] * ice.area
        evap = exchange.vapor[:, 1:]
        arriving = snowfall + (-evap).clamp_min(0)
        departing = evap.clamp_min(0)
        store = rho * ice.volume
        remaining = store + arriving - departing
        if (remaining < 0).any():
            raise ValueError("exchange exhausts frozen category; reduce dt")
        snow_available = initial.snow_mass + arriving
        snow1 = snow_available - torch.minimum(departing, snow_available)
        underlying = store - initial.snow_mass
        footprint = torch.where(
            underlying > 0,
            torch.minimum(remaining / safe(underlying), torch.ones_like(store)),
            (remaining > 0).to(store.dtype),
        )
        ice1 = replace(ice, volume=remaining / rho, area=ice.area * footprint)
        dm = (
            exchange.rain + exchange.snow * tiles.fraction[:, 0] - exchange.vapor[:, 0]
        )[:, None]
        h = torch.cat((s.h[:, :1] + dm / s.column.rho0, s.h[:, 1:]), -1)
        if (h[:, :1] < self.snow.phase.min_liquid).any():
            raise ValueError("exchange exhausts liquid top layer; reduce dt")
        salinity = s.salt_h / h
        if ((salinity <= 0) | (salinity >= 42)).any():
            raise ValueError("exchange leaves ocean salinity domain")
        temp_h = torch.cat((h[:, :1] * s.temperature()[:, :1], s.temp_h[:, 1:]), -1)
        trial = s.with_tensors(s.u_e, h, temp_h, s.salt_h, s.tracer_h)
        candidate = replace(
            initial,
            coupled=replace(old, ice=ice1, ocean=replace(old.ocean, ocean=trial)),
            snow_mass=snow1,
        )
        # Both native discretizations use the transpose of the SAME cell wind
        # reconstruction. Cancellation is exact in generalized edge momentum.
        into_ocean = (exchange.rain + exchange.snow * tiles.fraction[:, 0])[
            :, None
        ] * exchange.precipitation_velocity - exchange.vapor[
            :, 0, None
        ] * exchange.vapor_velocity[
            :, 0
        ]
        into_ice = snowfall.sum(-1, keepdim=True) * exchange.precipitation_velocity - (
            exchange.vapor[:, 1:, None] * exchange.vapor_velocity[:, 1:]
        ).sum(1)
        Iw, Ii = (self.wind.transpose((A * x)[:, None]) for x in (into_ocean, into_ice))
        Jw, Ji = (
            self.wind.transpose((-A * x)[:, None])
            for x in (exchange.stress[:, 0], exchange.stress[:, 1:].sum(1))
        )
        mechanical = H.model.energy.base.mechanical
        M0, M1 = mechanical.metric(s)[:, :1], mechanical.metric(trial)[:, :1]
        N0, N1 = H.metric(old), H.metric(candidate.coupled)
        pw, pi = M0 * s.u_e[:, :1] + Iw, N0 * ice.velocity + Ii
        if ((N1 == 0) & ((pi + Ji) != 0)).any():
            raise ValueError("nonzero momentum enters an empty ice edge")
        uw, ui = pw / safe(M1), pi / safe(N1)
        uw1, ui1 = uw + Jw / safe(M1), torch.where(N1 > 0, ui + Ji / safe(N1), 0.0)
        # Finite work of the stress kick AFTER water incorporation, using the
        # exact native midpoint. The atmosphere receives its negative.
        stress_ke = (uw + 0.5 * Jw / safe(M1)) * Jw + (ui + 0.5 * Ji / safe(N1)) * Ji
        work = (
            ConstantDensityEnergy.interpolation_transpose(mesh, stress_ke)[:, 0]
            / A[:, 0]
        )
        exchange = exchange.with_stress_work(-work)
        trial = trial.with_tensors(
            torch.cat((uw1, s.u_e[:, 1:]), -1), h, temp_h, s.salt_h, s.tracer_h
        )
        candidate = replace(
            candidate,
            coupled=replace(
                candidate.coupled,
                ice=replace(ice1, velocity=ui1),
                ocean=replace(old.ocean, ocean=trial),
            ),
        )
        residual = (
            M1 * uw1
            - M0 * s.u_e[:, :1]
            + N1 * ui1
            - N0 * ice.velocity
            - Iw
            - Ii
            - Jw
            - Ji
        )
        return candidate, exchange, (Iw + Jw, Ii + Ji, residual)

    def finish(self, initial, candidate, exchange, impulses, *, storage_change=None):
        H = self.H
        s = candidate.coupled.ocean.ocean
        A = s.mesh.cell_area
        before, provisional = H.terms(initial.coupled), H.terms(candidate.coupled)
        change = sum(provisional[k] - before[k] for k in before) / A
        storage = torch.zeros_like(change) if storage_change is None else storage_change
        thermal = exchange.surface_energy - change - storage
        mass = H.model.energy.mass_area(s) * s.h[:, :1]
        specific = torch.cat(
            ((A * thermal)[:, None] / mass, torch.zeros_like(s.h[:, 1:])), 1
        )
        end, _, _ = RoquetWorkspace(H.model.ocean, s).heat(
            s, s, specific, allow_cooling=True
        )
        result = replace(
            candidate,
            coupled=replace(
                candidate.coupled, ocean=replace(candidate.coupled.ocean, ocean=end)
            ),
        )
        self.snow.check(result)
        after = H.terms(result.coupled)
        error = (
            sum(after[k] - before[k] for k in before) / A
            + storage
            - exchange.surface_energy
        )
        scale = (
            sum(v.abs() for v in before.values()) / A + exchange.surface_energy.abs()
        )
        if (
            not torch.isfinite(error).all()
            or (
                error.abs() > 2048 * torch.finfo(A.dtype).eps * scale.clamp_min(1)
            ).any()
        ):
            raise FloatingPointError("shared boundary energy solve failed")
        s0 = initial.coupled.ocean.ocean
        water = (
            s.column.rho0 * (end.h - s0.h).sum(-1)
            + (H.mass(result.coupled) - H.mass(initial.coupled))[:, 0]
            + exchange.water
        )
        Iw, Ii, momentum = impulses
        return result, SurfaceReport(error, water, momentum, thermal, Iw, Ii)
