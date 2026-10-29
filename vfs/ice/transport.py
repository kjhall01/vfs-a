"""Conservative unstructured transport for the one-category sea-ice state.

The thermodynamic sea-ice state stores *in-ice* thickness and snow depth,
whereas the finite-volume quantities are their grid-cell means:
``A``, ``A h``, and ``A s``.  This component transports those extensive
quantities over the native Voronoi/MPAS C grid with the same edge orientation
and finite-volume divergence used by the atmosphere and ocean.  It therefore
does not assume an icosahedral mesh.

It is deliberately a one-category scheme.  When convergent transport would
make compactness ``A`` exceed local ocean area, compactness is capped and the
unchanged ice volume is represented as greater mean thickness.  That is a
bounded ridging closure, not a thickness-distribution model.  The diagnostic
area removed by this operation is exposed explicitly.

The EVP velocity and stress are carried as restart fields, but not advected by
this scalar map.  They remain owned by the separate native C-grid EVP
momentum map.  That avoids silently treating a cell-vector reconstruction as
a conservative C-grid momentum transport scheme; a proper vector-invariant
ice-momentum advection operator remains a distinct piece of work.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
import torch.nn as nn

from ..atmosphere.operators.fluxes import HorizontalScalarFluxPackage
from ..core.registry import register_component
from ..surface.fractions import SurfaceFractions
from ..surface.utils import field_values
from .state import SeaIceState


def _cell(values: torch.Tensor, *, name: str, n_cells: int) -> torch.Tensor:
    value = field_values(values)
    if value.ndim == 1:
        value = value[:, None]
    if tuple(value.shape) != (int(n_cells), 1):
        raise ValueError(f"{name} must have shape ({int(n_cells)}, 1), got {tuple(value.shape)}")
    return value


def _edge(values: torch.Tensor, *, name: str, n_edges: int) -> torch.Tensor:
    value = field_values(values)
    if value.ndim == 1:
        value = value[:, None]
    if tuple(value.shape) != (int(n_edges), 1):
        raise ValueError(f"{name} must have shape ({int(n_edges)}, 1), got {tuple(value.shape)}")
    return value


@dataclass(frozen=True)
class SeaIceTransportStepResult:
    """Endpoint and scalar diagnostics from one bounded transport map."""

    state: SeaIceState
    fractions: SurfaceFractions
    diagnostics: dict[str, torch.Tensor]


class SeaIceTransport(nn.Module):
    """Conservative native-mesh transport and bounded ridging.

    ``max_courant`` is enforced by deterministic internal subcycling.  The
    default ``upwind`` scheme is a donor-cell update. ``fct_muscl`` adds the
    atmosphere's limited-MUSCL face reconstruction through a shared-edge
    flux-corrected-transport limiter.  It therefore retains the donor update's
    positivity and exact global conservation for compactness, ice volume, and
    snow volume, rather than merely assuming that a reconstructed face value
    is sufficient for a forward-Euler thickness update.
    """

    recipe_name = "sea_ice_transport"

    def __init__(
        self,
        *,
        ice_density: float = 917.0,
        max_courant: float = 0.5,
        max_substeps: int = 4096,
        min_ice_fraction: float = 1.0e-10,
        scheme: str = "upwind",
    ) -> None:
        super().__init__()
        if float(ice_density) <= 0.0:
            raise ValueError("ice_density must be positive")
        if not 0.0 < float(max_courant) <= 1.0:
            raise ValueError("max_courant must be in (0, 1]")
        if int(max_substeps) < 1:
            raise ValueError("max_substeps must be at least one")
        if float(min_ice_fraction) <= 0.0:
            raise ValueError("min_ice_fraction must be positive")
        if str(scheme) not in {"upwind", "fct_muscl"}:
            raise ValueError("scheme must be 'upwind' or 'fct_muscl'")
        self.ice_density = float(ice_density)
        self.max_courant = float(max_courant)
        self.max_substeps = int(max_substeps)
        self.min_ice_fraction = float(min_ice_fraction)
        self.scheme = str(scheme)
        self._muscl = HorizontalScalarFluxPackage.muscl_horizontal()

    @staticmethod
    def _edge_ocean_velocity(mesh: Any, velocity: torch.Tensor, ocean_fraction: torch.Tensor) -> torch.Tensor:
        """Suppress transport through an edge without ocean on both sides."""

        c0 = mesh.cellsOnEdge[:, 0].to(device=velocity.device, dtype=torch.long)
        c1 = mesh.cellsOnEdge[:, 1].to(device=velocity.device, dtype=torch.long)
        water_edge = torch.minimum(ocean_fraction[c0], ocean_fraction[c1]).clamp(0.0, 1.0)
        return velocity * water_edge

    @staticmethod
    def _upwind(mesh: Any, cell_values: torch.Tensor, velocity: torch.Tensor) -> torch.Tensor:
        c0 = mesh.cellsOnEdge[:, 0].to(device=cell_values.device, dtype=torch.long)
        c1 = mesh.cellsOnEdge[:, 1].to(device=cell_values.device, dtype=torch.long)
        return torch.where(velocity >= 0.0, cell_values[c0], cell_values[c1])

    @staticmethod
    def _neighbor_bounds(mesh: Any, cell_values: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Return cell/edge-neighbor extrema for a ``[nCells, 1]`` field."""

        n_cells = int(mesh.nCells)
        device = cell_values.device
        cell_index = torch.arange(n_cells, device=device, dtype=torch.long)[:, None]
        safe_edges = mesh.safe_edges_on_cell.to(device=device, dtype=torch.long)
        edge_cells = mesh.safe_cells_on_edge.to(device=device, dtype=torch.long)
        first = edge_cells[safe_edges, 0]
        second = edge_cells[safe_edges, 1]
        neighbor = torch.where(first == cell_index, second, first)
        valid = mesh.valid_edges_on_cell.to(device=device)
        neighbor = torch.where(valid, neighbor, cell_index)
        values = cell_values[neighbor]
        values = torch.where(valid[..., None], values, cell_values[:, None, :])
        return torch.minimum(cell_values, values.min(dim=1).values), torch.maximum(cell_values, values.max(dim=1).values)

    @staticmethod
    def _scatter_cell_amount(
        edge_amount: torch.Tensor,
        cells: torch.Tensor,
        n_cells: int,
    ) -> torch.Tensor:
        out = torch.zeros(int(n_cells), 1, dtype=edge_amount.dtype, device=edge_amount.device)
        return out.scatter_add(0, cells[:, None], edge_amount)

    def _fct_flux(self, mesh: Any, cell_values: torch.Tensor, velocity: torch.Tensor, dt: float) -> torch.Tensor:
        """Return a FCT-limited MUSCL flux with shared-edge conservation.

        The antidiffusive edge amount is accepted only where both adjacent
        cells can remain in their local finite-volume bounds after the
        donor-cell update.  The result is a single edge flux, so divergence
        preserves the global cell-area integral exactly.
        """

        low = velocity * self._upwind(mesh, cell_values, velocity)
        left, right = self._muscl._reconstructed_sides(mesh, cell_values)
        high = velocity * torch.where(velocity >= 0.0, left, right)
        low_next = cell_values - float(dt) * mesh.div(low)
        q_min, q_max = self._neighbor_bounds(mesh, cell_values)
        q_min = torch.minimum(q_min, low_next)
        q_max = torch.maximum(q_max, low_next)
        dtype, device = cell_values.dtype, cell_values.device
        c0 = mesh.cellsOnEdge[:, 0].to(device=device, dtype=torch.long)
        c1 = mesh.cellsOnEdge[:, 1].to(device=device, dtype=torch.long)
        area = mesh.cell_area.to(dtype=dtype, device=device)[:, None]
        edge_amount = float(dt) * mesh.dv_edge.to(dtype=dtype, device=device)[:, None] * (high - low)
        # Correction amounts received by the left/right cells. Positive is an
        # increase of the cell-integrated quantity.
        gain0 = -edge_amount
        gain1 = edge_amount
        positive = self._scatter_cell_amount(gain0.clamp_min(0.0), c0, int(mesh.nCells)) + self._scatter_cell_amount(
            gain1.clamp_min(0.0), c1, int(mesh.nCells)
        )
        negative = self._scatter_cell_amount((-gain0).clamp_min(0.0), c0, int(mesh.nCells)) + self._scatter_cell_amount(
            (-gain1).clamp_min(0.0), c1, int(mesh.nCells)
        )
        tiny = torch.finfo(dtype).tiny
        r_plus = torch.minimum(torch.ones_like(positive), area * (q_max - low_next).clamp_min(0.0) / positive.clamp_min(tiny))
        r_minus = torch.minimum(torch.ones_like(negative), area * (low_next - q_min).clamp_min(0.0) / negative.clamp_min(tiny))
        positive_edge = edge_amount >= 0.0
        limiter_positive = torch.minimum(r_minus[c0], r_plus[c1])
        limiter_negative = torch.minimum(r_plus[c0], r_minus[c1])
        limiter = torch.where(positive_edge, limiter_positive, limiter_negative).clamp(0.0, 1.0)
        return low + limiter * (high - low)

    def _advect(self, mesh: Any, cell_values: torch.Tensor, velocity: torch.Tensor, dt: float) -> torch.Tensor:
        flux = (
            velocity * self._upwind(mesh, cell_values, velocity)
            if self.scheme == "upwind"
            else self._fct_flux(mesh, cell_values, velocity, dt)
        )
        return cell_values - float(dt) * mesh.div(flux)

    @staticmethod
    def _outflow_rate(mesh: Any, velocity: torch.Tensor, n_cells: int) -> torch.Tensor:
        """Return the per-cell donor CFL rate for a globally oriented edge flow."""

        dtype, device = velocity.dtype, velocity.device
        c0 = mesh.cellsOnEdge[:, 0].to(device=device, dtype=torch.long)
        c1 = mesh.cellsOnEdge[:, 1].to(device=device, dtype=torch.long)
        length = mesh.dv_edge.to(dtype=dtype, device=device)[:, None]
        outflow = torch.zeros(int(n_cells), 1, dtype=dtype, device=device)
        outflow.scatter_add_(0, c0[:, None], length * velocity.clamp_min(0.0))
        outflow.scatter_add_(0, c1[:, None], length * (-velocity).clamp_min(0.0))
        area = mesh.cell_area.to(dtype=dtype, device=device)[:, None]
        return outflow / area.clamp_min(torch.finfo(dtype).tiny)

    def _n_substeps(self, mesh: Any, velocity: torch.Tensor, dt: float) -> int:
        rate = self._outflow_rate(mesh, velocity, int(mesh.nCells)).max()
        courant = float((rate * float(dt)).detach().cpu())
        steps = max(1, int(torch.ceil(torch.as_tensor(courant / float(self.max_courant))).item()))
        if steps > int(self.max_substeps):
            raise ValueError(
                f"sea-ice transport requires {steps} substeps at dt={float(dt):g}; "
                f"max_substeps={self.max_substeps}"
            )
        return steps

    @staticmethod
    def _integral(mesh: Any, value: torch.Tensor) -> torch.Tensor:
        area = mesh.cell_area.to(dtype=value.dtype, device=value.device)[:, None]
        return (value * area).sum()

    def step(
        self,
        state: SeaIceState,
        fractions: SurfaceFractions,
        *,
        mesh: Any,
        dt: float,
    ) -> SeaIceTransportStepResult:
        """Transport one category of sea ice over ``dt`` on ``mesh``.

        No-op behavior for a state without EVP edge velocity keeps pure
        thermodynamic configurations valid and makes this component opt-in.
        """

        dt = float(dt)
        if not torch.isfinite(torch.as_tensor(dt)) or dt <= 0.0:
            raise ValueError("sea-ice transport dt must be finite and positive")
        n_cells, n_edges = int(mesh.nCells), int(mesh.nEdges)
        fraction = _cell(state.sea_ice_fraction, name="sea_ice_fraction", n_cells=n_cells).clamp(0.0, 1.0)
        thickness = _cell(state.sea_ice_thickness, name="sea_ice_thickness", n_cells=n_cells).clamp_min(0.0)
        snow = _cell(state.snow_on_sea_ice, name="snow_on_sea_ice", n_cells=n_cells).clamp_min(0.0)
        temperature = _cell(
            state.sea_ice_surface_temperature,
            name="sea_ice_surface_temperature",
            n_cells=n_cells,
        )
        ocean = _cell(fractions.ocean_fraction, name="ocean_fraction", n_cells=n_cells).clamp(0.0, 1.0)
        if state.sea_ice_u_e is None:
            zero = torch.zeros((), dtype=fraction.dtype, device=fraction.device)
            return SeaIceTransportStepResult(
                state,
                fractions,
                {
                    "sea_ice_transport_substeps": zero,
                    "sea_ice_transport_max_courant": zero,
                    "sea_ice_transport_ridged_area": zero,
                    "sea_ice_transport_ice_mass_residual": zero,
                    "sea_ice_transport_snow_mass_residual": zero,
                },
            )
        ice_velocity = _edge(state.sea_ice_u_e, name="sea_ice_u_e", n_edges=n_edges).to(
            dtype=fraction.dtype,
            device=fraction.device,
        )
        velocity = self._edge_ocean_velocity(mesh, ice_velocity, ocean)
        n_substeps = self._n_substeps(mesh, velocity, dt)
        sub_dt = dt / float(n_substeps)
        max_courant = self._outflow_rate(mesh, velocity, n_cells).max() * sub_dt

        # These are all cell-area densities.  Transporting them—not h and s
        # themselves—is what conserves total ice and snow water mass.
        area_fraction = fraction
        ice_volume = fraction * thickness
        snow_volume = fraction * snow
        thermal_volume = ice_volume * temperature
        stress = tuple(
            None
            if getattr(state, name) is None
            else area_fraction * _cell(getattr(state, name), name=name, n_cells=n_cells)
            for name in ("sea_ice_stress_xx", "sea_ice_stress_xy", "sea_ice_stress_yy")
        )
        initial_ice_mass = self._integral(mesh, ice_volume)
        initial_snow_mass = self._integral(mesh, snow_volume)
        ridged_area = torch.zeros((), dtype=fraction.dtype, device=fraction.device)

        for _ in range(n_substeps):
            area_raw = self._advect(mesh, area_fraction, velocity, sub_dt)
            next_ice_volume = self._advect(mesh, ice_volume, velocity, sub_dt).clamp_min(0.0)
            next_snow_volume = self._advect(mesh, snow_volume, velocity, sub_dt).clamp_min(0.0)
            next_thermal_volume = self._advect(mesh, thermal_volume, velocity, sub_dt)
            next_stress = tuple(
                None if value is None else self._advect(mesh, value, velocity, sub_dt) for value in stress
            )
            area_raw = area_raw.clamp_min(0.0)
            area_fraction = torch.minimum(area_raw, ocean)
            ridged_area = ridged_area + self._integral(mesh, (area_raw - area_fraction).clamp_min(0.0))
            ice_volume, snow_volume, thermal_volume = next_ice_volume, next_snow_volume, next_thermal_volume
            stress = next_stress

        tiny = torch.finfo(fraction.dtype).tiny
        occupied = area_fraction > float(self.min_ice_fraction)
        next_thickness = torch.where(occupied, ice_volume / area_fraction.clamp_min(tiny), torch.zeros_like(ice_volume))
        next_snow = torch.where(occupied, snow_volume / area_fraction.clamp_min(tiny), torch.zeros_like(snow_volume))
        has_ice_volume = ice_volume > float(self.min_ice_fraction)
        next_temperature = torch.where(
            has_ice_volume,
            thermal_volume / ice_volume.clamp_min(tiny),
            temperature,
        )
        next_stress_fields = tuple(
            None
            if value is None
            else torch.where(occupied, value / area_fraction.clamp_min(tiny), torch.zeros_like(value))
            for value in stress
        )
        next_state = SeaIceState(
            area_fraction,
            next_thickness,
            next_snow,
            next_temperature,
            ice_velocity,
            next_stress_fields[0],
            next_stress_fields[1],
            next_stress_fields[2],
        )
        # Snow coverage has only a combined land/sea-ice ownership field in
        # SurfaceFractions.  Limit the pre-existing diagnostic overlay to its
        # new valid ownership; its material mass is the snow_volume above.
        snow_cover = torch.minimum(
            _cell(fractions.snow_fraction, name="snow_fraction", n_cells=n_cells),
            (fractions.land_fraction.to(dtype=fraction.dtype, device=fraction.device) + area_fraction).clamp(0.0, 1.0),
        )
        next_fractions = fractions.with_overlays(sea_ice_fraction=area_fraction, snow_fraction=snow_cover)
        ice_residual = self._integral(mesh, ice_volume) - initial_ice_mass
        snow_residual = self._integral(mesh, snow_volume) - initial_snow_mass
        diagnostics = {
            "sea_ice_transport_substeps": torch.as_tensor(float(n_substeps), dtype=fraction.dtype, device=fraction.device),
            "sea_ice_transport_max_courant": max_courant,
            "sea_ice_transport_ridged_area": ridged_area,
            "sea_ice_transport_ice_mass_residual": ice_residual * float(self.ice_density),
            "sea_ice_transport_snow_mass_residual": snow_residual,
            "sea_ice_transport_ice_mass": self._integral(mesh, ice_volume) * float(self.ice_density),
            "sea_ice_transport_snow_mass": self._integral(mesh, snow_volume),
            "sea_ice_transport_fraction_max": area_fraction.max(),
        }
        return SeaIceTransportStepResult(next_state, next_fractions, diagnostics)


register_component("sea_ice_transport", SeaIceTransport)


__all__ = ["SeaIceTransport", "SeaIceTransportStepResult"]
