"""Native-mesh elastic--viscous--plastic sea-ice momentum closure.

The closure keeps edge-normal ice velocity and cell-centred symmetric stress
in :class:`~vfs.ice.state.SeaIceState`.  It uses the VFS Voronoi C-grid's
gradient, velocity reconstruction, and finite-volume edge traction geometry:

* Hibler-style compactness-dependent ice strength and elliptic VP target
  stress are evaluated from local reconstructed strain rates;
* an elastic relaxation advances that target stress over bounded physical
  substeps (the EVP part of the closure);
* atmosphere--ice and ocean--ice drag are returned as distinct, equal and
  opposite stress pairs for the coupled source to apply to their owning
  prognostic states;
* internal stress is a shared-face finite-volume divergence, so it has no
  global external impulse on the closed native mesh.

This is deliberately a momentum/rheology component only.  Ice advection,
ridging, and a thickness distribution are not silently implied by it; the
existing conservative thermodynamic map continues to own thickness and
fraction evolution.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
import torch.nn as nn

from ..core.registry import register_component
from ..field import Field
from ..surface.utils import field_values
from .state import SeaIceState


def _cell(values: Field | torch.Tensor, *, name: str) -> torch.Tensor:
    value = field_values(values)
    if value.ndim == 1:
        value = value[:, None]
    if value.ndim != 2 or int(value.shape[-1]) != 1:
        raise ValueError(f"{name} must have shape [nCells, 1] or [nCells], got {tuple(value.shape)}")
    return value


def _edge(values: Field | torch.Tensor, *, name: str, n_edges: int) -> torch.Tensor:
    value = field_values(values)
    if value.ndim == 1:
        value = value[:, None]
    if tuple(value.shape) != (int(n_edges), 1):
        raise ValueError(f"{name} must have shape ({int(n_edges)}, 1), got {tuple(value.shape)}")
    return value


@dataclass(frozen=True)
class SeaIceEVPStepResult:
    """One bounded sea-ice EVP physical-map update.

    ``stress_on_ice_from_air_edge`` and ``stress_on_ice_from_ocean_edge`` are
    both positive in the direction of ice acceleration.  The source adapter
    applies their exact opposites to atmosphere and ocean respectively.
    """

    state: SeaIceState
    stress_on_ice_from_air_edge: torch.Tensor
    stress_on_ice_from_ocean_edge: torch.Tensor
    internal_stress_edge: torch.Tensor
    diagnostics: dict[str, torch.Tensor]


class SeaIceEVP(nn.Module):
    """Bounded elastic--viscous--plastic sea-ice momentum closure.

    The implementation follows a frozen-coefficient EVP relaxation: each
    physical substep reconstructs strain, builds a Hibler elliptic VP target
    stress, relaxes the stored stress towards it, then advances native
    edge-normal ice velocity under air, ocean, and internal stress.  The
    relaxation time is a physical-map control, not an ARK stage coefficient.
    """

    recipe_name = "sea_ice_evp"

    def __init__(
        self,
        *,
        ice_density: float = 917.0,
        air_density: float = 1.25,
        ocean_density: float = 1025.0,
        air_drag_coefficient: float = 1.3e-3,
        ocean_drag_coefficient: float = 5.5e-3,
        ice_strength: float = 2.75e4,
        compactness: float = 20.0,
        ellipse_aspect_ratio: float = 2.0,
        elastic_relaxation_seconds: float = 300.0,
        strain_rate_floor: float = 1.0e-9,
        max_bulk_viscosity: float = 1.0e10,
        min_thickness: float = 1.0e-2,
        min_fraction: float = 1.0e-8,
    ) -> None:
        super().__init__()
        for name, value in (
            ("ice_density", ice_density),
            ("air_density", air_density),
            ("ocean_density", ocean_density),
            ("ice_strength", ice_strength),
            ("compactness", compactness),
            ("ellipse_aspect_ratio", ellipse_aspect_ratio),
            ("elastic_relaxation_seconds", elastic_relaxation_seconds),
            ("strain_rate_floor", strain_rate_floor),
            ("max_bulk_viscosity", max_bulk_viscosity),
            ("min_thickness", min_thickness),
            ("min_fraction", min_fraction),
        ):
            if float(value) <= 0.0:
                raise ValueError(f"{name} must be positive")
        for name, value in (("air_drag_coefficient", air_drag_coefficient), ("ocean_drag_coefficient", ocean_drag_coefficient)):
            if float(value) < 0.0:
                raise ValueError(f"{name} must be nonnegative")
        self.ice_density = float(ice_density)
        self.air_density = float(air_density)
        self.ocean_density = float(ocean_density)
        self.air_drag_coefficient = float(air_drag_coefficient)
        self.ocean_drag_coefficient = float(ocean_drag_coefficient)
        self.ice_strength = float(ice_strength)
        self.compactness = float(compactness)
        self.ellipse_aspect_ratio = float(ellipse_aspect_ratio)
        self.elastic_relaxation_seconds = float(elastic_relaxation_seconds)
        self.strain_rate_floor = float(strain_rate_floor)
        self.max_bulk_viscosity = float(max_bulk_viscosity)
        self.min_thickness = float(min_thickness)
        self.min_fraction = float(min_fraction)

    def initialize(self, state: SeaIceState, mesh: Any) -> SeaIceState:
        """Attach zero EVP velocity/stress fields to a thermodynamic state."""

        n_cells = int(mesh.nCells)
        n_edges = int(mesh.nEdges)
        fraction = _cell(state.sea_ice_fraction, name="sea_ice_fraction")
        if int(fraction.shape[0]) != n_cells:
            raise ValueError("sea-ice state and EVP mesh have different nCells")
        dtype, device = fraction.dtype, fraction.device
        u_e = torch.zeros(n_edges, 1, dtype=dtype, device=device) if state.sea_ice_u_e is None else _edge(
            state.sea_ice_u_e,
            name="sea_ice_u_e",
            n_edges=n_edges,
        ).to(dtype=dtype, device=device)
        stress = []
        for name in ("sea_ice_stress_xx", "sea_ice_stress_xy", "sea_ice_stress_yy"):
            value = getattr(state, name)
            stress.append(
                torch.zeros_like(fraction)
                if value is None
                else _cell(value, name=name).to(dtype=dtype, device=device)
            )
        return SeaIceState(
            fraction,
            state.sea_ice_thickness,
            state.snow_on_sea_ice,
            state.sea_ice_surface_temperature,
            u_e,
            stress[0],
            stress[1],
            stress[2],
        )

    def step(
        self,
        state: SeaIceState,
        *,
        mesh: Any,
        atmosphere_u_e: torch.Tensor,
        ocean_u_e: torch.Tensor,
        dt: float,
    ) -> SeaIceEVPStepResult:
        """Advance one physical EVP substep and return exchange stresses."""

        dt = float(dt)
        if not torch.isfinite(torch.as_tensor(dt)) or dt <= 0.0:
            raise ValueError("EVP dt must be finite and positive")
        state = self.initialize(state, mesh)
        n_cells, n_edges = int(mesh.nCells), int(mesh.nEdges)
        fraction = _cell(state.sea_ice_fraction, name="sea_ice_fraction")
        thickness = _cell(state.sea_ice_thickness, name="sea_ice_thickness").clamp_min(0.0)
        if tuple(fraction.shape) != (n_cells, 1) or tuple(thickness.shape) != (n_cells, 1):
            raise ValueError("sea-ice EVP cell fields must match mesh.nCells")
        dtype, device = fraction.dtype, fraction.device
        ice_u = _edge(state.sea_ice_u_e, name="sea_ice_u_e", n_edges=n_edges).to(dtype=dtype, device=device)
        air_u = _edge(atmosphere_u_e, name="atmosphere_u_e", n_edges=n_edges).to(dtype=dtype, device=device)
        ocean_u = _edge(ocean_u_e, name="ocean_u_e", n_edges=n_edges).to(dtype=dtype, device=device)
        fraction_edge = mesh.cell_to_edge(fraction).to(dtype=dtype, device=device).clamp(0.0, 1.0)
        thickness_edge = mesh.cell_to_edge(thickness).to(dtype=dtype, device=device).clamp_min(0.0)
        active = (fraction_edge > float(self.min_fraction)) & (thickness_edge > float(self.min_thickness))
        ice_u = torch.where(active, ice_u, torch.zeros_like(ice_u))

        stress_xx, stress_xy, stress_yy, strain_rate = self._relaxed_stress(state, mesh, fraction, thickness, ice_u, dt)
        internal_edge = self._internal_stress_edge(
            mesh,
            stress_xx,
            stress_xy,
            stress_yy,
        )
        internal_edge = torch.where(active, internal_edge, torch.zeros_like(internal_edge))
        internal_edge = self._remove_global_edge_impulse(mesh, internal_edge)
        internal_global = self._edge_force_vector(mesh, internal_edge)
        relative_air = air_u - ice_u
        relative_ocean = ocean_u - ice_u
        tau_air = (
            float(self.air_density)
            * float(self.air_drag_coefficient)
            * fraction_edge
            * relative_air.abs()
            * relative_air
        )
        tau_ocean = (
            float(self.ocean_density)
            * float(self.ocean_drag_coefficient)
            * fraction_edge
            * relative_ocean.abs()
            * relative_ocean
        )
        tau_air = torch.where(active, tau_air, torch.zeros_like(tau_air))
        tau_ocean = torch.where(active, tau_ocean, torch.zeros_like(tau_ocean))
        mass_edge = float(self.ice_density) * fraction_edge * thickness_edge
        total_stress = tau_air + tau_ocean + internal_edge
        acceleration = torch.where(active, total_stress / mass_edge.clamp_min(torch.finfo(dtype).tiny), torch.zeros_like(total_stress))
        next_u = torch.where(active, ice_u + dt * acceleration, torch.zeros_like(ice_u))
        next_state = SeaIceState(
            fraction,
            thickness,
            state.snow_on_sea_ice,
            state.sea_ice_surface_temperature,
            next_u,
            stress_xx,
            stress_xy,
            stress_yy,
        )
        control_area = mesh.edge_side_area.sum(dim=-1, keepdim=True).to(dtype=dtype, device=device)
        normal = mesh.edge_normal.to(dtype=dtype, device=device)
        air_impulse = (float(dt) * control_area * tau_air * normal).sum(dim=0)
        ocean_impulse = (float(dt) * control_area * tau_ocean * normal).sum(dim=0)
        update_residual = mass_edge * (next_u - ice_u) / dt - total_stress
        diagnostics = {
            "sea_ice_evp_active_edge_fraction": active.to(dtype=dtype).mean(),
            "sea_ice_evp_air_stress_rms": tau_air.square().mean().sqrt(),
            "sea_ice_evp_ocean_stress_rms": tau_ocean.square().mean().sqrt(),
            "sea_ice_evp_internal_stress_rms": internal_edge.square().mean().sqrt(),
            "sea_ice_evp_strain_rate_rms": strain_rate.square().mean().sqrt(),
            "sea_ice_evp_stress_rms": (
                stress_xx.square() + 2.0 * stress_xy.square() + stress_yy.square()
            ).mean().sqrt(),
            "sea_ice_evp_update_residual_max_abs": update_residual.abs().max(),
            "sea_ice_evp_internal_force_global_norm": internal_global.square().sum().sqrt(),
            "sea_ice_evp_air_impulse_x": air_impulse[0],
            "sea_ice_evp_air_impulse_y": air_impulse[1],
            "sea_ice_evp_ocean_impulse_x": ocean_impulse[0],
            "sea_ice_evp_ocean_impulse_y": ocean_impulse[1],
            "sea_ice_evp_momentum_x": next_state.momentum_storage(mesh=mesh, ice_density=self.ice_density)[0],
            "sea_ice_evp_momentum_y": next_state.momentum_storage(mesh=mesh, ice_density=self.ice_density)[1],
        }
        return SeaIceEVPStepResult(next_state, tau_air, tau_ocean, internal_edge, diagnostics)

    def _relaxed_stress(
        self,
        state: SeaIceState,
        mesh: Any,
        fraction: torch.Tensor,
        thickness: torch.Tensor,
        ice_u: torch.Tensor,
        dt: float,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return stress relaxed towards a Hibler elliptic VP target."""

        dtype, device = ice_u.dtype, ice_u.device
        velocity = mesh.cell_wind_vector(ice_u)[..., 0, :]
        basis1 = mesh.cell_rec_basis1.to(dtype=dtype, device=device)
        basis2 = mesh.cell_rec_basis2.to(dtype=dtype, device=device)
        u1 = (velocity * basis1).sum(dim=-1, keepdim=True)
        u2 = (velocity * basis2).sum(dim=-1, keepdim=True)
        grad_u1 = mesh.cell_wind_vector(mesh.grad(u1))[..., 0, :]
        grad_u2 = mesh.cell_wind_vector(mesh.grad(u2))[..., 0, :]
        du1_ds1 = (grad_u1 * basis1).sum(dim=-1, keepdim=True)
        du1_ds2 = (grad_u1 * basis2).sum(dim=-1, keepdim=True)
        du2_ds1 = (grad_u2 * basis1).sum(dim=-1, keepdim=True)
        du2_ds2 = (grad_u2 * basis2).sum(dim=-1, keepdim=True)
        e11 = du1_ds1
        e22 = du2_ds2
        e12 = 0.5 * (du1_ds2 + du2_ds1)
        aspect2 = float(self.ellipse_aspect_ratio) ** 2
        strain_invariant = torch.sqrt(
            (
                (e11.square() + e22.square()) * (1.0 + 1.0 / aspect2)
                + 4.0 * e12.square() / aspect2
                + 2.0 * e11 * e22 * (1.0 - 1.0 / aspect2)
            ).clamp_min(float(self.strain_rate_floor) ** 2)
        )
        pressure = float(self.ice_strength) * thickness * fraction * torch.exp(
            -float(self.compactness) * (1.0 - fraction)
        )
        bulk = (pressure / (2.0 * strain_invariant)).clamp(max=float(self.max_bulk_viscosity))
        shear = bulk / aspect2
        divergence = e11 + e22
        target_xx = 2.0 * shear * e11 + (bulk - shear) * divergence - 0.5 * pressure
        target_xy = 2.0 * shear * e12
        target_yy = 2.0 * shear * e22 + (bulk - shear) * divergence - 0.5 * pressure
        relaxation = min(1.0, float(dt) / float(self.elastic_relaxation_seconds))
        previous = (
            _cell(state.sea_ice_stress_xx, name="sea_ice_stress_xx").to(dtype=dtype, device=device),
            _cell(state.sea_ice_stress_xy, name="sea_ice_stress_xy").to(dtype=dtype, device=device),
            _cell(state.sea_ice_stress_yy, name="sea_ice_stress_yy").to(dtype=dtype, device=device),
        )
        active_cell = (fraction > float(self.min_fraction)) & (thickness > float(self.min_thickness))
        stress = tuple(
            torch.where(active_cell, old + relaxation * (target - old), torch.zeros_like(old))
            for old, target in zip(previous, (target_xx, target_xy, target_yy), strict=True)
        )
        return (*stress, strain_invariant)

    @staticmethod
    def _internal_stress_edge(
        mesh: Any,
        stress_xx: torch.Tensor,
        stress_xy: torch.Tensor,
        stress_yy: torch.Tensor,
    ) -> torch.Tensor:
        """Finite-volume shared-face stress divergence projected to edges."""

        dtype, device = stress_xx.dtype, stress_xx.device
        basis1 = mesh.cell_rec_basis1.to(dtype=dtype, device=device)
        basis2 = mesh.cell_rec_basis2.to(dtype=dtype, device=device)
        sigma = (
            stress_xx[..., None] * basis1[:, :, None] * basis1[:, None, :]
            + stress_xy[..., None]
            * (basis1[:, :, None] * basis2[:, None, :] + basis2[:, :, None] * basis1[:, None, :])
            + stress_yy[..., None] * basis2[:, :, None] * basis2[:, None, :]
        )
        cells = mesh.cellsOnEdge.to(device=device)
        sigma_edge = 0.5 * (sigma[cells[:, 0]] + sigma[cells[:, 1]])
        normal = mesh.edge_normal.to(dtype=dtype, device=device)
        traction = torch.einsum("eij,ej->ei", sigma_edge, normal)
        edge_length = mesh.dv_edge.to(dtype=dtype, device=device)[:, None]
        weighted = traction * edge_length
        force_cell = torch.zeros(int(mesh.nCells), 3, dtype=dtype, device=device)
        force_cell.index_add_(0, cells[:, 0], weighted)
        force_cell.index_add_(0, cells[:, 1], -weighted)
        area = mesh.cell_area.to(dtype=dtype, device=device)[:, None]
        divergence = force_cell / area.clamp_min(torch.finfo(dtype).tiny)
        divergence_edge = mesh.cell_to_edge(divergence).to(dtype=dtype, device=device)
        edge_force = (divergence_edge * normal).sum(dim=-1, keepdim=True)
        return edge_force

    @staticmethod
    def _edge_force_vector(mesh: Any, normal_stress: torch.Tensor) -> torch.Tensor:
        """Return global force for an edge-normal stress on native controls."""

        dtype, device = normal_stress.dtype, normal_stress.device
        area = mesh.edge_side_area.sum(dim=-1, keepdim=True).to(dtype=dtype, device=device)
        normal = mesh.edge_normal.to(dtype=dtype, device=device)
        return (area * normal_stress * normal).sum(dim=0)

    @classmethod
    def _remove_global_edge_impulse(cls, mesh: Any, normal_stress: torch.Tensor) -> torch.Tensor:
        """Project an internal edge stress onto its zero-global-force subspace.

        The cell finite-volume construction balances shared face forces exactly.
        Reconstructing that divergence onto C-grid edges is not an adjoint
        operation, however, and can reintroduce a global impulse.  Remove the
        minimum-control-area weighted normal correction so the *applied* edge
        stress retains the internal-force invariant.
        """

        dtype, device = normal_stress.dtype, normal_stress.device
        area = mesh.edge_side_area.sum(dim=-1).to(dtype=dtype, device=device)
        normal = mesh.edge_normal.to(dtype=dtype, device=device)
        metric = torch.einsum("e,ei,ej->ij", area, normal, normal)
        inverse_metric = torch.linalg.pinv(metric)
        corrected = normal_stress
        # A second pass suppresses the floating-point residual from the first
        # global reduction without changing the local finite-volume stencil.
        for _ in range(2):
            force = cls._edge_force_vector(mesh, corrected)
            correction = normal @ (inverse_metric @ force)
            corrected = corrected - correction[:, None]
        return corrected


register_component("sea_ice_evp", SeaIceEVP)


__all__ = ["SeaIceEVP", "SeaIceEVPStepResult"]
