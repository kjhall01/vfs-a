"""Named, auditable dry operator-set recipes.

The canonical dry RHS remains :class:`DryHydrostaticDynamics`.  Each class in
this module names one compatible set of strategy overrides for a controlled
experiment; the compact ``DryOperatorSet`` catalog at the end is the preferred
way for applications to discover and build those recipes.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch

from ..core.registry import register_component
from ..core.state import State
from .dry_dynamics import DryHydrostaticDynamics
from .conservative import DryHydrostaticConservativeDynamics
from .operators.fluxes import HorizontalScalarFluxPackage
from .dry_strategies import (
    BiharmonicDivergenceDamping,
    ExnerSecantSigmaPressureGradient,
    FCTTracerTransport,
    LogPressureSigmaPressureGradient,
    MildCellSupportedVerticalMomentumTransport,
    TopRayleighSponge,
    UnscaledReferenceCoordinateWork,
    VelocityHyperdiffusion,
    VirtualTemperatureThermodynamics,
)


@register_component("dry_hydrostatic_virtual_temperature")
class DryHydrostaticVirtualTemperatureDynamics(DryHydrostaticDynamics):
    """Experimental dry-pressure virtual-temperature coupling recipe."""

    recipe_name = "virtual_temperature"

    def __init__(self, *args, pressure_gradient=None, **kwargs) -> None:
        if "thermodynamics" in kwargs:
            raise TypeError(
                "DryHydrostaticVirtualTemperatureDynamics owns its thermodynamics strategy"
            )
        super().__init__(
            *args,
            thermodynamics=VirtualTemperatureThermodynamics(),
            pressure_gradient=pressure_gradient,
            **kwargs,
        )

    def _vapor_mixing_ratio(self, state: State) -> torch.Tensor | None:
        return self.thermodynamics.vapor_mixing_ratio(state)

    def _virtual_temperature_factor_full(self, state: State) -> torch.Tensor | None:
        return self.thermodynamics.virtual_temperature_factor_full(state)

    def _virtual_temperature_factor_interface(self, state: State) -> torch.Tensor | None:
        return self.thermodynamics.virtual_temperature_factor_interface(state)

    def _virtual_theta_interface(self, state: State, theta_i: torch.Tensor) -> torch.Tensor:
        return self.thermodynamics.virtual_theta_interface(state, theta_i)

    def _theta_full_for_column(self, state: State, theta_i: torch.Tensor) -> torch.Tensor:
        return self.thermodynamics.theta_full_for_column(state, theta_i)

    def _virtual_temperature_full(self, state: State, theta_i: torch.Tensor) -> torch.Tensor:
        return self.thermodynamics.virtual_temperature_full(state, theta_i)

    def _virtual_specific_volume_full(self, state: State, theta_i: torch.Tensor) -> torch.Tensor:
        return self.thermodynamics.virtual_specific_volume_full(state, theta_i)


@register_component("dry_hydrostatic_unscaled_coord_work")
class DryHydrostaticUnscaledCoordWorkDynamics(DryHydrostaticDynamics):
    """Comparison dynamics using the original unscaled coordinate-work algebra."""

    recipe_name = "unscaled_coord_work"

    def __init__(self, *args, **kwargs) -> None:
        if "coordinate_work" in kwargs:
            raise TypeError(
                "DryHydrostaticUnscaledCoordWorkDynamics owns its coordinate-work strategy"
            )
        super().__init__(*args, coordinate_work=UnscaledReferenceCoordinateWork(), **kwargs)


@register_component("dry_hydrostatic_mild_vertical_momentum")
class DryHydrostaticMildVerticalMomentumDynamics(DryHydrostaticDynamics):
    """Experimental archived mild vertical momentum flux path.

    This changes only the vertical edge-normal momentum transport term.  The
    continuity path, CP mtheta transport, coordinate work, PGF, Bernoulli term,
    and optional diffusion strategies remain inherited from the base dynamics.
    """

    recipe_name = "mild_vertical_momentum"

    def __init__(self, *args, **kwargs) -> None:
        if "momentum_transport" in kwargs:
            raise TypeError(
                "DryHydrostaticMildVerticalMomentumDynamics owns its momentum_transport strategy"
            )
        super().__init__(
            *args,
            momentum_transport=MildCellSupportedVerticalMomentumTransport(),
            **kwargs,
        )


@register_component("dry_hydrostatic_centered_mtheta_transport")
class DryHydrostaticCenteredMthetaTransportDynamics(DryHydrostaticDynamics):
    """Experimental JW-A diagnostic using centered horizontal CP mtheta fluxes.

    The canonical dry core keeps first-order donor/upwind CP mtheta transport
    for robustness and positivity.  This variant changes only the horizontal
    CP mtheta transport stencil to a centered edge value so JW-A maintenance
    probes can separate donor-cell diffusion from the rest of the balanced-jet
    drift.  It is not a production default and should not be used for tracer or
    moist-climate claims without separate monotonicity/positivity checks.
    """

    recipe_name = "centered_mtheta_transport"

    def _mtheta_tendency(
        self,
        state: State,
        u_e: torch.Tensor,
        theta_i: torch.Tensor,
        dp: torch.Tensor,
        *,
        dp_e: torch.Tensor | None = None,
        mass_flux: torch.Tensor | None = None,
    ) -> torch.Tensor:
        mesh = state.mesh
        if mass_flux is None:
            dp_e = mesh.cell_to_edge(dp) if dp_e is None else dp_e
            mass_flux = dp_e * u_e

        c0 = mesh.cellsOnEdge[:, 0]
        c1 = mesh.cellsOnEdge[:, 1]
        theta_upper_e = 0.5 * (theta_i[..., c0, :-1] + theta_i[..., c1, :-1])
        theta_lower_e = 0.5 * (theta_i[..., c0, 1:] + theta_i[..., c1, 1:])

        div_upper = mesh.div(0.5 * mass_flux * theta_upper_e)
        div_lower = mesh.div(0.5 * mass_flux * theta_lower_e)

        dmtheta_dt = torch.zeros_like(theta_i)
        dmtheta_dt[..., :-1] -= div_upper
        dmtheta_dt[..., 1:] -= div_lower
        return dmtheta_dt


@register_component("dry_hydrostatic_kite_centered_mtheta_transport")
class DryHydrostaticKiteCenteredMthetaTransportDynamics(DryHydrostaticDynamics):
    """Experimental CP ``mtheta_i`` transport using edge-kite scalar remapping.

    This diagnostic branch differs from ``centered_mtheta_transport`` only in
    how it places the transported intensive scalar on edges.  The arithmetic
    centered branch uses ``0.5 * (theta_left + theta_right)``.  This branch uses
    ``mesh.cell_to_edge(theta_i)``, which is the mesh-owned side-area weighted
    remap over the two cell subtriangles that form the edge kite.  The remapped
    value is still an intensive ``theta`` value; the subtriangle areas are not
    folded into the transported scalar magnitude.  On the current orthogonal
    Voronoi geometry the two edge-side areas are equal to roundoff, so this is
    normally numerically equivalent to arithmetic centered transport.  It is
    retained as an explicit geometry-owned remap path for diagnostics.
    """

    recipe_name = "kite_centered_mtheta_transport"

    def _mtheta_tendency(
        self,
        state: State,
        u_e: torch.Tensor,
        theta_i: torch.Tensor,
        dp: torch.Tensor,
        *,
        dp_e: torch.Tensor | None = None,
        mass_flux: torch.Tensor | None = None,
    ) -> torch.Tensor:
        mesh = state.mesh
        if mass_flux is None:
            dp_e = mesh.cell_to_edge(dp) if dp_e is None else dp_e
            mass_flux = dp_e * u_e

        theta_i_e = mesh.cell_to_edge(theta_i)
        theta_upper_e = theta_i_e[..., :-1]
        theta_lower_e = theta_i_e[..., 1:]

        div_upper = mesh.div(0.5 * mass_flux * theta_upper_e)
        div_lower = mesh.div(0.5 * mass_flux * theta_lower_e)

        dmtheta_dt = torch.zeros_like(theta_i)
        dmtheta_dt[..., :-1] -= div_upper
        dmtheta_dt[..., 1:] -= div_lower
        return dmtheta_dt


@register_component("dry_hydrostatic_mass_kite_centered_mtheta_transport")
class DryHydrostaticMassKiteCenteredMthetaTransportDynamics(DryHydrostaticDynamics):
    """Experimental CP ``mtheta_i`` transport with mass-weighted kite remap.

    This branch remaps the conservative interface amount and interface mass to
    the edge kite, then divides back to an intensive transported scalar:

    ``theta_edge = cell_to_edge(dm_i * theta_i) / cell_to_edge(dm_i)``.

    It is the conservative-density analogue of ``mesh.cell_to_edge(theta_i)``.
    The flux is still the existing shared edge mass flux, so the horizontal
    ``mtheta_i`` tendency remains globally conservative.
    """

    recipe_name = "mass_kite_centered_mtheta_transport"

    def _mtheta_tendency(
        self,
        state: State,
        u_e: torch.Tensor,
        theta_i: torch.Tensor,
        dp: torch.Tensor,
        *,
        dp_e: torch.Tensor | None = None,
        mass_flux: torch.Tensor | None = None,
    ) -> torch.Tensor:
        mesh = state.mesh
        if mass_flux is None:
            dp_e = mesh.cell_to_edge(dp) if dp_e is None else dp_e
            mass_flux = dp_e * u_e

        dm = self._interface_mass(dp)
        dm_e = mesh.cell_to_edge(dm)
        mtheta_e = mesh.cell_to_edge(dm * theta_i)
        theta_i_e = mtheta_e / dm_e.clamp_min(torch.finfo(theta_i.dtype).tiny)
        theta_upper_e = theta_i_e[..., :-1]
        theta_lower_e = theta_i_e[..., 1:]

        div_upper = mesh.div(0.5 * mass_flux * theta_upper_e)
        div_lower = mesh.div(0.5 * mass_flux * theta_lower_e)

        dmtheta_dt = torch.zeros_like(theta_i)
        dmtheta_dt[..., :-1] -= div_upper
        dmtheta_dt[..., 1:] -= div_lower
        return dmtheta_dt


@register_component("dry_hydrostatic_muscl_mtheta_transport")
class DryHydrostaticMUSCLMthetaTransportDynamics(DryHydrostaticDynamics):
    """Experimental limited linear CP ``mtheta_i`` horizontal transport.

    This is a conservative MUSCL-style diagnostic branch for JW-A/JW-B scalar
    transport experiments.  It reconstructs a limited linear value of
    recovered ``theta_i`` from each cell to each edge, then uses the existing
    upwind mass-flux direction to choose the donor-side reconstructed value.
    The limiter reduces the reconstruction back toward donor-cell upwind near
    local extrema, so this is less diffusive than first-order donor transport
    in smooth regions without being the unlimited centered branch.
    """

    recipe_name = "muscl_mtheta_transport"

    def __init__(
        self,
        *args,
        muscl_blend: float = 1.0,
        muscl_limiter: str = "sqrt_min",
        muscl_limiter_epsilon: float = 1.0e-6,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        blend = float(muscl_blend)
        if blend < 0.0 or blend > 1.0:
            raise ValueError("muscl_blend must be between 0 and 1")
        limiter = str(muscl_limiter)
        if limiter not in {"sqrt_min", "tanh_bound"}:
            raise ValueError("muscl_limiter must be 'sqrt_min' or 'tanh_bound'")
        self.muscl_blend = blend
        self.muscl_limiter = limiter
        self.muscl_limiter_epsilon = float(muscl_limiter_epsilon)

    @staticmethod
    def _cell_tangent_basis(state: State) -> tuple[torch.Tensor, torch.Tensor]:
        mesh = state.mesh
        xyz = mesh.cell_xyz.to(dtype=state.mtheta_i.dtype, device=state.mtheta_i.device)
        z_axis = torch.zeros_like(xyz)
        z_axis[..., 2] = 1.0
        x_axis = torch.zeros_like(xyz)
        x_axis[..., 0] = 1.0
        e1_z = torch.cross(z_axis, xyz, dim=-1)
        e1_x = torch.cross(x_axis, xyz, dim=-1)
        use_x = e1_z.norm(dim=-1, keepdim=True) < 1.0e-8
        e1 = torch.where(use_x, e1_x, e1_z)
        e1 = e1 / e1.norm(dim=-1, keepdim=True).clamp_min(torch.finfo(e1.dtype).tiny)
        e2 = torch.cross(xyz, e1, dim=-1)
        e2 = e2 / e2.norm(dim=-1, keepdim=True).clamp_min(torch.finfo(e2.dtype).tiny)
        return e1, e2

    def _cell_limited_gradients(
        self,
        state: State,
        phi_c: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return tangent-plane least-squares gradients and local bounds."""

        mesh = state.mesh
        dtype = phi_c.dtype
        device = phi_c.device
        n_cells = int(mesh.nCells)
        leading_ndim = phi_c.ndim - 2
        lead = (1,) * leading_ndim

        cell_index = torch.arange(n_cells, device=device, dtype=torch.long).view(n_cells, 1)
        safe_edges = mesh.safe_edges_on_cell.to(device=device).long()
        edge_cells = mesh.safe_cells_on_edge.to(device=device).long()
        neighbor0 = edge_cells[safe_edges, 0]
        neighbor1 = edge_cells[safe_edges, 1]
        neighbor_index = torch.where(neighbor0 == cell_index, neighbor1, neighbor0)
        valid = mesh.valid_edges_on_cell.to(device=device)
        neighbor_index = torch.where(valid, neighbor_index, cell_index)

        cell_xyz = mesh.cell_xyz.to(dtype=dtype, device=device)
        neighbor_xyz = cell_xyz[neighbor_index]
        center_xyz = cell_xyz[:, None, :]
        tangent = neighbor_xyz - (neighbor_xyz * center_xyz).sum(dim=-1, keepdim=True) * center_xyz
        tangent = tangent / tangent.norm(dim=-1, keepdim=True).clamp_min(torch.finfo(dtype).tiny)
        dist = mesh.dc_edge.to(dtype=dtype, device=device)[safe_edges]
        dist = torch.where(valid, dist, torch.zeros_like(dist))

        basis_x, basis_y = self._cell_tangent_basis(state)
        x = dist * (tangent * basis_x[:, None, :]).sum(dim=-1)
        y = dist * (tangent * basis_y[:, None, :]).sum(dim=-1)

        neighbor_phi = phi_c[..., neighbor_index, :]
        center_phi = phi_c.unsqueeze(-2)
        valid_view = valid.view(*lead, n_cells, -1, 1)
        neighbor_phi = torch.where(valid_view, neighbor_phi, center_phi)
        dphi = neighbor_phi - center_phi

        x_view = x.view(*lead, n_cells, -1, 1)
        y_view = y.view(*lead, n_cells, -1, 1)
        A11 = (x * x).sum(dim=-1).to(dtype=dtype).view(*lead, n_cells, 1)
        A12 = (x * y).sum(dim=-1).to(dtype=dtype).view(*lead, n_cells, 1)
        A22 = (y * y).sum(dim=-1).to(dtype=dtype).view(*lead, n_cells, 1)
        b1 = (x_view * dphi).sum(dim=-2)
        b2 = (y_view * dphi).sum(dim=-2)
        det = (A11 * A22 - A12.square()).clamp_min(torch.finfo(dtype).tiny)
        grad_x = (A22 * b1 - A12 * b2) / det
        grad_y = (-A12 * b1 + A11 * b2) / det

        phi_min = torch.minimum(phi_c, neighbor_phi.min(dim=-2).values)
        phi_max = torch.maximum(phi_c, neighbor_phi.max(dim=-2).values)
        return grad_x, grad_y, phi_min, phi_max, basis_x, basis_y

    def _smooth_limited_delta(
        self,
        phi: torch.Tensor,
        delta: torch.Tensor,
        phi_min: torch.Tensor,
        phi_max: torch.Tensor,
    ) -> torch.Tensor:
        """Apply a smooth min(1, local-bound-ratio) slope limiter."""

        tiny = torch.finfo(phi.dtype).tiny
        allowed = torch.where(delta >= 0.0, phi_max - phi, phi - phi_min).clamp_min(0.0)
        if self.muscl_limiter == "tanh_bound":
            # A differentiable saturation toward the local one-sided bound.
            # This removes the sharp min(1, ratio) transition from the
            # full-strength MUSCL branch while remaining strictly bounded.
            sign = torch.sign(delta)
            scale = allowed.clamp_min(tiny)
            return sign * allowed * torch.tanh(delta.abs() / scale)

        ratio = allowed / delta.abs().clamp_min(tiny)
        eps = torch.as_tensor(float(self.muscl_limiter_epsilon), dtype=phi.dtype, device=phi.device)
        one = torch.ones((), dtype=phi.dtype, device=phi.device)
        # Smooth approximation to min(ratio, 1).  The final clamp only removes
        # tiny negative/over-one roundoff from the smooth square-root formula.
        limiter = 0.5 * (ratio + one - torch.sqrt((ratio - one).square() + eps.square()))
        limiter = limiter.clamp(0.0, 1.0)
        return limiter * delta

    def _limited_edge_values(self, state: State, phi_c: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Return reconstructed edge values from cellsOnEdge[:,0] and [:,1]."""

        mesh = state.mesh
        dtype = phi_c.dtype
        device = phi_c.device
        leading_ndim = phi_c.ndim - 2
        lead = (1,) * leading_ndim
        grad_x, grad_y, phi_min, phi_max, basis_x, basis_y = self._cell_limited_gradients(state, phi_c)

        c0 = mesh.cellsOnEdge[:, 0].to(device=device).long()
        c1 = mesh.cellsOnEdge[:, 1].to(device=device).long()
        edge_xyz = mesh.edge_xyz.to(dtype=dtype, device=device)
        cell_xyz = mesh.cell_xyz.to(dtype=dtype, device=device)
        edge_view = edge_xyz

        def side_values(cells: torch.Tensor) -> torch.Tensor:
            center_xyz = cell_xyz[cells]
            tangent = edge_view - (edge_view * center_xyz).sum(dim=-1, keepdim=True) * center_xyz
            tangent = tangent / tangent.norm(dim=-1, keepdim=True).clamp_min(torch.finfo(dtype).tiny)
            dot = (edge_view * center_xyz).sum(dim=-1).clamp(-1.0, 1.0)
            dist = float(mesh.radius) * torch.acos(dot)
            rx = dist * (tangent * basis_x[cells]).sum(dim=-1)
            ry = dist * (tangent * basis_y[cells]).sum(dim=-1)
            phi = phi_c[..., cells, :]
            delta = grad_x[..., cells, :] * rx.view(*lead, -1, 1) + grad_y[..., cells, :] * ry.view(*lead, -1, 1)
            limited = self._smooth_limited_delta(
                phi,
                delta,
                phi_min[..., cells, :],
                phi_max[..., cells, :],
            )
            return phi + limited

        return side_values(c0), side_values(c1)

    def _blend_muscl_edge_values(
        self,
        state: State,
        theta_i: torch.Tensor,
        mass_flux: torch.Tensor,
        theta_upper_muscl: torch.Tensor,
        theta_lower_muscl: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Blend reconstructed edge values back toward donor-cell upwind."""

        if self.muscl_blend >= 1.0:
            return theta_upper_muscl, theta_lower_muscl

        mesh = state.mesh
        c0 = mesh.cellsOnEdge[:, 0].to(device=theta_i.device).long()
        c1 = mesh.cellsOnEdge[:, 1].to(device=theta_i.device).long()
        theta_upper_upwind = torch.where(mass_flux >= 0.0, theta_i[..., c0, :-1], theta_i[..., c1, :-1])
        theta_lower_upwind = torch.where(mass_flux >= 0.0, theta_i[..., c0, 1:], theta_i[..., c1, 1:])
        blend = torch.as_tensor(self.muscl_blend, dtype=theta_i.dtype, device=theta_i.device)
        theta_upper_e = theta_upper_upwind + blend * (theta_upper_muscl - theta_upper_upwind)
        theta_lower_e = theta_lower_upwind + blend * (theta_lower_muscl - theta_lower_upwind)
        return theta_upper_e, theta_lower_e

    def _mtheta_tendency(
        self,
        state: State,
        u_e: torch.Tensor,
        theta_i: torch.Tensor,
        dp: torch.Tensor,
        *,
        dp_e: torch.Tensor | None = None,
        mass_flux: torch.Tensor | None = None,
    ) -> torch.Tensor:
        mesh = state.mesh
        if mass_flux is None:
            dp_e = mesh.cell_to_edge(dp) if dp_e is None else dp_e
            mass_flux = dp_e * u_e

        theta_left, theta_right = self._limited_edge_values(state, theta_i)
        theta_upper_muscl = torch.where(mass_flux >= 0.0, theta_left[..., :-1], theta_right[..., :-1])
        theta_lower_muscl = torch.where(mass_flux >= 0.0, theta_left[..., 1:], theta_right[..., 1:])
        theta_upper_e, theta_lower_e = self._blend_muscl_edge_values(
            state,
            theta_i,
            mass_flux,
            theta_upper_muscl,
            theta_lower_muscl,
        )

        div_upper = mesh.div(0.5 * mass_flux * theta_upper_e)
        div_lower = mesh.div(0.5 * mass_flux * theta_lower_e)

        dmtheta_dt = torch.zeros_like(theta_i)
        dmtheta_dt[..., :-1] -= div_upper
        dmtheta_dt[..., 1:] -= div_lower
        return dmtheta_dt


@register_component("dry_hydrostatic_flux_package_muscl")
class DryHydrostaticFluxPackageMUSCLDynamics(DryHydrostaticDynamics):
    """Field-aware donor MUSCL-horizontal transport through a reusable package.

    The canonical dry core remains unchanged.  This opt-in recipe imports the
    donor flux-package boundary so the exact active faces can be inspected and
    later paired with separately validated momentum work.
    """

    recipe_name = "flux_package_muscl_horizontal"

    def __init__(
        self,
        *args,
        muscl_blend: float = 1.0,
        muscl_limiter_epsilon: float = 1.0e-6,
        transport_flux_package: HorizontalScalarFluxPackage | None = None,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.transport_flux_package = (
            HorizontalScalarFluxPackage.muscl_horizontal(
                blend=float(muscl_blend),
                limiter_epsilon=float(muscl_limiter_epsilon),
            )
            if transport_flux_package is None
            else transport_flux_package
        )
        if self.transport_flux_package.scheme != "muscl":
            raise ValueError("DryHydrostaticFluxPackageMUSCLDynamics requires a MUSCL flux package")

    def _mtheta_tendency(
        self,
        state: State,
        u_e: torch.Tensor,
        theta_i: torch.Tensor,
        dp: torch.Tensor,
        *,
        dp_e: torch.Tensor | None = None,
        mass_flux: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if mass_flux is None:
            dp_e = state.mesh.cell_to_edge(dp) if dp_e is None else dp_e
            mass_flux = dp_e * u_e
        return self.transport_flux_package.scalar_tendency(state, theta_i, mass_flux)


@register_component("dry_hydrostatic_soft_muscl_mtheta_transport")
class DryHydrostaticSoftMUSCLMthetaTransportDynamics(DryHydrostaticMUSCLMthetaTransportDynamics):
    """Softer active ``mtheta_i`` MUSCL diagnostic branch.

    This branch keeps the same conservative finite-volume flux form as the
    full-strength MUSCL probe, but changes two independent knobs:

    - ``muscl_blend`` blends the high-order edge value back toward donor-cell
      upwind, reducing the amount of antidiffusive correction in active theta.
    - ``muscl_limiter='tanh_bound'`` replaces the hard min-ratio transition
      with a smooth bounded saturation near local extrema.

    It is intentionally a separate named branch so previous ``muscl`` results
    remain reproducible.
    """

    recipe_name = "soft_muscl_mtheta_transport"

    def __init__(
        self,
        *args,
        muscl_blend: float = 0.5,
        muscl_limiter: str = "tanh_bound",
        **kwargs,
    ) -> None:
        super().__init__(*args, muscl_blend=muscl_blend, muscl_limiter=muscl_limiter, **kwargs)








@register_component("dry_hydrostatic_fct_mtheta_transport")
class DryHydrostaticFCTMthetaTransportDynamics(DryHydrostaticDynamics):
    """Experimental active-scalar FCT probe for CP ``mtheta_i`` advection.

    This is a diagnostic branch, not a default candidate. It limits the
    combined horizontal and vertical advective fluxes for recovered
    ``theta_i = mtheta_i / dm`` while leaving the existing coordinate-work
    correction untouched. Because that coordinate-work correction was derived
    for donor-cell vertical scalar transport, this variant must pass separate
    energy and JW-A maintenance gates before it can be considered scientifically
    meaningful.
    """

    recipe_name = "fct_mtheta_transport"

    def __init__(self, *args, mtheta_tendency_dt: float | None = None, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._mtheta_fct = FCTTracerTransport(tendency_dt=mtheta_tendency_dt)

    def _mtheta_advection_tendency(
        self,
        state: State,
        theta_i: torch.Tensor,
        primitives: dict[str, torch.Tensor],
    ) -> torch.Tensor:
        mesh = state.mesh
        n_interfaces = state.column.n_levels + 1
        leading = theta_i.shape[:-2]
        batch = 1
        for size in leading:
            batch *= int(size)

        theta_b = theta_i.reshape(batch, mesh.nCells, n_interfaces)
        mass_flux = primitives["mass_flux"].reshape(batch, mesh.nEdges, state.column.n_levels)
        omega = primitives["omega"].reshape(batch, mesh.nCells, state.column.n_levels)
        dm = primitives["dm"].reshape(batch, mesh.nCells, n_interfaces)
        dB = (state.column.B_interface[1:] - state.column.B_interface[:-1]).to(
            dtype=theta_i.dtype,
            device=theta_i.device,
        )
        ddp_dt = (dB.reshape(1, 1, state.column.n_levels) * primitives["dps_dt"]).reshape(
            batch,
            mesh.nCells,
            state.column.n_levels,
        )
        ddm_dt = self._interface_mass(ddp_dt)

        c0 = mesh.cellsOnEdge[:, 0]
        c1 = mesh.cellsOnEdge[:, 1]
        theta0 = theta_b[:, c0, :]
        theta1 = theta_b[:, c1, :]

        low_h = torch.zeros(
            batch,
            mesh.nEdges,
            n_interfaces,
            1,
            dtype=theta_i.dtype,
            device=theta_i.device,
        )
        high_h = torch.zeros_like(low_h)
        theta_upper_low = torch.where(mass_flux >= 0.0, theta0[..., :-1], theta1[..., :-1])
        theta_lower_low = torch.where(mass_flux >= 0.0, theta0[..., 1:], theta1[..., 1:])
        theta_upper_high = 0.5 * (theta0[..., :-1] + theta1[..., :-1])
        theta_lower_high = 0.5 * (theta0[..., 1:] + theta1[..., 1:])
        low_h[..., :-1, 0] += 0.5 * mass_flux * theta_upper_low
        low_h[..., 1:, 0] += 0.5 * mass_flux * theta_lower_low
        high_h[..., :-1, 0] += 0.5 * mass_flux * theta_upper_high
        high_h[..., 1:, 0] += 0.5 * mass_flux * theta_lower_high

        low_v = torch.zeros(
            batch,
            mesh.nCells,
            n_interfaces + 1,
            1,
            dtype=theta_i.dtype,
            device=theta_i.device,
        )
        high_v = torch.zeros_like(low_v)
        theta_up = torch.where(omega >= 0.0, theta_b[..., :-1], theta_b[..., 1:])
        theta_center = 0.5 * (theta_b[..., :-1] + theta_b[..., 1:])
        low_v[:, :, 1:-1, 0] = omega * theta_up
        high_v[:, :, 1:-1, 0] = omega * theta_center

        limited_h, limited_v = self._mtheta_fct._fct_limiter(
            state,
            theta_b[..., None],
            dm,
            ddm_dt,
            low_h,
            high_h,
            low_v,
            high_v,
            dt=self._mtheta_fct._stage_dt(self),
        )
        horiz = -self._mtheta_fct._div_tracer_flux(mesh, limited_h)[..., 0]
        vert = (limited_v[:, :, :-1, 0] - limited_v[:, :, 1:, 0])
        return (horiz + vert).reshape(*leading, mesh.nCells, n_interfaces)

    def mtheta_tendency_terms(
        self,
        state: State,
        *,
        primitives: dict[str, torch.Tensor] | None = None,
    ) -> dict[str, torch.Tensor]:
        ctx = self._transport_primitives(state) if primitives is None else primitives
        theta = ctx["theta"]
        advective = self._mtheta_advection_tendency(state, theta, ctx)
        coord_work = self._canonical_mtheta_coordinate_work(state, theta, ctx["omega"], primitives=ctx)
        zero = torch.zeros_like(advective)
        return {
            "horizontal": zero,
            "vertical": zero,
            "advective": advective,
            "coord_work": coord_work,
            "total": advective + coord_work,
        }


@register_component("dry_hydrostatic_log_pressure_sigma")
class DryHydrostaticLogPressureSigmaDynamics(DryHydrostaticDynamics):
    """Explicit dry dynamics using a log-pressure sigma PGF correction.

    The canonical base dynamics already dispatches to this correction for GL2
    terrain-aware columns. This named variant remains useful for side-by-side
    audits and configs that should force the log-pressure sigma PGF directly.
    """

    recipe_name = "log_pressure_sigma"

    def __init__(self, *args, **kwargs) -> None:
        if "pressure_gradient" in kwargs:
            raise TypeError(
                "DryHydrostaticLogPressureSigmaDynamics owns its pressure-gradient strategy"
            )
        super().__init__(
            *args,
            pressure_gradient=LogPressureSigmaPressureGradient(),
            **kwargs,
        )


@register_component("dry_hydrostatic_exner_secant_sigma")
class DryHydrostaticExnerSecantSigmaDynamics(DryHydrostaticDynamics):
    """Explicit dry dynamics using an Exner-secant sigma PGF correction.

    This is an opt-in terrain-PGF research branch.  It keeps the same
    full-level geopotential placement as the active column but writes the
    sigma correction in the hydrostatic variable used by the column,
    ``theta_eff * grad(Pi_full)``, instead of the isothermal identity
    ``R_d T * grad(log p_full)``.
    """

    recipe_name = "exner_secant_sigma"

    def __init__(self, *args, **kwargs) -> None:
        if "pressure_gradient" in kwargs:
            raise TypeError(
                "DryHydrostaticExnerSecantSigmaDynamics owns its pressure-gradient strategy"
            )
        super().__init__(
            *args,
            pressure_gradient=ExnerSecantSigmaPressureGradient(),
            **kwargs,
        )


@register_component("dry_hydrostatic_velocity_hyperdiffusion")
class DryHydrostaticVelocityHyperdiffusionDynamics(DryHydrostaticDynamics):
    """Experimental dry dynamics with opt-in velocity hyperdiffusion.

    This branch adds a scale-selective fourth-order edge-velocity damping term
    through the mesh-owned vector Laplacian. It is intended for JW/HS
    regularization experiments, not for proving topographic-rest exactness.
    """

    recipe_name = "velocity_hyperdiffusion"

    def __init__(
        self,
        *args,
        hyperdiffusion_nu4: float | None = None,
        hyperdiffusion_tau_grid_seconds: float | None = 168.0 * 3600.0,
        hyperdiffusion_reference_length: float | None = None,
        hyperdiffusion_grid_eigenvalue_factor: float = 4096.0,
        **kwargs,
    ) -> None:
        if "momentum_diffusion" in kwargs:
            raise TypeError(
                "DryHydrostaticVelocityHyperdiffusionDynamics owns its momentum_diffusion strategy"
            )
        if hyperdiffusion_nu4 is not None:
            hyperdiffusion_tau_grid_seconds = None
        super().__init__(
            *args,
            momentum_diffusion=VelocityHyperdiffusion(
                nu4=hyperdiffusion_nu4,
                tau_grid_seconds=hyperdiffusion_tau_grid_seconds,
                reference_length=hyperdiffusion_reference_length,
                grid_eigenvalue_factor=hyperdiffusion_grid_eigenvalue_factor,
            ),
            **kwargs,
        )


@register_component("dry_hydrostatic_divergence_damping")
class DryHydrostaticDivergenceDampingDynamics(DryHydrostaticDynamics):
    """Experimental dry dynamics with opt-in biharmonic divergence damping."""

    recipe_name = "divergence_damping"

    def __init__(
        self,
        *args,
        divergence_damping_nu4: float | None = None,
        divergence_damping_tau_grid_seconds: float | None = 24.0 * 3600.0,
        divergence_damping_reference_length: float | None = None,
        divergence_damping_grid_eigenvalue_factor: float = 4096.0,
        **kwargs,
    ) -> None:
        if "momentum_diffusion" in kwargs:
            raise TypeError("DryHydrostaticDivergenceDampingDynamics owns its momentum_diffusion strategy")
        if divergence_damping_nu4 is not None:
            divergence_damping_tau_grid_seconds = None
        super().__init__(
            *args,
            momentum_diffusion=BiharmonicDivergenceDamping(
                nu4=divergence_damping_nu4,
                tau_grid_seconds=divergence_damping_tau_grid_seconds,
                reference_length=divergence_damping_reference_length,
                grid_eigenvalue_factor=divergence_damping_grid_eigenvalue_factor,
            ),
            **kwargs,
        )


@register_component("dry_hydrostatic_top_sponge")
class DryHydrostaticTopSpongeDynamics(DryHydrostaticDynamics):
    """Experimental dry dynamics with opt-in top Rayleigh sponge."""

    recipe_name = "top_sponge"

    def __init__(
        self,
        *args,
        sponge_tau_top_seconds: float = 24.0 * 3600.0,
        sponge_top_fraction: float = 0.25,
        sponge_taper_power: float = 2.0,
        **kwargs,
    ) -> None:
        if "momentum_diffusion" in kwargs:
            raise TypeError("DryHydrostaticTopSpongeDynamics owns its momentum_diffusion strategy")
        super().__init__(
            *args,
            momentum_diffusion=TopRayleighSponge(
                tau_top_seconds=float(sponge_tau_top_seconds),
                top_fraction=float(sponge_top_fraction),
                taper_power=float(sponge_taper_power),
            ),
            **kwargs,
        )


@dataclass(frozen=True)
class DryOperatorSet:
    """A named, compatible dry-dynamics operator bundle.

    This is construction metadata, not a second runtime dispatch layer.  The
    returned object is still a ``DryHydrostaticDynamics`` subclass with the
    ordinary fast ``tendency(state)`` interface.
    """

    name: str
    dynamics_type: type[DryHydrostaticDynamics]
    description: str

    def build(self, **kwargs: Any) -> DryHydrostaticDynamics:
        """Construct this named operator set with its explicit parameters."""

        return self.dynamics_type(**kwargs)


_DRY_OPERATOR_SETS: dict[str, DryOperatorSet] = {
    "conservative_diagonal": DryOperatorSet("conservative_diagonal", DryHydrostaticConservativeDynamics, "matched dry operators with diagonal kinetic energy"),
    "canonical": DryOperatorSet("canonical", DryHydrostaticDynamics, "canonical upwind dry core"),
    "virtual_temperature": DryOperatorSet(
        "virtual_temperature", DryHydrostaticVirtualTemperatureDynamics, "virtual-temperature hydrostatics"
    ),
    "unscaled_coord_work": DryOperatorSet(
        "unscaled_coord_work", DryHydrostaticUnscaledCoordWorkDynamics, "archived coordinate-work comparison"
    ),
    "mild_vertical_momentum": DryOperatorSet(
        "mild_vertical_momentum", DryHydrostaticMildVerticalMomentumDynamics, "mild vertical wind transport"
    ),
    "centered_mtheta_transport": DryOperatorSet(
        "centered_mtheta_transport", DryHydrostaticCenteredMthetaTransportDynamics, "centered CP scalar transport"
    ),
    "kite_centered_mtheta_transport": DryOperatorSet(
        "kite_centered_mtheta_transport", DryHydrostaticKiteCenteredMthetaTransportDynamics, "kite-remapped CP scalar transport"
    ),
    "mass_kite_centered_mtheta_transport": DryOperatorSet(
        "mass_kite_centered_mtheta_transport",
        DryHydrostaticMassKiteCenteredMthetaTransportDynamics,
        "mass-weighted kite CP scalar transport",
    ),
    "muscl_mtheta_transport": DryOperatorSet(
        "muscl_mtheta_transport", DryHydrostaticMUSCLMthetaTransportDynamics, "legacy MUSCL CP scalar transport"
    ),
    "flux_package_muscl": DryOperatorSet(
        "flux_package_muscl", DryHydrostaticFluxPackageMUSCLDynamics, "reusable MUSCL flux-package transport"
    ),
    "soft_muscl_mtheta_transport": DryOperatorSet(
        "soft_muscl_mtheta_transport", DryHydrostaticSoftMUSCLMthetaTransportDynamics, "soft-limited MUSCL transport"
    ),
    "fct_mtheta_transport": DryOperatorSet(
        "fct_mtheta_transport", DryHydrostaticFCTMthetaTransportDynamics, "FCT CP scalar transport"
    ),
    "log_pressure_sigma": DryOperatorSet(
        "log_pressure_sigma", DryHydrostaticLogPressureSigmaDynamics, "log-pressure sigma PGF"
    ),
    "exner_secant_sigma": DryOperatorSet(
        "exner_secant_sigma", DryHydrostaticExnerSecantSigmaDynamics, "Exner-secant sigma PGF"
    ),
    "velocity_hyperdiffusion": DryOperatorSet(
        "velocity_hyperdiffusion", DryHydrostaticVelocityHyperdiffusionDynamics, "opt-in velocity hyperdiffusion"
    ),
    "divergence_damping": DryOperatorSet(
        "divergence_damping", DryHydrostaticDivergenceDampingDynamics, "opt-in divergence damping"
    ),
    "top_sponge": DryOperatorSet("top_sponge", DryHydrostaticTopSpongeDynamics, "opt-in top Rayleigh sponge"),
}


def dry_operator_sets() -> dict[str, DryOperatorSet]:
    """Return the named dry bundles without exposing the mutable registry."""

    return dict(_DRY_OPERATOR_SETS)


def get_dry_operator_set(name: str) -> DryOperatorSet:
    """Look up one named dry operator bundle with a helpful error."""

    try:
        return _DRY_OPERATOR_SETS[str(name)]
    except KeyError as error:
        available = ", ".join(sorted(_DRY_OPERATOR_SETS))
        raise ValueError(f"unknown dry operator set {name!r}; available: {available}") from error


def build_dry_dynamics(operator_set: str = "canonical", **kwargs: Any) -> DryHydrostaticDynamics:
    """Build the canonical dry RHS with one named compatible operator set."""

    return get_dry_operator_set(operator_set).build(**kwargs)


__all__ = [
    "DryOperatorSet",
    "DryHydrostaticCenteredMthetaTransportDynamics",
    "DryHydrostaticDivergenceDampingDynamics",
    "DryHydrostaticExnerSecantSigmaDynamics",
    "DryHydrostaticFCTMthetaTransportDynamics",
    "DryHydrostaticFluxPackageMUSCLDynamics",
    "DryHydrostaticKiteCenteredMthetaTransportDynamics",
    "DryHydrostaticLogPressureSigmaDynamics",
    "DryHydrostaticMassKiteCenteredMthetaTransportDynamics",
    "DryHydrostaticMildVerticalMomentumDynamics",
    "DryHydrostaticMUSCLMthetaTransportDynamics",
    "DryHydrostaticSoftMUSCLMthetaTransportDynamics",
    "DryHydrostaticTopSpongeDynamics",
    "DryHydrostaticUnscaledCoordWorkDynamics",
    "DryHydrostaticVelocityHyperdiffusionDynamics",
    "DryHydrostaticVirtualTemperatureDynamics",
    "build_dry_dynamics",
    "dry_operator_sets",
    "get_dry_operator_set",
]
