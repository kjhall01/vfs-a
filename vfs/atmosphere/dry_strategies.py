"""Narrow strategy objects for existing dry-dynamics variants."""

from __future__ import annotations

import math
from typing import Any

import torch

from ..core.state import State
from ..moisture import (
    EPSILON_WATER_DRY_AIR as _EPSILON_WATER_DRY_AIR,
    VAPOR_QUANTITY_ATTR as _VAPOR_QUANTITY_ATTR,
    VAPOR_QUANTITY_DRY_MIXING_RATIO as _VAPOR_QUANTITY_DRY_MIXING_RATIO,
    VAPOR_QUANTITY_SPECIFIC_HUMIDITY as _VAPOR_QUANTITY_SPECIFIC_HUMIDITY,
    VAPOR_TRACER_INDEX_ATTR as _VAPOR_TRACER_INDEX_ATTR,
)


class UpwindTracerTransport:
    """Conservative first-order upwind passive tracer transport."""

    name = "upwind"

    def tendency(
        self,
        dyn: Any,
        state: State,
        u_e: torch.Tensor,
        q: torch.Tensor,
        dp: torch.Tensor,
        mu: torch.Tensor,
    ) -> torch.Tensor:
        mesh = state.mesh
        n_levels = state.column.n_levels
        n_tracers = q.shape[-1]
        leading = q.shape[:-3]

        dp_e = mesh.cell_to_edge(dp)
        mass_flux = dp_e * u_e

        c0 = mesh.cellsOnEdge[:, 0]
        c1 = mesh.cellsOnEdge[:, 1]
        q0 = q[..., c0, :, :]
        q1 = q[..., c1, :, :]
        q_edge = torch.where(mass_flux[..., None] >= 0.0, q0, q1)
        flux = mass_flux[..., None] * q_edge

        flat_flux = flux.reshape(*leading, mesh.nEdges, n_levels * n_tracers)
        horiz = -mesh.div(flat_flux).reshape(*leading, mesh.nCells, n_levels, n_tracers)

        q_int = torch.zeros(
            *leading,
            mesh.nCells,
            n_levels + 1,
            n_tracers,
            dtype=q.dtype,
            device=q.device,
        )
        if n_levels > 1:
            q_down = q[..., :-1, :]
            q_up = q[..., 1:, :]
            q_int[..., 1:-1, :] = torch.where(mu[..., 1:-1, None] >= 0.0, q_down, q_up)
        flux_v = mu[..., None] * q_int
        vert = flux_v[..., :-1, :] - flux_v[..., 1:, :]

        return horiz + vert


class EdgeUpwindVerticalMomentumTransport:
    """Current edge-remapped donor vertical momentum transport."""

    name = "edge_upwind"

    @staticmethod
    def _upwind_full_to_interface(
        u_e: torch.Tensor, interface_flux: torch.Tensor
    ) -> torch.Tensor:
        top = u_e[..., :1]
        bot = u_e[..., -1:]
        if u_e.shape[-1] > 1:
            donor = torch.where(
                interface_flux[..., 1:-1] >= 0.0,
                u_e[..., :-1],
                u_e[..., 1:],
            )
            return torch.cat([top, donor, bot], dim=-1)
        return torch.cat([top, bot], dim=-1)

    def tendency(
        self,
        dyn: Any,
        state: State,
        u_e: torch.Tensor,
        mu: torch.Tensor,
        dp: torch.Tensor,
        *,
        dp_e: torch.Tensor | None = None,
    ) -> torch.Tensor:
        del dyn
        mu_e = state.cell_to_edge(mu)  # [..., nEdges, N+1]
        dp_e = state.cell_to_edge(dp) if dp_e is None else dp_e  # [..., nEdges, N]
        u_adv = self._upwind_full_to_interface(u_e, mu_e)

        # Raw-velocity advective/remap form, not a mass-weighted content update.
        return (
            mu_e[..., :-1] * (u_adv[..., :-1] - u_e)
            - mu_e[..., 1:] * (u_adv[..., 1:] - u_e)
        ) / dp_e


class MildCellSupportedVerticalMomentumTransport(EdgeUpwindVerticalMomentumTransport):
    """Archived mild left/right cell-supported vertical momentum flux.

    This builds donor-cell vertical momentum fluxes from the two cell-side
    interface mass fluxes adjacent to each edge, then geometry-averages those
    fluxes back to the edge.  The quotient/compressive correction still uses
    the native edge-averaged interface mass flux.
    """

    name = "mild_cell_supported"

    @staticmethod
    def _edge_side_average(
        state: State, left: torch.Tensor, right: torch.Tensor
    ) -> torch.Tensor:
        mesh = state.mesh
        dtype = left.dtype
        device = left.device
        shape = (1,) * (left.ndim - 2) + (mesh.nEdges, 1)
        a0 = mesh.edge_side_area[:, 0].to(dtype=dtype, device=device).reshape(shape)
        a1 = mesh.edge_side_area[:, 1].to(dtype=dtype, device=device).reshape(shape)
        averaged = (left * a0 + right * a1) / (a0 + a1).clamp_min(
            torch.finfo(dtype).tiny
        )

        cells = mesh.cellsOnEdge.to(device=device)
        valid0 = (cells[:, 0] >= 0).reshape(shape)
        valid1 = (cells[:, 1] >= 0).reshape(shape)
        one_sided = torch.where(valid0, left, right)
        return torch.where(valid0 & valid1, averaged, one_sided)

    def tendency(
        self,
        dyn: Any,
        state: State,
        u_e: torch.Tensor,
        mu: torch.Tensor,
        dp: torch.Tensor,
        *,
        dp_e: torch.Tensor | None = None,
    ) -> torch.Tensor:
        del dyn
        mesh = state.mesh
        safe_cells = mesh.safe_cells_on_edge.to(device=mu.device).long()
        mu_left = mu[..., safe_cells[:, 0], :]
        mu_right = mu[..., safe_cells[:, 1], :]

        u_if_left = self._upwind_full_to_interface(u_e, mu_left)
        u_if_right = self._upwind_full_to_interface(u_e, mu_right)
        momentum_flux_edge = self._edge_side_average(
            state,
            mu_left * u_if_left,
            mu_right * u_if_right,
        )

        mu_e = state.cell_to_edge(mu)
        dp_e = state.cell_to_edge(dp) if dp_e is None else dp_e
        return (
            momentum_flux_edge[..., :-1]
            - momentum_flux_edge[..., 1:]
            + u_e * (mu_e[..., 1:] - mu_e[..., :-1])
        ) / dp_e


class BlendedMildUpwindVerticalMomentumTransport(EdgeUpwindVerticalMomentumTransport):
    """Fixed-alpha blend between canonical upwind and mild vertical momentum.

    This opt-in diagnostic/oracle strategy does not define a new transport
    formula. It blends the two existing vertical momentum tendencies as
    ``upwind + alpha * (mild - upwind)``.
    """

    name = "blended_mild_upwind"

    def __init__(self, alpha: float = 0.0, *, alpha_max: float = 1.0) -> None:
        alpha_max_f = float(alpha_max)
        if alpha_max_f < 0.0 or alpha_max_f > 1.0:
            raise ValueError("alpha_max must be between 0 and 1")
        alpha_f = float(alpha)
        if alpha_f < 0.0 or alpha_f > alpha_max_f:
            raise ValueError("alpha must be between 0 and alpha_max")
        self.alpha = alpha_f
        self.alpha_max = alpha_max_f
        self.upwind = EdgeUpwindVerticalMomentumTransport()
        self.mild = MildCellSupportedVerticalMomentumTransport()
        self.last_alpha_stats: dict[str, float] = {}

    def _alpha_tensor(self, like: torch.Tensor) -> torch.Tensor:
        alpha = min(max(float(self.alpha), 0.0), float(self.alpha_max))
        self.last_alpha_stats = {
            "alpha_mean": alpha,
            "alpha_min": alpha,
            "alpha_max": alpha,
        }
        return torch.as_tensor(alpha, dtype=like.dtype, device=like.device)

    def tendency(
        self,
        dyn: Any,
        state: State,
        u_e: torch.Tensor,
        mu: torch.Tensor,
        dp: torch.Tensor,
        *,
        dp_e: torch.Tensor | None = None,
    ) -> torch.Tensor:
        upwind = self.upwind.tendency(dyn, state, u_e, mu, dp, dp_e=dp_e)
        alpha = self._alpha_tensor(upwind)
        if bool(torch.all(alpha == 0.0).item()):
            return upwind
        mild = self.mild.tendency(dyn, state, u_e, mu, dp, dp_e=dp_e)
        return upwind + alpha * (mild - upwind)


class FCTTracerTransport:
    """Experimental bounded flux-corrected passive tracer transport.

    This ports the old survivor-branch Stage510 idea into the current strategy
    system.  The low-order flux is donor-cell/upwind; the high-order flux is a
    centered tracer flux; the returned flux adds only the admissible
    antidiffusive part.  The limiter is conservative because it limits a single
    shared edge/interface flux before taking divergence.

    The limiter needs the explicit stage length to decide how much
    antidiffusive mass can be accepted without leaving local tracer bounds.
    Current integrators expose that as ``dyn._current_tendency_dt`` while
    preserving the ordinary ``dyn.tendency(state)`` public API.
    """

    name = "fct"

    def __init__(self, tendency_dt: float | None = None) -> None:
        self.tendency_dt = None if tendency_dt is None else float(tendency_dt)

    def _stage_dt(self, dyn: Any) -> float:
        value = getattr(dyn, "_current_tendency_dt", None)
        if value is None:
            value = self.tendency_dt
        if value is None:
            raise ValueError(
                "FCTTracerTransport requires the current explicit stage dt. "
                "Use vfs.core.integrators.ssprk3/rk4/forward_euler or construct "
                "FCTTracerTransport(tendency_dt=...)."
            )
        dt = float(value)
        if dt <= 0.0:
            raise ValueError("FCTTracerTransport tendency_dt must be positive")
        return dt

    @staticmethod
    def _div_tracer_flux(mesh, flux: torch.Tensor) -> torch.Tensor:
        """Divergence for flux shaped ``[B, nEdges, nLevels, nTracers]``."""

        batch, n_edges, n_levels, n_tracers = flux.shape
        div = mesh.div(flux.reshape(batch, n_edges, n_levels * n_tracers))
        return div.reshape(batch, mesh.nCells, n_levels, n_tracers)

    @staticmethod
    def _local_bounds(
        state: State, q: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Cell/level/tracer bounds over horizontal neighbors and adjacent levels."""

        mesh = state.mesh
        batch, n_cells, n_levels, _n_tracers = q.shape
        device = q.device
        cell_index = torch.arange(n_cells, device=device, dtype=torch.long).view(
            n_cells, 1
        )
        safe_edges = mesh.safe_edges_on_cell.to(device=device).long()
        edge_cells = mesh.safe_cells_on_edge.to(device=device).long()
        neighbor0 = edge_cells[safe_edges, 0]
        neighbor1 = edge_cells[safe_edges, 1]
        neighbor_index = torch.where(neighbor0 == cell_index, neighbor1, neighbor0)
        valid = mesh.valid_edges_on_cell.to(device=device)
        neighbor_index = torch.where(valid, neighbor_index, cell_index)

        neighbor_values = q[:, neighbor_index, :, :]
        center_values = q.unsqueeze(2)
        neighbor_values = torch.where(
            valid.view(1, n_cells, -1, 1, 1),
            neighbor_values,
            center_values,
        )
        q_min = torch.minimum(q, neighbor_values.min(dim=2).values)
        q_max = torch.maximum(q, neighbor_values.max(dim=2).values)

        if n_levels > 1:
            upper = torch.cat([q[:, :, :1, :], q[:, :, :-1, :]], dim=2)
            lower = torch.cat([q[:, :, 1:, :], q[:, :, -1:, :]], dim=2)
            q_min = torch.minimum(torch.minimum(q_min, upper), lower)
            q_max = torch.maximum(torch.maximum(q_max, upper), lower)
        return q_min, q_max

    def _fct_limiter(
        self,
        state: State,
        q: torch.Tensor,
        dp: torch.Tensor,
        ddp_dt: torch.Tensor,
        low_h: torch.Tensor,
        high_h: torch.Tensor,
        low_v: torch.Tensor,
        high_v: torch.Tensor,
        *,
        dt: float,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return limited horizontal/vertical content fluxes."""

        mesh = state.mesh
        batch, n_cells, n_levels, n_tracers = q.shape
        device = q.device
        dtype = q.dtype

        horizontal_tendency_low = -self._div_tracer_flux(mesh, low_h)
        vertical_tendency_low = low_v[:, :, :-1, :] - low_v[:, :, 1:, :]

        area = mesh.cell_area.to(device=device, dtype=dtype).view(1, n_cells, 1, 1)
        edge_length = mesh.dv_edge.to(device=device, dtype=dtype).view(
            1, mesh.nEdges, 1, 1
        )
        content_amount = area * dp[..., None] * q
        mass_amount_new = (
            area
            * (dp + float(dt) * ddp_dt).clamp_min(torch.finfo(dtype).tiny)[..., None]
        )
        mass_amount_new = mass_amount_new.expand_as(q)
        content_amount_low = content_amount + float(dt) * area * (
            horizontal_tendency_low + vertical_tendency_low
        )
        q_low = content_amount_low / mass_amount_new

        q_min, q_max = self._local_bounds(state, q)
        q_min = torch.minimum(q_min, q_low)
        q_max = torch.maximum(q_max, q_low)

        anti_h = edge_length * (high_h - low_h)
        anti_v = area * (high_v - low_v)

        node_ids = torch.arange(
            n_cells * n_levels * n_tracers, device=device, dtype=torch.long
        ).reshape(n_cells, n_levels, n_tracers)
        left_cells = mesh.safe_cells_on_edge[:, 0].to(device=device).long()
        right_cells = mesh.safe_cells_on_edge[:, 1].to(device=device).long()
        left_nodes = node_ids[left_cells]
        right_nodes = node_ids[right_cells]
        horizontal_valid = (mesh.cellsOnEdge[:, 0] >= 0).to(device=device) & (
            mesh.cellsOnEdge[:, 1] >= 0
        ).to(device=device)

        if n_levels > 1:
            upper_nodes = node_ids[:, :-1, :]
            lower_nodes = node_ids[:, 1:, :]
        else:
            upper_nodes = torch.zeros(
                (n_cells, 0, n_tracers), device=device, dtype=torch.long
            )
            lower_nodes = torch.zeros(
                (n_cells, 0, n_tracers), device=device, dtype=torch.long
            )

        flat_mass = mass_amount_new.reshape(batch, -1)
        flat_q_low = q_low.reshape(batch, -1)
        flat_q_min = q_min.reshape(batch, -1)
        flat_q_max = q_max.reshape(batch, -1)
        left_flat = left_nodes.reshape(-1)
        right_flat = right_nodes.reshape(-1)
        horiz_valid_flat = (
            horizontal_valid[:, None, None].expand_as(left_nodes).reshape(-1)
        )
        upper_flat = upper_nodes.reshape(-1)
        lower_flat = lower_nodes.reshape(-1)

        lambda_h = torch.ones_like(anti_h)
        lambda_v = torch.ones_like(anti_v)

        for batch_index in range(batch):
            q_plus = torch.zeros((flat_q_low.shape[-1],), device=device, dtype=dtype)
            q_minus = torch.zeros_like(q_plus)

            horizontal_amount = (float(dt) * anti_h[batch_index]).reshape(-1)
            amount = horizontal_amount[horiz_valid_flat]
            left_valid = left_flat[horiz_valid_flat]
            right_valid = right_flat[horiz_valid_flat]
            q_plus.scatter_add_(0, left_valid, torch.clamp(-amount, min=0.0))
            q_minus.scatter_add_(0, left_valid, torch.clamp(amount, min=0.0))
            q_plus.scatter_add_(0, right_valid, torch.clamp(amount, min=0.0))
            q_minus.scatter_add_(0, right_valid, torch.clamp(-amount, min=0.0))

            if n_levels > 1:
                vertical_amount = (float(dt) * anti_v[batch_index, :, 1:-1, :]).reshape(
                    -1
                )
                q_plus.scatter_add_(
                    0, upper_flat, torch.clamp(-vertical_amount, min=0.0)
                )
                q_minus.scatter_add_(
                    0, upper_flat, torch.clamp(vertical_amount, min=0.0)
                )
                q_plus.scatter_add_(
                    0, lower_flat, torch.clamp(vertical_amount, min=0.0)
                )
                q_minus.scatter_add_(
                    0, lower_flat, torch.clamp(-vertical_amount, min=0.0)
                )

            p_plus = (flat_q_max[batch_index] - flat_q_low[batch_index]).clamp_min(
                0.0
            ) * flat_mass[batch_index]
            p_minus = (flat_q_low[batch_index] - flat_q_min[batch_index]).clamp_min(
                0.0
            ) * flat_mass[batch_index]
            r_plus = torch.where(
                q_plus > 0.0,
                torch.minimum(
                    torch.ones_like(p_plus),
                    p_plus / q_plus.clamp_min(torch.finfo(dtype).tiny),
                ),
                torch.ones_like(p_plus),
            )
            r_minus = torch.where(
                q_minus > 0.0,
                torch.minimum(
                    torch.ones_like(p_minus),
                    p_minus / q_minus.clamp_min(torch.finfo(dtype).tiny),
                ),
                torch.ones_like(p_minus),
            )

            left_r_plus = r_plus[left_nodes]
            left_r_minus = r_minus[left_nodes]
            right_r_plus = r_plus[right_nodes]
            right_r_minus = r_minus[right_nodes]
            candidate_h = anti_h[batch_index]
            horizontal_lambda = torch.where(
                candidate_h >= 0.0,
                torch.minimum(left_r_minus, right_r_plus),
                torch.minimum(left_r_plus, right_r_minus),
            )
            lambda_h[batch_index] = torch.where(
                horizontal_valid[:, None, None],
                horizontal_lambda,
                torch.zeros_like(horizontal_lambda),
            )

            if n_levels > 1:
                upper_r_plus = r_plus[upper_nodes]
                upper_r_minus = r_minus[upper_nodes]
                lower_r_plus = r_plus[lower_nodes]
                lower_r_minus = r_minus[lower_nodes]
                candidate_v = anti_v[batch_index, :, 1:-1, :]
                lambda_v[batch_index, :, 1:-1, :] = torch.where(
                    candidate_v >= 0.0,
                    torch.minimum(upper_r_minus, lower_r_plus),
                    torch.minimum(upper_r_plus, lower_r_minus),
                )

        limited_h_amount = edge_length * low_h + lambda_h * anti_h
        limited_v_amount = area * low_v + lambda_v * anti_v
        return (
            limited_h_amount / edge_length.clamp_min(torch.finfo(dtype).tiny),
            limited_v_amount / area.clamp_min(torch.finfo(dtype).tiny),
        )

    def tendency(
        self,
        dyn: Any,
        state: State,
        u_e: torch.Tensor,
        q: torch.Tensor,
        dp: torch.Tensor,
        mu: torch.Tensor,
    ) -> torch.Tensor:
        dt = self._stage_dt(dyn)
        mesh = state.mesh
        n_levels = state.column.n_levels
        n_tracers = q.shape[-1]
        leading = q.shape[:-3]
        batch = 1
        for size in leading:
            batch *= int(size)

        q_b = q.reshape(batch, mesh.nCells, n_levels, n_tracers)
        dp_b = dp.reshape(batch, mesh.nCells, n_levels)
        mu_b = mu.reshape(batch, mesh.nCells, n_levels + 1)
        dp_e = mesh.cell_to_edge(dp)
        mass_flux = (dp_e * u_e).reshape(batch, mesh.nEdges, n_levels)
        layer_div = mesh.div(mass_flux)
        dps_dt = -layer_div.sum(dim=-1, keepdim=True)
        dB = (state.column.B_interface[1:] - state.column.B_interface[:-1]).to(
            dtype=dp.dtype,
            device=dp.device,
        )
        ddp_dt = (dB.reshape(1, 1, n_levels) * dps_dt).reshape(
            batch, mesh.nCells, n_levels
        )

        c0 = mesh.cellsOnEdge[:, 0]
        c1 = mesh.cellsOnEdge[:, 1]
        q0 = q_b[:, c0, :, :]
        q1 = q_b[:, c1, :, :]
        q_edge_low = torch.where(mass_flux[..., None] >= 0.0, q0, q1)
        q_edge_high = 0.5 * (q0 + q1)
        low_h = mass_flux[..., None] * q_edge_low
        high_h = mass_flux[..., None] * q_edge_high

        q_int_low = torch.zeros(
            batch,
            mesh.nCells,
            n_levels + 1,
            n_tracers,
            dtype=q.dtype,
            device=q.device,
        )
        q_int_high = torch.zeros_like(q_int_low)
        if n_levels > 1:
            q_down = q_b[:, :, :-1, :]
            q_up = q_b[:, :, 1:, :]
            q_int_low[:, :, 1:-1, :] = torch.where(
                mu_b[:, :, 1:-1, None] >= 0.0, q_down, q_up
            )
            q_int_high[:, :, 1:-1, :] = 0.5 * (q_down + q_up)
        low_v = mu_b[..., None] * q_int_low
        high_v = mu_b[..., None] * q_int_high

        limited_h, limited_v = self._fct_limiter(
            state,
            q_b,
            dp_b,
            ddp_dt,
            low_h,
            high_h,
            low_v,
            high_v,
            dt=dt,
        )
        horiz = -self._div_tracer_flux(mesh, limited_h)
        vert = limited_v[:, :, :-1, :] - limited_v[:, :, 1:, :]
        return (horiz + vert).reshape(*leading, mesh.nCells, n_levels, n_tracers)


class VelocityHyperdiffusion:
    """Opt-in fourth-order scale-selective damping for edge velocity.

    The operator is built from the mesh-owned compatible vector Laplacian:

    ``du/dt = -nu4 * L(L(u))``.

    Since ``L`` is negative semidefinite under the VFS edge metric, this
    biharmonic form damps grid-scale kinetic energy while leaving the dry-core
    default unchanged unless the strategy is explicitly installed.
    """

    name = "velocity_hyperdiffusion"

    def __init__(
        self,
        *,
        nu4: float | None = None,
        tau_grid_seconds: float | None = None,
        reference_length: float | None = None,
        grid_eigenvalue_factor: float = 4096.0,
    ) -> None:
        if nu4 is not None and tau_grid_seconds is not None:
            raise ValueError("Specify either nu4 or tau_grid_seconds, not both")
        if nu4 is None and tau_grid_seconds is None:
            tau_grid_seconds = 168.0 * 3600.0
        if nu4 is not None and float(nu4) < 0.0:
            raise ValueError("VelocityHyperdiffusion nu4 must be nonnegative")
        if tau_grid_seconds is not None and float(tau_grid_seconds) <= 0.0:
            raise ValueError("VelocityHyperdiffusion tau_grid_seconds must be positive")
        if reference_length is not None and float(reference_length) <= 0.0:
            raise ValueError("VelocityHyperdiffusion reference_length must be positive")
        if float(grid_eigenvalue_factor) <= 0.0:
            raise ValueError(
                "VelocityHyperdiffusion grid_eigenvalue_factor must be positive"
            )
        self.nu4 = None if nu4 is None else float(nu4)
        self.tau_grid_seconds = (
            None if tau_grid_seconds is None else float(tau_grid_seconds)
        )
        self.reference_length = (
            None if reference_length is None else float(reference_length)
        )
        self.grid_eigenvalue_factor = float(grid_eigenvalue_factor)

    def reference_length_tensor(self, state: State) -> torch.Tensor:
        """Return the horizontal length used to convert timescale to ``nu4``."""

        if self.reference_length is not None:
            return torch.as_tensor(
                self.reference_length,
                dtype=state.u_e.dtype,
                device=state.u_e.device,
            )
        area = state.mesh.cell_area.to(dtype=state.u_e.dtype, device=state.u_e.device)
        return area.mean().sqrt()

    def coefficient(self, state: State) -> torch.Tensor:
        """Return the biharmonic coefficient in mesh units."""

        if self.nu4 is not None:
            return torch.as_tensor(
                self.nu4, dtype=state.u_e.dtype, device=state.u_e.device
            )
        ell = self.reference_length_tensor(state)
        assert self.tau_grid_seconds is not None
        return ell.pow(4) / (
            float(self.grid_eigenvalue_factor) * float(self.tau_grid_seconds)
        )

    def tendency(self, dyn: Any, state: State) -> torch.Tensor:
        del dyn
        lap_u = state.vector_laplacian_edge(state.u_e)
        return -self.coefficient(state) * state.vector_laplacian_edge(lap_u)


class BiharmonicDivergenceDamping:
    """Opt-in fourth-order damping of divergent edge velocity.

    The operator is the divergent part of the vector biharmonic:

    ``du/dt = -nu4 * grad(div(grad(div(u))))``.

    With VFS's mimetic ``grad``/``div`` pairing, ``grad`` is the negative
    adjoint of ``div`` under the edge/cell metrics, so this term dissipates
    divergent kinetic energy without directly applying the curl part of the
    vector Laplacian.  It is intended as a diagnostic regularization path for
    fast/divergent vertical modes, not as a default dry-core term.
    """

    name = "biharmonic_divergence_damping"
    term_name = "divergence_damping"

    def __init__(
        self,
        *,
        nu4: float | None = None,
        tau_grid_seconds: float | None = None,
        reference_length: float | None = None,
        grid_eigenvalue_factor: float = 4096.0,
    ) -> None:
        if nu4 is not None and tau_grid_seconds is not None:
            raise ValueError("Specify either nu4 or tau_grid_seconds, not both")
        if nu4 is None and tau_grid_seconds is None:
            tau_grid_seconds = 24.0 * 3600.0
        if nu4 is not None and float(nu4) < 0.0:
            raise ValueError("BiharmonicDivergenceDamping nu4 must be nonnegative")
        if tau_grid_seconds is not None and float(tau_grid_seconds) <= 0.0:
            raise ValueError(
                "BiharmonicDivergenceDamping tau_grid_seconds must be positive"
            )
        if reference_length is not None and float(reference_length) <= 0.0:
            raise ValueError(
                "BiharmonicDivergenceDamping reference_length must be positive"
            )
        if float(grid_eigenvalue_factor) <= 0.0:
            raise ValueError(
                "BiharmonicDivergenceDamping grid_eigenvalue_factor must be positive"
            )
        self.nu4 = None if nu4 is None else float(nu4)
        self.tau_grid_seconds = (
            None if tau_grid_seconds is None else float(tau_grid_seconds)
        )
        self.reference_length = (
            None if reference_length is None else float(reference_length)
        )
        self.grid_eigenvalue_factor = float(grid_eigenvalue_factor)

    def reference_length_tensor(self, state: State) -> torch.Tensor:
        """Return the horizontal length used to convert timescale to ``nu4``."""

        if self.reference_length is not None:
            return torch.as_tensor(
                self.reference_length,
                dtype=state.u_e.dtype,
                device=state.u_e.device,
            )
        area = state.mesh.cell_area.to(dtype=state.u_e.dtype, device=state.u_e.device)
        return area.mean().sqrt()

    def coefficient(self, state: State) -> torch.Tensor:
        """Return the biharmonic divergence-damping coefficient."""

        if self.nu4 is not None:
            return torch.as_tensor(
                self.nu4, dtype=state.u_e.dtype, device=state.u_e.device
            )
        ell = self.reference_length_tensor(state)
        assert self.tau_grid_seconds is not None
        return ell.pow(4) / (
            float(self.grid_eigenvalue_factor) * float(self.tau_grid_seconds)
        )

    def tendency(self, dyn: Any, state: State) -> torch.Tensor:
        del dyn
        div_u = state.div(state.u_e)
        grad_div_u = state.grad(div_u)
        div_grad_div_u = state.div(grad_div_u)
        return -self.coefficient(state) * state.grad(div_grad_div_u)


class TopRayleighSponge:
    """Opt-in smooth Rayleigh damping near the model top.

    This is a deliberately simple top sponge:

    ``du/dt = -sigma(k) * u``

    where ``sigma`` is zero below the selected top fraction of model levels and
    ramps smoothly to ``1 / tau_top_seconds`` at the top model level.  It is
    intended for controlled upper-boundary / upper-level-mode experiments.  It
    is not part of the default dry core.
    """

    name = "top_rayleigh_sponge"
    term_name = "top_sponge"

    def __init__(
        self,
        *,
        tau_top_seconds: float = 24.0 * 3600.0,
        top_fraction: float = 0.25,
        taper_power: float = 2.0,
        target_u_e: torch.Tensor | None = None,
    ) -> None:
        if float(tau_top_seconds) <= 0.0:
            raise ValueError("TopRayleighSponge tau_top_seconds must be positive")
        if not (0.0 < float(top_fraction) <= 1.0):
            raise ValueError("TopRayleighSponge top_fraction must be in (0, 1]")
        if float(taper_power) <= 0.0:
            raise ValueError("TopRayleighSponge taper_power must be positive")
        self.tau_top_seconds = float(tau_top_seconds)
        self.top_fraction = float(top_fraction)
        self.taper_power = float(taper_power)
        self.target_u_e = None if target_u_e is None else target_u_e.detach().clone()

    def profile(self, state: State) -> torch.Tensor:
        """Return the per-level damping rate ``sigma(k)`` in ``s^-1``.

        Model level index 0 is the top level in the current atmosphere layout.
        The profile is intentionally cell-independent so it broadcasts cleanly
        over edge velocity and remains cheap on every backend.
        """

        n_levels = int(state.column.n_levels)
        n_sponge = max(
            1, min(n_levels, int(self.top_fraction * n_levels + 0.999999999))
        )
        level = torch.arange(n_levels, dtype=state.u_e.dtype, device=state.u_e.device)
        raw = ((float(n_sponge) - level) / float(n_sponge)).clamp_min(0.0)
        weight = raw.pow(float(self.taper_power))
        return weight / float(self.tau_top_seconds)

    def tendency(self, dyn: Any, state: State) -> torch.Tensor:
        del dyn
        sigma = self.profile(state)
        if self.target_u_e is None:
            anomaly = state.u_e
        else:
            target = self.target_u_e.to(dtype=state.u_e.dtype, device=state.u_e.device)
            if target.shape != state.u_e.shape:
                raise ValueError(
                    "TopRayleighSponge target_u_e shape must match state.u_e; "
                    f"got {tuple(target.shape)} vs {tuple(state.u_e.shape)}"
                )
            anomaly = state.u_e - target
        return -anomaly * sigma

    def set_target_from_state(self, state: State) -> None:
        """Set the Rayleigh target to the current edge velocity.

        This is useful for idealized balanced-flow diagnostics where a
        top-sponge should damp departures from the balanced initial wind rather
        than relax the balanced jet itself toward rest.
        """

        self.target_u_e = state.u_e.detach().clone()


def _vertical_dct_basis(
    n_levels: int,
    *,
    dtype: torch.dtype,
    device: torch.device | str | None,
) -> torch.Tensor:
    n = int(n_levels)
    level = torch.arange(n, dtype=dtype, device=device)
    mode = torch.arange(n, dtype=dtype, device=device).reshape(n, 1)
    basis = torch.cos(math.pi * (level + 0.5) * mode / float(n))
    basis[0, :] = basis[0, :] / math.sqrt(float(n))
    if n > 1:
        basis[1:, :] = basis[1:, :] * math.sqrt(2.0 / float(n))
    return basis


def _high_vertical_mode_fraction_per_edge(
    u_e: torch.Tensor, *, high_mode_start: int | None = None
) -> torch.Tensor:
    n_levels = int(u_e.shape[-1])
    basis = _vertical_dct_basis(n_levels, dtype=u_e.dtype, device=u_e.device)
    coeff = torch.einsum("el,ml->em", u_e, basis)
    start = (
        int(high_mode_start)
        if high_mode_start is not None
        else max(1, int(math.ceil(0.5 * n_levels)))
    )
    start = max(0, min(start, n_levels))
    total = coeff.square().sum(dim=-1)
    high = (
        coeff[:, start:].square().sum(dim=-1)
        if start < n_levels
        else torch.zeros_like(total)
    )
    return high / total.clamp_min(torch.finfo(u_e.dtype).tiny)


def _high_vertical_mode_reconstruction(
    u_e: torch.Tensor,
    *,
    high_mode_start: int | None = None,
) -> torch.Tensor:
    n_levels = int(u_e.shape[-1])
    basis = _vertical_dct_basis(n_levels, dtype=u_e.dtype, device=u_e.device)
    coeff = torch.einsum("el,ml->em", u_e, basis)
    start = (
        int(high_mode_start)
        if high_mode_start is not None
        else max(1, int(math.ceil(0.5 * n_levels)))
    )
    start = max(0, min(start, n_levels))
    low = torch.zeros_like(coeff)
    if start < n_levels:
        low[:, start:] = coeff[:, start:]
    return torch.einsum("em,ml->el", low, basis)


class _AdaptiveCoefficientMixin:
    """Shared explicit bounded coefficient law for adaptive damping strategies."""

    def __init__(
        self,
        *,
        tau_min_seconds: float,
        tau_max_seconds: float | None = None,
        nu_min: float | None = None,
        nu_max: float | None = None,
        reference_length: float | None = None,
        grid_eigenvalue_factor: float = 4096.0,
        bias: float = -4.0,
        div_weight: float = 2.0,
        lap_weight: float = 1.0,
        high_mode_weight: float = 3.0,
        divergence_scale: float = 1.0e-6,
        laplacian_scale: float = 1.0e-13,
        high_mode_scale: float = 0.05,
        high_mode_start: int | None = None,
    ) -> None:
        if tau_min_seconds is not None and float(tau_min_seconds) <= 0.0:
            raise ValueError("tau_min_seconds must be positive")
        if tau_max_seconds is not None and float(tau_max_seconds) <= 0.0:
            raise ValueError("tau_max_seconds must be positive")
        if reference_length is not None and float(reference_length) <= 0.0:
            raise ValueError("reference_length must be positive")
        if float(grid_eigenvalue_factor) <= 0.0:
            raise ValueError("grid_eigenvalue_factor must be positive")
        self.tau_min_seconds = float(tau_min_seconds)
        self.tau_max_seconds = (
            None if tau_max_seconds is None else float(tau_max_seconds)
        )
        self.nu_min = None if nu_min is None else float(nu_min)
        self.nu_max = None if nu_max is None else float(nu_max)
        self.reference_length = (
            None if reference_length is None else float(reference_length)
        )
        self.grid_eigenvalue_factor = float(grid_eigenvalue_factor)
        self.bias = float(bias)
        self.div_weight = float(div_weight)
        self.lap_weight = float(lap_weight)
        self.high_mode_weight = float(high_mode_weight)
        self.divergence_scale = float(divergence_scale)
        self.laplacian_scale = float(laplacian_scale)
        self.high_mode_scale = float(high_mode_scale)
        self.high_mode_start = None if high_mode_start is None else int(high_mode_start)

    def reference_length_tensor(self, state: State) -> torch.Tensor:
        if self.reference_length is not None:
            return torch.as_tensor(
                self.reference_length, dtype=state.u_e.dtype, device=state.u_e.device
            )
        area = state.mesh.cell_area.to(dtype=state.u_e.dtype, device=state.u_e.device)
        return area.mean().sqrt()

    def _nu_bounds(self, state: State) -> tuple[torch.Tensor, torch.Tensor]:
        dtype = state.u_e.dtype
        device = state.u_e.device
        if self.nu_max is not None:
            high = torch.as_tensor(self.nu_max, dtype=dtype, device=device)
        else:
            ell = self.reference_length_tensor(state)
            high = ell.pow(4) / (
                float(self.grid_eigenvalue_factor) * float(self.tau_min_seconds)
            )
        if self.nu_min is not None:
            low = torch.as_tensor(self.nu_min, dtype=dtype, device=device)
        elif self.tau_max_seconds is None:
            low = torch.zeros((), dtype=dtype, device=device)
        else:
            ell = self.reference_length_tensor(state)
            low = ell.pow(4) / (
                float(self.grid_eigenvalue_factor) * float(self.tau_max_seconds)
            )
        return low, high

    def features(self, state: State, *, laplacian_proxy: torch.Tensor) -> torch.Tensor:
        """Return local dimensionless coefficient-policy features.

        Features are ordered as divergence magnitude, operator roughness, and
        high vertical-mode fraction.  They are local edge/level tensors and are
        shared by hand-designed and learned bounded coefficient laws.
        """

        div = state.div(state.u_e)
        div_edge = state.cell_to_edge(div.abs())
        high = _high_vertical_mode_fraction_per_edge(
            state.u_e, high_mode_start=self.high_mode_start
        )[:, None]
        div_feature = torch.log1p(div_edge / max(self.divergence_scale, 1.0e-300))
        lap_feature = torch.log1p(
            laplacian_proxy.abs() / max(self.laplacian_scale, 1.0e-300)
        )
        high_feature = high / max(self.high_mode_scale, 1.0e-300)
        return torch.stack(
            [
                div_feature,
                lap_feature,
                high_feature.expand_as(div_feature),
            ],
            dim=-1,
        )

    def activation(
        self, state: State, *, laplacian_proxy: torch.Tensor
    ) -> torch.Tensor:
        features = self.features(state, laplacian_proxy=laplacian_proxy)
        logits = (
            self.bias
            + self.div_weight * features[..., 0]
            + self.lap_weight * features[..., 1]
            + self.high_mode_weight * features[..., 2]
        )
        return torch.sigmoid(logits).clamp(0.0, 1.0)

    def coefficient(self, state: State) -> torch.Tensor:
        raise NotImplementedError


class AdaptiveDivergenceDamping(_AdaptiveCoefficientMixin):
    """Bounded local coefficient version of biharmonic divergence damping."""

    name = "adaptive_divergence_damping"
    term_name = "adaptive_divergence_damping"

    def __init__(self, **kwargs) -> None:
        kwargs.setdefault("tau_min_seconds", 6.0 * 3600.0)
        kwargs.setdefault("tau_max_seconds", None)
        super().__init__(**kwargs)

    def _base_operator(self, state: State) -> torch.Tensor:
        div_u = state.div(state.u_e)
        grad_div_u = state.grad(div_u)
        div_grad_div_u = state.div(grad_div_u)
        return -state.grad(div_grad_div_u)

    def coefficient(self, state: State) -> torch.Tensor:
        low, high = self._nu_bounds(state)
        activation = self.activation(
            state, laplacian_proxy=state.grad(state.div(state.u_e))
        )
        return low + (high - low).clamp_min(0.0) * activation

    def tendency(self, dyn: Any, state: State) -> torch.Tensor:
        del dyn
        return self.coefficient(state) * self._base_operator(state)




class AdaptiveVelocityHyperdiffusion(_AdaptiveCoefficientMixin):
    """Bounded local coefficient version of vector velocity hyperdiffusion."""

    name = "adaptive_velocity_hyperdiffusion"
    term_name = "adaptive_hyperdiffusion"

    def __init__(self, **kwargs) -> None:
        kwargs.setdefault("tau_min_seconds", 24.0 * 3600.0)
        kwargs.setdefault("tau_max_seconds", None)
        super().__init__(**kwargs)

    def _base_operator(self, state: State) -> torch.Tensor:
        lap_u = state.vector_laplacian_edge(state.u_e)
        return -state.vector_laplacian_edge(lap_u)

    def coefficient(self, state: State) -> torch.Tensor:
        low, high = self._nu_bounds(state)
        activation = self.activation(
            state, laplacian_proxy=state.vector_laplacian_edge(state.u_e)
        )
        return low + (high - low).clamp_min(0.0) * activation

    def tendency(self, dyn: Any, state: State) -> torch.Tensor:
        del dyn
        return self.coefficient(state) * self._base_operator(state)


class AdaptiveVerticalModeDamping:
    """Damp high vertical modes with a bounded local Rayleigh rate."""

    name = "adaptive_vertical_mode_damping"
    term_name = "adaptive_vertical_mode_damping"

    def __init__(
        self,
        *,
        tau_min_seconds: float = 6.0 * 3600.0,
        tau_max_seconds: float | None = None,
        bias: float = -4.0,
        high_mode_weight: float = 6.0,
        high_mode_scale: float = 0.05,
        high_mode_start: int | None = None,
    ) -> None:
        if float(tau_min_seconds) <= 0.0:
            raise ValueError("tau_min_seconds must be positive")
        if tau_max_seconds is not None and float(tau_max_seconds) <= 0.0:
            raise ValueError("tau_max_seconds must be positive")
        self.tau_min_seconds = float(tau_min_seconds)
        self.tau_max_seconds = (
            None if tau_max_seconds is None else float(tau_max_seconds)
        )
        self.bias = float(bias)
        self.high_mode_weight = float(high_mode_weight)
        self.high_mode_scale = float(high_mode_scale)
        self.high_mode_start = None if high_mode_start is None else int(high_mode_start)

    def coefficient(self, state: State) -> torch.Tensor:
        high = _high_vertical_mode_fraction_per_edge(
            state.u_e, high_mode_start=self.high_mode_start
        )[:, None]
        activation = torch.sigmoid(
            self.bias
            + self.high_mode_weight * high / max(self.high_mode_scale, 1.0e-300)
        )
        rate_max = 1.0 / float(self.tau_min_seconds)
        rate_min = (
            0.0 if self.tau_max_seconds is None else 1.0 / float(self.tau_max_seconds)
        )
        return (
            torch.as_tensor(rate_min, dtype=state.u_e.dtype, device=state.u_e.device)
            + (rate_max - rate_min) * activation
        )

    def tendency(self, dyn: Any, state: State) -> torch.Tensor:
        del dyn
        high_u = _high_vertical_mode_reconstruction(
            state.u_e, high_mode_start=self.high_mode_start
        )
        return -self.coefficient(state) * high_u


class CompositeMomentumDiffusion:
    """Sum a small set of named momentum-diffusion strategies."""

    name = "composite_momentum_diffusion"
    term_name = "adaptive_composite_damping"

    def __init__(self, *strategies) -> None:
        if not strategies:
            raise ValueError(
                "CompositeMomentumDiffusion requires at least one strategy"
            )
        self.strategies = tuple(strategies)

    def term_tendencies(self, dyn: Any, state: State) -> dict[str, torch.Tensor]:
        out: dict[str, torch.Tensor] = {}
        for strategy in self.strategies:
            name = str(
                getattr(
                    strategy,
                    "term_name",
                    getattr(strategy, "name", type(strategy).__name__),
                )
            )
            out[name] = strategy.tendency(dyn, state)
        return out

    def tendency(self, dyn: Any, state: State) -> torch.Tensor:
        terms = self.term_tendencies(dyn, state)
        total = None
        for value in terms.values():
            total = value if total is None else total + value
        assert total is not None
        return total


class ScaleNormalizedCoordinateWork:
    """Float32-safe canonical scalar coordinate-work correction."""

    name = "scale_normalized"

    def tendency(
        self, dyn: Any, state: State, theta_i: torch.Tensor, omega: torch.Tensor
    ) -> torch.Tensor:
        return dyn._coordinate_work_scale_normalized_impl(state, theta_i, omega)

    def tendency_with_primitives(
        self,
        dyn: Any,
        state: State,
        theta_i: torch.Tensor,
        omega: torch.Tensor,
        *,
        primitives: dict[str, torch.Tensor] | None = None,
    ) -> torch.Tensor:
        return dyn._coordinate_work_scale_normalized_impl(
            state, theta_i, omega, primitives=primitives
        )


class UnscaledReferenceCoordinateWork:
    """Original unscaled coordinate-work algebra for controlled comparisons."""

    name = "unscaled_reference"

    def tendency(
        self, dyn: Any, state: State, theta_i: torch.Tensor, omega: torch.Tensor
    ) -> torch.Tensor:
        return dyn._coordinate_work_unscaled_reference_impl(state, theta_i, omega)


class DryThermodynamics:
    """Dry-air thermodynamics, hydrostatic geopotential, and PGF coupling."""

    name = "dry"

    def pgf_components(
        self, dyn: Any, state: State, theta_i: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return state.column.pgf_components(theta_i, state.ps)

    def energy_gradient_mtheta_for_coord_work(
        self,
        dyn: Any,
        state: State,
        theta_i: torch.Tensor,
        *,
        primitives: dict[str, torch.Tensor] | None = None,
    ) -> torch.Tensor:
        return dyn._dry_energy_gradient_mtheta_for_coord_work(
            state,
            theta_i,
            dp=None if primitives is None else primitives.get("dp"),
            dm=None if primitives is None else primitives.get("dm"),
        )

    def thermo_fields(self, dyn: Any, state: State) -> dict[str, torch.Tensor]:
        theta = state.theta()
        return {
            "theta": theta,
            "theta_full": 0.5 * (theta[..., :-1] + theta[..., 1:]),
            "temperature": state.temperature(),
            "geopotential_interface": state.column.geopotential_interface(
                theta, state.ps
            ),
            "geopotential": state.geopotential(),
            "exner_interface": state.exner_interface(),
            "exner_full": state.exner_full(),
        }

    def total_potential_energy(self, dyn: Any, state: State) -> torch.Tensor:
        return dyn._dry_total_potential_energy(state)

    def total_internal_energy(self, dyn: Any, state: State) -> torch.Tensor:
        return dyn._dry_total_internal_energy(state)


class LogPressureSigmaPressureGradient:
    """Dry PGF variant with an edge-consistent log-pressure sigma correction.

    The standard hybrid-coordinate PGF correction uses
    ``cell_to_edge(alpha * B) * grad(ps)``.  That is the continuous formula in
    average-times-difference form.  For terrain-rest states the bottom-level
    cancellation is more naturally written as ``R_d T grad(log(p_full))``.
    This strategy keeps the active column's full-level geopotential placement
    and only changes that sigma correction.
    """

    name = "log_pressure_sigma"

    def __init__(self, thermodynamics: Any | None = None) -> None:
        self.thermodynamics = (
            DryThermodynamics() if thermodynamics is None else thermodynamics
        )

    def pgf_components(
        self, dyn: Any, state: State, theta_i: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return self.thermodynamics.pgf_components(dyn, state, theta_i)

    def _temperature_for_sigma_correction(
        self,
        state: State,
        theta_i: torch.Tensor,
    ) -> torch.Tensor:
        virtual_temperature = getattr(
            self.thermodynamics, "virtual_temperature_full", None
        )
        if callable(virtual_temperature):
            return virtual_temperature(state, theta_i)
        return state.column.temperature_full(theta_i, state.ps)

    def sigma_correction_term(
        self, dyn: Any, state: State, theta_i: torch.Tensor
    ) -> torch.Tensor:
        """Return ``-R_d * T_edge * grad(log(p_full))`` on edges."""

        T_full = self._temperature_for_sigma_correction(state, theta_i)
        log_p_full = torch.log(state.column.full_pressure(state.ps))
        return -state.column.R_d * state.cell_to_edge(T_full) * state.grad(log_p_full)

    def pgf_term(self, dyn: Any, state: State, theta_i: torch.Tensor) -> torch.Tensor:
        phi_full, _alpha_B = self.pgf_components(dyn, state, theta_i)
        return -state.grad(phi_full) + self.sigma_correction_term(dyn, state, theta_i)


class ExnerSecantSigmaPressureGradient:
    """Dry PGF variant with an Exner-secant sigma correction.

    ``LogPressureSigmaPressureGradient`` is exact for the isothermal
    hypsometric rest identity ``dPhi = -R_d T dlog(p)``.  This variant targets
    the discrete hydrostatic relation used by the Charney-Phillips column:
    ``dPhi = -theta dPi``.  Its sigma correction uses an edge mean of the
    effective full-level potential temperature multiplying ``grad(Pi_full)``.

    The branch is intentionally opt-in.  It is a principled diagnostic
    candidate for vertically varying reference profiles, not a replacement for
    the validated log-pressure terrain path until the rest/JW/HS gates support
    it.
    """

    name = "exner_secant_sigma"

    def __init__(self, thermodynamics: Any | None = None) -> None:
        self.thermodynamics = (
            DryThermodynamics() if thermodynamics is None else thermodynamics
        )

    def pgf_components(
        self, dyn: Any, state: State, theta_i: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return self.thermodynamics.pgf_components(dyn, state, theta_i)

    def _exner_full_for_sigma_correction(self, state: State) -> torch.Tensor:
        # The terrain-aware GL2 branch places velocity/full-level quantities at
        # the actual layer pressure midpoint.  Use that same pressure point for
        # the Exner-slope correction.
        return state.column.exner_full_pmid(state.ps)

    def _theta_for_sigma_correction(
        self, state: State, theta_i: torch.Tensor
    ) -> torch.Tensor:
        virtual_temperature = getattr(
            self.thermodynamics, "virtual_temperature_full", None
        )
        pi_full = self._exner_full_for_sigma_correction(state)
        if callable(virtual_temperature):
            # alpha dp = R_d T_v dlog(p) = theta_v dPi, with
            # theta_v = c_p T_v / Pi.
            T_v = virtual_temperature(state, theta_i)
            return (
                state.column.c_p
                * T_v
                / pi_full.clamp_min(torch.finfo(theta_i.dtype).tiny)
            )
        if state.column.phi_full_mode == "direct_u_gl2_adjoint":
            return state.column.theta_full_pmid_linear_exner(theta_i, state.ps)
        return 0.5 * (theta_i[..., :-1] + theta_i[..., 1:])

    def sigma_correction_term(
        self, dyn: Any, state: State, theta_i: torch.Tensor
    ) -> torch.Tensor:
        """Return ``-theta_eff_edge * grad(Pi_full)`` on edges."""

        theta_full = self._theta_for_sigma_correction(state, theta_i)
        pi_full = self._exner_full_for_sigma_correction(state)
        return -state.cell_to_edge(theta_full) * state.grad(pi_full)

    def pgf_term(self, dyn: Any, state: State, theta_i: torch.Tensor) -> torch.Tensor:
        phi_full, _alpha_B = self.pgf_components(dyn, state, theta_i)
        return -state.grad(phi_full) + self.sigma_correction_term(dyn, state, theta_i)


class TerrainAwarePressureGradient:
    """Canonical dry PGF dispatcher with a terrain-aware GL2 branch.

    Ordinary columns keep the historical hybrid-coordinate PGF correction
    ``-cell_to_edge(alpha * B) * grad(ps)``.  GL2 terrain-aware columns use the
    validated log-pressure sigma correction, which better preserves
    hypsometric topographic rest without changing the flat/JW paths.
    """

    name = "terrain_aware_log_pressure_sigma"
    terrain_phi_full_mode = "direct_u_gl2_adjoint"

    def __init__(
        self,
        thermodynamics: Any | None = None,
        terrain_pressure_gradient: LogPressureSigmaPressureGradient | None = None,
    ) -> None:
        self.thermodynamics = (
            DryThermodynamics() if thermodynamics is None else thermodynamics
        )
        self.terrain_pressure_gradient = (
            LogPressureSigmaPressureGradient(self.thermodynamics)
            if terrain_pressure_gradient is None
            else terrain_pressure_gradient
        )

    def _use_terrain_log_pressure(self, state: State) -> bool:
        return (
            getattr(state.column, "phi_full_mode", None) == self.terrain_phi_full_mode
        )

    def pgf_components(
        self, dyn: Any, state: State, theta_i: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return self.thermodynamics.pgf_components(dyn, state, theta_i)

    def sigma_correction_term(
        self, dyn: Any, state: State, theta_i: torch.Tensor
    ) -> torch.Tensor:
        if self._use_terrain_log_pressure(state):
            return self.terrain_pressure_gradient.sigma_correction_term(
                dyn, state, theta_i
            )
        _phi_full, alpha_B = self.pgf_components(dyn, state, theta_i)
        return -state.cell_to_edge(alpha_B) * state.grad(state.ps)

    def pgf_term(self, dyn: Any, state: State, theta_i: torch.Tensor) -> torch.Tensor:
        if self._use_terrain_log_pressure(state):
            return self.terrain_pressure_gradient.pgf_term(dyn, state, theta_i)
        phi_full, alpha_B = self.pgf_components(dyn, state, theta_i)
        return -state.grad(phi_full) - state.cell_to_edge(alpha_B) * state.grad(
            state.ps
        )


class VirtualTemperatureThermodynamics:
    """Virtual-temperature thermodynamics driven by named passive vapor."""

    name = "virtual_temperature"

    def vapor_mixing_ratio(self, state: State) -> torch.Tensor | None:
        q = state.tracers()
        if q is None:
            return None
        value = (state.attrs or {}).get(_VAPOR_TRACER_INDEX_ATTR)
        if value is None:
            return None
        try:
            index = int(value)
        except (TypeError, ValueError):
            return None
        if index < 0 or index >= int(q.shape[-1]):
            return None

        vapor = q[..., index]
        quantity = str(
            (state.attrs or {}).get(
                _VAPOR_QUANTITY_ATTR, _VAPOR_QUANTITY_DRY_MIXING_RATIO
            )
        )
        if quantity == _VAPOR_QUANTITY_SPECIFIC_HUMIDITY:
            return vapor / (1.0 - vapor).clamp_min(torch.finfo(vapor.dtype).tiny)
        return vapor

    def virtual_temperature_factor_full(self, state: State) -> torch.Tensor | None:
        r_v = self.vapor_mixing_ratio(state)
        if r_v is None:
            return None
        denom = (1.0 + r_v).clamp_min(torch.finfo(r_v.dtype).tiny)
        return (1.0 + r_v / _EPSILON_WATER_DRY_AIR) / denom

    def virtual_temperature_factor_interface(self, state: State) -> torch.Tensor | None:
        factor = self.virtual_temperature_factor_full(state)
        if factor is None:
            return None
        if factor.shape[-1] == 1:
            return torch.cat([factor, factor], dim=-1)
        interior = 0.5 * (factor[..., :-1] + factor[..., 1:])
        return torch.cat([factor[..., :1], interior, factor[..., -1:]], dim=-1)

    def virtual_theta_interface(
        self, state: State, theta_i: torch.Tensor
    ) -> torch.Tensor:
        factor_i = self.virtual_temperature_factor_interface(state)
        if factor_i is None:
            return theta_i
        return theta_i * factor_i

    def theta_full_for_column(
        self, state: State, theta_i: torch.Tensor
    ) -> torch.Tensor:
        if state.column.phi_full_mode == "direct_u_gl2_adjoint":
            return state.column.theta_full_pmid_linear_exner(theta_i, state.ps)
        return 0.5 * (theta_i[..., :-1] + theta_i[..., 1:])

    def virtual_temperature_full(
        self, state: State, theta_i: torch.Tensor
    ) -> torch.Tensor:
        temperature = state.column.temperature_full(theta_i, state.ps)
        factor = self.virtual_temperature_factor_full(state)
        if factor is None:
            return temperature
        return temperature * factor

    def virtual_specific_volume_full(
        self, state: State, theta_i: torch.Tensor
    ) -> torch.Tensor:
        temperature_v = self.virtual_temperature_full(state, theta_i)
        return state.column.R_d * temperature_v / state.column.full_pressure(state.ps)

    def pgf_components(
        self, dyn: Any, state: State, theta_i: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        factor_i = self.virtual_temperature_factor_interface(state)
        if factor_i is None:
            return DryThermodynamics().pgf_components(dyn, state, theta_i)

        theta_v_i = theta_i * factor_i
        phi_full = state.column.geopotential_full(theta_v_i, state.ps)
        alpha_full = self.virtual_specific_volume_full(state, theta_i)
        return phi_full, alpha_full * state.column.B_full

    def energy_gradient_mtheta_for_coord_work(
        self,
        dyn: Any,
        state: State,
        theta_i: torch.Tensor,
        *,
        primitives: dict[str, torch.Tensor] | None = None,
    ) -> torch.Tensor:
        factor_i = self.virtual_temperature_factor_interface(state)
        if factor_i is None:
            return DryThermodynamics().energy_gradient_mtheta_for_coord_work(
                dyn,
                state,
                theta_i,
                primitives=primitives,
            )

        theta_v_i = theta_i * factor_i
        return factor_i * dyn._dry_energy_gradient_mtheta_for_coord_work(
            state,
            theta_v_i,
            dp=None if primitives is None else primitives.get("dp"),
            dm=None if primitives is None else primitives.get("dm"),
        )

    def thermo_fields(self, dyn: Any, state: State) -> dict[str, torch.Tensor]:
        theta = state.theta()
        theta_full = self.theta_full_for_column(state, theta)
        factor_full = self.virtual_temperature_factor_full(state)
        if factor_full is None:
            factor_full = torch.ones_like(theta_full)
        theta_v = self.virtual_theta_interface(state, theta)
        theta_v_full = theta_full * factor_full
        temperature = state.column.temperature_full(theta, state.ps)
        temperature_v = temperature * factor_full
        return {
            "theta": theta,
            "theta_full": theta_full,
            "temperature": temperature,
            "theta_virtual": theta_v,
            "theta_virtual_full": theta_v_full,
            "virtual_temperature": temperature_v,
            "specific_volume": state.column.R_d
            * temperature_v
            / state.column.full_pressure(state.ps),
            "geopotential_interface": state.column.geopotential_interface(
                theta_v, state.ps
            ),
            "geopotential": state.column.geopotential_full(theta_v, state.ps),
            "exner_interface": state.exner_interface(),
            "exner_full": state.exner_full(),
        }

    def total_potential_energy(self, dyn: Any, state: State) -> torch.Tensor:
        dyn.check_state_compatible(state)
        dp = state.layer_dp()
        theta_v = self.virtual_theta_interface(state, state.theta())
        phi_full = state.column.geopotential_full(theta_v, state.ps)
        return (
            state.mesh.integrate_cell((phi_full * dp).sum(dim=-1, keepdim=True))
            / dyn.gravity
        )

    def total_internal_energy(self, dyn: Any, state: State) -> torch.Tensor:
        dyn.check_state_compatible(state)
        c_v = state.column.c_p - state.column.R_d
        dp = state.layer_dp()
        temperature_v = self.virtual_temperature_full(state, state.theta())
        return (
            state.mesh.integrate_cell(
                (c_v * temperature_v * dp).sum(dim=-1, keepdim=True)
            )
            / dyn.gravity
        )


__all__ = [
    "BlendedMildUpwindVerticalMomentumTransport",
    "BiharmonicDivergenceDamping",
    "DryThermodynamics",
    "EdgeUpwindVerticalMomentumTransport",
    "ExnerSecantSigmaPressureGradient",
    "FCTTracerTransport",
    "LogPressureSigmaPressureGradient",
    "MildCellSupportedVerticalMomentumTransport",
    "ScaleNormalizedCoordinateWork",
    "TerrainAwarePressureGradient",
    "TopRayleighSponge",
    "UnscaledReferenceCoordinateWork",
    "UpwindTracerTransport",
    "VelocityHyperdiffusion",
    "VirtualTemperatureThermodynamics",
]
