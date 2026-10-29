"""Integrated mass and energy diagnostics for dry dynamics."""

from __future__ import annotations

import torch

from ..core.state import State


# Numerical functions bound explicitly by DryHydrostaticDynamics.
def total_mass(self, state: State) -> torch.Tensor:
    """
    Total atmospheric mass per unit area, integrated horizontally.

    mass = (1/g) * Σ_k integrate_cell(dp_k)   [Pa * m² / (m/s²)] = [kg]

    Returns a scalar (or [...] tensor for batched states).
    """
    self.check_state_compatible(state)
    return state.total_mass()

def total_tracer_mass(self, state: State) -> torch.Tensor | None:
    """
    Total passive tracer mass for each tracer.

    tracer_m is q * dp, so the physical integral is
    (1/g) * ∫Σ_k tracer_m_k dA.  Returns None for tracer-free states.
    """
    self.check_state_compatible(state)
    return state.total_tracer_mass()

def total_kinetic_energy(self, state: State) -> torch.Tensor:
    """
    Total kinetic energy integrated over the domain.

    KE = (1/g) * Σ_k integrate_cell(ke_k * dp_k)

    Returns a scalar (or [...] tensor).
    """
    self.check_state_compatible(state)
    dp = state.layer_dp()
    ke = state.ke_cell()                                  # [..., nCells, N]
    return state.mesh.integrate_cell(
        (ke * dp).sum(dim=-1, keepdim=True)
    ) / self.gravity

def kinetic_energy_gradient_u(self, state: State) -> torch.Tensor:
    """
    Closed-form ``d(KE)/d(u_e)`` for the active mesh kinetic energy.

    This includes the transpose of the tangential-velocity reconstruction
    used by ``mesh.ke_cell``. It is the correct wind-space gradient for
    exact pressure-work diagnostics; the older diagonal edge metric is only
    a lightweight smoke approximation.
    """
    self.check_state_compatible(state)
    return self.kinetic_energy_gradient_u_terms(state)["total"]

def _kinetic_energy_edge_weight(self, state: State) -> torch.Tensor:
    """Return edge/layer mass weights used by the mesh KE metric."""
    mesh = state.mesh
    dp = state.layer_dp()
    c0 = mesh.cellsOnEdge[:, 0]
    c1 = mesh.cellsOnEdge[:, 1]
    side_area = mesh.edge_side_area.to(dtype=state.u_e.dtype, device=state.u_e.device)
    return (
        side_area[:, 0, None] * dp[..., c0, :]
        + side_area[:, 1, None] * dp[..., c1, :]
    ) / float(self.gravity)

def kinetic_energy_gradient_u_terms(self, state: State) -> dict[str, torch.Tensor]:
    """
    Split ``d(KE)/d(u_e)`` into normal and tangential-metric pieces.

    The normal piece is the diagonal edge mass metric multiplying the
    normal wind. The tangential piece is the adjoint contribution from the
    TRiSK tangential reconstruction used by ``mesh.ke_cell``. Their sum is
    exactly the gradient returned by ``kinetic_energy_gradient_u``.
    """
    self.check_state_compatible(state)
    weight = self._kinetic_energy_edge_weight(state)
    tangential = state.tangential_velocity_adjoint(weight * state.tangential_velocity(state.u_e))
    normal = weight * state.u_e
    return {
        "normal": normal,
        "tangential": tangential,
        "total": normal + tangential,
    }

def scalar_energy_gradients(self, state: State) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Closed-form dry-energy gradients for scalar prognostic fields.

    Returns ``(dE/dmtheta_i, dE/dps)`` for the dry energy functional used by
    ``total_energy``.  The pressure gradient is taken at fixed prognostic
    ``mtheta_i``, so it includes both explicit pressure dependence and the
    induced ``theta = mtheta/dm(ps)`` dependence.
    """
    self.check_state_compatible(state)
    theta = state.theta()
    dp = state.layer_dp()
    dm = self._interface_mass(dp)
    grad_mtheta = self._dry_energy_gradient_mtheta_for_coord_work(state, theta, dp=dp, dm=dm)
    grad_ps = self._dry_energy_gradient_ps(state, theta, dp=dp, dm=dm, grad_mtheta=grad_mtheta)
    return grad_mtheta, grad_ps

def scalar_pressure_energy_gradient_terms(self, state: State) -> dict[str, torch.Tensor]:
    """
    Decompose ``dE/dps`` into physically interpretable closed-form pieces.

    The terms sum to the pressure component returned by
    ``scalar_energy_gradients``.  They are diagnostic inspection fields:
    changing ``ps`` at fixed prognostic ``mtheta`` changes layer mass,
    geopotential/internal-energy placement, and recovered theta.
    """
    self.check_state_compatible(state)
    theta = state.theta()
    dp = state.layer_dp()
    dm = self._interface_mass(dp)
    grad_mtheta = self._dry_energy_gradient_mtheta_for_coord_work(state, theta, dp=dp, dm=dm)
    return self._dry_energy_gradient_ps_terms(state, theta, dp=dp, dm=dm, grad_mtheta=grad_mtheta)

def scalar_pressure_energy_gradient_local_terms(self, state: State) -> dict[str, dict[str, torch.Tensor]]:
    """
    Local ``dE/dps`` decomposition before column summation.

    Returns ``layer`` pieces for kinetic-mass, potential, and internal
    energy, ``interface`` pieces for theta/mass coupling, and ``column``
    pieces whose values sum to ``scalar_pressure_energy_gradient_terms``.
    """
    self.check_state_compatible(state)
    theta = state.theta()
    dp = state.layer_dp()
    dm = self._interface_mass(dp)
    grad_mtheta = self._dry_energy_gradient_mtheta_for_coord_work(state, theta, dp=dp, dm=dm)
    return self._dry_energy_gradient_ps_local_terms(state, theta, dp=dp, dm=dm, grad_mtheta=grad_mtheta)

def scalar_energy_rate_closed(
    self,
    state: State,
    mtheta_tendency: torch.Tensor,
    ps_tendency: torch.Tensor,
) -> torch.Tensor:
    """Exact dry-energy rate from scalar tendencies using closed-form gradients."""
    grad_mtheta, grad_ps = self.scalar_energy_gradients(state)
    return (grad_mtheta * mtheta_tendency).sum() + (grad_ps * ps_tendency).sum()

def scalar_energy_rate_terms_closed(
    self,
    state: State,
    mtheta_terms: dict[str, torch.Tensor],
    ps_terms: dict[str, torch.Tensor],
) -> dict[str, dict[str, torch.Tensor] | torch.Tensor]:
    """
    Closed-form dry-energy rates for scalar tendency components.

    This is an inspection helper for pressure-work debugging.  It uses the
    same analytic ``dE/dmtheta`` and ``dE/dps`` as ``total_energy`` and
    returns rates for each named scalar term plus their totals.
    """
    grad_mtheta, grad_ps = self.scalar_energy_gradients(state)
    mtheta_rates = {name: (grad_mtheta * value).sum() for name, value in mtheta_terms.items()}
    ps_rates = {name: (grad_ps * value).sum() for name, value in ps_terms.items()}
    zero = torch.zeros((), dtype=state.ps.dtype, device=state.ps.device)
    mtheta_total = mtheta_rates.get(
        "total",
        sum((v for k, v in mtheta_rates.items() if k != "total"), zero),
    )
    ps_total = ps_rates.get(
        "total",
        sum((v for k, v in ps_rates.items() if k != "total"), zero),
    )
    return {
        "mtheta_i": mtheta_rates,
        "ps": ps_rates,
        "mtheta_total": mtheta_total,
        "ps_total": ps_total,
        "total": mtheta_total + ps_total,
    }

def scalar_pressure_energy_rate_terms_closed(
    self,
    state: State,
    ps_terms: dict[str, torch.Tensor],
) -> dict[str, dict[str, torch.Tensor]]:
    """Closed-form ps-energy-rate decomposition by ps-gradient piece."""
    grad_terms = self.scalar_pressure_energy_gradient_terms(state)
    return {
        component: {
            name: (grad * value).sum()
            for name, value in ps_terms.items()
        }
        for component, grad in grad_terms.items()
    }

def wind_energy_rate_terms_closed(
    self,
    state: State,
    u_terms: dict[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    """Closed-form kinetic-energy rates for named wind tendency terms."""
    grad_u = self.kinetic_energy_gradient_u(state)
    return {name: (grad_u * value).sum() for name, value in u_terms.items()}

def energy_rate_closed(self, state: State, tendency: State) -> torch.Tensor:
    """Closed-form dry-energy directional derivative for a State-shaped RHS."""
    grad_u = self.kinetic_energy_gradient_u(state)
    grad_mtheta, grad_ps = self.scalar_energy_gradients(state)
    return (
        (grad_u * tendency.u_e).sum()
        + (grad_mtheta * tendency.mtheta_i).sum()
        + (grad_ps * tendency.ps).sum()
    )

def _dry_total_potential_energy(self, state: State) -> torch.Tensor:
    """
    Total potential energy (geopotential) integrated over the domain.

    PE = (1/g) * Σ_k integrate_cell(phi_full_k * dp_k)

    Returns a scalar (or [...] tensor).
    """
    self.check_state_compatible(state)
    dp       = state.layer_dp()
    phi_full = state.geopotential()
    return state.mesh.integrate_cell(
        (phi_full * dp).sum(dim=-1, keepdim=True)
    ) / self.gravity

def _dry_total_internal_energy(self, state: State) -> torch.Tensor:
    """
    Total internal energy integrated over the domain.

    IE = (c_v / g) * Σ_k integrate_cell(T_full_k * dp_k)
       = (c_v / c_p / g) * Σ_k integrate_cell(theta_bar_k * pi_full_k * dp_k)

    For a dry ideal gas: c_v = c_p - R_d.

    Returns a scalar (or [...] tensor).
    """
    self.check_state_compatible(state)
    c_v   = state.column.c_p - state.column.R_d
    dp    = state.layer_dp()
    T     = state.temperature()
    return state.mesh.integrate_cell(
        (c_v * T * dp).sum(dim=-1, keepdim=True)
    ) / self.gravity

def total_potential_energy(self, state: State) -> torch.Tensor:
    """Total potential energy for the active thermodynamics strategy."""
    return self.thermodynamics.total_potential_energy(self, state)

def total_internal_energy(self, state: State) -> torch.Tensor:
    """Total internal energy for the active thermodynamics strategy."""
    return self.thermodynamics.total_internal_energy(self, state)

def total_energy(self, state: State) -> torch.Tensor:
    """
    Total dry energy: KE + PE + IE.

    This is the conserved quantity for a dry adiabatic hydrostatic
    atmosphere. The canonical scalar coordinate-work correction improves
    the pressure-work/vertical-transport exchange; time discretization can
    still introduce finite residuals.
    """
    return (
        self.total_kinetic_energy(state)
        + self.total_potential_energy(state)
        + self.total_internal_energy(state)
    )

__all__ = ['total_mass', 'total_tracer_mass', 'total_kinetic_energy', 'kinetic_energy_gradient_u', '_kinetic_energy_edge_weight', 'kinetic_energy_gradient_u_terms', 'scalar_energy_gradients', 'scalar_pressure_energy_gradient_terms', 'scalar_pressure_energy_gradient_local_terms', 'scalar_energy_rate_closed', 'scalar_energy_rate_terms_closed', 'scalar_pressure_energy_rate_terms_closed', 'wind_energy_rate_terms_closed', 'energy_rate_closed', '_dry_total_potential_energy', '_dry_total_internal_energy', 'total_potential_energy', 'total_internal_energy', 'total_energy']
