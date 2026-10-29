"""Hybrid-pressure transport primitives for dry atmospheric dynamics."""

from __future__ import annotations

import torch

from ..core.state import State


# Numerical functions bound explicitly by DryHydrostaticDynamics.
def _transport_primitives(self, state: State) -> dict[str, torch.Tensor]:
    """Shared mass/transport primitives used by dynamics and diagnostics."""

    self.check_state_compatible(state)
    dp = state.layer_dp()
    dm = self._interface_mass(dp)
    self._validate_mass_geometry(state, dp, dm)
    theta = state.mtheta_i / dm
    dp_e = state.cell_to_edge(dp)
    mass_flux = state.u_e * dp_e
    layer_div = self._layer_divergence(state, state.u_e, dp, dp_e=dp_e, mass_flux=mass_flux)
    dps_dt = self._ps_tendency(layer_div)
    mu = self._vertical_mass_flux(state, layer_div, dps_dt)
    omega = self._omega_full(mu)
    return {
        "dp": dp,
        "dp_e": dp_e,
        "dm": dm,
        "theta": theta,
        "mass_flux": mass_flux,
        "layer_div": layer_div,
        "dps_dt": dps_dt,
        "mu": mu,
        "omega": omega,
    }

def ps_tendency_terms(
    self,
    state: State,
    *,
    primitives: dict[str, torch.Tensor] | None = None,
) -> dict[str, torch.Tensor]:
    """Return the surface-pressure continuity terms owned by transport."""

    ctx = self._transport_primitives(state) if primitives is None else primitives
    return {
        "mass_flux": ctx["dps_dt"],
        "total": ctx["dps_dt"],
    }

def _layer_thickness(self, state: State) -> torch.Tensor:
    """
    Pressure thickness of each full level.

    Returns dp [..., nCells, N]  (always positive).
    """
    return state.layer_dp()

def _interface_mass(self, dp: torch.Tensor) -> torch.Tensor:
    """
    Mass associated with each CP interface.

    The CP theta at interface k is associated with the average of
    the pressure thicknesses of the layers immediately above and below.

    dm_0     = 0.5 * dp_0          (top interface: only layer below)
    dm_k     = 0.5*(dp_{k-1}+dp_k) (interior: average of two layers)
    dm_N     = 0.5 * dp_{N-1}      (surface interface: only layer above)

    Input:  dp   [..., nCells, N]
    Output: dm   [..., nCells, N+1]
    """
    half = 0.5 * dp
    # dm for interior interfaces: 0.5*(dp_{k-1} + dp_k)
    interior = half[..., :-1] + half[..., 1:]          # [..., nCells, N-1]
    # Concatenate top, interior, surface
    return torch.cat([half[..., :1], interior, half[..., -1:]], dim=-1)

def _layer_divergence(
    self,
    state: State,
    u_e: torch.Tensor,
    dp:  torch.Tensor,
    *,
    dp_e: torch.Tensor | None = None,
    mass_flux: torch.Tensor | None = None,
) -> torch.Tensor:
    """
    Horizontal mass-flux divergence in each full level.

        div_k = div( u_e_k * cell_to_edge(dp_k) )

    This quantity is needed both for ∂ps/∂t and for the vertical
    mass flux recurrence, so it is computed once and shared.

    Input:  u_e [..., nEdges, N]
            dp  [..., nCells, N]
    Output:     [..., nCells, N]
    """
    if mass_flux is None:
        dp_e = state.cell_to_edge(dp) if dp_e is None else dp_e
        mass_flux = u_e * dp_e
    return state.div(mass_flux)               # [..., nCells, N]

def _ps_tendency(self, layer_div: torch.Tensor) -> torch.Tensor:
    """
    Surface pressure tendency: vertical integral of layer divergence.

    ∂ps/∂t = -Σ_k div_k

    Input:  layer_div [..., nCells, N]
    Output:           [..., nCells, 1]
    """
    return -layer_div.sum(dim=-1, keepdim=True)

def _vertical_mass_flux(
    self,
    state: State,
    layer_div: torch.Tensor,
    dps_dt:    torch.Tensor,
) -> torch.Tensor:
    """
    Downward mass flux at each CP interface (Pa/s).

    Derived from the layer continuity equation by a downward prefix
    cumulative sum starting from the top boundary condition μ_0 = 0:

        μ_k = -Σ_{j=0}^{k-1} [ δB_j * ∂ps/∂t + div_j ]

    where δB_j = B_{j+1} - B_j accounts for the time tendency of
    the hybrid coordinate.

    Boundary conditions:
        μ_0 = 0  (no mass through model top)
        μ_N ≈ 0  (self-consistent if ps obeys its budget)

    Positive μ means downward (toward higher pressure).

    Input:  layer_div [..., nCells, N]   from _layer_divergence()
            dps_dt    [..., nCells, 1]   from _ps_tendency()
    Output: mu        [..., nCells, N+1]
    """
    # residual_k = δB_k * dps_dt + div_k
    # dB: [N], dps_dt: [..., nCells, 1] → broadcasts to [..., nCells, N]
    dB = state.column.B_interface[1:] - state.column.B_interface[:-1]
    dB = dB.to(dtype=layer_div.dtype, device=layer_div.device)
    residual = dB * dps_dt + layer_div           # [..., nCells, N]

    # Exclusive prefix sum: μ_k = -Σ_{j<k} residual_j
    # cumsum[k] = Σ_{j=0}^{k} residual_j
    # μ = cat([0, -cumsum[:-1]], which shifts right by one
    neg_cum = -torch.cumsum(residual, dim=-1)    # [..., nCells, N]
    zero    = torch.zeros_like(neg_cum[..., :1]) # [..., nCells, 1]
    return torch.cat([zero, neg_cum], dim=-1)    # [..., nCells, N+1]

def _omega_full(self, mu: torch.Tensor) -> torch.Tensor:
    """
    Vertical mass flux at full levels: average of bounding interfaces.

        ω_k = ½ (μ_k + μ_{k+1})

    The same averaging as the hydrostatic layer-mean θ̄_k — required
    for energy consistency of the vertical theta flux.

    Input:  mu    [..., nCells, N+1]
    Output: omega [..., nCells, N]
    """
    return 0.5 * (mu[..., :-1] + mu[..., 1:])

__all__ = ['_transport_primitives', 'ps_tendency_terms', '_layer_thickness', '_interface_mass', '_layer_divergence', '_ps_tendency', '_vertical_mass_flux', '_omega_full']
