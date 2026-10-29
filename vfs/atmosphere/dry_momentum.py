"""Vector-invariant momentum operators for dry atmospheric dynamics."""

from __future__ import annotations

import torch

from ..core.state import State


# Numerical functions bound explicitly by DryHydrostaticDynamics.
def _trisk_rotation(
    self, state: State, u_e: torch.Tensor | None = None
) -> torch.Tensor:
    """
    M-skew-projected exact-Thuburn PV-flux rotation.

    The raw rotation operator R satisfies:

        M R + R^T M = 0

    Then variable q is inserted pairwise:

        rot_e = sum_e' 0.5 * (q_e + q_e') R[e,e'] u_e'

    This preserves skewness when q is symmetrically placed on edge pairs.
    """
    mesh = state.mesh
    if u_e is None:
        u_e = state.u_e
    return mesh.trisk_pv_flux_rotation(u_e, self.rotation_rate)

# ------------------------------------------------------------------
# Tendency equations
# ------------------------------------------------------------------
def _u_vertical(self, state: State, u_e, mu, dp, *, dp_e=None):
    return self.momentum_transport.tendency(
        self,
        state,
        u_e,
        mu,
        dp,
        dp_e=dp_e,
    )

def _momentum_diffusion_tendency(self, state: State) -> torch.Tensor:
    """Return optional velocity diffusion / sponge tendency."""

    if getattr(self, "momentum_diffusion", None) is None:
        return torch.zeros_like(state.u_e)
    return self.momentum_diffusion.tendency(self, state)

def _momentum_diffusion_term_name(self) -> str:
    strategy = getattr(self, "momentum_diffusion", None)
    if strategy is None:
        return "momentum_diffusion"
    return str(
        getattr(
            strategy, "term_name", getattr(strategy, "name", "momentum_diffusion")
        )
    )

def momentum_tendency_terms(
    self,
    state: State,
    *,
    primitives: dict[str, torch.Tensor] | None = None,
) -> dict[str, torch.Tensor]:
    """Return momentum-equation terms owned by the momentum operator."""

    ctx = self._transport_primitives(state) if primitives is None else primitives
    theta = ctx["theta"]
    dp = ctx["dp"]
    mu = ctx["mu"]
    rotation = self._trisk_rotation(state)
    pgf_term_details = None
    pgf_terms = getattr(self.pressure_gradient, "pgf_terms", None)
    if callable(pgf_terms):
        pgf_term_details = pgf_terms(self, state, theta)
        pgf = pgf_term_details["pgf"]
    else:
        pgf = self._pgf_term_for_state(state, theta)
    bernoulli = -state.grad(state.ke_cell())
    vertical = self._u_vertical(state, state.u_e, mu, dp, dp_e=ctx.get("dp_e"))
    diffusion_details = None
    strategy = getattr(self, "momentum_diffusion", None)
    term_tendencies = getattr(strategy, "term_tendencies", None)
    if callable(term_tendencies):
        diffusion_details = term_tendencies(self, state)
        diffusion = sum(diffusion_details.values(), torch.zeros_like(state.u_e))
    else:
        diffusion = self._momentum_diffusion_tendency(state)
    out = {
        "rotation": rotation,
        "pgf": pgf,
        "bernoulli": bernoulli,
        "vertical": vertical,
        "momentum_diffusion": diffusion,
        "diffusion": diffusion,
        "total": rotation + pgf + bernoulli + vertical + diffusion,
    }
    if pgf_term_details is not None:
        out["pgf_base"] = pgf_term_details["pgf_base"]
        out["pgf_correction"] = pgf_term_details["pgf_correction"]
        if "pgf_correction_raw" in pgf_term_details:
            out["pgf_correction_raw"] = pgf_term_details["pgf_correction_raw"]
        if "pgf_correction_cap" in pgf_term_details:
            out["pgf_correction_cap"] = pgf_term_details["pgf_correction_cap"]
    term_name = self._momentum_diffusion_term_name()
    if diffusion_details is not None:
        out.update(diffusion_details)
    out[term_name] = diffusion
    if term_name == "velocity_hyperdiffusion":
        out["hyperdiffusion"] = diffusion
    elif term_name == "divergence_damping":
        out["divergence_damping"] = diffusion
    return out

def _pgf_components_for_state(
    self,
    state: State,
    theta_i: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return full-level geopotential and alpha*B for the active PGF strategy."""

    return self.pressure_gradient.pgf_components(self, state, theta_i)

def _pgf_sigma_correction_for_state(
    self,
    state: State,
    theta_i: torch.Tensor,
    *,
    alpha_B: torch.Tensor | None = None,
) -> torch.Tensor:
    """Return the terrain/sigma correction term in the u-equation."""

    sigma_correction = getattr(
        self.pressure_gradient, "sigma_correction_term", None
    )
    if callable(sigma_correction):
        return sigma_correction(self, state, theta_i)
    if alpha_B is None:
        _phi_full, alpha_B = self._pgf_components_for_state(state, theta_i)
    return -state.cell_to_edge(alpha_B) * state.grad(state.ps)

def _pgf_term_for_state(
    self,
    state: State,
    theta_i: torch.Tensor,
) -> torch.Tensor:
    """Return the full hydrostatic PGF term, excluding kinetic energy."""

    pgf_term = getattr(self.pressure_gradient, "pgf_term", None)
    if callable(pgf_term):
        return pgf_term(self, state, theta_i)
    phi_full, alpha_B = self._pgf_components_for_state(state, theta_i)
    return -state.grad(phi_full) + self._pgf_sigma_correction_for_state(
        state,
        theta_i,
        alpha_B=alpha_B,
    )

def _u_tendency(
    self,
    state: State,
    u_e: torch.Tensor,
    theta_i: torch.Tensor,
) -> torch.Tensor:
    """
    Horizontal momentum tendency (no vertical advection).

    ∂u_e/∂t = rotation + bernoulli_gradient + pgf_sigma_correction

    rotation:
        -(ζ + f)_e * u_t_e

    bernoulli_gradient:
        -grad(KE + phi_full)

    pgf_sigma_correction:
        -cell_to_edge(alpha_B) * grad(ps)

    Input:  u_e     [..., nEdges,  N]
            theta_i [..., nCells,  N+1]
            ps      [..., nCells,  1]
    Output:         [..., nEdges,  N]
    """
    # --- Rotation term (TRiSK PV-flux form) ---
    rot = self._trisk_rotation(state, u_e)  # [..., nEdges, N]

    # --- Kinetic-energy Bernoulli gradient plus active PGF strategy ---
    # KE at cell centres uses the same TRiSK weights as u_t, preserving
    # the zero-work rotation property.  The PGF is delegated to the active
    # pressure-gradient strategy so opt-in wrappers affect the fast path.
    ke = state.ke_cell(u_e)  # [..., nCells, N]
    grad_ke = state.grad(ke)  # [..., nEdges, N]
    pgf = self._pgf_term_for_state(state, theta_i)  # [..., nEdges, N]

    return rot - grad_ke + pgf  # [..., nEdges, N]

__all__ = ['_trisk_rotation', '_u_vertical', '_momentum_diffusion_tendency', '_momentum_diffusion_term_name', 'momentum_tendency_terms', '_pgf_components_for_state', '_pgf_sigma_correction_for_state', '_pgf_term_for_state', '_u_tendency']
