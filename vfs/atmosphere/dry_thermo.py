"""Charney-Phillips thermodynamic transport and coordinate work."""

from __future__ import annotations

import torch

from ..core.state import State


# Numerical functions bound explicitly by DryHydrostaticDynamics.
def _cell_interface_energy_weight(self, state: State, dm: torch.Tensor) -> torch.Tensor:
    """Cell-area/interface-mass weight used by prototype coordinate work."""

    shape = [1] * dm.ndim
    shape[-2] = state.mesh.nCells
    shape[-1] = 1
    area = state.mesh.cell_area.reshape(shape).to(dtype=dm.dtype, device=dm.device)
    return area * dm / float(self.gravity)

def _cell_full_energy_weight(self, state: State, dp: torch.Tensor) -> torch.Tensor:
    """Cell-area/full-layer energy weight area*dp/g."""

    shape = [1] * dp.ndim
    shape[-2] = state.mesh.nCells
    shape[-1] = 1
    area = state.mesh.cell_area.reshape(shape).to(dtype=dp.dtype, device=dp.device)
    return area * dp / float(self.gravity)

def _dry_energy_gradient_mtheta_for_coord_work(
    self,
    state: State,
    theta_i: torch.Tensor,
    *,
    dp: torch.Tensor | None = None,
    dm: torch.Tensor | None = None,
) -> torch.Tensor:
    """
    Return dE/dmtheta for the current thermodynamic state.

    Closed-form adjoint of the thermodynamic energy with ps fixed.  KE has
    no mtheta dependence.  The result is tied to the active full-level
    placement mode used by total_potential_energy/total_internal_energy.
    """

    ps = state.ps
    column = state.column
    dp = state.layer_dp() if dp is None else dp
    dm = self._interface_mass(dp) if dm is None else dm
    w_full = self._cell_full_energy_weight(state, dp)
    grad_theta = torch.zeros_like(theta_i)

    mode = column.phi_full_mode
    pi_i = column.exner_interface(ps)

    # Potential-energy adjoint.
    if mode == "interface_mean":
        psi_i = torch.zeros_like(theta_i)
        psi_i[..., :-1] += 0.5 * w_full
        psi_i[..., 1:] += 0.5 * w_full
        grad_theta = grad_theta + column.hydrostatic_adjoint_rhs(psi_i, ps)
    elif mode in ("linear_pfull_adjoint", "direct_u_linear_adjoint"):
        pi_full = column.exner_full_pmid(ps)
        partial = pi_i[..., 1:] - pi_full
        psi_i = torch.zeros_like(theta_i)
        psi_i[..., 1:] += w_full
        grad_theta = grad_theta + column.hydrostatic_adjoint_rhs(psi_i, ps)
        grad_theta[..., :-1] += 0.5 * w_full * partial
        grad_theta[..., 1:] += 0.5 * w_full * partial
    elif mode == "direct_u_gl2_adjoint":
        pi_u = column.exner_full_pmid(ps)
        delta = (pi_i[..., 1:] - pi_i[..., :-1]).clamp_min(torch.finfo(theta_i.dtype).tiny)
        s = (pi_u - pi_i[..., :-1]) / delta
        coef_top = 0.5 * delta * (1.0 - s).square()
        coef_bot = 0.5 * delta * (1.0 - s.square())
        psi_i = torch.zeros_like(theta_i)
        psi_i[..., 1:] += w_full
        grad_theta = grad_theta + column.hydrostatic_adjoint_rhs(psi_i, ps)
        grad_theta[..., :-1] += w_full * coef_top
        grad_theta[..., 1:] += w_full * coef_bot
    else:
        raise RuntimeError(f"unsupported phi_full_mode={mode!r}")

    # Internal-energy adjoint.
    c_v = column.c_p - column.R_d
    if mode == "direct_u_gl2_adjoint":
        pi_full = column.exner_full_pmid(ps)
        delta = (pi_i[..., 1:] - pi_i[..., :-1]).clamp_min(torch.finfo(theta_i.dtype).tiny)
        s = (pi_full - pi_i[..., :-1]) / delta
        scale = w_full * c_v * (pi_full / column.c_p)
        grad_theta[..., :-1] += scale * (1.0 - s)
        grad_theta[..., 1:] += scale * s
    else:
        if mode in ("linear_pfull_adjoint", "direct_u_linear_adjoint"):
            pi_full = column.exner_full_pmid(ps)
        else:
            pi_full = column.exner_full(ps)
        scale = 0.5 * w_full * c_v * (pi_full / column.c_p)
        grad_theta[..., :-1] += scale
        grad_theta[..., 1:] += scale

    return grad_theta / dm.clamp_min(torch.finfo(dm.dtype).tiny)

def _dry_energy_gradient_ps_local_terms(
    self,
    state: State,
    theta_i: torch.Tensor,
    *,
    dp: torch.Tensor | None = None,
    dm: torch.Tensor | None = None,
    grad_mtheta: torch.Tensor | None = None,
) -> dict[str, dict[str, torch.Tensor]]:
    """
    Local closed-form ``dE/dps`` pieces at fixed prognostic ``mtheta``.

    The dry energy depends on ``ps`` directly through layer thicknesses and
    Exner-pressure placement, and indirectly because
    ``theta = mtheta / dm(dp(ps))``.  The indirect part is

        -sum_i (dE/dmtheta_i) * theta_i * d(dm_i)/dps.

    The column-summed terms mirror ``total_kinetic_energy``,
    ``total_potential_energy``, and ``total_internal_energy``:

    - ``kinetic_mass``: pressure-thickness weighting of existing KE;
    - ``potential``: explicit ``ps`` dependence of layer mass and phi;
    - ``internal``: explicit ``ps`` dependence of layer mass and T;
    - ``theta_mass_coupling``: induced theta change at fixed mtheta.
    """

    ps = state.ps
    column = state.column
    dp = state.layer_dp() if dp is None else dp
    dm = self._interface_mass(dp) if dm is None else dm
    grad_mtheta = (
        self._dry_energy_gradient_mtheta_for_coord_work(state, theta_i, dp=dp, dm=dm)
        if grad_mtheta is None
        else grad_mtheta
    )

    dtype = ps.dtype
    device = ps.device
    n_levels = int(column.n_levels)
    layer_shape = [1] * dp.ndim
    layer_shape[-1] = n_levels
    interface_shape = [1] * theta_i.ndim
    interface_shape[-1] = n_levels + 1

    B_i = column.B_interface.to(dtype=dtype, device=device).reshape(interface_shape)
    dB_layer = (
        (column.B_interface[1:] - column.B_interface[:-1])
        .to(dtype=dtype, device=device)
        .reshape(layer_shape)
    )
    dB_layer = dB_layer.expand_as(dp)
    dmdps = self._interface_mass(dB_layer)

    area_shape = [1] * dp.ndim
    area_shape[-2] = state.mesh.nCells
    area_shape[-1] = 1
    area_over_g = state.mesh.cell_area.reshape(area_shape).to(dtype=dtype, device=device) / float(self.gravity)

    p_i = column.interface_pressure(ps)
    pi_i = column.exner_interface(ps)
    dpi_i = column.kappa * pi_i / p_i.clamp_min(torch.finfo(dtype).tiny) * B_i
    delta_pi = pi_i[..., 1:] - pi_i[..., :-1]
    ddelta_pi = dpi_i[..., 1:] - dpi_i[..., :-1]

    theta_top = theta_i[..., :-1]
    theta_bot = theta_i[..., 1:]
    theta_avg = 0.5 * (theta_top + theta_bot)
    d_integrand = theta_avg * ddelta_pi
    d_suffix = torch.flip(torch.cumsum(torch.flip(d_integrand, dims=[-1]), dim=-1), dims=[-1])
    dphi_i = torch.cat([d_suffix, torch.zeros_like(d_suffix[..., :1])], dim=-1)

    p_full_mid = column.full_pressure(ps)
    pi_mid = column.exner_full_pmid(ps)
    B_mid = 0.5 * (B_i[..., :-1] + B_i[..., 1:])
    dpi_mid = column.kappa * pi_mid / p_full_mid.clamp_min(torch.finfo(dtype).tiny) * B_mid

    mode = column.phi_full_mode
    if mode == "interface_mean":
        dphi_full = 0.5 * (dphi_i[..., :-1] + dphi_i[..., 1:])
        dpi_full = 0.5 * (dpi_i[..., :-1] + dpi_i[..., 1:])
        theta_full = theta_avg
        dtheta_full = torch.zeros_like(theta_full)
    elif mode in ("linear_pfull_adjoint", "direct_u_linear_adjoint"):
        dphi_full = dphi_i[..., 1:] + theta_avg * (dpi_i[..., 1:] - dpi_mid)
        dpi_full = dpi_mid
        theta_full = theta_avg
        dtheta_full = torch.zeros_like(theta_full)
    elif mode == "direct_u_gl2_adjoint":
        a = pi_i[..., :-1]
        b = pi_i[..., 1:]
        u = pi_mid
        da = dpi_i[..., :-1]
        db = dpi_i[..., 1:]
        du = dpi_mid
        delta = (b - a).clamp_min(torch.finfo(dtype).tiny)
        ddelta = db - da
        s = (u - a) / delta
        ds = ((du - da) * delta - (u - a) * ddelta) / delta.square().clamp_min(
            torch.finfo(dtype).tiny
        )
        coef_top = 0.5 * delta * (1.0 - s).square()
        coef_bot = 0.5 * delta * (1.0 - s.square())
        dcoef_top = 0.5 * ddelta * (1.0 - s).square() - delta * (1.0 - s) * ds
        dcoef_bot = 0.5 * ddelta * (1.0 - s.square()) - delta * s * ds
        dpartial = dcoef_top * theta_top + dcoef_bot * theta_bot
        dphi_full = dphi_i[..., 1:] + dpartial
        dpi_full = dpi_mid
        theta_full = theta_top + s * (theta_bot - theta_top)
        dtheta_full = ds * (theta_bot - theta_top)
    else:
        raise RuntimeError(f"unsupported phi_full_mode={mode!r}")

    phi_full = column.geopotential_full(theta_i, ps)
    temperature = column.temperature_full(theta_i, ps)
    pi_temperature = column.exner_full(ps) if mode == "interface_mean" else pi_mid
    dtemperature = (dtheta_full * pi_temperature + theta_full * dpi_full) / column.c_p
    if mode == "direct_u_gl2_adjoint":
        dtemperature = (dtheta_full * pi_mid + theta_full * dpi_mid) / column.c_p

    c_v = column.c_p - column.R_d
    ke = state.ke_cell()
    kinetic_mass_layer = area_over_g * dB_layer * ke
    potential_layer = area_over_g * (dB_layer * phi_full + dp * dphi_full)
    internal_layer = area_over_g * (dB_layer * c_v * temperature + dp * c_v * dtemperature)
    theta_mass_coupling_interface = -(grad_mtheta * theta_i * dmdps)

    kinetic_mass = kinetic_mass_layer.sum(dim=-1, keepdim=True)
    potential = potential_layer.sum(dim=-1, keepdim=True)
    internal = internal_layer.sum(dim=-1, keepdim=True)
    theta_mass_coupling = theta_mass_coupling_interface.sum(dim=-1, keepdim=True)
    total = kinetic_mass + potential + internal + theta_mass_coupling
    return {
        "layer": {
            "kinetic_mass": kinetic_mass_layer,
            "potential": potential_layer,
            "internal": internal_layer,
        },
        "interface": {
            "theta_mass_coupling": theta_mass_coupling_interface,
        },
        "column": {
            "kinetic_mass": kinetic_mass,
            "potential": potential,
            "internal": internal,
            "theta_mass_coupling": theta_mass_coupling,
            "total": total,
        },
    }

def _dry_energy_gradient_ps_terms(
    self,
    state: State,
    theta_i: torch.Tensor,
    *,
    dp: torch.Tensor | None = None,
    dm: torch.Tensor | None = None,
    grad_mtheta: torch.Tensor | None = None,
) -> dict[str, torch.Tensor]:
    """Column-summed closed-form ``dE/dps`` pieces."""

    local = self._dry_energy_gradient_ps_local_terms(
        state,
        theta_i,
        dp=dp,
        dm=dm,
        grad_mtheta=grad_mtheta,
    )
    return {
        name: value
        for name, value in local["column"].items()
    }

def _dry_energy_gradient_ps(
    self,
    state: State,
    theta_i: torch.Tensor,
    *,
    dp: torch.Tensor | None = None,
    dm: torch.Tensor | None = None,
    grad_mtheta: torch.Tensor | None = None,
) -> torch.Tensor:
    """Closed-form ``dE/dps`` at fixed prognostic ``mtheta``."""

    terms = self._dry_energy_gradient_ps_terms(
        state,
        theta_i,
        dp=dp,
        dm=dm,
        grad_mtheta=grad_mtheta,
    )
    return terms["total"]

def _coordinate_work_unscaled_reference_impl(
    self,
    state: State,
    theta_i: torch.Tensor,
    omega: torch.Tensor,
) -> torch.Tensor:
    """
    Unfitted mass-weighted local coordinate-work correction.

    For vertical full-level face k, the robust upwind scalar flux differs
    from the centred energy-compatible flux by

        delta_F[k] = omega[k] * (theta_up[k] - 0.5*(theta_i[k]+theta_i[k+1])).

    With G=dE/dmtheta, the local energy defect is

        delta_E[k] = delta_F[k] * (G[k+1] - G[k]).

    The correction adds adjacent-interface tendencies C[k], C[k+1]
    satisfying G[k] C[k] + G[k+1] C[k+1] = -delta_E[k], choosing the
    minimum norm under sum(C^2 / (cell_area*dm/g)).  This keeps the
    upwind transport path intact; it only adds the conjugate coordinate
    work implied by that upwind-vs-centred flux defect.
    """
    G = self._energy_gradient_mtheta_for_coord_work(state, theta_i)
    theta_bar = 0.5 * (theta_i[..., :-1] + theta_i[..., 1:])
    theta_up = torch.where(omega >= 0.0, theta_i[..., :-1], theta_i[..., 1:])
    delta_F = omega * (theta_up - theta_bar)

    G_upper = G[..., :-1]
    G_lower = G[..., 1:]
    target = -delta_F * (G_lower - G_upper)

    dp = state.layer_dp()
    dm = self._interface_mass(dp)
    weight = self._cell_interface_energy_weight(state, dm).clamp_min(torch.finfo(dm.dtype).tiny)
    w_upper = weight[..., :-1]
    w_lower = weight[..., 1:]
    denom = (w_upper * G_upper.square() + w_lower * G_lower.square()).clamp_min(torch.finfo(G.dtype).tiny)

    C_upper = target * w_upper * G_upper / denom
    C_lower = target * w_lower * G_lower / denom
    out = torch.zeros_like(theta_i)
    out[..., :-1] += C_upper
    out[..., 1:] += C_lower
    return out

def _coordinate_work_scale_normalized_impl(
    self,
    state: State,
    theta_i: torch.Tensor,
    omega: torch.Tensor,
    *,
    primitives: dict[str, torch.Tensor] | None = None,
) -> torch.Tensor:
    """
    Scale-normalized form of the canonical coordinate-work correction.

    This is algebraically equivalent to
    `_canonical_mtheta_coordinate_work_unscaled_reference`, but avoids
    forming `weight * G**2`.  That product is O(1e46) in standard
    Held-Suarez cases and overflows float32 even though the final
    correction is small and finite.
    """
    G = self._energy_gradient_mtheta_for_coord_work(state, theta_i, primitives=primitives)
    return self._coordinate_work_scale_normalized_from_gradient(
        state,
        theta_i,
        omega,
        G,
        dp=None if primitives is None else primitives.get("dp"),
        dm=None if primitives is None else primitives.get("dm"),
    )

def _coordinate_work_scale_normalized_from_gradient(
    self,
    state: State,
    theta_i: torch.Tensor,
    omega: torch.Tensor,
    G: torch.Tensor,
    *,
    dp: torch.Tensor | None = None,
    dm: torch.Tensor | None = None,
) -> torch.Tensor:
    """Apply the scale-normalized coordinate-work algebra for a given energy gradient."""

    theta_bar = 0.5 * (theta_i[..., :-1] + theta_i[..., 1:])
    theta_up = torch.where(omega >= 0.0, theta_i[..., :-1], theta_i[..., 1:])
    delta_F = omega * (theta_up - theta_bar)

    G_upper = G[..., :-1]
    G_lower = G[..., 1:]
    g_scale = torch.maximum(G_upper.abs(), G_lower.abs()).clamp_min(torch.finfo(G.dtype).tiny)
    g_upper = G_upper / g_scale
    g_lower = G_lower / g_scale
    scaled_target = -delta_F * (g_lower - g_upper)

    if dm is None:
        dp = state.layer_dp() if dp is None else dp
        dm = self._interface_mass(dp)
    weight = self._cell_interface_energy_weight(state, dm).clamp_min(torch.finfo(dm.dtype).tiny)
    w_upper = weight[..., :-1]
    w_lower = weight[..., 1:]
    denom = (w_upper * g_upper.square() + w_lower * g_lower.square()).clamp_min(torch.finfo(G.dtype).tiny)

    C_upper = scaled_target * w_upper * g_upper / denom
    C_lower = scaled_target * w_lower * g_lower / denom
    out = torch.zeros_like(theta_i)
    out[..., :-1] += C_upper
    out[..., 1:] += C_lower
    return out


def _mtheta_vertical(self, theta_i: torch.Tensor, omega: torch.Tensor) -> torch.Tensor:
    """
    Positivity-oriented vertical transport for CP mtheta.

    omega[..., j] is the full-level mass flux between CP interfaces j and j+1.
    Positive omega is downward: from interface j to interface j+1.

    This transports the donor interface's own theta, not the layer-mean theta.
    """
    theta_up = torch.where(
        omega >= 0.0,
        theta_i[..., :-1],   # downward: donor is upper interface j
        theta_i[..., 1:],    # upward: donor is lower interface j+1
    )

    Fv = omega * theta_up  # flux from j to j+1, positive downward

    out = torch.zeros_like(theta_i)
    out[..., :-1] -= Fv    # leaving upper interface when Fv > 0
    out[..., 1:]  += Fv    # entering lower interface when Fv > 0
    return out


def _energy_gradient_mtheta_for_coord_work(
    self,
    state: State,
    theta_i: torch.Tensor,
    *,
    primitives: dict[str, torch.Tensor] | None = None,
) -> torch.Tensor:
    """Return dE/dmtheta for the active thermodynamics strategy."""

    return self.thermodynamics.energy_gradient_mtheta_for_coord_work(
        self,
        state,
        theta_i,
        primitives=primitives,
    )

def _canonical_mtheta_coordinate_work_unscaled_reference(
    self,
    state: State,
    theta_i: torch.Tensor,
    omega: torch.Tensor,
) -> torch.Tensor:
    """Return the original unscaled coordinate-work reference algebra."""

    return self._coordinate_work_unscaled_reference_impl(state, theta_i, omega)

def _canonical_mtheta_coordinate_work(
    self,
    state: State,
    theta_i: torch.Tensor,
    omega: torch.Tensor,
    *,
    primitives: dict[str, torch.Tensor] | None = None,
) -> torch.Tensor:
    """Return coordinate work from the active coordinate-work strategy."""

    if primitives is not None:
        tendency_with_primitives = getattr(self.coordinate_work, "tendency_with_primitives", None)
        if tendency_with_primitives is not None:
            return tendency_with_primitives(self, state, theta_i, omega, primitives=primitives)
    return self.coordinate_work.tendency(self, state, theta_i, omega)


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
    """
    Positivity-oriented horizontal advection tendency for CP mtheta.

    Each full-level mass flux F_j = dp_j_e * u_j contributes half its
    flux divergence to the upper CP interface j and half to the lower
    CP interface j+1.

    Unlike the theta_full scheme, this transports the actual CP-interface
    theta_i[k] for each receiving interface. For uniform theta, it reduces
    exactly to theta * dm_t,horiz.
    """
    mesh = state.mesh

    # Full-level horizontal mass flux: [..., nEdges, N]
    if mass_flux is None:
        dp_e = mesh.cell_to_edge(dp) if dp_e is None else dp_e
        mass_flux = dp_e * u_e

    c0 = mesh.cellsOnEdge[:, 0]
    c1 = mesh.cellsOnEdge[:, 1]

    # For each full level j, use the same mass flux to advect the upper
    # interface theta_i[j] and lower interface theta_i[j+1] separately.
    theta_upper_c0 = theta_i[..., c0, :-1]   # [..., nEdges, N]
    theta_upper_c1 = theta_i[..., c1, :-1]
    theta_lower_c0 = theta_i[..., c0, 1:]    # [..., nEdges, N]
    theta_lower_c1 = theta_i[..., c1, 1:]

    theta_upper_e = torch.where(mass_flux >= 0.0, theta_upper_c0, theta_upper_c1)
    theta_lower_e = torch.where(mass_flux >= 0.0, theta_lower_c0, theta_lower_c1)

    # Half of each full-layer mass flux belongs to each adjacent CP interface.
    div_upper = mesh.div(0.5 * mass_flux * theta_upper_e)  # contribution to interface j
    div_lower = mesh.div(0.5 * mass_flux * theta_lower_e)  # contribution to interface j+1

    dmtheta_dt = torch.zeros_like(theta_i)
    dmtheta_dt[..., :-1] -= div_upper
    dmtheta_dt[..., 1:] -= div_lower
    return dmtheta_dt

def _mtheta_advection_tendency(
    self,
    state: State,
    theta_i: torch.Tensor,
    primitives: dict[str, torch.Tensor],
) -> torch.Tensor:
    """Return the advective CP ``mtheta_i`` tendency.

    The canonical path is the historical split form: horizontal donor-cell
    CP-interface transport plus vertical donor-cell transport. Experimental
    subclasses can override this combined hook when the limiter needs to
    see horizontal and vertical scalar fluxes at the same time.
    """

    return self._mtheta_tendency(
        state,
        state.u_e,
        theta_i,
        primitives["dp"],
        dp_e=primitives.get("dp_e"),
        mass_flux=primitives.get("mass_flux"),
    ) + self._mtheta_vertical(theta_i, primitives["omega"])

def mtheta_tendency_terms(
    self,
    state: State,
    *,
    primitives: dict[str, torch.Tensor] | None = None,
) -> dict[str, torch.Tensor]:
    """Return CP potential-temperature terms owned by thermodynamics."""

    ctx = self._transport_primitives(state) if primitives is None else primitives
    theta = ctx["theta"]
    horizontal = self._mtheta_tendency(
        state,
        state.u_e,
        theta,
        ctx["dp"],
        dp_e=ctx.get("dp_e"),
        mass_flux=ctx.get("mass_flux"),
    )
    vertical = self._mtheta_vertical(theta, ctx["omega"])
    advective = self._mtheta_advection_tendency(state, theta, ctx)
    coord_work = self._canonical_mtheta_coordinate_work(state, theta, ctx["omega"], primitives=ctx)
    return {
        "horizontal": horizontal,
        "vertical": vertical,
        "advective": advective,
        "coord_work": coord_work,
        "total": advective + coord_work,
    }

__all__ = ['_cell_interface_energy_weight', '_cell_full_energy_weight', '_dry_energy_gradient_mtheta_for_coord_work', '_dry_energy_gradient_ps_local_terms', '_dry_energy_gradient_ps_terms', '_dry_energy_gradient_ps', '_coordinate_work_unscaled_reference_impl', '_coordinate_work_scale_normalized_impl', '_coordinate_work_scale_normalized_from_gradient', '_mtheta_vertical', '_energy_gradient_mtheta_for_coord_work', '_canonical_mtheta_coordinate_work_unscaled_reference', '_canonical_mtheta_coordinate_work', '_mtheta_tendency', '_mtheta_advection_tendency', 'mtheta_tendency_terms']
