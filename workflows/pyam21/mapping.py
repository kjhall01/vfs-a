"""Explicit CP/column and work-adjoint cell/edge mappings for this workflow."""

import torch
from workflows.momentum import KineticWindMap
from vfs.atmosphere.moist_cp import interface_pressure_mass, reconstruct_moist_cp


def average(x):
    return 0.5 * (x[:, :-1] + x[:, 1:])


def distribute(amount):
    """Layer extensive amount -> two adjacent CP nodes, with half to each."""
    zero = torch.zeros_like(amount[:, :1])
    return 0.5 * (torch.cat((amount, zero), 1) + torch.cat((zero, amount), 1))


def species_increment(fractions, amount):
    """Distribute additions equally; withdrawals use each node's species share.

    A layer draws only from its two contributions to the CP mass weights. If
    every layer's endpoint species amount is nonnegative, withdrawals cannot
    exhaust a CP species. This avoids negative condensate at sharp cloud edges.
    No clipping or reduction of the requested column increment is performed.
    """
    pair = fractions[:, :-1] + fractions[:, 1:]
    denominator = torch.where(pair > 0, pair, torch.ones_like(pair))
    upper_share = torch.where(amount < 0, fractions[:, :-1] / denominator, 0.5)
    zero = torch.zeros_like(amount[:, :1])
    return torch.cat((amount * upper_share, zero), 1) + torch.cat(
        (zero, amount * (1 - upper_share)), 1
    )


def species_endpoint(contents, dp, amount, endpoint_fraction):
    """Same donor map, evaluated from remaining amounts when a layer loses water.

    Subtracting a nearly exhausted store can create a spurious negative CP amount.
    The already-computed AM2 endpoint instead supplies the remaining layer store.
    Nonnegative available stores define donor weights; admitted signed roundoff
    cannot act as a negative donor. The endpoint itself is never clipped. A
    species with no sources anywhere in a column retains its contents bitwise.
    Additions retain the half-to-each convention for physical nonnegative stores.
    """
    fractions = contents / interface_pressure_mass(dp)[..., None]
    remaining = dp[..., None] * endpoint_fraction
    old_upper = 0.5 * dp[..., None] * fractions[:, :-1]
    old_lower = 0.5 * dp[..., None] * fractions[:, 1:]
    # These are interpolation weights, not clipped prognostic amounts. Their
    # normalization transfers precisely the supplied (even signed) endpoint.
    upper_weight = (old_upper + 0.5 * amount.clamp_min(0)).clamp_min(0)
    lower_weight = (old_lower + 0.5 * amount.clamp_min(0)).clamp_min(0)
    total_weight = upper_weight + lower_weight
    denominator = torch.where(
        total_weight > 0, total_weight, torch.ones_like(total_weight)
    )
    share = torch.where(total_weight > 0, upper_weight / denominator, 0.5)
    upper, lower = remaining * share, remaining * (1 - share)
    zero = torch.zeros_like(upper[:, :1])
    result = torch.cat((upper, zero), 1) + torch.cat((zero, lower), 1)
    return torch.where((amount == 0).all(1, keepdim=True), contents, result)


class ColumnMap:
    def __init__(self, model):
        self.model = model
        mesh = model.mesh
        xyz = mesh.cell_xyz
        radius = torch.linalg.vector_norm(xyz[:, :2], dim=-1)
        if (radius <= torch.finfo(xyz.dtype).eps).any():
            raise ValueError(
                "column east/north basis is undefined at a geographic pole"
            )
        self.east = (
            torch.stack((-xyz[:, 1], xyz[:, 0], torch.zeros_like(radius)), -1)
            / radius[:, None]
        )
        self.north = torch.linalg.cross(xyz, self.east)
        self.wind = KineticWindMap(mesh)
        self.coefficients = self.wind.coefficients

    def vector(self, wind):
        return self.wind.vector(wind)

    def transpose(self, covector):
        return self.wind.transpose(covector)

    def metric(self, state):
        mesh = state.mesh
        return (
            (mesh.dc_edge * mesh.dv_edge)[:, None]
            * mesh.cell_to_edge(state.column.interface_pressure(state.ps).diff())
            / self.model.dynamics.kernel.gravity
        )

    def columns(self, state, cloud, omega=None):
        """Water and enthalpy preserving readout, with the VFS hydrostatic height.

        T is selected so that the VFS mixture enthalpy at each full level equals
        that layer's native two-point enthalpy quadrature. It is not an arithmetic
        temperature interpolation. No readout is written back into the state.
        """
        model = self.model
        thermo = model.dynamics.thermodynamics
        p = state.column.interface_pressure(state.ps)
        dp = p.diff()
        quad = reconstruct_moist_cp(
            p, state.entropy_content, state.water_content, thermo
        )
        qw = state.water_content / interface_pressure_mass(dp)[..., None]
        q = average(qw)
        mixture = torch.cat((1 - q.sum(-1, keepdim=True), q), -1)
        h = thermo.enthalpy(quad["temperature"], quad["fractions"]).mean(-1)
        T = thermo.T_ref + (
            h - thermo.enthalpy_reference(mixture)
        ) / thermo.heat_capacity_p(mixture)
        # Simmons-Burridge pressure, stable for thin layers and positive top.
        ratio = dp / p[:, :-1]
        pfull = p[:, 1:] * torch.exp(-1 + torch.log1p(ratio) / ratio)
        phi = model.dynamics.energy_terms(state)["geopotential_interface"]
        zhalf = (phi - phi[:, -1:]) / model.dynamics.kernel.gravity
        zfull = zhalf[:, 1:] + (
            phi[:, :-1] - phi[:, 1:]
        ) / model.dynamics.kernel.gravity * (
            torch.log(p[:, 1:] / pfull) / torch.log(p[:, 1:] / p[:, :-1])
        )
        v = self.vector(state.u_e)
        u = (v * self.east[:, None]).sum(-1)
        north = (v * self.north[:, None]).sum(-1)
        R = torch.cat((q, average(cloud)[..., None]), -1)
        # The caller supplies physical pressure velocity, not the hybrid-grid
        # relative flux called omega inside VFS transport.
        omega = torch.zeros_like(T) if omega is None else omega
        if (
            omega.shape != T.shape
            or omega.dtype != T.dtype
            or omega.device != T.device
            or not torch.isfinite(omega).all()
        ):
            raise ValueError(
                "omega must be finite physical Dp/Dt [cell,level], matching state"
            )
        return dict(T=T, u=u, v=north, tracers=R, ps=state.ps[:, 0]), dict(
            pfull=pfull, phalf=p, zfull=zfull, zhalf=zhalf, omega=omega
        )

    def energy(self, state):
        """Full native atmosphere + fixed-pressure lid energy by cell, J/m2."""
        d = self.model.dynamics
        p = state.column.interface_pressure(state.ps)
        quad = reconstruct_moist_cp(
            p, state.entropy_content, state.water_content, d.thermodynamics
        )
        h = d.thermodynamics.enthalpy(quad["temperature"], quad["fractions"]).mean(-1)
        ke = d.kernel.cell_kinetic_energy(d.transport_state(state))
        phi_s = torch.as_tensor(state.column.phi_s, dtype=p.dtype, device=p.device)
        return (p.diff() * (h + ke)).sum(-1) / d.kernel.gravity + (
            p[:, -1] - p[:, 0]
        ) * phi_s / d.kernel.gravity
