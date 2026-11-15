"""Conservative AM2 energy transfer and an independent open-system ledger.

AM2's layer thermal/latent + kinetic increments are transferred as energy,
not as a temperature increment. Boundary water enthalpies use the host mixture
datum. Any AM2 energy residual is retained, never balanced away by the adapter.
"""

from dataclasses import dataclass, replace
import torch
from pyam21.constants import CP_AIR, HLV, HLF
from vfs.atmosphere.moist_cp import interface_pressure_mass, reconstruct_moist_cp


@dataclass(frozen=True)
class WaterBoundary:
    """Temperatures (K) and tangential Cartesian velocities (m/s) at the surface.

    These are explicit boundary assumptions, not precipitation diagnostics
    supplied by AM2. Positive vapor flux uses SST and a stationary surface by
    default; departing vapor and precipitation use the old lowest air level.
    Rain and snow temperatures may be specified separately by a future coupler.
    """

    vapor_temperature: torch.Tensor
    rain_temperature: torch.Tensor
    snow_temperature: torch.Tensor
    vapor_velocity: torch.Tensor
    precipitation_velocity: torch.Tensor

    @classmethod
    def prescribed_ocean(cls, adapter, state, column, out):
        entering = out["flux_q"] >= 0
        air_velocity = adapter.mapping.vector(state.atmosphere.u_e)[:, -1]
        return cls(
            torch.where(entering, state.surface["t_surf"], column["T"][:, -1]),
            column["T"][:, -1],
            column["T"][:, -1],
            torch.where(
                entering[:, None], torch.zeros_like(air_velocity), air_velocity
            ),
            air_velocity,
        )

    def validate(self, adapter, reference):
        for name in ("vapor_temperature", "rain_temperature", "snow_temperature"):
            value = getattr(self, name)
            if (
                value.shape != reference.shape
                or value.dtype != reference.dtype
                or value.device != reference.device
                or not torch.isfinite(value).all()
                or (value <= 0).any()
            ):
                raise ValueError(
                    "boundary temperatures must be positive matching cell tensors"
                )
        for name in ("vapor_velocity", "precipitation_velocity"):
            value = getattr(self, name)
            if (
                value.shape != (*reference.shape, 3)
                or value.dtype != reference.dtype
                or value.device != reference.device
                or not torch.isfinite(value).all()
            ):
                raise ValueError(
                    "boundary velocities must be finite matching cell Cartesian tensors"
                )
            radial = (value * adapter.model.mesh.cell_xyz).sum(-1)
            tol = 128 * torch.finfo(value.dtype).eps * value.abs().amax(-1).clamp_min(1)
            if (radial.abs() > tol).any():
                raise ValueError("boundary velocities must be tangential")


def bottom_only(value, layers):
    """A surface amount belongs to the bottom layer, never all layers."""
    return torch.cat((torch.zeros_like(layers[:, :-1]), value[:, None]), 1)


def exchanges(adapter, state, dt, column, geometry, out, boundary=None):
    """Independent rates and fixed-carrier AM2 increments, all in J/m².

    Surface stress acts on a stationary prescribed ocean: external stress work
    is zero. AM2's dissipative heating can therefore return atmospheric KE loss
    to heat. A missing or imperfect upstream return remains in am2_residual.
    """
    boundary = (
        WaterBoundary.prescribed_ocean(adapter, state, column, out)
        if boundary is None
        else boundary
    )
    boundary.validate(adapter, column["ps"])
    th = adapter.model.dynamics.thermodynamics
    mass = geometry["phalf"].diff() / adapter.model.dynamics.kernel.gravity
    E, rain, snow = (dt * out[k] for k in ("flux_q", "lprec", "fprec"))
    if (rain < 0).any() or (snow < 0).any():
        raise ValueError("precipitation fluxes must be nonnegative")
    hv = th.L_v_ref + th.cp_v * (boundary.vapor_temperature - th.T_ref)
    hl = th.cp_l * (boundary.rain_temperature - th.T_ref)
    hi = -th.L_f_ref + th.cp_i * (boundary.snow_temperature - th.T_ref)
    vapor_h, rain_h, snow_h = E * hv, -rain * hl, -snow * hi
    phi = torch.as_tensor(state.atmosphere.column.phi_s, dtype=E.dtype, device=E.device)
    potential = (E - rain - snow) * phi
    mass_ke = E * 0.5 * boundary.vapor_velocity.square().sum(-1) - (
        rain + snow
    ) * 0.5 * boundary.precipitation_velocity.square().sum(-1)
    water_momentum = (
        E[:, None] * boundary.vapor_velocity
        - (rain + snow)[:, None] * boundary.precipitation_velocity
    )
    layer_momentum = torch.zeros((*mass.shape, 3), dtype=mass.dtype, device=mass.device)
    layer_momentum = torch.cat((layer_momentum[:, :-1], water_momentum[:, None]), 1)
    impulse = adapter.mapping.transpose(
        adapter.model.mesh.cell_area[:, None, None] * layer_momentum
    )
    dT, dR = dt * out["tdt"], dt * out["rdt"]
    if adapter.energy_mode == "conservative" and "endpoint_tracers" in out:
        dT = out["endpoint_temperature"] - column["T"]
        dR = out["endpoint_tracers"] - column["tracers"]
    proxy_layer = mass * (CP_AIR * dT + HLV * dR[..., 0] - HLF * dR[..., 2])
    du, dv = dt * out["udt"], dt * out["vdt"]
    kinetic_layer = mass * (
        (column["u"] + 0.5 * du) * du + (column["v"] + 0.5 * dv) * dv
    )
    radiation = out["radiation"]
    rad_heat = torch.zeros_like(E)
    rad_boundary = torch.zeros_like(E)
    if radiation is not None:
        net = radiation["dfsw"] - radiation["ufsw"] - radiation["flxnet"]
        rad_boundary = dt * (net[:, 0] - net[:, -1])
        rad_heat = (mass * CP_AIR * dt * radiation["tdt_rad"]).sum(-1)
    sensible = dt * out["flux_t"]
    am2_boundary = sensible + HLV * E + HLF * snow + rad_boundary
    datum = vapor_h + rain_h + snow_h - HLV * E - HLF * snow
    terms = dict(
        sensible=sensible,
        vapor_enthalpy=vapor_h,
        rain_enthalpy=rain_h,
        snow_enthalpy=snow_h,
        water_potential=potential,
        water_kinetic=mass_ke,
        radiation=rad_boundary,
        stress_work=torch.zeros_like(E),
    )
    external = sum(terms.values())
    residual = proxy_layer.sum(-1) + kinetic_layer.sum(-1) - am2_boundary
    # Preserve the upstream increment, with only the declared surface reference
    # and mechanical transport added. No subtraction of the upstream residual.
    target = (
        proxy_layer + kinetic_layer + bottom_only(datum + potential + mass_ke, mass)
    )
    return dict(
        boundary=boundary,
        terms=terms,
        external=external,
        target=target,
        am2_residual=residual,
        am2_kinetic=kinetic_layer.sum(-1),
        am2_thermal=proxy_layer.sum(-1),
        am2_boundary=am2_boundary,
        datum=datum,
        mass_impulse=impulse,
        radiation_residual=rad_heat - rad_boundary,
    )


def layer_energy(adapter, state):
    """Native full-H ownership by cell and layer, J/m² (includes the lid)."""
    d = adapter.model.dynamics
    p = state.column.interface_pressure(state.ps)
    quad = reconstruct_moist_cp(
        p, state.entropy_content, state.water_content, d.thermodynamics
    )
    h = d.thermodynamics.enthalpy(quad["temperature"], quad["fractions"]).mean(-1)
    ke = d.kernel.cell_kinetic_energy(d.transport_state(state))
    phi = torch.as_tensor(state.column.phi_s, dtype=p.dtype, device=p.device)
    return p.diff() / d.kernel.gravity * (h + ke + phi[..., None])


def transfer_energy(adapter, before, provisional, target, *, max_iterations=12):
    """Invert each layer's enthalpy constraint on the native CP entropy grid.

    The L constraints have L+1 CP log-temperature unknowns. Each Newton step
    takes the minimum Euclidean-norm log-temperature increment. The analytic
    Jacobian has only two nonzero entries per row. No forward autograd, global
    energy rescaling, state clipping or cross-column heat transfer is used.
    Failure to attain a positive finite state rejects the whole physics call.
    """
    d = adapter.model.dynamics
    th = d.thermodynamics
    p = provisional.column.interface_pressure(provisional.ps)
    dp = p.diff()
    mass = interface_pressure_mass(dp)
    q = provisional.water_content / mass[..., None]
    mix = torch.cat((1 - q.sum(-1, keepdim=True), q), -1)
    cp = th.heat_capacity_p(mix)
    capacity = mass * cp
    p0 = before.column.interface_pressure(before.ps)
    old = reconstruct_moist_cp(p0, before.entropy_content, before.water_content, th)
    cp0 = th.heat_capacity_p(old["fractions"])
    h0 = th.enthalpy(old["temperature"], old["fractions"])
    ke0 = d.kernel.cell_kinetic_energy(d.transport_state(before))
    ke1 = d.kernel.cell_kinetic_energy(d.transport_state(provisional))
    phi = torch.as_tensor(before.column.phi_s, dtype=p.dtype, device=p.device)
    mechanical = (
        dp * ke1 - p0.diff() * ke0 + (dp - p0.diff()) * phi[..., None]
    ) / d.kernel.gravity
    desired = target - mechanical
    x = p.new_tensor([0.5 - 0.5 / 3**0.5, 0.5 + 0.5 / 3**0.5])
    entropy = provisional.entropy_content
    initial_entropy = entropy
    scale = (p0.diff() / d.kernel.gravity * (cp0 * old["temperature"]).mean(-1)).abs()
    tolerance = 128 * torch.finfo(p.dtype).eps * scale.clamp_min(1)
    for iteration in range(max_iterations + 1):
        quad = reconstruct_moist_cp(p, entropy, provisional.water_content, th)
        T, qg = quad["temperature"], quad["fractions"]
        cpg = th.heat_capacity_p(qg)
        # Evaluate the finite enthalpy difference without subtracting two large
        # column totals. All mass, composition and pressure effects are included.
        dh = (
            cpg * (T - old["temperature"])
            + (cpg - cp0) * (old["temperature"] - th.T_ref)
            + th.enthalpy_reference(qg)
            - th.enthalpy_reference(old["fractions"])
        )
        change = (dp * dh.mean(-1) + (dp - p0.diff()) * h0.mean(-1)) / d.kernel.gravity
        residual = change - desired
        if (
            not torch.isfinite(T).all()
            or not torch.isfinite(residual).all()
            or (T <= 0).any()
        ):
            raise FloatingPointError(
                "nonfinite/positive-temperature failure in CP energy transfer"
            )
        # Two Newton maps retain the implicit derivative even if a zero-valued
        # differentiable source happens to satisfy the initial constraint.
        if (residual.abs() <= tolerance).all() and (
            iteration >= 2 or not torch.is_grad_enabled()
        ):
            end = replace(provisional, entropy_content=entropy)
            return end, dict(
                layer_residual=residual,
                iterations=iteration,
                entropy_increment=entropy - initial_entropy,
                log_temperature_increment=(entropy - initial_entropy) / capacity,
            )
        if iteration == max_iterations:
            break
        # d H_layer / d log(T_CP), with pressure and composition fixed.
        left = dp / d.kernel.gravity * cp[:, :-1] * (T * (1 - x)).mean(-1)
        right = dp / d.kernel.gravity * cp[:, 1:] * (T * x).mean(-1)
        length = dp.shape[1]
        J = (
            torch.diag_embed(left)
            @ torch.eye(length, length + 1, dtype=p.dtype, device=p.device)
            + torch.diag_embed(right)
            @ torch.eye(length + 1, dtype=p.dtype, device=p.device)[1:]
        )
        # Row scaling avoids pressure-layer-size conditioning in the solve.
        row_scale = torch.linalg.vector_norm(J, dim=-1)
        A = J / row_scale[..., None]
        multiplier = torch.linalg.solve(
            A @ A.transpose(-1, -2), (residual / row_scale)[..., None]
        )
        delta = -(A.transpose(-1, -2) @ multiplier).squeeze(-1)
        # A bounded Newton step is a solver line search, not a clipped state.
        factor = torch.minimum(
            torch.ones_like(delta[:, :1]),
            0.25 / delta.abs().amax(-1, keepdim=True).clamp_min(0.25),
        )
        entropy = entropy + capacity * (factor * delta)
    raise FloatingPointError("CP energy transfer did not converge; no state committed")
