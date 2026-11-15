"""Category phase equilibrium with conservative ocean/ice momentum exchange.

The phase solve uses the nonmechanical terms of the SAME Hamiltonian. The
subsequent donor-velocity momentum transfer conserves KE+ocean TKE, so no energy
is missing from that solve. No state clipping or post-step energy repair.
"""

from dataclasses import dataclass, replace
import math
import torch
from ridging import RidgingState, CategoryIce
from roquet import RoquetWorkspace
from ocean_ice import safe
from vfs.ocean.energy import ConstantDensityEnergy


@dataclass(frozen=True)
class CategoryPhaseReport:
    external_heat: torch.Tensor  # J/cell
    energy_change_minus_heat: (
        torch.Tensor
    )  # J/cell; includes conservative edge work redistribution
    water_residual: torch.Tensor  # kg/cell
    momentum_residual: torch.Tensor  # kg m/s, per edge generalized momentum
    kinetic_to_tke: torch.Tensor  # J/cell, nonnegative


class CategoryThermodynamics:
    """Fresh zero-layer ice, empirical CT liquidus, two-category geometry.

    Growth/melt scales both category volumes. Footprint scales at fixed category
    thickness until coverage reaches one, after which growth thickens both.
    New ice starts in category zero at nucleation_thickness. This geometry law
    is explicit and idealized; it is not a thermodynamic thickness-distribution
    model such as Icepack. All categories vanish on complete melting.
    """

    def __init__(self, coupled, *, nucleation_thickness=0.1, min_liquid=1.0):
        if any(
            not math.isfinite(x) or x <= 0 for x in (nucleation_thickness, min_liquid)
        ):
            raise ValueError("positive finite thickness parameters required")
        self.coupled = coupled
        self.H = coupled.hamiltonian
        self.nucleation_thickness = nucleation_thickness
        self.min_liquid = min_liquid

    def energy_values(self, initial, top, ct, salinity, eta, ice):
        """Exactly the phase-varying terms of H; unchanged lower layers cancel.

        Reuse the Roquet parcel-potential kernel and its quadrature. Only the
        top liquid layer changes in this phase path, so no deep-column EOS work
        belongs inside the scalar root iterations.
        """
        H = self.H
        s = initial.ocean.ocean
        p = H.slab.parameters
        node, weight = H.model.energy.base._quadrature(top)
        z = eta - 0.5 * top
        psi, *_ = H.model.energy.base.parcel_potential(
            ct[..., None],
            salinity[..., None],
            z[..., None] + top[..., None] * (node - 0.5),
        )
        rho = s.column.rho0
        g = s.column.gravity
        V = ice.volume.sum(-1, keepdim=True)
        m = p.density * V
        value = (
            rho * top * (H.model.config.cp * ct + (psi * weight).sum(-1))
            + 0.5 * rho * g * eta.square()
        )
        value = value - p.latent_heat * m + g * (m * eta + m.square() / (2 * p.density))
        value = value + H.coefficient(initial) * (
            (ice.volume * ice.thickness()).sum(-1, keepdim=True) - V.square()
        )
        return torch.where(H.model.energy.mass_area(s) > 0, value, 0.0)

    def thermal_energy(self, state):
        s = state.ocean.ocean
        return self.energy_values(
            state,
            s.h[:, :1],
            s.temperature()[:, :1],
            s.salinity()[:, :1],
            s.column.eta(s.h),
            state.ice,
        )

    def root_energy(self, initial, m):
        s = initial.ocean.ocean
        change = (self.H.mass(initial) - m) / s.column.rho0
        top = s.h[:, :1] + change
        salt = s.salt_h[:, :1] / top
        ice, *_ = self.geometry(initial, m)
        return self.energy_values(
            initial,
            top,
            -self.H.slab.parameters.freezing_slope * salt,
            salt,
            s.column.eta(s.h) + change,
            ice,
        )

    def geometry(self, initial, m):
        old = self.H.mass(initial)
        ice = initial.ice
        rho = self.H.slab.parameters.density
        total = ice.area.sum(-1, keepdim=True)
        ratio = m / safe(old)
        factor = torch.minimum(ratio, 1 / safe(total))
        a_existing = ice.area * factor
        v_existing = ice.volume * ratio
        new_fraction = torch.minimum(
            m / (rho * self.nucleation_thickness), torch.ones_like(m)
        )
        a_new = torch.cat((new_fraction, torch.zeros_like(m)), -1)
        v_new = torch.cat((m / rho, torch.zeros_like(m)), -1)
        a = torch.where(old > 0, a_existing, a_new)
        v = torch.where(old > 0, v_existing, v_new)
        # Directional geometry derivatives along the declared phase path.
        da_existing = torch.where(ratio < 1 / safe(total), ice.area / safe(old), 0.0)
        da_new = torch.cat(
            (
                torch.where(
                    new_fraction < 1,
                    torch.full_like(m, 1 / (rho * self.nucleation_thickness)),
                    0.0,
                ),
                torch.zeros_like(m),
            ),
            -1,
        )
        da = torch.where(old > 0, da_existing, da_new)
        dv = torch.where(
            old > 0,
            ice.volume / safe(old),
            torch.cat((torch.full_like(m, 1 / rho), torch.zeros_like(m)), -1),
        )
        virtual = torch.where(
            old > 0,
            ice.thickness(),
            torch.full_like(ice.area, self.nucleation_thickness),
        )
        thickness = torch.where(a > 0, v / safe(a), virtual)
        return CategoryIce(a, v, ice.velocity), da, dv, thickness

    def phase_state(self, initial, m):
        s = initial.ocean.ocean
        old = self.H.mass(initial)
        wet = self.H.model.energy.mass_area(s) > 0
        h = torch.cat((s.h[:, :1] + (old - m) / s.column.rho0, s.h[:, 1:]), -1)
        th = torch.cat(
            (
                torch.where(
                    wet,
                    -self.H.slab.parameters.freezing_slope * s.salt_h[:, :1],
                    s.temp_h[:, :1],
                ),
                s.temp_h[:, 1:],
            ),
            -1,
        )
        physical = s.with_tensors(s.u_e, h, th, s.salt_h, s.tracer_h)
        ice, *_ = self.geometry(initial, m)
        return RidgingState(replace(initial.ocean, ocean=physical), ice)

    def derivative(self, initial, state, m):
        s = state.ocean.ocean
        H = self.H
        rho = s.column.rho0
        g = s.column.gravity
        rhoi = H.slab.parameters.density
        area = s.mesh.cell_area[:, None]
        base = H.model.energy.base
        dh = (base.gradients(s)["h"] - base.mechanical.kinetic_thickness_gradient(s))[
            :, :1
        ] / area
        _, da, dv, h = self.geometry(initial, m)
        V = m / rhoi
        return (
            -H.slab.parameters.latent_heat
            + g * s.column.eta(s.h)
            + g * m / rhoi
            - (dh + g * m) / rho
            + H.coefficient(state)
            * ((2 * h * dv - h.square() * da).sum(-1, keepdim=True) - 2 * V / rhoi)
        )

    def exchange_momentum(self, initial, end):
        H = self.H
        s = initial.ocean.ocean
        t = end.ocean.ocean
        mechanical = H.model.energy.base.mechanical
        Mi0, Mi1 = H.metric(initial), H.metric(end)
        Mo0, Mo1 = mechanical.metric(s)[:, :1], mechanical.metric(t)[:, :1]
        delta = Mi1 - Mi0
        ui = initial.ice.velocity
        uo = s.u_e[:, :1]
        # Keep the donor velocity exactly. Subtracting nearly equal momenta
        # then dividing by a vanishing remaining ice mass is catastrophic at
        # complete melt. Recipient updates are bounded convex velocity mixes.
        freeze = delta >= 0
        new_ui = torch.where(
            Mi1 > 0, torch.where(freeze, ui + delta / safe(Mi1) * (uo - ui), ui), 0.0
        )
        new_uo = torch.where(freeze, uo, uo + (-delta) / safe(Mo1) * (ui - uo))
        physical = t.with_tensors(
            torch.cat((new_uo, t.u_e[:, 1:]), -1), t.h, t.temp_h, t.salt_h, t.tracer_h
        )
        # Inelastic mixing at the recipient; pure transfer from the donor.
        reduced = torch.where(
            delta >= 0, delta * Mi0 / safe(Mi1), (-delta) * Mo0 / safe(Mo1)
        )
        gain = 0.5 * reduced * (uo - ui).square()
        cells = ConstantDensityEnergy.interpolation_transpose(s.mesh, gain)
        mass = H.model.energy.mass_area(t)
        dk = torch.cat((cells / safe(mass), torch.zeros_like(t.h[:, 1:])), -1)
        out = RidgingState(
            replace(end.ocean, ocean=physical, tke_h=end.ocean.tke_h + dk),
            replace(end.ice, velocity=new_ui),
        )
        residual = Mi1 * new_ui + Mo1 * new_uo - Mi0 * ui - Mo0 * uo
        return out, cells[:, 0], residual

    def apply(self, initial, dt, *, heat_flux):
        self.coupled.check(initial)
        if not math.isfinite(dt) or dt <= 0:
            raise ValueError("positive finite dt required")
        H = self.H
        s = initial.ocean.ocean
        m0 = H.mass(initial)
        q = heat_flux
        wet = H.model.energy.mass_area(s) > 0
        if (
            q.shape != m0.shape
            or q.dtype != s.h.dtype
            or q.device != s.h.device
            or not torch.isfinite(q).all()
            or (q[~wet] != 0).any()
        ):
            raise ValueError(
                "heat flux must match [cell,1], finite and zero on dry cells"
            )
        if (s.h[:, :1][wet] < self.min_liquid).any():
            raise ValueError("top liquid layer too thin")
        target = self.thermal_energy(initial) + dt * q
        zero = torch.zeros_like(m0)
        melted = self.phase_state(initial, zero)
        melt_energy = self.thermal_energy(melted)
        frozen = wet & (target < melt_energy)
        floor = torch.maximum(
            torch.full_like(m0, self.min_liquid),
            s.salt_h[:, :1] / (42 * (1 - 64 * torch.finfo(s.h.dtype).eps)),
        )
        upper = torch.where(
            wet, torch.maximum(m0 + s.column.rho0 * (s.h[:, :1] - floor), zero), zero
        )
        if (frozen & (target < self.root_energy(initial, upper))).any():
            raise ValueError("cooling exhausts liquid/EOS bracket")
        lower = zero
        with torch.no_grad():
            for _ in range(56):
                mid = 0.5 * (lower + upper)
                warm = self.root_energy(initial, mid) > target
                lower = torch.where(warm, mid, lower)
                upper = torch.where(warm, upper, mid)
        m = torch.where(frozen, 0.5 * (lower + upper), zero).detach()
        for _ in range(3):
            state = self.phase_state(initial, m)
            derivative = self.derivative(initial, state, m)
            if (derivative[wet] >= 0).any():
                raise ValueError("nonmonotone phase-energy relation")
            m = torch.where(
                frozen, m - (self.thermal_energy(state) - target) / derivative, zero
            )
        if (m < 0).any() or not torch.isfinite(m).all():
            raise ValueError("inadmissible phase root")
        end = self.phase_state(initial, m)
        t = end.ocean.ocean
        extra = torch.where(wet & ~frozen, target - melt_energy, zero)
        heat = torch.cat(
            (extra / (s.column.rho0 * t.h[:, :1]), torch.zeros_like(t.h[:, 1:])), -1
        )
        physical, _, _ = RoquetWorkspace(H.model.ocean, t).heat(
            t, t, heat, allow_cooling=True
        )
        end = replace(end, ocean=replace(end.ocean, ocean=physical))
        end, gain, momentum = self.exchange_momentum(initial, end)
        self.coupled.check(end)
        area = s.mesh.cell_area[:, None]
        external = (area * dt * q)[:, 0]
        energy = sum(H.terms(end).values()) - sum(H.terms(initial).values()) - external
        water = (
            area
            * (
                s.column.rho0 * (end.ocean.ocean.h - s.h).sum(-1, keepdim=True)
                + H.mass(end)
                - m0
            )
        )[:, 0]
        return end, CategoryPhaseReport(external, energy, water, momentum, gain)

    def air_flux(self, state, air_temperature, *, air_conductance=10.0):
        """Category-weighted zero-layer conduction plus open-water heat exchange."""
        s = state.ocean.ocean
        p = self.H.slab.parameters
        if (
            air_temperature.shape != s.h[:, :1].shape
            or not torch.isfinite(air_temperature).all()
            or not math.isfinite(air_conductance)
            or air_conductance <= 0
        ):
            raise ValueError(
                "finite [cell,1] air temperature and positive conductance required"
            )
        base = p.freezing_temperature(s.salinity()[:, :1])
        resistance = state.ice.thickness() / p.conductivity
        q = (air_temperature - base) / (1 / air_conductance + resistance)
        surface = torch.minimum(base + q * resistance, torch.zeros_like(resistance))
        covered = (state.ice.area * air_conductance * (air_temperature - surface)).sum(
            -1, keepdim=True
        )
        open_water = (
            (1 - state.ice.area.sum(-1, keepdim=True))
            * air_conductance
            * (air_temperature - s.temperature()[:, :1])
        )
        return torch.where(
            self.H.model.energy.mass_area(s) > 0, covered + open_water, 0.0
        )
