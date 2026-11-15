"""Stationary, full-cell zero-layer ice coupled to the actual Roquet Hamiltonian.

Ice is fresh and has latent, but no sensible, heat storage. Horizontal dynamics,
partial coverage, snow, ice momentum and rheology are deliberately unsupported.
The moving lower ice boundary follows the liquid column's top. No ocean or ice
state is clipped or repaired after the thermodynamic solve.
"""

from dataclasses import dataclass, replace
import math
import torch
from vfs.ocean.stratified import TKEState
from vfs.ocean.stratified.energy_columns import column_terms
from eddy_energy import EddyState
from roquet import RoquetOcean, RoquetWorkspace


@dataclass(frozen=True)
class IceParameters:
    density: float = 917.0  # kg/m3
    latent_heat: float = 3.34e5  # J/kg, liquid/solid energy datum difference
    freezing_slope: float = 0.054  # degC/(g/kg), idealized CT liquidus
    conductivity: float = 2.1  # W/(m K)

    def __post_init__(self):
        if not all(
            math.isfinite(x) and x > 0
            for x in (
                self.density,
                self.latent_heat,
                self.freezing_slope,
                self.conductivity,
            )
        ):
            raise ValueError("positive finite ice parameters required")

    def freezing_temperature(self, salinity):
        return -self.freezing_slope * salinity


@dataclass(frozen=True)
class SeaIceState:
    mass: torch.Tensor  # kg/m2 of complete horizontal cell; pure fresh ice

    def thickness(self, parameters):
        return self.mass / parameters.density


@dataclass(frozen=True)
class IceOceanState:
    ocean: TKEState  # includes mesoscale reservoir if EddyState
    ice: SeaIceState


class IceHamiltonian:
    """H = existing ocean H + A[-Lf*m + g*m*eta + g*m^2/(2*rho_i)].

    Fully covered planar columns: ice bottom at eta, centre at eta+h_i/2.
    The derivative dH_ice/deta = A*g*m is the future ocean loading term;
    it must enter horizontal dynamics before this column model can be coupled
    to a moving ocean. The current API therefore rejects nonzero velocities.
    """

    def __init__(self, ocean_model, parameters=IceParameters()):
        if not isinstance(ocean_model, RoquetOcean):
            raise TypeError("matched RoquetOcean required")
        self.model, self.parameters = ocean_model, parameters

    def column_terms(self, state):
        s = state.ocean.ocean
        terms, _ = column_terms(self.model.energy, s)
        mass = self.model.energy.mass_area(s)
        terms["turbulent_kinetic"] = (mass * state.ocean.tke_h).sum(-1)
        terms["mesoscale_eddy"] = (
            (mass * state.ocean.eddy_h).sum(-1)
            if isinstance(state.ocean, EddyState)
            else torch.zeros_like(terms["kinetic"])
        )
        area = s.mesh.cell_area[:, None]
        m = state.ice.mass
        g = s.column.gravity
        eta = s.column.eta(s.h)
        terms["ice_latent"] = (-area * self.parameters.latent_heat * m).squeeze(-1)
        terms["ice_gravitational"] = (
            area * g * (m * eta + m.square() / (2 * self.parameters.density))
        ).squeeze(-1)
        return terms

    def total_energy(self, state):
        return sum(x.sum() for x in self.column_terms(state).values())

    def energy_per_area(self, state):
        return (
            sum(self.column_terms(state).values()) / state.ocean.ocean.mesh.cell_area
        )[:, None]

    def phase_mass_derivative(self, state):
        # Along CT=-mu*SA and constant salt_h, top temp_h=-mu*salt_h
        # is constant. h changes by -dm/rho0. Thus no cp term is missing.
        s = state.ocean.ocean
        rho = s.column.rho0
        g = s.column.gravity
        area = s.mesh.cell_area[:, None]
        m = state.ice.mass
        dh = self.model.energy.base.gradients(s)["h"][:, :1] / area
        return (
            -self.parameters.latent_heat
            + g * s.column.eta(s.h)
            + g * m / self.parameters.density
            - (dh + g * m) / rho
        )


@dataclass(frozen=True)
class IceStepReport:
    external_heat: torch.Tensor  # J/cell
    ice_mass_change: torch.Tensor  # kg/cell, positive freezing
    ocean_energy_change: torch.Tensor  # J/cell
    ice_energy_change: torch.Tensor  # J/cell
    energy_residual: torch.Tensor  # J/cell
    water_residual: torch.Tensor  # kg/cell
    salt_residual: torch.Tensor  # kg/cell


class SeaIceThermodynamics:
    """Solve phase equilibrium AND full-H energy together, independently per cell.

    If ice survives: CT_top = -mu*SA_top. If all ice melts: mass=0 and
    surplus heat warms the ocean with the existing full-H quadratic solve.
    Salt, passive amounts, TKE and eddy contents stay in the liquid. The phase
    law is an explicit idealization, not a TEOS chemical-equilibrium derivation.
    """

    def __init__(self, ocean_model, parameters=IceParameters(), *, min_liquid=1.0):
        if not math.isfinite(min_liquid) or min_liquid <= 0:
            raise ValueError("positive finite minimum liquid thickness required")
        self.hamiltonian = IceHamiltonian(ocean_model, parameters)
        self.min_liquid = float(min_liquid)

    def check(self, state):
        if not isinstance(state, IceOceanState):
            raise TypeError("IceOceanState required")
        H = self.hamiltonian
        s = state.ocean.ocean
        H.model.check(state.ocean)
        m = state.ice.mass
        if (
            m.shape != s.h[:, :1].shape
            or m.dtype != s.h.dtype
            or m.device != s.h.device
        ):
            raise ValueError("ice mass must match top-layer shape/dtype/device")
        if not torch.isfinite(m).all() or (m < 0).any():
            raise ValueError("finite nonnegative ice mass required")
        wet = H.model.energy.mass_area(s) > 0
        if (m[~wet] != 0).any():
            raise ValueError("ice on dry cells is unsupported")
        if (s.u_e != 0).any():
            raise ValueError(
                "stationary columns only: moving ocean needs ice loading and momentum coupling"
            )
        if (s.h[:, :1][wet] < self.min_liquid).any():
            raise ValueError("top liquid layer too thin")

    def apply(self, state, dt, *, heat_flux):
        self.check(state)
        if not math.isfinite(float(dt)) or dt <= 0:
            raise ValueError("positive finite dt required")
        H = self.hamiltonian
        p = H.parameters
        s = state.ocean.ocean
        area = s.mesh.cell_area[:, None]
        rho = s.column.rho0
        wet = H.model.energy.mass_area(s) > 0
        q = heat_flux
        if (
            not isinstance(q, torch.Tensor)
            or q.shape != state.ice.mass.shape
            or q.dtype != s.h.dtype
            or q.device != s.h.device
        ):
            raise ValueError("heat flux must match ice mass shape/dtype/device")
        if not torch.isfinite(q).all() or (q[~wet] != 0).any():
            raise ValueError("finite heat flux, zero on dry cells required")
        initial = H.energy_per_area(state)
        target = initial + dt * q
        old_mass = state.ice.mass

        def phase(m):
            top = s.h[:, :1] + (old_mass - m) / rho
            h = torch.cat((top, s.h[:, 1:]), -1)
            top_content = torch.where(
                wet, -p.freezing_slope * s.salt_h[:, :1], s.temp_h[:, :1]
            )
            physical = s.with_tensors(
                s.u_e,
                h,
                torch.cat((top_content, s.temp_h[:, 1:]), -1),
                s.salt_h,
                s.tracer_h,
            )
            return IceOceanState(replace(state.ocean, ocean=physical), SeaIceState(m))

        zero = torch.zeros_like(old_mass)
        melted = phase(zero)
        melt_energy = H.energy_per_area(melted)
        frozen = wet & (target < melt_energy)
        # Stay inside the existing ocean EOS salinity range and positive h.
        salt_floor = s.salt_h[:, :1] / (42.0 * (1 - 64 * torch.finfo(s.h.dtype).eps))
        floor = torch.maximum(torch.full_like(salt_floor, self.min_liquid), salt_floor)
        # Within roundoff of the EOS salinity ceiling, a zero-ice state still permits
        # warming/melting but has no freezing bracket. This bounds a search
        # interval, not a prognostic ice or water store.
        upper = torch.where(
            wet, torch.maximum(old_mass + rho * (s.h[:, :1] - floor), zero), zero
        )
        lower = zero
        if (frozen & (target < H.energy_per_area(phase(upper)))).any():
            raise ValueError(
                "cooling exhausts liquid/EOS phase bracket; reduce forcing or dt"
            )
        # Bracketing is a root search, not a differentiable iteration. Analytic
        # Newton steps below restore implicit sensitivities on a fixed branch.
        with torch.no_grad():
            for _ in range(56):
                mid = (lower + upper) * 0.5
                too_warm = H.energy_per_area(phase(mid)) > target
                lower = torch.where(too_warm, mid, lower)
                upper = torch.where(too_warm, upper, mid)
        m = torch.where(frozen, (lower + upper) * 0.5, zero).detach()
        for _ in range(3):
            candidate = phase(m)
            derivative = H.phase_mass_derivative(candidate)
            if (derivative[wet] >= 0).any():
                raise ValueError("nonmonotone phase-energy relation")
            correction = (H.energy_per_area(candidate) - target) / derivative
            m = torch.where(frozen, m - correction, zero)
        if (m < 0).any() or not torch.isfinite(m).all():
            raise ValueError("inadmissible phase root")
        end = phase(m)
        extra = torch.where(wet & ~frozen, target - melt_energy, zero)
        physical = end.ocean.ocean
        heat = torch.cat(
            (extra / (rho * physical.h[:, :1]), torch.zeros_like(physical.h[:, 1:])), -1
        )
        heated, _, _ = RoquetWorkspace(H.model.ocean, physical).heat(
            physical, physical, heat, allow_cooling=True
        )
        end = IceOceanState(replace(end.ocean, ocean=heated), end.ice)
        self.check(end)
        before_terms = H.column_terms(state)
        after_terms = H.column_terms(end)
        ice_keys = ("ice_latent", "ice_gravitational")
        ice_change = sum(after_terms[k] - before_terms[k] for k in ice_keys)
        ocean_change = sum(
            after_terms[k] - before_terms[k] for k in before_terms if k not in ice_keys
        )
        external = (area * dt * q).squeeze(-1)
        water = (
            area * (rho * (heated.h - s.h).sum(-1, keepdim=True) + m - old_mass)
        ).squeeze(-1)
        salt = (
            area * rho * ((heated.salt_h - s.salt_h) / 1000).sum(-1, keepdim=True)
        ).squeeze(-1)
        return end, IceStepReport(
            external,
            (area * (m - old_mass)).squeeze(-1),
            ocean_change,
            ice_change,
            ocean_change + ice_change - external,
            water,
            salt,
        )


@dataclass(frozen=True)
class ConductiveIceBoundary:
    """Zero heat-capacity slab plus finite air-side resistance; W/m2 into system.

    Prescribed air temperature and conductance are idealized forcing, not a
    bulk atmospheric boundary layer. Ice surface cannot exceed 0 C; positive
    surplus then supplies surface melt. No snow, radiation or sensible ice store.
    """

    parameters: IceParameters = IceParameters()
    air_conductance: float = 10.0  # W/(m2 K)

    def __post_init__(self):
        if not math.isfinite(self.air_conductance) or self.air_conductance <= 0:
            raise ValueError("positive finite air conductance required")

    def flux(self, state, air_temperature):
        s = state.ocean.ocean
        m = state.ice.mass
        if (
            air_temperature.shape != m.shape
            or air_temperature.dtype != m.dtype
            or air_temperature.device != m.device
            or not torch.isfinite(air_temperature).all()
        ):
            raise ValueError("air temperature must match ice mass and be finite")
        base = self.parameters.freezing_temperature(s.salinity()[:, :1])
        resistance = state.ice.thickness(self.parameters) / self.parameters.conductivity
        q = (air_temperature - base) / (1 / self.air_conductance + resistance)
        surface = torch.minimum(base + q * resistance, torch.zeros_like(base))
        covered = self.air_conductance * (air_temperature - surface)
        open_water = self.air_conductance * (air_temperature - s.temperature()[:, :1])
        return torch.where(m > 0, covered, open_water)
