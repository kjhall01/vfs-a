"""Compression-only ice stress and two-category redistribution.

An isolated mechanics stage: no advection, ocean dynamics, phase change, drag,
wind, Coriolis or shear rheology. All tendencies are instantaneous. RK errors
are left visible; no finite-step energy repair or prognostic clipping occurs.
"""

from dataclasses import dataclass, replace
import math
import torch
from vfs.ice.ice import IceParameters, IceHamiltonian, IceOceanState, SeaIceState
from vfs.ocean.temperature.roquet import RoquetWorkspace
from vfs.ocean.energy import ConstantDensityEnergy
from vfs.ocean.stratified import TKEState


@dataclass(frozen=True)
class CategoryIce:
    area: torch.Tensor  # [cell,2], fractions of horizontal cell area
    volume: torch.Tensor  # [cell,2], m3 ice per m2 cell area
    velocity: torch.Tensor  # [edge,1], common ice edge-normal velocity, m/s

    def thickness(self):
        denominator = torch.where(self.area > 0, self.area, torch.ones_like(self.area))
        return self.volume / denominator


@dataclass(frozen=True)
class RidgingState:
    ocean: TKEState  # also accepts the existing EddyState subclass
    ice: CategoryIce


class RidgingHamiltonian:
    """Existing full ocean/ice H + ice KE + subcell hydrostatic variance energy.

    V=sum V_k, S=sum V_k^2/a_k, C=rho_i*g*(1-rho_i/rho0)/2.
    H_extra=sum A*C*(S-V^2). Together with the existing slab ice PE,
    this represents hydrostatic ice keels and displaced reference-density water
    beneath different thickness categories, at fixed mean liquid height eta.
    Subcell stratification is neglected. A single fully covered category reduces
    exactly to the earlier slab functional. This is an explicit model extension.
    """

    def __init__(self, model, parameters=IceParameters()):
        self.slab = IceHamiltonian(model, parameters)
        if parameters.density >= model.ocean.base.column.rho0:
            raise ValueError(
                "floating ice requires density below reference water density"
            )

    @property
    def model(self):
        return self.slab.model

    def coefficient(self, state):
        column = state.ocean.ocean.column
        rho = self.slab.parameters.density
        return 0.5 * rho * column.gravity * (1 - rho / column.rho0)

    def mass(self, state):
        return self.slab.parameters.density * state.ice.volume.sum(-1, keepdim=True)

    def metric(self, state):
        mesh = state.ocean.ocean.mesh
        return (mesh.dc_edge * mesh.dv_edge)[:, None] * mesh.cell_to_edge(
            self.mass(state)
        )

    def terms(self, state):
        s, ice = state.ocean.ocean, state.ice
        terms = self.slab.column_terms(
            IceOceanState(state.ocean, SeaIceState(self.mass(state)))
        )
        second = (ice.volume * ice.thickness()).sum(-1)
        mean = ice.volume.sum(-1)
        terms["ridge_potential"] = (
            s.mesh.cell_area * self.coefficient(state) * (second - mean.square())
        )
        terms["ice_kinetic"] = 0.5 * ConstantDensityEnergy.interpolation_transpose(
            s.mesh, self.metric(state) * ice.velocity.square()
        ).squeeze(-1)
        return terms

    def total_energy(self, state):
        return sum(v.sum() for v in self.terms(state).values())


@dataclass(frozen=True)
class RidgingTendency:
    area: torch.Tensor
    volume: torch.Tensor
    velocity: torch.Tensor
    temp_h: torch.Tensor


@dataclass(frozen=True)
class RidgingReport:
    compression: torch.Tensor  # [cell,1], s^-1
    pressure: torch.Tensor  # [cell,1], vertically integrated pressure, N/m
    transfer: torch.Tensor  # [cell,1], thin -> thick volume, m/s
    kinetic_power: torch.Tensor  # W/cell (may have either local sign)
    potential_power: torch.Tensor  # W/cell, >=0
    heating_power: torch.Tensor  # W/cell, >=0
    energy_redistribution: torch.Tensor  # W/cell, conservative stress-work transport


class CompressionRidging:
    """A testable constitutive prototype, not VP/EVP or a coverage limiter.

    c=max(-div u,0), chi=a^crowding_power*c/(c+strain_rate), F=chi*c*V_thin.
    Transfer F at unchanged donor/receiver thickness. A new thick category is
    born at ridge_ratio*donor thickness. Its directional derivative is used
    at birth; backward differentiation across birth is not claimed.

    P=(1+friction)*C*chi*V_thin*(h_thick-h_thin)+bulk_viscosity*chi*a*c
    during compression, zero otherwise. Thus P*c funds ridge PE plus nonnegative
    frictional heat. Friction and viscosity are uncalibrated constitutive choices.
    """

    def __init__(
        self,
        model,
        parameters=IceParameters(),
        *,
        ridge_ratio=3.0,
        friction=1.0,
        bulk_viscosity=0.0,
        crowding_power=4.0,
        strain_rate=1e-6,  # s^-1; regularizes resistance near zero compression
    ):
        values = (ridge_ratio, friction, bulk_viscosity, crowding_power, strain_rate)
        if (
            not all(math.isfinite(x) for x in values)
            or ridge_ratio <= 1
            or min(values[1:]) < 0
            or strain_rate <= 0
        ):
            raise ValueError("require ratio>1 and finite nonnegative coefficients")
        self.hamiltonian = RidgingHamiltonian(model, parameters)
        self.ridge_ratio, self.friction = ridge_ratio, friction
        self.bulk_viscosity, self.crowding_power = bulk_viscosity, crowding_power
        self.strain_rate = strain_rate

    def active_edges(self, state):
        H = self.hamiltonian
        active = H.metric(state) > 0
        domain = H.model.ocean.domain
        if domain is not None:
            active = active & domain.active_edge[:, None]
        return active

    def check(self, state):
        H = self.hamiltonian
        H.model.check(state.ocean)
        s, ice = state.ocean.ocean, state.ice
        for x, shape in [
            (ice.area, (s.mesh.nCells, 2)),
            (ice.volume, (s.mesh.nCells, 2)),
            (ice.velocity, (s.mesh.nEdges, 1)),
        ]:
            if (
                x.shape != shape
                or x.dtype != s.h.dtype
                or x.device != s.h.device
                or not torch.isfinite(x).all()
            ):
                raise ValueError(
                    "ice tensors must have matching shape, dtype, device and finite values"
                )
        if (
            (ice.area < 0).any()
            or (ice.volume < 0).any()
            or (ice.area.sum(-1) > 1).any()
        ):
            raise ValueError("nonnegative stores and total coverage <=1 required")
        if ((ice.area == 0) != (ice.volume == 0)).any():
            raise ValueError("empty categories require both zero area and zero volume")
        h = ice.thickness()
        both = (ice.area > 0).all(-1)
        if (h[both, 1] < h[both, 0]).any():
            raise ValueError("receiver category must be at least as thick as donor")
        wet = H.model.energy.mass_area(s)[:, 0] > 0
        if (ice.area[~wet] != 0).any() or (ice.volume[~wet] != 0).any():
            raise ValueError("dry cells cannot contain ice")
        if (ice.velocity[~self.active_edges(state)] != 0).any():
            raise ValueError("zero ice velocity required on massless or closed edges")

    def tendency(self, state, *, _validated=False):
        # The coupled RHS has already checked this exact, unmodified state.
        # Public standalone calls always retain the complete check.
        if not _validated:
            self.check(state)
        H = self.hamiltonian
        s, ice = state.ocean.ocean, state.ice
        mesh = s.mesh
        h = ice.thickness()
        donor = h[:, :1]
        receiver = torch.where(ice.area[:, 1:] > 0, h[:, 1:], self.ridge_ratio * donor)
        coverage = ice.area.sum(-1, keepdim=True)
        compression = (-mesh.div(ice.velocity)).clamp_min(0)
        chi = (
            coverage.pow(self.crowding_power)
            * compression
            / (compression + self.strain_rate)
        )
        transfer = chi * compression * ice.volume[:, :1]
        safe = lambda x: torch.where(x > 0, x, torch.ones_like(x))
        da = torch.cat((-transfer / safe(donor), transfer / safe(receiver)), -1)
        dv = torch.cat((-transfer, transfer), -1)
        cost = H.coefficient(state) * chi * ice.volume[:, :1] * (receiver - donor)
        viscous = self.bulk_viscosity * chi * coverage * compression
        pressure = torch.where(
            compression > 0,
            (1 + self.friction) * cost + viscous,
            torch.zeros_like(cost),
        )
        metric = H.metric(state)
        mass_edge = mesh.cell_to_edge(H.mass(state))
        acceleration = torch.where(
            self.active_edges(state), -mesh.grad(pressure) / safe(mass_edge), 0.0
        )
        potential = cost * compression
        heating = (self.friction * cost + viscous) * compression
        # Full-H caloric conjugate: warming also changes Roquet potential energy.
        workspace = RoquetWorkspace(H.model.ocean, s)
        capacity = (
            workspace.reference.cp_theta
            + 2 * workspace.a * s.temperature()
            + workspace.b
        )
        if (capacity <= 0).any():
            raise ValueError("positive full-H thermal conjugate required")
        dtemp = torch.cat(
            (heating / (s.column.rho0 * capacity[:, :1]), torch.zeros_like(s.h[:, 1:])),
            -1,
        )
        area = mesh.cell_area
        kinetic = ConstantDensityEnergy.interpolation_transpose(
            mesh, metric * ice.velocity * acceleration
        ).squeeze(-1)
        potential = area * potential[:, 0]
        heating = area * heating[:, 0]
        redistribution = kinetic + potential + heating
        return RidgingTendency(da, dv, acceleration, dtemp), RidgingReport(
            compression, pressure, transfer, kinetic, potential, heating, redistribution
        )

    @staticmethod
    def increment(state, rhs, dt):
        s, ice = state.ocean.ocean, state.ice
        physical = s.with_tensors(
            s.u_e, s.h, s.temp_h + dt * rhs.temp_h, s.salt_h, s.tracer_h
        )
        return RidgingState(
            replace(state.ocean, ocean=physical),
            CategoryIce(
                ice.area + dt * rhs.area,
                ice.volume + dt * rhs.volume,
                ice.velocity + dt * rhs.velocity,
            ),
        )

    def step(self, state, dt):
        """Isolated RK4 mechanics experiment; every stage is checked, never repaired."""
        if not math.isfinite(dt) or dt <= 0:
            raise ValueError("positive finite timestep required")
        k1, _ = self.tendency(state)
        k2, _ = self.tendency(self.increment(state, k1, 0.5 * dt))
        k3, _ = self.tendency(self.increment(state, k2, 0.5 * dt))
        k4, _ = self.tendency(self.increment(state, k3, dt))
        avg = RidgingTendency(
            *(
                sum(
                    w * getattr(k, name)
                    for w, k in [(1, k1), (2, k2), (2, k3), (1, k4)]
                )
                / 6
                for name in ("area", "volume", "velocity", "temp_h")
            )
        )
        end = self.increment(state, avg, dt)
        self.check(end)
        return end
